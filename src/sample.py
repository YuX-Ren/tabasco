import torch
import argparse
import pickle
import lightning as L
from tabasco.callbacks.ema import EMAOptimizer
from tabasco.models.lightning_tabasco import LightningTabasco
from tabasco.chem.convert import MoleculeConverter
from tensordict import TensorDict
from tabasco.data.lmdb_datamodule import LmdbDataModule
import datamol as dm
torch.set_float32_matmul_precision("high")
L.seed_everything(42)
import numpy as np
import tqdm
from pymatgen.analysis.structure_matcher import StructureMatcher
from tabasco.chem.crystal_matcher import array_dict_to_crystal
# Manually setting the configuration dictionary (cfg)
cfg = {
    "data_dir": "./data/processed_mp20_train.pt",
    "val_data_dir": "./data/processed_mp20_val.pt",
    "test_data_dir": "./data/processed_mp20_test.pt",
    "lmdb_dir": "./data/lmdb_mp_20",
    "add_random_rotation": True,
    "add_random_permutation": False,
    "reorder_to_smiles_order": True,
    "remove_hydrogens": True,
    "batch_size": 256,
    "num_workers": 0,
    "train_materials": True
}

def batch_frac_to_cart_coords_with_lattice(
    frac_coords: torch.Tensor, lattice: torch.Tensor
) -> torch.Tensor:
    num_atoms = frac_coords.shape[1]
    lattice_nodes = torch.repeat_interleave(lattice.unsqueeze(1), num_atoms, dim=1)
    pos = torch.einsum("bni,bnij->bnj", frac_coords, lattice_nodes)  # cart coords
    return pos

def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)

def lattice_matrix_to_params(matrix):
    lengths = np.sqrt(np.sum(matrix ** 2, axis=1))

    angles = np.zeros(3)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[i] = abs_cap(np.dot(matrix[j], matrix[k]) /
                            (lengths[j] * lengths[k]))
    angles = np.arccos(angles) * 180.0 / np.pi
    return lengths, angles

def compute_volume(batch_lattice):
    """Compute volume from batched lattice matrix

    batch_lattice: (N, 3, 3)
    """
    vector_a, vector_b, vector_c = torch.unbind(batch_lattice, dim=1)
    return torch.abs(torch.einsum('bi,bi->b', vector_a,
                                  torch.cross(vector_b, vector_c, dim=1)))

def kabsch_algorithm(P, Q):
    """
    Kabsch algorithm to align two sets of points P and Q using numpy.
    
    P: numpy.ndarray, shape (N, 3) - Reference structure (aligned)
    Q: numpy.ndarray, shape (N, 3) - Generated structure to align
    
    Returns: 
        R: numpy.ndarray, shape (3, 3) - The optimal rotation matrix
        aligned_Q: numpy.ndarray, shape (N, 3) - The aligned Q
    """
    # Step 1: Compute centroids of both point clouds
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    # Step 2: Center the coordinates (subtract centroids)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # Step 3: Compute the covariance matrix
    H = np.dot(P_centered.T, Q_centered)
    
    # Step 4: Singular Value Decomposition (SVD)
    U, _, Vt = np.linalg.svd(H)
    
    # Step 5: Calculate the optimal rotation matrix R
    R = np.dot(Vt.T, U.T)
    
    # Step 6: If the determinant is negative, the rotation matrix is a reflection, so we flip the sign of Vt
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    # Step 7: Apply the rotation to Q
    aligned_Q = np.dot(Q_centered, R.T) + centroid_P  # Recenter to the original centroid
    
    return R, aligned_Q

def compute_rmsd_with_kabsch(batch, out_batch, materials_match = False):
    """
    Compute the RMSD with Kabsch alignment, ignoring padded atoms.
    
    batch: dict, contains the reference coordinates
    out_batch: dict, contains the generated coordinates
    
    Returns:
        rmsd: float - The RMSD after alignment
    """
    coords_ref = batch["coords"]
    coords_gen = out_batch["coords"]
    lattices_ref = batch["lattices"]
    lattices_gen = out_batch["lattices"]
    coords_ref = batch_frac_to_cart_coords_with_lattice(coords_ref, lattices_ref)
    coords_gen = batch_frac_to_cart_coords_with_lattice(coords_gen, lattices_gen)
    real_mask = ~out_batch["padding_mask"]  # ~mask to get True for valid atoms

    # Ensure coordinates are on the same device (e.g., CUDA)
    coords_ref = coords_ref.cpu().numpy()
    coords_gen = coords_gen.cpu().numpy()
    real_mask = real_mask.cpu().numpy()

    # Initialize RMSD accumulator
    rmsds = 0.0

    # Iterate through each molecule in the batch
    for i in range(coords_ref.shape[0]):
        # Extract individual molecule's coords (ignoring padding mask)
        ref_coords = coords_ref[i][real_mask[i]]
        gen_coords = coords_gen[i][real_mask[i]]
        if ref_coords.shape[0] > 0:  # Ensure there are valid atoms
            # Apply Kabsch alignment to this pair of molecules
            _, aligned_gen_coords = kabsch_algorithm(ref_coords, gen_coords)
            # Calculate the squared differences between the aligned coords
            sq_diff = np.mean((aligned_gen_coords - ref_coords)**2, axis=-1)

            # Average over the valid (non-masked) atoms
            per_molecule_rmsd = np.sqrt(np.mean(sq_diff))
            # print(per_molecule_rmsd)
            # Accumulate RMSD over all molecules
            if materials_match:
                per_molecule_rmsd = per_molecule_rmsd * (len(ref_coords) / compute_volume(lattices_ref[i])) ** (1/3)
            rmsds += per_molecule_rmsd.item()
            
    # Return average RMSD over all valid molecules in the batch
    return rmsds

def sample_batch(
    lightning_module: L.LightningModule,
    batch: TensorDict,
    ema_optimizer: EMAOptimizer | None,
    num_steps: int,
) -> TensorDict:
    """Generate a batch of molecules.

    Args:
        lightning_module: Loaded PocketSynth checkpoint.
        batch: Batch of molecules to sample from.
        ema_optimizer: Optional EMA wrapper; if given, swaps in EMA weights.
        num_steps: Number of diffusion steps per trajectory.

    Returns:
        TensorDict with keys `coords`, `atomics`, `padding_mask`.
    """

    if ema_optimizer is None:
        with torch.no_grad():
            out_batch = lightning_module.sample(
                batch=batch, num_steps=num_steps
            )
    else:
        with torch.no_grad() and ema_optimizer.swap_ema_weights():
            out_batch = lightning_module.sample(
                batch=batch, num_steps=num_steps
            )

    return out_batch


def export_batch_to_pickle(out_batch: TensorDict, out_path: str):
    """Serialize generated molecules and basic metrics.

    Saves two files: `<out_path>.pkl` containing a Python list of RDKit
    molecules (with `None` for invalid ones) and `<out_path>.sdf` containing
    only the valid molecules.
    """
    mol_converter = MoleculeConverter()
    generated_mols = mol_converter.from_batch(out_batch, sanitize=False)

    print(f"Saving generated mols to {out_path}...")
    with open(out_path, "wb") as f:
        pickle.dump(generated_mols, f)

def export_batch_to_sdf(out_batch: TensorDict, out_path: str):
    """Serialize generated molecules and basic metrics.
    """
    mol_converter = MoleculeConverter()
    
    generated_mols = mol_converter.from_batch(out_batch, sanitize=False)
    out_path = out_path.replace(".pkl", ".sdf")
    dm.to_sdf(generated_mols, urlpath=out_path)

def parse_args():
    """Return CLI arguments parsed with `argparse`."""
    parser = argparse.ArgumentParser(description="Run PocketSynth generation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--num_mols",
        type=int,
        default=100,
        help="Number of molecules to generate (default: 100)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of steps to generate (default: 100)",
    )
    parser.add_argument(
        "--ema_strength", type=float, default=None, help="EMA strength (default: None)"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=None,
        help="Path to the output file ending in .pkl (default: None)",
    )
    return parser.parse_args()


def main():
    """Main entry-point: parse args, load model, sample, export."""
    matcher = StructureMatcher(stol=0.2, angle_tol=5, ltol=0.2)
    args = parse_args()
    num_mols = args.num_mols
    num_steps = args.num_steps

    # Load the PocketSynth model checkpoint
    lightning_module = LightningTabasco.load_from_checkpoint(args.checkpoint)

    # Initialize datamodule manually using cfg
    datamodule = LmdbDataModule(
        **cfg
    )
    datamodule.setup()
    lightning_module.model.net.eval()

    if torch.cuda.is_available():
        lightning_module.to("cuda")

    # EMA optimizer (if specified)
    if args.ema_strength is not None:
        adam_opt = torch.optim.Adam(lightning_module.model.net.parameters())
        ema_optimizer = EMAOptimizer(
            adam_opt,
            "cuda" if torch.cuda.is_available() else "cpu",
            args.ema_strength,
        )
    else:
        ema_optimizer = None

    out_batch_list = []

    # Sampling from validation data loader
    rmsds = 0
    num_graphs = 0
    idx = 0
    for batch in tqdm.tqdm(datamodule.test_dataloader()):
        batch = batch.to("cuda")
        out_batch = sample_batch(
            lightning_module=lightning_module,
            batch=batch,
            ema_optimizer=ema_optimizer,
            num_steps=num_steps,
        )
        out_batch_list.append(out_batch)

        # for out_mol, batch_mol in zip(out_batch, batch):
        #     try:
        #         out_atom_types = lightning_module.mol_converter.get_atom_types_from_tensor(out_mol)
        #         batch_atom_types = lightning_module.mol_converter.get_atom_types_from_tensor(batch_mol)
        #         if out_atom_types != batch_atom_types:
        #             print(f"out atom types: {out_atom_types}, batch atom types: {batch_atom_types}")
        #         lengths, angles = lattice_matrix_to_params(batch_mol["lattices"].detach().cpu().numpy())
        #         batch_mol = array_dict_to_crystal({
        #             "atom_types": np.array(batch_atom_types),
        #             "frac_coords": batch_mol["coords"][~batch_mol["padding_mask"]].detach().cpu().numpy(),
        #             "lengths": lengths,
        #             "angles": angles,
        #             "sample_idx": idx,
        #         })
        #         out_lengths, out_angles = lattice_matrix_to_params(out_mol["lattices"].detach().cpu().numpy())
        #         out_mol = array_dict_to_crystal({
        #             "atom_types": np.array(out_atom_types),
        #             "frac_coords": out_mol["coords"][~out_mol["padding_mask"]].detach().cpu().numpy(),
        #             "lengths": out_lengths,
        #             "angles": out_angles,
        #             "sample_idx": idx,
        #         })
        #         rmsd = matcher.get_rms_dist(batch_mol.structure, out_mol.structure)
        #         print(f"rmsd: {rmsd}")
        #     except Exception as e:
        #         print(f"error: {e}")


        # calculate the rmsd between the generated and reference molecules
        # for mol, target_mol in zip(batch,out_batch):
        per_batch_rmsd = compute_rmsd_with_kabsch(batch, out_batch)
        # sq_diff = (out_batch["coords"] - batch["coords"]).pow(2).sum(dim=-1)
        # sq_diff = sq_diff * (~out_batch["padding_mask"])
        # per_batch_mse = sq_diff.sum(dim=-1)/(~out_batch["padding_mask"]).sum(dim=-1)
        # rmsd = per_batch_mse.sqrt().mean()
        num_graphs += batch['padding_mask'].shape[0]
        rmsds += per_batch_rmsd
        for out_mol, batch_mol in zip(out_batch, batch):
            out_atom_types = lightning_module.mol_converter.get_atom_types_from_tensor(out_mol)
            batch_atom_types = lightning_module.mol_converter.get_atom_types_from_tensor(batch_mol)
            print(f"out atom types: {out_atom_types},\n batch atom types: {batch_atom_types}\n\n",file=open("atom_types.txt", "a"))
            if out_atom_types != batch_atom_types:
                print(f"out atom types: {out_atom_types}, batch atom types: {batch_atom_types}")
        # break
    # rmsds /= len(datamodule.test_dataloader())
    rmsd = rmsds/num_graphs
    print(f"rmsd: {rmsd}")
    mse = out_batch["coords"] - batch["coords"]
    mse = mse.pow(2).mean()
    print(f"mse: {mse}")
    lattice_mse = out_batch["lattices"] - batch["lattices"]
    lattice_mse = lattice_mse.pow(2).mean()
    print(f"lattice_mse: {lattice_mse}")
    # compare the generated and reference molecules by atom type

    # Concatenate results from all batches
    out_batch = torch.cat(out_batch_list, dim=0)

    # Export the sampled batch to pickle if specified
    # if args.output_path is not None:
    #     export_batch_to_sdf(batch, args.output_path.replace(".pkl", "_ref.pkl"))
    #     export_batch_to_sdf(out_batch, args.output_path)


if __name__ == "__main__":
    main()
