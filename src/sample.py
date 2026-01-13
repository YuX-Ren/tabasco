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

# Manually setting the configuration dictionary (cfg)
cfg = {
    "data_dir": "./data/processed_qm9_train.pt",
    "val_data_dir": "./data/processed_qm9_val.pt",
    "test_data_dir": "./data/processed_qm9_test.pt",
    "lmdb_dir": "./data/lmdb_qm9",
    # "data_dir": "./data/processed_geom_train.pt",
    # "val_data_dir": "./data/processed_geom_val.pt",
    # "test_data_dir": "./data/processed_geom_test.pt",
    # "lmdb_dir": "./data/lmdb_geom",
    "add_random_rotation": False,
    "add_random_permutation": False,
    "reorder_to_smiles_order": True,
    "remove_hydrogens": True,
    "batch_size": 256,
    "num_workers": 0
}

def apply_ema_weights_to_model(model: torch.nn.Module, ckpt_path: str, device: str = "cpu"):
    """
    Manually load EMA weights from the checkpoint's optimizer_states and apply them to the model.
    """
    print(f"Loading checkpoint from {ckpt_path} to extract EMA weights...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "optimizer_states" not in checkpoint or not checkpoint["optimizer_states"]:
        print("WARNING: No optimizer_states found in checkpoint! Using original weights.")
        return

    # Assuming single optimizer
    opt_state = checkpoint["optimizer_states"][0]

    if "ema" not in opt_state:
        print("WARNING: No 'ema' key found in optimizer state! Using original weights.")
        return

    ema_params_list = opt_state["ema"]
    model_params = list(model.parameters())

    if len(model_params) != len(ema_params_list):
        print(f"ERROR: Model has {len(model_params)} params, but EMA has {len(ema_params_list)} params.")
        return

    print("Overwriting model weights with EMA weights...")
    with torch.no_grad():
        for param, ema_param in zip(model_params, ema_params_list):
            # Ensure dtype and device match
            param.data.copy_(ema_param.to(device=param.device, dtype=param.dtype))
            
    print("Successfully applied EMA weights to the model!")

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

def compute_rmsd_with_kabsch(batch, out_batch):
    """
    Compute the RMSD with Kabsch alignment, ignoring padded atoms.
    
    batch: dict, contains the reference coordinates
    out_batch: dict, contains the generated coordinates
    
    Returns:
        rmsd: float - The RMSD after alignment
    """
    coords_ref = batch["coords"]
    coords_gen = out_batch["coords"]
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
    args = parse_args()
    num_mols = args.num_mols
    num_steps = args.num_steps

    # Load the PocketSynth model checkpoint
    lightning_module = LightningTabasco.load_from_checkpoint(args.checkpoint, weights_only=False)
    apply_ema_weights_to_model(lightning_module.model, args.checkpoint)
    lightning_module.model.net.train_diffusion = False
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
    for batch in tqdm.tqdm(datamodule.test_dataloader()):
        batch = batch.to("cuda")
        out_batch = sample_batch(
            lightning_module=lightning_module,
            batch=batch,
            ema_optimizer=ema_optimizer,
            num_steps=num_steps,
        )
        out_batch_list.append(out_batch)
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
            if out_atom_types != batch_atom_types:
                print(f"out atom types: {out_atom_types}, batch atom types: {batch_atom_types}")
        # break
    # rmsds /= len(datamodule.test_dataloader())
    rmsd = rmsds/num_graphs
    print(f"rmsd: {rmsd}")
    mse = out_batch["coords"] - batch["coords"]
    mse = mse.pow(2).mean()
    print(f"mse: {mse}")
    # compare the generated and reference molecules by atom type

    # Concatenate results from all batches
    out_batch = torch.cat(out_batch_list, dim=0)

    # Export the sampled batch to pickle if specified
    if args.output_path is not None:
        export_batch_to_sdf(batch, args.output_path.replace(".sdf", "_ref.sdf"))
        export_batch_to_sdf(out_batch, args.output_path)


if __name__ == "__main__":
    main()
