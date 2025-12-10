"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
from functools import partial
from typing import Any, Dict, List

import numpy as np
import torch
import wandb
from pymatgen.analysis.structure_matcher import StructureMatcher
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm

from src.eval.crystal import Crystal
from src.ase_notebook import AseView
from src.utils import joblib_map

def compute_volume(batch_lattice):
    """Compute volume from batched lattice matrix

    batch_lattice: (3, 3)
    """
    vector_a, vector_b, vector_c = np.split(batch_lattice, 3, axis=0)
    return np.abs(np.einsum('ij,ij->i', vector_a,
                                  np.cross(vector_b, vector_c, axis=1)))[0]


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
    coords_ref = batch["frac_coords"]
    coords_gen = out_batch["frac_coords"]
    lattices_ref = lattice_params_to_matrix_numpy(batch["lengths"][0], batch["angles"][0])
    lattices_gen = lattice_params_to_matrix_numpy(out_batch["lengths"], out_batch["angles"])
    coords_ref = frac_to_cart_coords(coords_ref, batch["lengths"], batch["angles"], len(batch["frac_coords"]), lattices=lattices_ref)
    coords_gen = frac_to_cart_coords(coords_gen, out_batch["lengths"], out_batch["angles"], len(out_batch["frac_coords"]), lattices=lattices_gen)

    # Iterate through each molecule in the batch
        # Extract individual molecule's coords (ignoring padding mask)
    ref_coords = coords_ref
    gen_coords = coords_gen
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
            per_molecule_rmsd = per_molecule_rmsd * (len(ref_coords) / compute_volume(lattices_ref)) ** (1/3)
        per_molecule_rmsd
            
    # Return average RMSD over all valid molecules in the batch
    return per_molecule_rmsd


def lattice_params_to_matrix_numpy(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: np.ndarray of shape (3), unit A
    angles: torch.Tensor of shape (3), unit degree
    """
    angles_r = np.deg2rad(angles)
    coses = np.cos(angles_r)
    sins = np.sin(angles_r)

    val = (coses[0] * coses[1] - coses[2]) / (sins[0] * sins[1])
    # Sometimes rounding errors result in values slightly > 1.
    val = np.clip(val, -1., 1.)
    gamma_star = np.arccos(val)
    vector_a = np.array([
        lengths[0] * sins[1],
        0,
        lengths[0] * coses[1]])
    vector_b = np.array([
        -lengths[1] * sins[0] * np.cos(gamma_star),
        lengths[1] * sins[0] * np.sin(gamma_star),
        lengths[1] * coses[0]])
    vector_c = np.array([
        0,
        0,
        lengths[2]])

    return np.array([vector_a, vector_b, vector_c])


def frac_to_cart_coords(
    frac_coords,
    lengths,
    angles,
    num_atoms,
    regularized = True,
    lattices = None
):
    if regularized:
        frac_coords = frac_coords % 1.
    if lattices is None:
        lattices = lattice_params_to_matrix_numpy(lengths, angles)
    lattice_nodes = np.tile(lattices, (num_atoms, 1, 1))
    pos = np.einsum("ni,nij->nj", frac_coords, lattice_nodes) # cart coords
    return pos

ase_view = AseView(
    rotations="45x,45y,45z",
    atom_font_size=16,
    axes_length=30,
    canvas_size=(400, 400),
    zoom=1.2,
    show_bonds=False,
    # uc_dash_pattern=(.6, .4),
    atom_show_label=True,
    canvas_background_opacity=0.0,
)
# ase_view.add_miller_plane(1, 0, 0, color="green")


class CrystalReconstructionEvaluator:
    """Evaluator for crystal reconstruction tasks. Can be used within a Lightning module, appending
    predictions and ground truths during training and computing metrics at the end of an epoch, or
    can be used as a standalone object to evaluate predictions on a dataset.

    Args:
        stol (float): StructureMatcher tolerance for matching sites.
        angle_tol (float): StructureMatcher tolerance for matching angles.
        ltol (float): StructureMatcher tolerance for matching lengths.
    """

    def __init__(self, stol=0.5, angle_tol=10, ltol=0.3, device="cpu"):
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.pred_arrays_list = []  # list of Dict[str, np.array] predictions
        self.gt_arrays_list = []  # list of Dict[str, np.array] ground truths
        self.pred_crys_list = []  # list of Crystal predictions
        self.gt_crys_list = []  # list of Crystal ground truths
        self.device = device

    def append_pred_array(self, pred: Dict[str, np.array]):
        """Append a prediction to the evaluator."""
        self.pred_arrays_list.append(pred)

    def append_gt_array(self, gt: Dict[str, np.array]):
        """Append a ground truth to the evaluator."""
        self.gt_arrays_list.append(gt)

    def clear(self):
        """Clear the stored predictions and ground truths, to be used at the end of an epoch."""
        self.pred_arrays_list = []
        self.gt_arrays_list = []
        self.pred_crys_list = []
        self.gt_crys_list = []

    def _arrays_to_crystals(self, save: bool = False, save_dir: str = ""):
        """Convert stored predictions and ground truths to Crystal objects for evaluation."""
        print(f"{save_dir}/pred     +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        self.pred_crys_list = joblib_map(
            partial(
                array_dict_to_crystal,
                save=save,
                save_dir_name=f"{save_dir}/pred",
            ),
            self.pred_arrays_list,
            n_jobs=-4,
            inner_max_num_threads=1,
            desc=f"    Pred to Crystal",
            total=len(self.pred_arrays_list),
        )
        self.gt_crys_list = joblib_map(
            partial(
                array_dict_to_crystal,
                save=save,
                save_dir_name=f"{save_dir}/gt",
            ),
            self.gt_arrays_list,
            n_jobs=-4,
            inner_max_num_threads=1,
            desc=f"    G.T. to Crystal",
            total=len(self.gt_arrays_list),
        )

    def _get_metrics(self, pred, gt, is_valid):
        if not is_valid:
            return float("inf")
        try:
            rms_dist = self.matcher.get_rms_dist(pred.structure, gt.structure)
            rms_dist = float("inf") if rms_dist is None else rms_dist[0]
            return rms_dist
        except Exception:
            return float("inf")

    def get_metrics(self, save: bool = False, save_dir: str = "") -> Dict[str, Any]:
        """Compute the match rate and avg. RMS distance between predictions and ground truths.

        Note: self.rms_dists can be used to access RMSD per sample but is not returned.

        Returns:
            Dict: Dictionary of metrics, including match rate and avg. RMSD.
        """
        assert len(self.pred_arrays_list) == len(
            self.gt_arrays_list
        ), "Number of predictions and ground truths must match."

        # # Convert predictions and ground truths to Crystal objects
        # self._arrays_to_crystals(save, save_dir)

        # # Check validity of predictions and ground truths
        # validity = [
        #     c1.valid and c2.valid for c1, c2 in zip(self.pred_crys_list, self.gt_crys_list)
        # ]

        # self.rms_dists = []
        # for i in tqdm(range(len(self.pred_crys_list)), desc="   Reconstruction eval"):
        #     self.rms_dists.append(
        #         self._get_metrics(self.pred_crys_list[i], self.gt_crys_list[i], validity[i])
        #     )
        self.rms_dists = []
        for i in tqdm(range(len(self.pred_arrays_list)), desc="   Reconstruction eval"):
            self.rms_dists.append(
                compute_rmsd_with_kabsch(self.gt_arrays_list[i], self.pred_arrays_list[i], materials_match=True)
            )
        self.rms_dists = torch.tensor(self.rms_dists, device=self.device)
        match_rate = (~torch.isinf(self.rms_dists)).long()
        if match_rate.sum() == 0:
            # No valid predictions --> return large RMSD for logging purposes
            return {
                "match_rate": match_rate,
                "rms_dist": torch.tensor([10.0] * len(match_rate), device=self.device),
            }
        else:
            return {
                "match_rate": match_rate,
                "rms_dist": self.rms_dists[~torch.isinf(self.rms_dists)],
            }

    def get_wandb_table(self, current_epoch: int = 0, save_dir: str = "") -> wandb.Table:
        """Create a wandb.Table object with the results of the evaluation."""
        pred_table = wandb.Table(
            columns=[
                "Epoch",
                "Sample idx",
                "Num atoms",
                "RMSD",
                "Match?",
                "Valid?",
                "Comp valid?",
                "Struct valid?",
                "True atom types",
                "Pred atom types",
                "True lengths",
                "Pred lengths",
                "True angles",
                "Pred angles",
                "True 2D",
                "Pred 2D",
            ]
        )
        for idx in range(len(self.pred_crys_list)):
            sample_idx = self.gt_crys_list[idx].sample_idx
            assert sample_idx == self.pred_crys_list[idx].sample_idx

            num_atoms = len(self.gt_crys_list[idx].atom_types)

            rmsd = self.rms_dists[idx]

            match = rmsd != float("inf")

            true_atom_types = " ".join([str(int(t)) for t in self.gt_crys_list[idx].atom_types])

            pred_atom_types = " ".join([str(int(t)) for t in self.pred_crys_list[idx].atom_types])

            true_lengths = " ".join([f"{l:.2f}" for l in self.gt_crys_list[idx].lengths])

            true_angles = " ".join([f"{a:.2f}" for a in self.gt_crys_list[idx].angles])

            pred_lengths = " ".join([f"{l:.2f}" for l in self.pred_crys_list[idx].lengths])

            pred_angles = " ".join([f"{a:.2f}" for a in self.pred_crys_list[idx].angles])

            # 2D structures
            try:
                true_2d = ase_view.make_wandb_image(
                    self.gt_crys_list[idx].structure,
                    center_in_uc=False,
                )
            except Exception as e:
                # log.error(f"Failed to load 2D structure for true sample {sample_idx}.")
                true_2d = None
            try:
                pred_2d = ase_view.make_wandb_image(
                    self.pred_crys_list[idx].structure,
                    center_in_uc=False,
                )
            except Exception as e:
                # log.error(f"Failed to load 2D structure for pred sample {sample_idx}.")
                pred_2d = None

            # Update table
            pred_table.add_data(
                current_epoch,
                sample_idx,
                num_atoms,
                rmsd,
                match,
                self.pred_crys_list[idx].valid,
                self.pred_crys_list[idx].comp_valid,
                self.pred_crys_list[idx].struct_valid,
                true_atom_types,
                pred_atom_types,
                true_lengths,
                pred_lengths,
                true_angles,
                pred_angles,
                true_2d,
                pred_2d,
            )
            # TODO What if Structures were to be saved as PDB, too? (for 3D)

        return pred_table


def array_dict_to_crystal(
    x: dict[str, np.ndarray],
    save: bool = False,
    save_dir_name: str = "",
) -> Crystal:
    """Method to convert a dictionary of numpy arrays to a Crystal object which is compatible with
    StructureMatcher (used for evaluations). Previously called 'safe_crystal', as it return a
    generic crystal if the input is invalid.

    Adapted from: https://github.com/facebookresearch/flowmm

    Args:
        x: Dictionary of numpy arrays with keys:
            - 'frac_coords': Fractional coordinates of atoms.
            - 'atom_types': Atomic numbers of atoms.
            - 'lengths': Lengths of the lattice vectors.
            - 'angles': Angles between the lattice vectors.
            - 'sample_idx': Index of the sample in the dataset.
        save: Whether to save the crystal as a CIF file.
        save_dir_name: Directory to save the CIF file.

    Returns:
        Crystal: Crystal object, optionally saved as a CIF file.
    """
    # Check if the lattice angles are in a valid range
    try:
        if np.all(50 < x["angles"]) and np.all(x["angles"] < 130):
            crys = Crystal(x)
            print(x)
            if crys.valid and save:
                os.makedirs(save_dir_name, exist_ok=True)
                crys.structure.to(os.path.join(save_dir_name, f"crystal_{x['sample_idx']}.cif"))
        else:
            # returns an absurd crystal
            crys = Crystal(
                {
                    "frac_coords": np.zeros_like(x["frac_coords"]),
                    "atom_types": np.zeros_like(x["atom_types"]),
                    "lengths": 100 * np.ones_like(x["lengths"]),
                    "angles": np.ones_like(x["angles"]) * 90,
                    "sample_idx": x["sample_idx"],
                }
            )
    except:
        crys = Crystal(
            {
                "frac_coords": np.zeros_like(x["frac_coords"]),
                "atom_types": np.zeros_like(x["atom_types"]),
                "lengths": 100 * np.ones_like(x["lengths"]),
                "angles": np.ones_like(x["angles"]) * 90,
                "sample_idx": x["sample_idx"],
            }
        )
    return crys