from typing import Optional, Callable
import torch
from tensordict import TensorDict
from torch import Tensor, nn

from tabasco.flow.path import FlowPath
from tabasco.utils.metric_utils import split_losses_by_time


class InterDistancesLoss(nn.Module):
    """Mean-squared error between predicted and reference inter-atomic distance matrices."""

    def __init__(
        self,
        distance_threshold: Optional[float] = None,
        sqrd: bool = False,
        key: str = "coords",
        key_pad_mask: str = "padding_mask",
        time_factor: Optional[Callable] = None,
    ):
        """Initialize the loss module.

        Args:
            distance_threshold: If provided, only atom pairs with distance <= threshold
                contribute to the loss. Units must match the coordinate system.
            sqrd: When `True` the raw *squared* distances are used instead of their square-root.
                Set this to `True` if you have pre-squared your training targets.
            key: Key that stores coordinates inside `TensorDict` objects.
            key_pad_mask: Key that stores the boolean padding mask inside `TensorDict` objects.
            time_factor: Optional callable `f(t)` that rescales the per-pair loss as a
                function of the interpolation time `t`.
        """
        super().__init__()
        self.key = key
        self.key_pad_mask = key_pad_mask
        self.distance_threshold = distance_threshold
        self.sqrd = sqrd
        self.mse_loss = nn.MSELoss(reduction="none")
        self.time_factor = time_factor

    def inter_distances(self, coords1, coords2, eps: float = 1e-6) -> Tensor:
        """Compute pairwise distances between two coordinate sets.

        Args:
            coords1: Coordinate tensor of shape `(N, 3)`.
            coords2: Coordinate tensor of shape `(M, 3)`.
            eps: Numerical stability term added before `sqrt` when `sqrd` is `False`.

        Returns:
            Tensor of shape `(N, M)` containing pairwise distances. Values are squared
            distances when the instance was created with `sqrd=True`.
        """
        if self.sqrd:
            return torch.cdist(coords1, coords2, p=2) ** 2
        else:
            squared_dist = torch.cdist(coords1, coords2, p=2) ** 2
            return torch.sqrt(squared_dist + eps)

    def forward(
        self, path: FlowPath, pred: TensorDict, compute_stats: bool = True
    ) -> Tensor:
        """Compute the inter-distance MSE loss.

        Args:
            path: `FlowPath` containing ground-truth endpoint coordinates and the
                interpolation time `t`.
            pred: `TensorDict` with predicted coordinates under the same `key` as
                specified during initialization.
            compute_stats: If `True` additionally returns distance-loss statistics binned
                by time for logging purposes.

        Returns:
            - loss:         Scalar tensor with the mean loss.
            - stats_dict:   Dictionary with binned loss statistics. Empty when
                `compute_stats` is `False`.
        """
        real_mask = 1 - path.x_1[self.key_pad_mask].float()
        real_mask = real_mask.unsqueeze(-1)

        pred_coords = pred[self.key]
        true_coords = path.x_1[self.key]

        pred_dists = self.inter_distances(pred_coords, pred_coords)
        true_dists = self.inter_distances(true_coords, true_coords)

        mask_2d = torch.matmul(real_mask, real_mask.transpose(-1, -2))

        # Add distance threshold mask (0 for pairs where distance > threshold)
        if self.distance_threshold is not None:
            distance_mask = (true_dists <= self.distance_threshold).float()
            combined_mask = mask_2d * distance_mask
        else:
            combined_mask = mask_2d

        dists_loss = self.mse_loss(pred_dists, true_dists)
        dists_loss = dists_loss * combined_mask

        if self.time_factor:
            dists_loss = dists_loss * self.time_factor(path.t)

        if compute_stats:
            binned_losses = split_losses_by_time(path.t, dists_loss, 5)
            stats_dict = {
                **{f"dists_loss_bin_{i}": loss for i, loss in enumerate(binned_losses)},
            }
        else:
            stats_dict = {}

        dists_loss = dists_loss.mean()
        return dists_loss, stats_dict


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

import numpy as np
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

def compute_rmsd_with_kabsch(batch, out_batch, materials_match = False, materials_weight = 0.1):
    """
    Compute the RMSD with Kabsch alignment, ignoring padded atoms.
    
    batch: dict, contains the reference coordinates
    out_batch: dict, contains the generated coordinates
    
    Returns:
        rmsd: float - The RMSD after alignment
    """
    coords_ref = batch["coords"]
    coords_gen = out_batch["coords"]
    if materials_match:
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
                per_molecule_rmsd = per_molecule_rmsd * (len(ref_coords) / compute_volume(lattices_ref[i])) ** (1/3) * materials_weight
            rmsds += per_molecule_rmsd.item()
            
    # Return average RMSD over all valid molecules in the batch
    return rmsds/coords_ref.shape[0]