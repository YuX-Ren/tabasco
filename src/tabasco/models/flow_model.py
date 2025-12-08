import random
from copy import deepcopy
from typing import Dict, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor

from tabasco.flow.interpolate import Interpolant
from tabasco.flow.path import FlowPath
from tabasco.flow.utils import HistogramTimeDistribution
from tabasco.models.components.losses import InterDistancesLoss, compute_rmsd_with_kabsch
from tabasco.data.transforms import apply_random_rotation, frac_to_cart_coords, cart_to_frac_coords, apply_random_translation


class FlowMatchingModel(nn.Module):
    """Flow-matching diffusion model for 3-D molecule generation.

    Typical usage:
    - `forward`:    called during training to compute loss and optional stats.
    - `sample`:     runs the Euler sampler to generate new molecules at inference.
    """

    def __init__(
        self,
        net: nn.Module,
        coords_interpolant: Interpolant,
        frac_coords_interpolant: Interpolant,
        atomics_interpolant: Interpolant,
        time_distribution: str = "uniform",
        time_alpha_factor: float = 2.0,
        interdist_loss: InterDistancesLoss = None,
        num_random_augmentations: Optional[int] = None,
        sample_schedule: str = "linear",
        compile: bool = False,
        lattices_interpolant: Interpolant = None,
        train_materials = False,
    ):
        """Args:
        net: The neural network predicting velocity fields.
        coords_interpolant: Interpolant for Cartesian coordinates.
        frac_coords_interpolant: Interpolant for fractional coordinates.
        atomics_interpolant: Interpolant for one-hot atom types.
        time_distribution: `uniform`, `beta`, or `histogram`.
        time_alpha_factor: Alpha for beta distribution (ignored otherwise).
        interdist_loss: Optional additional loss on inter-atomic distances.
        num_random_augmentations: Number of random rotations per sample.
        sample_schedule: `linear`, `power`, or `log` schedule in `sample`.
        compile: If True, passes the network through `torch.compile`.
        """
        super().__init__()
        self.net = net
        self.train_materials = train_materials
        if compile:
            self.net = torch.compile(self.net)

        self.atomics_interpolant = atomics_interpolant
        self.coords_interpolant = coords_interpolant
        self.frac_coords_interpolant = frac_coords_interpolant
        self.interdist_loss = interdist_loss
        self.time_alpha_factor = time_alpha_factor
        self.lattices_interpolant = lattices_interpolant
        if time_distribution == "uniform":
            self.time_distribution = torch.distributions.Uniform(0, 1)
        elif time_distribution == "beta":
            self.time_distribution = torch.distributions.Beta(time_alpha_factor, 1)
        elif time_distribution == "histogram":
            # TODO: chore: make pretty
            print("Using histogram time distribution")
            self.time_distribution = HistogramTimeDistribution(
                torch.tensor([0.05, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4])
            )
        else:
            raise ValueError(f"Invalid time distribution: {time_distribution}")

        assert num_random_augmentations is None or num_random_augmentations >= 0, (
            "num_random_augmentations must be non-negative or None"
        )

        self.num_random_augmentations = num_random_augmentations
        self.sample_schedule = sample_schedule

    def set_data_stats(self, stats: Dict):
        """Set the data statistics."""
        self.data_stats = stats

    def _call_net(self,batch_ori, batch_t, t):
        """Wrapper around `self.net` for `torch.compile` compatibility."""
        if batch_ori["data_type"][0] == 1:
            coords, atom_logits, lattices, kl_loss = self.net(
                batch_ori["coords"], batch_ori["atomics"], batch_ori["padding_mask"],batch_t["coords"], batch_t["atomics"], t,
                lattices_ori = batch_ori["lattices"],
                lattices_t = batch_t["lattices"],
                data_type = batch_ori["data_type"]
            )
        else:
            coords, atom_logits, kl_loss = self.net(
                batch_ori["coords"], batch_ori["atomics"], batch_ori["padding_mask"],batch_t["coords"], batch_t["atomics"], t,
                data_type = batch_ori["data_type"]
            )
        return TensorDict(
            {
                "coords": coords,
                "atomics": atom_logits,
                **({"lattices": lattices} if batch_ori["data_type"][0] == 1 else {}),
                "padding_mask": batch_ori["padding_mask"],
            },
            batch_size=batch_ori["padding_mask"].shape[0],
        ), kl_loss

    def forward(self, batch, compute_stats: bool = True):
        """Compute training loss and optional stats."""

        if self.num_random_augmentations:
            if batch["data_type"][0] == 1:
                batch = apply_random_translation(
                    batch, n_augmentations=self.num_random_augmentations
                )
            else:
                batch = apply_random_rotation(
                    batch, n_augmentations=self.num_random_augmentations
                )
        if batch["data_type"][0] == 1:
            num_atoms = (~batch["padding_mask"]).sum(dim=-1).unsqueeze(-1).unsqueeze(-1)
            batch["lattices"] = batch["lattices"] / num_atoms**(1/3)
            batch["coords"] = (batch["coords"] + 0.5) % 1.0 - 0.5



        path = self._create_path(batch)
        pred, kl_loss = self._call_net(path.x_1, path.x_t, path.t)

        loss, stats_dict = self._compute_loss(path, pred, compute_stats, kl_loss)
        return loss, stats_dict

    def _create_path(
        self,
        x_1: TensorDict,
        t: Optional[Tensor] = None,
        noise_batch: Optional[TensorDict] = None,
    ) -> FlowPath:
        """Generate `(x_0, x_t, dx_t)` tensors for a random or given time `t`."""

        batch_size = x_1["padding_mask"].shape[0]
        pad_mask = x_1["padding_mask"]

        if t is None:
            t = self.time_distribution.sample((batch_size,))
            t = t.to(x_1.device)

        if noise_batch is None:
            noise_batch = self._sample_noise_like_batch(x_1)

        if x_1["data_type"][0] == 0:
            x_0_coords, x_t_coords, dx_t_coords = self.coords_interpolant.create_path(
                x_1=x_1, t=t, x_0=noise_batch
            )

        else:
            x_0_coords, x_t_coords, dx_t_coords = self.frac_coords_interpolant.create_path(
                x_1=x_1, t=t, x_0=noise_batch
            )
        x_0_atomics, x_t_atomics, dx_t_atomics = self.atomics_interpolant.create_path(
            x_1=x_1, t=t, x_0=noise_batch
        )
        if x_1["data_type"][0] == 1:
            x_0_lattices, x_t_lattices, dx_t_lattices = self.lattices_interpolant.create_path(
                x_1=x_1, t=t, x_0=noise_batch
            )
        # TODO: feat: bonds
        # x_0_bonds, x_t_bonds, dx_t_bonds = self.bonds_interpolant.sample_noise(t, x_1["bonds"])

        x_0 = TensorDict(
            {
                "coords": x_0_coords,
                "atomics": x_0_atomics,
                "padding_mask": pad_mask,
                **({"lattices": x_0_lattices} if x_1["data_type"][0] == 1 else {}),
                "data_type": x_1["data_type"],
            },
            batch_size=batch_size,
        )

        x_t = TensorDict(
            {
                "coords": x_t_coords,
                "atomics": x_t_atomics,
                "padding_mask": pad_mask,
                **({"lattices": x_t_lattices} if x_1["data_type"][0] == 1 else {}),
                "data_type": x_1["data_type"],
            },
            batch_size=batch_size,
        )

        dx_t = TensorDict(
            {
                "coords": dx_t_coords,
                "atomics": dx_t_atomics,
                "padding_mask": pad_mask,
                **({"lattices": dx_t_lattices} if x_1["data_type"][0] == 1 else {}),
                "data_type": x_1["data_type"],
            },
            batch_size=batch_size,
        )

        x_0 = x_0.to(x_1.device)
        x_t = x_t.to(x_1.device)
        dx_t = dx_t.to(x_1.device)

        return FlowPath(x_0=x_0, x_t=x_t, dx_t=dx_t, x_1=x_1, t=t)

    def _compute_loss(
        self, path: FlowPath, pred: TensorDict, compute_stats: bool = True, kl_loss: Tensor = 0.0
    ) -> Tensor:
        """Compute and sum coordinate, atom-type, and optional inter-distance losses."""

        atomics_loss, atomics_stats = self.atomics_interpolant.compute_loss(
            path, pred, compute_stats
        )
        if path.x_1["data_type"][0] == 0:
            coords_loss, coord_stats = self.coords_interpolant.compute_loss(
                path, pred, compute_stats
            )
        else:
            coords_loss, coord_stats = self.frac_coords_interpolant.compute_loss(
                path, pred, compute_stats
            )
        if path.x_1["data_type"][0] == 1:
            lattices_loss, lattices_stats = self.lattices_interpolant.compute_loss(
                path, pred, compute_stats
            )
        else:
            lattices_loss, lattices_stats = 0, {}
        if self.interdist_loss:
            dists_loss, dists_stats = self.interdist_loss(path, pred, compute_stats)
        else:
            dists_loss, dists_stats = 0, {}

        if compute_stats:
            pesudo_pred = self.sample(batch=path.x_1, num_steps=2)
            rmsd = compute_rmsd_with_kabsch(path.x_1, pesudo_pred)
            stats_dict = {
                "atomics_loss": atomics_loss,
                "coords_loss": coords_loss,
                "kl_loss": kl_loss,
                "rmsd": rmsd,
                **({"lattices_loss": lattices_loss} if path.x_1["data_type"][0] == 1 else {}),
                **atomics_stats,
                **coord_stats,
                **lattices_stats,
                **dists_stats,
                "rmsd": rmsd,
            }

            atomics_logit_norm = pred["atomics"].norm(dim=-1)
            atomics_logit_max, _ = pred["atomics"].max(dim=-1)
            atomics_logit_min, _ = pred["atomics"].min(dim=-1)
            stats_dict["atomics_logit_norm"] = atomics_logit_norm.mean().item()
            stats_dict["atomics_logit_max"] = atomics_logit_max.mean().item()
            stats_dict["atomics_logit_min"] = atomics_logit_min.mean().item()

            coords_logit_norm = pred["coords"].norm(dim=-1).mean().item()
            stats_dict["coords_logit_norm"] = coords_logit_norm
        else:
            stats_dict = {}

        total_loss = atomics_loss + coords_loss + dists_loss + lattices_loss + kl_loss
        return total_loss, stats_dict

    def _get_sample_schedule(self, num_steps: int) -> Tensor:
        """Return monotonically increasing schedule `T` in `[0,1]`.

        Based on approach in Proteina
        """

        eff_num_steps = num_steps + 1

        if self.sample_schedule == "linear":
            T = torch.linspace(0, 1, eff_num_steps)

        elif self.sample_schedule == "power":
            T = torch.linspace(0, 1, eff_num_steps)
            T = T**2

        elif self.sample_schedule == "log":
            T = 1.0 - torch.logspace(-2, 0, eff_num_steps).flip(0)
            T = T - torch.amin(T)
            T = T / torch.amax(T)

        else:
            raise ValueError(f"Invalid sample schedule: {self.sample_schedule}")

        return T

    def sample(
        self,
        batch: Optional[TensorDict] = None,
        num_steps: int = 100,
        batch_size: Optional[int] = None,
        return_trajectories: bool = False,
    ):
        """Sample molecules.

        Args:
            batch: Optional reference batch whose padding mask/shape determine
                the noise tensor. If `None`, shapes are drawn from
                `self.data_stats`.
            num_steps: Number of Euler steps.
            batch_size: Required when `batch` is `None`.
            return_trajectories: If True, also return intermediate snapshots.
        """
        if batch["data_type"][0] == 1:
            num_atoms = (~batch["padding_mask"]).sum(dim=-1).unsqueeze(-1).unsqueeze(-1)
            batch["lattices"] = batch["lattices"] / num_atoms**(1/3)
            batch["coords"] = (batch["coords"] + 0.5) % 1.0 - 0.5
        x_t = self._sample_noise_like_batch(batch, batch_size)
        if return_trajectories:
            trajectories = []

        T = self._get_sample_schedule(num_steps)
        T = T.to(x_t.device)[:, None]
        T = T.repeat(1, x_t["coords"].shape[0])

        for i in range(1, len(T)):
            t = T[i - 1]
            dt = T[i] - T[i - 1]

            x_t = self._step(batch, x_t, t, dt)
            if return_trajectories:
                trajectories.append(deepcopy(x_t.detach().cpu()))
        if x_t["data_type"][0] == 1:
            num_atoms = (~x_t["padding_mask"]).sum(dim=-1).unsqueeze(-1).unsqueeze(-1)
            x_t["lattices"] = x_t["lattices"] * num_atoms**(1/3)
            batch["lattices"] = batch["lattices"] * num_atoms**(1/3)
            # x_t["coords"] = x_t["coords"] % 1.0 
            # batch["coords"] = batch["coords"] % 1.0 
        if return_trajectories:
            return x_t, trajectories

        return x_t

    def _step(self, batch, x_t, t, step_size):
        """Single Euler step at time `t` using model-predicted velocity."""
        with torch.no_grad():
            out_batch, _ = self._call_net(batch, x_t, t)

        if x_t["data_type"][0] == 0:
            x_t["coords"] = self.coords_interpolant.step(x_t, out_batch, t, step_size)
        else:
            x_t["coords"] = self.frac_coords_interpolant.step(x_t, out_batch, t, step_size)
        x_t["atomics"] = self.atomics_interpolant.step(x_t, out_batch, t, step_size)
        if x_t["data_type"][0] == 1:
            x_t["lattices"] = self.lattices_interpolant.step(x_t, out_batch, t, step_size)
        return x_t

    def _sample_noise_like_batch(
        self, batch: Optional[TensorDict] = None, batch_size: Optional[int] = None
    ):
        """Draw coordinate and atom-type noise compatible with `batch`."""
        # Determine device
        device = batch.device if batch is not None else next(self.parameters()).device

        if batch is None:
            assert hasattr(self, "data_stats"), "self.data_stats not set"
            assert batch_size is not None, (
                "Batch size must be provided when batch is None"
            )

            max_num_atoms = self.data_stats["max_num_atoms"]
            sampled_num_atoms = random.choices(  # nosec B311
                list(self.data_stats["num_atoms_histogram"].keys()),
                weights=list(self.data_stats["num_atoms_histogram"].values()),
                k=batch_size,
            )
            sampled_num_atoms = torch.tensor(sampled_num_atoms)

            pad_mask = (
                torch.arange(max_num_atoms)[None, :] >= sampled_num_atoms[:, None]
            )
            coord_shape = (batch_size, max_num_atoms, self.data_stats["spatial_dim"])
            atomics_shape = (batch_size, max_num_atoms, self.data_stats["atom_dim"])
            lattices_shape = (batch_size, 3, 3)
        else:
            pad_mask = batch["padding_mask"]
            coord_shape = batch["coords"].shape
            atomics_shape = batch["atomics"].shape
            if batch["data_type"][0] == 1:
                lattices_shape = batch["lattices"].shape
            else:
                lattices_shape = None

        if batch["data_type"][0] == 0:
            coord_noise = self.coords_interpolant.sample_noise(coord_shape, pad_mask)
        else:
            coord_noise = self.frac_coords_interpolant.sample_noise(coord_shape, pad_mask)
        atomics_noise = self.atomics_interpolant.sample_noise(atomics_shape, pad_mask)
        if batch["data_type"][0] == 1:
            lattices_noise = self.lattices_interpolant.sample_noise(lattices_shape, batch.device)
        noise_batch = TensorDict(
            {
                "coords": coord_noise,
                "atomics": atomics_noise,
                "padding_mask": pad_mask,
                **({"lattices": lattices_noise} if batch["data_type"][0] == 1 else {}),
                "data_type": batch["data_type"],
            },
            batch_size=pad_mask.shape[0],
        )
        noise_batch = noise_batch.to(device)

        return noise_batch
