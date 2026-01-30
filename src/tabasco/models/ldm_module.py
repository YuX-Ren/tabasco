"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
import random
from typing import Any, Dict, Literal
from tabasco.models.lightning_tabasco import LightningTabasco
from tabasco.utils.utils import apply_ema_weights_to_model
import numpy as np
import pandas as pd
import torch
import wandb
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from tqdm import tqdm
from tensordict import TensorDict
import shutil
from tabasco.data.transforms import apply_random_rotation_one
from torch.nn import ModuleDict
from torchmetrics import MeanMetric


from tabasco.chem.constants import restypes_with_x, ATOM_NAMES_OFFSET
from tabasco.chem.utils import save_aa_coords

IDX_TO_DATASET = {
    0: "mp20",
    1: "qm9",
    2: "pdb",
}
DATASET_TO_IDX = {
    "mp20": 0,  # periodic
    "qm9": 1,  # non-periodic
    "pdb": 2,  # protein
}
class LatentDiffusionLitModule(LightningModule):

    def __init__(
        self,
        autoencoder_ckpt: str,
        denoiser: torch.nn.Module,
        interpolant: DictConfig,
        augmentations: DictConfig,
        sampling: DictConfig,
        conditioning: DictConfig,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scheduler_frequency: str,
        compile: bool,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        lightning_module = LightningTabasco.load_from_checkpoint(autoencoder_ckpt,weights_only=False)
        apply_ema_weights_to_model(lightning_module.model, autoencoder_ckpt)

        self.autoencoder_ckpt = autoencoder_ckpt
        log.info(f"Loading Autoencoder ckpt: {autoencoder_ckpt}")
        self.autoencoder = lightning_module.model
        # freeze autoencoder
        self.autoencoder.requires_grad_(False)
        self.autoencoder.eval()
        self.train_metrics = ModuleDict(
            {
                "loss": MeanMetric(),
                "x_loss": MeanMetric(),
                "x_loss t=[0,25)": MeanMetric(),
                "x_loss t=[25,50)": MeanMetric(),
                "x_loss t=[50,75)": MeanMetric(),
                "x_loss t=[75,100)": MeanMetric(),
                "t_avg": MeanMetric(),
                "dataset_idx": MeanMetric(),
            }
        )
        self.val_metrics = ModuleDict(
            {
                "pdb": ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "x_loss": MeanMetric(),
                        "x_loss t=[0,25)": MeanMetric(),
                        "x_loss t=[25,50)": MeanMetric(),
                        "x_loss t=[50,75)": MeanMetric(),
                        "x_loss t=[75,100)": MeanMetric(),
                        "t_avg": MeanMetric(),
                    }
                ),
            }
        )
        # denoiser model (second-stage model)
        self.denoiser = denoiser

        # interpolant for diffusion or flow matching training/sampling
        self.interpolant = interpolant

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []

    def forward(self, batch, sample_posterior: bool = True):
        # Encode batch to latent space
        with torch.no_grad():
            encoded_batch = self.autoencoder.encode(batch)
            x_1 = encoded_batch["x"]
            mask = batch['padding_mask']
            # Convert from PyG batch to dense batch with padding
            dense_encoded_batch = {"x_1": x_1, "token_mask": ~mask, "diffuse_mask": ~mask}

        # Corrupt batch using the interpolant
        self.interpolant.device = dense_encoded_batch["x_1"].device
        noisy_dense_encoded_batch = self.interpolant.corrupt_batch(dense_encoded_batch)

        # Prepare conditioning inputs to forward pass
        dataset_idx = batch['dataset_idx'] + 1  # 0 -> null class
        spacegroup = torch.zeros_like(batch['dataset_idx'])

        # Use self-conditioning for ~half training batches
        if (
            self.interpolant.self_condition
            and random.random() < self.interpolant.self_condition_prob
        ):
            with torch.no_grad():
                x_sc = self.denoiser(
                    x=noisy_dense_encoded_batch["x_t"],
                    t=noisy_dense_encoded_batch["t"],
                    dataset_idx=dataset_idx,
                    spacegroup=spacegroup,
                    mask=~mask,
                    x_sc=None,
                )
        else:
            x_sc = None

        # Run denoiser model
        pred_x = self.denoiser(
            x=noisy_dense_encoded_batch["x_t"],
            t=noisy_dense_encoded_batch["t"],
            dataset_idx=dataset_idx,
            spacegroup=spacegroup,
            mask=~mask,
            x_sc=x_sc,
        )

        return pred_x, noisy_dense_encoded_batch

    def criterion(
        self,
        noisy_dense_encoded_batch: Dict[str, torch.Tensor],
        pred_x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Compute MSE loss w/ masking for padded tokens
        gt_x_1 = noisy_dense_encoded_batch["x_1"]
        norm_scale = 1 - torch.min(noisy_dense_encoded_batch["t"].unsqueeze(-1), torch.tensor(0.9))
        x_error = (gt_x_1 - pred_x) / norm_scale
        loss_mask = (
            noisy_dense_encoded_batch["token_mask"] * noisy_dense_encoded_batch["diffuse_mask"]
        )
        loss_denom = torch.sum(loss_mask, dim=-1) * pred_x.size(-1)
        x_loss = torch.sum(x_error**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        loss_dict = {"loss": x_loss.mean(), "x_loss": x_loss}

        # add diffusion loss stratified across t
        num_bins = 4
        flat_losses = x_loss.detach().cpu().numpy().flatten()
        flat_t = noisy_dense_encoded_batch["t"].detach().cpu().numpy().flatten()
        bin_edges = np.linspace(0.0, 1.0 + 1e-3, num_bins + 1)
        bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
        t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
        t_binned_n = np.bincount(bin_idx)
        for t_bin in np.unique(bin_idx).tolist():
            bin_start = bin_edges[t_bin]
            bin_end = bin_edges[t_bin + 1]
            t_range = f"x_loss t=[{int(bin_start*100)},{int(bin_end*100)})"
            range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
            loss_dict[t_range] = range_loss
        loss_dict["t_avg"] = np.mean(flat_t)

        return loss_dict

    #####################################################################################################

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        for dataset in self.val_metrics.keys():
            for metric in self.val_metrics[dataset].values():
                metric.reset()

    def on_train_epoch_start(self) -> None:
        """Lightning hook that is called when a training epoch starts."""
        for metric in self.train_metrics.values():
            metric.reset()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        with torch.no_grad():
            batch = apply_random_rotation_one(
                batch
            )

        # forward pass
        pred_x, noisy_dense_encoded_batch = self.forward(batch)

        # calculate loss
        loss_dict = self.criterion(noisy_dense_encoded_batch, pred_x)

        # log relative proportions of datasets in batch
        loss_dict["dataset_idx"] = batch['dataset_idx'].detach().flatten()

        # update and log train metrics
        for k, v in loss_dict.items():
            self.train_metrics[k](v)
            self.log(
                f"train/{k}",
                self.train_metrics[k],
                on_step=True,
                on_epoch=False,
                prog_bar=False if k != "loss" else True,
            )

        # return loss or backpropagation will fail
        return loss_dict["loss"]

    #####################################################################################################

    def on_validation_epoch_start(self) -> None:
        self.on_evaluation_epoch_start(stage="val")

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 2) -> None:
        self.evaluation_step(batch, batch_idx, dataloader_idx, stage="val")

    def on_validation_epoch_end(self) -> None:
        self.on_evaluation_epoch_end(stage="val")

    #####################################################################################################

    def on_test_epoch_start(self) -> None:
        self.on_evaluation_epoch_start(stage="test")

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 2) -> None:
        self.evaluation_step(batch, batch_idx, dataloader_idx, stage="test")

    def on_test_epoch_end(self) -> None:
        self.on_evaluation_epoch_end(stage="test")

    #####################################################################################################

    def on_evaluation_epoch_start(self, stage: Literal["val", "test"]) -> None:
        "Lightning hook that is called when a validation/test epoch starts."
        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")
        for dataset in metrics.keys():
            for metric in metrics[dataset].values():
                metric.reset()

    def evaluation_step(
        self,
        batch,
        batch_idx: int,
        dataloader_idx: int,
        stage: Literal["val", "test"],
    ) -> None:
        """Perform a single evaluation step on a batch of data from the validation/test set."""

        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")[IDX_TO_DATASET[dataloader_idx]]

        # forward pass
        pred_x, noisy_dense_encoded_batch = self.forward(batch)

        # calculate loss
        loss_dict = self.criterion(noisy_dense_encoded_batch, pred_x)

        # update and log per-step val metrics
        for k, v in loss_dict.items():
            metrics[k](v)
            self.log(
                f"{stage}_{IDX_TO_DATASET[dataloader_idx]}/{k}",
                metrics[k],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                add_dataloader_idx=False,
            )

    def on_evaluation_epoch_end(self, stage: Literal["val", "test"]) -> None:
        """Lightning hook that is called when a validation/test epoch ends."""

        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")
        length_list = [70,100]
        for dataset in metrics.keys():
            for length in length_list:
                for samples_so_far in tqdm(
                    range(0, self.hparams.sampling.num_samples, self.hparams.sampling.batch_size),
                    desc=f"    Sampling",
                ):
                    # Perform sampling and decoding to crystal structures
                    out, batch, samples = self.sample_and_decode(
                        batch_size=self.hparams.sampling.batch_size,
                        cfg_scale=self.hparams.sampling.cfg_scale,
                        dataset_idx=DATASET_TO_IDX[dataset],
                        length=length,
                    )
                    self.compute_protein_metrics(out, length)

    def compute_protein_metrics(self, protein, length):
        token_mask = protein["token_mask"]
        samples = protein["coords"]
        generated_aatypes = protein["atomics"].argmax(dim=-1)
        aatypes_seq_list = []
        coords_list = []
        for i in range(protein.batch_size[0]):
            cur_aatypes = generated_aatypes[i][token_mask[i]]
            aatypes_seq_list.append("".join([restypes_with_x[x - ATOM_NAMES_OFFSET] for x in cur_aatypes]))
            coords_list.append(samples[i][token_mask[i]].cpu().numpy() * 4.0) # scale back to original units
        os.makedirs(self.hparams.sampling.save_dir +f"/epoch_{self.current_epoch}/sample_{length}", exist_ok=True)
        save_aa_coords(aatypes_seq_list, coords_list, savedir=self.hparams.sampling.save_dir +f"/epoch_{self.current_epoch}/sample_{length}")

    #####################################################################################################

    def sample_and_decode(
        self,
        batch_size,
        cfg_scale=4.0,
        dataset_idx=0,
        length=100,
    ):
        # sample random lengths from distribution: (B, 1)
        sample_lengths = torch.ones(batch_size, dtype=torch.int64, device=self.device) * length
        dataset_idx = torch.full(
            (batch_size,), dataset_idx + 1, dtype=torch.int64, device=self.device
        )
        # create spacegroup tensor
        spacegroup = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        # create token mask for visualization
        token_mask = torch.zeros(
            batch_size,
            max(sample_lengths),
            dtype=torch.bool,
            device=self.device,
        )
        for idx, length in enumerate(sample_lengths):
            token_mask[idx, :length] = True
        # create new samples from interpolant
        samples = self.interpolant.sample_with_classifier_free_guidance(
            batch_size=batch_size,
            num_tokens=max(sample_lengths),
            emb_dim=self.denoiser.d_x,
            model=self.denoiser,
            dataset_idx=dataset_idx,
            spacegroup=spacegroup,
            cfg_scale=cfg_scale,
            token_mask=token_mask,
        )
        # get final samples and remove padding (to PyG format)
        x = samples["clean_traj"][-1]

        batch = {
            "x": x,
            "num_atoms": sample_lengths,
            "batch": torch.repeat_interleave(
                torch.arange(len(sample_lengths), device=self.device), sample_lengths
            ),
            "token_idx": (torch.cumsum(token_mask, dim=-1, dtype=torch.int64) - 1)[token_mask],
        }
        # decode samples to crystal structures using frozen decoder
        out = self.autoencoder.decode(batch, padding_mask=~token_mask)
        out = TensorDict(
            {
                "coords": out["coords"],
                "atomics": out["atomics"],
                "token_mask": token_mask,
            },
            batch_size=out["coords"].shape[0],
        )
        return out, batch, samples