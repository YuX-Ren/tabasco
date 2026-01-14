"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import copy
import os
import random
import time
from typing import Any, Dict, Literal, Tuple
from tabasco.models.lightning_tabasco import LightningTabasco
from tabasco.utils.utils import apply_ema_weights_to_model
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torch.nn import ModuleDict
from torchmetrics import MeanMetric
from tqdm import tqdm
from tensordict import TensorDict
import math
from src.eval.crystal_generation import CrystalGenerationEvaluator
from src.eval.mof_generation import MOFGenerationEvaluator
from src.eval.molecule_generation import MoleculeGenerationEvaluator
from src.utils import pylogger
from tabasco.data.transforms import apply_random_rotation_one, apply_random_translation_one, frac_to_cart_coords_one

log = pylogger.RankedLogger(__name__)


IDX_TO_DATASET = {
    0: "qm9",
    1: "mp20",
    2: "qmof150",
}
DATASET_TO_IDX = {
    "qm9": 0,  # non-periodic
    "mp20": 1,  # periodic
    "qmof150": 1,  # periodic
}

class LatentDiffusionLitModule(LightningModule):
    """LightningModule for latent diffusion generative modellling of 3D atomic systems.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

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
        num_random_augmentations: int = 0,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.num_random_augmentations = num_random_augmentations
        self.autoencoder_ckpt = autoencoder_ckpt
        # denoiser model (second-stage model)
        self.denoiser = denoiser

        # interpolant for diffusion or flow matching training/sampling
        self.interpolant = interpolant

        # evaluator objects for computing metrics
        self.val_generation_evaluators = {
            "mp20": CrystalGenerationEvaluator(
                dataset_cif_list=pd.read_csv(
                    os.path.join(self.hparams.sampling.data_dir, f"mp_20_all.csv")
                )["cif"].tolist()
            ),
            "qm9": MoleculeGenerationEvaluator(
                dataset_smiles_list=torch.load(
                    os.path.join(self.hparams.sampling.data_dir, f"smiles.pt"),
                ),
                removeHs=self.hparams.sampling.removeHs,
            ),
        }
        self.test_generation_evaluators = copy.deepcopy(self.val_generation_evaluators)

        # metric objects for calculating and averaging across batches
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
                "mp20": ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "x_loss": MeanMetric(),
                        "x_loss t=[0,25)": MeanMetric(),
                        "x_loss t=[25,50)": MeanMetric(),
                        "x_loss t=[50,75)": MeanMetric(),
                        "x_loss t=[75,100)": MeanMetric(),
                        "t_avg": MeanMetric(),
                        "valid_rate": MeanMetric(),
                        "struct_valid_rate": MeanMetric(),
                        "comp_valid_rate": MeanMetric(),
                        "unique_rate": MeanMetric(),
                        "novel_rate": MeanMetric(),
                        "sampling_time": MeanMetric(),
                    }
                ),
                "qm9": ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "x_loss": MeanMetric(),
                        "x_loss t=[0,25)": MeanMetric(),
                        "x_loss t=[25,50)": MeanMetric(),
                        "x_loss t=[50,75)": MeanMetric(),
                        "x_loss t=[75,100)": MeanMetric(),
                        "t_avg": MeanMetric(),
                        "valid_rate": MeanMetric(),
                        "unique_rate": MeanMetric(),
                        "novel_rate": MeanMetric(),
                        "mol_pred_loaded": MeanMetric(),
                        "sanitization": MeanMetric(),
                        "inchi_convertible": MeanMetric(),
                        "all_atoms_connected": MeanMetric(),
                        "bond_lengths": MeanMetric(),
                        "bond_angles": MeanMetric(),
                        "internal_steric_clash": MeanMetric(),
                        "aromatic_ring_flatness": MeanMetric(),
                        "double_bond_flatness": MeanMetric(),
                        "internal_energy": MeanMetric(),
                        "sampling_time": MeanMetric(),
                        "diversity": MeanMetric(),
                        # "strain_energy_mean": MeanMetric(),
                        # "strain_energy_median": MeanMetric(),
                        "pb_valid_rate": MeanMetric(),
                    }
                ),
            }
        )
        self.test_metrics = copy.deepcopy(self.val_metrics)

        # load bincounts for sampling
        self.num_nodes_bincount = {
            "mp20": torch.nn.Parameter(
                torch.load(
                    os.path.join(self.hparams.sampling.data_dir, f"mp_20_num_nodes_bincount.pt"),
                    map_location="cpu",
                ),
                requires_grad=False,
            ),
            "qm9": torch.nn.Parameter(
                torch.load(
                    os.path.join(self.hparams.sampling.data_dir, f"qm9_num_nodes_bincount.pt"),
                    map_location="cpu",
                ),
                requires_grad=False,
            ),
        }
        self.spacegroups_bincount = {
            "mp20": torch.nn.Parameter(
                torch.load(
                    os.path.join(self.hparams.sampling.data_dir, f"spacegroups_bincount.pt"),
                    map_location="cpu",
                ),
                requires_grad=False,
            ),
            "qm9": None,
        }

    def forward(self, batch, sample_posterior: bool = True):
        # Encode batch to latent space
        with torch.no_grad():
            is_crystal = (batch["data_type"] == 1).flatten() # shape [B]
            is_molecule = ~is_crystal
            if is_molecule.any():
                mol_subbatch = batch[is_molecule] 
                encoded_mol_subbatch = self.autoencoder.encode(mol_subbatch)
                mol_mask = mol_subbatch['padding_mask']
                mol_mask = torch.cat([mol_mask,torch.ones(mol_mask.shape[0], 3, device=mol_mask.device, dtype=mol_mask.dtype)], dim=1)
                encoded_mol_subbatch["x"] = torch.cat([encoded_mol_subbatch["x"], torch.zeros(encoded_mol_subbatch["x"].shape[0], 3, encoded_mol_subbatch["x"].shape[2], device=encoded_mol_subbatch["x"].device, dtype=encoded_mol_subbatch["x"].dtype)], dim=1)

            if is_crystal.any():
                crys_subbatch = batch[is_crystal]
                encoded_crys_subbatch = self.autoencoder.encode(crys_subbatch)
                crys_mask = crys_subbatch['padding_mask']
                crys_mask = torch.cat([torch.zeros(crys_mask.shape[0], 3, device=crys_mask.device, dtype=crys_mask.dtype), crys_mask], dim=1)
            
            # encoded_batch = self.autoencoder.encode(batch)
            x_1 = torch.cat([encoded_mol_subbatch["x"], encoded_crys_subbatch["x"]], dim=0)
            mask = torch.cat([mol_mask, crys_mask], dim=0)
            dense_encoded_batch = {"x_1": x_1, "token_mask": ~mask, "diffuse_mask": ~mask}

        self.interpolant.device = dense_encoded_batch["x_1"].device
        noisy_dense_encoded_batch = self.interpolant.corrupt_batch(dense_encoded_batch)

        dataset_idx = batch['data_type'] + 1
        spacegroup = torch.zeros_like(batch['data_type'])

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

        pred_x = self.denoiser(
            x=noisy_dense_encoded_batch["x_t"],
            t=noisy_dense_encoded_batch["t"],
            dataset_idx=dataset_idx,
            spacegroup=spacegroup,
            mask=~mask,
            x_sc=x_sc,
        )

        return pred_x, noisy_dense_encoded_batch

    #####################################################################################################

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        for dataset in self.val_metrics.keys():
            for metric in self.val_metrics[dataset].values():
                metric.reset()

    def on_train_epoch_start(self) -> None:
        """Lightning hook that is called when a training epoch starts."""
        for metric in self.train_metrics.values():
            metric.reset()

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        with torch.no_grad():
            is_crystal = (batch["data_type"] == 1).flatten() # shape [B]
            is_molecule = ~is_crystal
            if is_molecule.any():
                mol_subbatch = batch[is_molecule] 
                mol_subbatch = apply_random_rotation_one(mol_subbatch)
                batch["coords"][is_molecule] = mol_subbatch["coords"]

            if is_crystal.any():
                crys_subbatch = batch[is_crystal]
                crys_subbatch = apply_random_translation_one(crys_subbatch)
                batch["coords"][is_crystal] = crys_subbatch["coords"]

        pred_x, noisy_dense_encoded_batch = self.forward(batch)

        loss_dict = self.criterion(noisy_dense_encoded_batch, pred_x)

        loss_dict["dataset_idx"] = batch['data_type'].detach().flatten()

        for k, v in loss_dict.items():
            self.train_metrics[k](v)
            self.log(
                f"train/{k}",
                self.train_metrics[k],
                on_step=True,
                on_epoch=False,
                prog_bar=False if k != "loss" else True,
            )

        return loss_dict["loss"]

    #####################################################################################################

    def on_validation_epoch_start(self) -> None:
        self.on_evaluation_epoch_start(stage="val")

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 1) -> None:
        self.evaluation_step(batch, batch_idx, dataloader_idx, stage="val")

    def on_validation_epoch_end(self) -> None:
        self.on_evaluation_epoch_end(stage="val")

    #####################################################################################################

    def on_test_epoch_start(self) -> None:
        self.on_evaluation_epoch_start(stage="test")

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = 1) -> None:
        self.evaluation_step(batch, batch_idx, dataloader_idx, stage="test")

    def on_test_epoch_end(self) -> None:
        self.on_evaluation_epoch_end(stage="test")

    #####################################################################################################

    def on_evaluation_epoch_start(self, stage: Literal["val", "test"]) -> None:
        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")
        for dataset in metrics.keys():
            for metric in metrics[dataset].values():
                metric.reset()
        generation_evaluators = getattr(self, f"{stage}_generation_evaluators")
        for dataset in generation_evaluators.keys():
            generation_evaluators[dataset].clear()

    def evaluation_step(
        self,
        batch,
        batch_idx: int,
        dataloader_idx: int,
        stage: Literal["val", "test"],
    ) -> None:
        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        metrics = getattr(self, f"{stage}_metrics")[IDX_TO_DATASET[dataloader_idx]]
        generation_evaluator = getattr(self, f"{stage}_generation_evaluators")[
            IDX_TO_DATASET[dataloader_idx]
        ]
        generation_evaluator.device = metrics["loss"].device

        pred_x, noisy_dense_encoded_batch = self.forward(batch)

        loss_dict = self.criterion(noisy_dense_encoded_batch, pred_x)

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
        if stage not in ["val", "test"]:
            raise ValueError("stage must be 'val' or 'test'.")
        
        metrics = getattr(self, f"{stage}_metrics")
        generation_evaluators = getattr(self, f"{stage}_generation_evaluators")

        world_size = self.trainer.world_size
        global_rank = self.global_rank

        for dataset in metrics.keys():
            generation_evaluators[dataset].device = metrics[dataset]["loss"].device
            t_start = time.time()
            
            if dataset == "mp20":
                continue
                total_samples = 10000
            else:
                total_samples = self.hparams.sampling.num_samples
            
            samples_per_gpu = math.ceil(total_samples / world_size)
            
            rank_sample_offset = global_rank * samples_per_gpu

            for samples_so_far in tqdm(
                range(0, samples_per_gpu, self.hparams.sampling.batch_size),
                desc=f"    Sampling (Rank {global_rank})",
            ):
                out, batch, samples = self.sample_and_decode(
                    num_nodes_bincount=self.num_nodes_bincount[dataset],
                    spacegroups_bincount=self.spacegroups_bincount[dataset],
                    batch_size=self.hparams.sampling.batch_size,
                    cfg_scale=self.hparams.sampling.cfg_scale,
                    dataset_idx=DATASET_TO_IDX[dataset],
                )
                
                start_idx = 0
                for idx_in_batch, num_atom in enumerate(batch["num_atoms"].tolist()):
                    # convert atom types to element symbols
                    _atom_types = (
                        out["atomics"].narrow(0, start_idx, num_atom).argmax(dim=1)
                    )  # take argmax
                    
                    _atom_types[_atom_types == 0] = 1  # atom type 0 -> 1 (H) to prevent crash
                    
                    '''multiply by the dataset_normalizer'''
                    if dataset == "qm9":
                        _pos = out["coords"].narrow(0, start_idx, num_atom) * 2.0
                        _lattices = None # qm9 不需要 lattices，初始化为 None 防止下面报错
                        _frac_coords = None
                    else:
                        _lattices = out["lattices"][idx_in_batch]
                        _frac_coords = out["coords"].narrow(0, start_idx, num_atom) % 1.0 
                        _pos = frac_to_cart_coords_one(_frac_coords, _lattices)
                    
                    current_global_sample_idx = rank_sample_offset + samples_so_far + idx_in_batch

                    generation_evaluators[dataset].append_pred_array(
                        {
                            "atom_types": _atom_types.detach().cpu().numpy(),
                            "pos": _pos.detach().cpu().numpy(),
                            **({"lattices": _lattices.detach().cpu().numpy()} if dataset == "mp20" else {"lattices": None}),
                            **({"frac_coords": _frac_coords.detach().cpu().numpy()} if dataset == "mp20" else {"frac_coords": None}),
                            "sample_idx": current_global_sample_idx, 
                        }
                    )
                    start_idx = start_idx + num_atom
            t_end = time.time()

            gen_metrics_dict = generation_evaluators[dataset].get_metrics(
                save=self.hparams.sampling.visualize,
                save_dir=self.hparams.sampling.save_dir + f"/{dataset}_{stage}",
            )
            gen_metrics_dict["sampling_time"] = t_end - t_start
            
            for k, v in gen_metrics_dict.items():
                metrics[dataset][k](v)
                self.log(
                    f"{stage}_{dataset}/{k}",
                    metrics[dataset][k],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False if k != "valid_rate" else True,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

            # if self.hparams.sampling.visualize and type(self.logger) == WandbLogger:
            #     pred_table = generation_evaluators[dataset].get_wandb_table(
            #         current_epoch=self.current_epoch,
            #         save_dir=self.hparams.sampling.save_dir
            #         + f"/{dataset}_{stage}",
            #     )
            #     self.logger.experiment.log(
            #         {f"{dataset}_{stage}_samples_table": pred_table}
            #     )

    #####################################################################################################

    def sample_and_decode(
        self,
        num_nodes_bincount,
        spacegroups_bincount,
        batch_size,
        cfg_scale=4.0,
        dataset_idx=0,
    ):
        # sample random lengths from distribution: (B, 1)
        if dataset_idx == 1:
            sample_lengths = torch.multinomial(
                num_nodes_bincount.float(),
                batch_size,
                replacement=True,
            ).to(self.device) + 3 # add 3 for lattices
        else:
            sample_lengths = torch.multinomial(
                num_nodes_bincount.float(),
                batch_size,
                replacement=True,
            ).to(self.device)

        # create dataset_idx tensor
        # NOTE 0 -> null class within DiT, while 0 -> MP20 elsewhere, so increment by 1
        dataset_idx = torch.full(
            (batch_size,), dataset_idx + 1, dtype=torch.int64, device=self.device
        )

        # create spacegroup tensor
        if not self.hparams.conditioning.spacegroup or spacegroups_bincount is None:
            # null spacegroup
            spacegroup = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        else:
            # sample random spacegroups from distribution: (B, 1)
            spacegroup = torch.multinomial(
                spacegroups_bincount.float(),
                batch_size,
                replacement=True,
            ).to(self.device)

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
        token_mask = token_mask[:,3:] if dataset_idx[0] - 1 == 1 else token_mask
        batch = {
            "x": x,
            "num_atoms": sample_lengths - 3 if dataset_idx[0] - 1 == 1 else sample_lengths, # subtract 3 for lattices
            "batch": torch.repeat_interleave(
                torch.arange(len(sample_lengths), device=self.device), sample_lengths
            ),
            "token_idx": (torch.cumsum(token_mask, dim=-1, dtype=torch.int64) - 1)[token_mask],
            "data_type": dataset_idx - 1,  # 0 -> null class
        }
        # decode samples to crystal structures using frozen decoder
        out = self.autoencoder.decode(batch, padding_mask=~token_mask)
        # print(out["atomics"][token_mask].shape)
        # print(out["atomics"].shape)
        out = {
                "coords": out["coords"][token_mask],
                "atomics": out["atomics"][token_mask],
                **({"lattices": out["lattices"]} if dataset_idx[0] - 1 == 1 else {}),
            }
        return out, batch, samples

    #####################################################################################################

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.autoencoder = torch.compile(self.autoencoder)
            self.denoiser = torch.compile(self.denoiser)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_mp20/valid_rate",
                    "interval": "epoch",
                    "frequency": self.hparams.scheduler_frequency,
                },
            }
        return {"optimizer": optimizer}

    def criterion(
        self,
        noisy_dense_encoded_batch: Dict[str, torch.Tensor],
        pred_x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        gt_x_1 = noisy_dense_encoded_batch["x_1"]
        norm_scale = 1 - torch.min(noisy_dense_encoded_batch["t"].unsqueeze(-1), torch.tensor(0.9))
        x_error = (gt_x_1 - pred_x) / norm_scale
        loss_mask = (
            noisy_dense_encoded_batch["token_mask"] * noisy_dense_encoded_batch["diffuse_mask"]
        )
        loss_denom = torch.sum(loss_mask, dim=-1) * pred_x.size(-1)
        x_loss = torch.sum(x_error**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        loss_dict = {"loss": x_loss.mean(), "x_loss": x_loss}

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

    def load_autoencoder(self, auto_encoder_module: LightningModule):
        # autoencoder models (first-stage model)
        checkpoint = torch.load(self.autoencoder_ckpt, weights_only=False)  # 加载 checkpoint
        auto_encoder_module.load_state_dict(checkpoint["state_dict"],strict=False)  # 加载 state_dict（模型权重）
        apply_ema_weights_to_model(auto_encoder_module, self.autoencoder_ckpt)
        log.info(f"Loading Autoencoder ckpt: {self.autoencoder_ckpt}")
        self.autoencoder = auto_encoder_module.model
        # freeze autoencoder
        self.autoencoder.requires_grad_(False)
        self.autoencoder.eval()