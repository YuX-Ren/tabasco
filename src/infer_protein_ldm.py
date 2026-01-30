import argparse
import os

import hydra
import lightning as L
import numpy as np
import torch
from omegaconf import OmegaConf

from tabasco.chem.utils import write_coords_to_pdb
from tabasco.models.proteinae_ldm import ProteinLDM


def load_cfg(config_root: str, config_name: str):
    if os.path.isabs(config_root):
        with hydra.initialize_config_dir(
            config_dir=config_root, version_base=hydra.__version__
        ):
            return hydra.compose(config_name=config_name)
    with hydra.initialize(config_path=config_root, version_base=hydra.__version__):
        return hydra.compose(config_name=config_name)


def resolve_ckpt_path(cfg_inf, override_ckpt: str | None):
    if override_ckpt:
        return override_ckpt
    ckpt_dir = cfg_inf.get("ckpt_path", None)
    ckpt_name = cfg_inf.get("ckpt_name", None)
    if ckpt_dir and ckpt_name:
        return os.path.join(ckpt_dir, ckpt_name)
    raise ValueError("Checkpoint path not provided. Use --ckpt or set ckpt_path + ckpt_name.")


def main():
    parser = argparse.ArgumentParser(description="ProteinLDM inference")
    parser.add_argument("--exp_config", type=str, default="training_pldm_200M_afdb_512")
    parser.add_argument("--inf_config", type=str, default="inference_ucond_pldm_200M_512")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="./inference_outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--coord_scale", type=float, default=4.0)
    args = parser.parse_args()

    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

    config_root = os.path.join(os.path.dirname(__file__), "..", "configs", "experiment_config")
    cfg_exp = load_cfg(config_root, args.exp_config)
    cfg_inf = load_cfg(config_root, args.inf_config)
    OmegaConf.set_struct(cfg_inf, False)
    if "inv_folding" not in cfg_inf:
        cfg_inf.inv_folding = False

    ckpt_path = resolve_ckpt_path(cfg_inf, args.ckpt)
    os.makedirs(args.out_dir, exist_ok=True)

    model = ProteinLDM(cfg_exp, store_dir=args.out_dir)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
    if missing:
        print(f"Warning: missing keys: {len(missing)}")
    if unexpected:
        print(f"Warning: unexpected keys: {len(unexpected)}")

    model.eval()
    model.to(args.device)
    model.inf_cfg = cfg_inf

    nres_lens = cfg_inf.get("nres_lens", [])
    if not nres_lens:
        raise ValueError("nres_lens is empty in inference config.")
    nsamples_per_len = int(cfg_inf.get("nsamples_per_len", 1))
    max_nsamples = int(cfg_inf.get("max_nsamples", nsamples_per_len))

    dt = torch.tensor(cfg_inf.get("dt", 0.0025), device=args.device)
    dt_latent = torch.tensor(cfg_inf.get("dt_latent", 0.0025), device=args.device)

    for nres in range(min(nres_lens), max(nres_lens) + 1):
        out_len_dir = os.path.join(args.out_dir, f"pdbs")
        os.makedirs(out_len_dir, exist_ok=True)
        remaining = nsamples_per_len
        batch_idx = 0
        while remaining > 0:
            cur_bs = min(max_nsamples, remaining)
            batch = {
                "nsamples": cur_bs,
                "nres": int(nres),
                "dt": dt,
                "dt_latent": dt_latent,
            }
            with torch.no_grad():
                out = model.predict_step(batch, batch_idx)
            coords = out.get("pred_coords", None)
            if coords is not None:
                coords = coords.detach().cpu().numpy()
                if coords.ndim == 4 and coords.shape[-2:] == (37, 3):
                    coords = coords[:, :, 1, :]  # CA atom
                coords = coords * float(args.coord_scale)
                for i in range(coords.shape[0]):
                    pdb_path = os.path.join(
                        out_len_dir, f"sample_{nres}_{i:03d}.pdb"
                    )
                    write_coords_to_pdb(coords[i].astype(np.float32), pdb_path)
            remaining -= cur_bs
            batch_idx += 1

    print(f"Saved samples to {args.out_dir}")


if __name__ == "__main__":
    main()
