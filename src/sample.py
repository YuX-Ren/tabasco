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

# Manually setting the configuration dictionary (cfg)
cfg = {
    "data_dir": "./data/processed_qm9_train.pt",
    "val_data_dir": "./data/processed_qm9_val.pt",
    "test_data_dir": "./data/processed_qm9_test.pt",
    "lmdb_dir": "./data/lmdb_qm9",
    "add_random_rotation": True,
    "add_random_permutation": False,
    "reorder_to_smiles_order": True,
    "remove_hydrogens": True,
    "batch_size": 256,
    "num_workers": 0
}

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
    for batch in datamodule.val_dataloader():
        batch = batch.to("cuda")
        out_batch = sample_batch(
            lightning_module=lightning_module,
            batch=batch,
            ema_optimizer=ema_optimizer,
            num_steps=num_steps,
        )
        out_batch_list.append(out_batch)
        break
    # calculate the rmsd between the generated and reference molecules
    rmsd = lightning_module.mol_converter.rmsd_calculation(out_batch, batch)
    print(f"rmsd: {rmsd}")
    mse = out_batch["coords"] - batch["coords"]
    mse = mse.pow(2).mean()
    print(f"mse: {mse}")
    # Concatenate results from all batches
    out_batch = torch.cat(out_batch_list, dim=0)

    # Export the sampled batch to pickle if specified
    if args.output_path is not None:
        export_batch_to_sdf(batch, args.output_path.replace(".pkl", "_ref.pkl"))
        export_batch_to_sdf(out_batch, args.output_path)


if __name__ == "__main__":
    main()
