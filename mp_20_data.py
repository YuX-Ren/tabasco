import os
import torch
from materials import preprocess
data_path = "data/materials"
cached_data = preprocess(
    os.path.join(data_path, "raw/val.csv"),
    niggli=True,
    primitive=False,
    graph_method="crystalnn",
    prop_list=["formation_energy_per_atom"],
    use_space_group=False,
    tol=0.01,
    num_workers=32
)
torch.save(cached_data, os.path.join(data_path, "processed_mp_20_val.pt"))