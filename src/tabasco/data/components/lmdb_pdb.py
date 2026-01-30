import os
import yaml
import pickle
from collections import Counter
import lmdb
from tqdm import tqdm
import torch
import numpy as np
from tensordict import TensorDict

from tabasco.data.components.lmdb_base import BaseLMDBDataset
from tabasco.utils import RankedLogger

logger = RankedLogger(__name__)

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og',
    # pseudo atoms
    'NA'
]

PROTEIN_OFFSET = len(chemical_symbols)

aa_map = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLU': 5, 'GLN': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
}

sorted_aa = sorted(aa_map.keys(), key=lambda k: aa_map[k])

chemical_symbols.extend(sorted_aa)

logger.info(f"Chemical symbols extended. AA start index: {PROTEIN_OFFSET}, Total symbols: {len(chemical_symbols)}")



class ProteinLMDBDataset(BaseLMDBDataset):
    def __init__(
        self,
        data_dir: str,  
        split: str,
        limit_samples: int = None,
        lmdb_dir: str = None,
        pad_to_max: bool = True,
        normalize_coef: float = 4.0,
    ):
        """
        Args:
            pad_to_max: 是否将所有样本 Padding 到数据集的最大序列长度。
        """
        super().__init__(
            split=split,
            single_sample=False,
            limit_samples=limit_samples,
            lmdb_dir=lmdb_dir,
        )
        self.split = split
        self.data_dir = data_dir
        self.pad_to_max = pad_to_max
        self.seq_len_list = []
        self.normalize_coef = normalize_coef
        if not (os.path.exists(self.lmdb_path)):
            self._process()
        else:
            logger.info(f"Loading existing LMDB from {self.lmdb_path}.")
            self._load_stats()

    def _update_stats(self, data_dict: dict):
        try:
            # 统计序列长度
            seq_len = data_dict["x"].shape[0]
            self.seq_len_list.append(seq_len)
        except Exception as e:
            logger.warning(f"Error updating stats: {e}")

    def compute_stats(self):
        if not self.seq_len_list:
            raise ValueError("No data processed.")
        example_datapoint = self.__getitem__(0)
        self.max_seq_len = max(self.seq_len_list)
        stats_dict = {
            "spatial_dim": example_datapoint["coords"].shape[1],
            "atom_dim": example_datapoint["atomics"].shape[1],
            "count": len(self.seq_len_list),
            "total_vocab_size": len(chemical_symbols)
        }
        self.stats_dict = stats_dict
        yaml.dump(
            stats_dict,
            open(os.path.join(self.lmdb_dir, f"{self.split}_stats.yaml"), "w"),
        )
        logger.info(f"Stats computed. Max seq len: {self.max_seq_len}")

    def _load_stats(self):
        stats_path = os.path.join(self.lmdb_dir, f"{self.split}_stats.yaml")
        with open(stats_path, "r") as f:
            self.stats_dict = yaml.safe_load(f)

    def get_stats(self):
        """Get the dataset statistics."""
        return self.stats_dict

    def _process(self):
        """读取 .pt 并存入 LMDB"""
        db = lmdb.open(
            self.lmdb_path,
            map_size=10 * (1024 * 1024 * 1024),
            create=True,
            subdir=False,
            readonly=False,
        )

        logger.info(f"Loading raw data from {self.data_dir}...")
        raw_data = torch.load(self.data_dir, weights_only=False) # list of objects or dicts

        idx_counter = 0
        print(f"Processing {len(raw_data)} samples...")
        with db.begin(write=True, buffers=True) as txn:
            for item in tqdm(raw_data):

                if isinstance(item, dict):
                    c_ca = item['coords_ca']
                    x_seq = item['x']
                    padding_mask = item['padding_mask']
                else:
                    c_ca = item.coords_ca
                    x_seq = item.x
                    padding_mask = item.padding_mask
                if isinstance(c_ca, torch.Tensor): c_ca = c_ca.numpy()
                if isinstance(x_seq, torch.Tensor): x_seq = x_seq.numpy()
                if isinstance(padding_mask, torch.Tensor): padding_mask = padding_mask.numpy()

                if x_seq.shape[0] == 0: continue

                store_dict = {
                    "coords_ca": c_ca,
                    "x": x_seq, 
                    "padding_mask": padding_mask,
                }

                self._update_stats(store_dict)
                txn.put(key=str(idx_counter).encode(), value=pickle.dumps(store_dict))
                idx_counter += 1

                if self.limit_samples and idx_counter >= self.limit_samples:
                    break
        
        db.close()
        self.compute_stats()

    def _get_atomics(self, global_indices: torch.Tensor) -> torch.Tensor:
        """
        生成全局符号表的 One-Hot 编码。
        shape: (N, len(chemical_symbols))
        """
        num_classes = len(chemical_symbols)
        atomics = torch.zeros(global_indices.shape[0], num_classes, dtype=torch.float32)
        atomics.scatter_(1, global_indices.unsqueeze(1), 1.0)
        return atomics

    def _to_tensor(self, data: dict) -> TensorDict:
        padding_mask = torch.tensor(data['padding_mask'])
        real_mask = ~padding_mask
        x_raw = torch.tensor(data['x'], dtype=torch.long).squeeze()   # [N] (0-19)
        coords = torch.tensor(data['coords_ca'], dtype=torch.float32) / self.normalize_coef # [N, 3]

        x_global = x_raw * real_mask + PROTEIN_OFFSET 
        atomics = self._get_atomics(x_global) * real_mask.unsqueeze(1)

        return TensorDict(
            {
                "atomics": atomics,    
                "coords": coords,      
                "padding_mask": padding_mask, 
                "dataset_idx": torch.tensor([2], dtype=torch.int32)
            }
        )

    def __getitem__(self, index: int) -> TensorDict:
        data_dict = super().__getitem__(index)
        
        tensor_out = self._to_tensor(data=data_dict)
        tensor_out["index"] = index
        return tensor_out