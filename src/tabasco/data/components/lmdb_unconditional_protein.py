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

# ==========================================
# 1. 定义符号表和全局偏移量
# ==========================================

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
    'NA']

# 记录扩展前的长度，这就是 氨基酸索引的 起始偏移量 (Offset)
# 例如，如果原表长度是 120，那么 'ALA' (0) 将变成 120
PROTEIN_OFFSET = len(chemical_symbols)

aa_map = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLU': 5, 'GLN': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
}

# 按照 index 0-19 的顺序提取 key，确保顺序正确
# sorted_aa 应该是 ['ALA', 'ARG', 'ASN', ...]
sorted_aa = sorted(aa_map.keys(), key=lambda k: aa_map[k])

# 将其加入到全局符号表中
chemical_symbols.extend(sorted_aa)

logger.info(f"Chemical symbols extended. AA start index: {PROTEIN_OFFSET}, Total symbols: {len(chemical_symbols)}")


# ==========================================
# 2. 蛋白质数据集类
# ==========================================

class ProteinLMDBDataset(BaseLMDBDataset):
    def __init__(
        self,
        data_dir: str,  # .pt 文件路径
        split: str,
        limit_samples: int = None,
        lmdb_dir: str = None,
        pad_to_max: bool = True,
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
        
        self.max_seq_len = max(self.seq_len_list)
        stats_dict = {
            "max_seq_len": self.max_seq_len,
            "min_seq_len": min(self.seq_len_list),
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
        self.max_seq_len = self.stats_dict["max_seq_len"]

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
        with db.begin(write=True, buffers=True) as txn:
            for item in tqdm(raw_data):
                # 1. 提取数据
                # 假设 item 是对象，或者是 dict
                if isinstance(item, dict):
                    c_ca = item['coords_ca']
                    x_seq = item['x']
                    # mask = item['padding_mask'] # 通常原始数据可能是全长的，不需要存 padding mask
                else:
                    c_ca = item.coords_ca
                    x_seq = item.x
                
                # 2. 转为 Numpy 存储 (节省空间)
                if isinstance(c_ca, torch.Tensor): c_ca = c_ca.numpy()
                if isinstance(x_seq, torch.Tensor): x_seq = x_seq.numpy()

                # 简单校验
                if x_seq.shape[0] == 0: continue

                store_dict = {
                    "coords_ca": c_ca,
                    "x": x_seq, # 这里存原始的 0-19，读取时再加 Offset，或者现在加也可以。
                                # 为了灵活性，通常建议存原始数据，读取时转换。
                                # 但为了代码清晰，我们在 _to_tensor 里做转换。
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
        # 将 index 转为 one-hot
        # global_indices 里的值已经是 120 ~ 139 了
        atomics = torch.zeros(global_indices.shape[0], num_classes, dtype=torch.float32)
        # scatter 或直接赋值
        # atomics[i, index] = 1
        atomics.scatter_(1, global_indices.unsqueeze(1), 1.0)
        return atomics

    def _to_tensor(self, data: dict, pad_to_size: int) -> TensorDict:
        # 1. 加载并转换类型
        x_raw = torch.tensor(data['x'], dtype=torch.long).squeeze()   # [N] (0-19)
        coords = torch.tensor(data['coords_ca'], dtype=torch.float32) # [N, 3]

        # =========================================================
        # 核心逻辑：应用 Offset，将 0-19 映射到全局 ID (如 120-139)
        # =========================================================
        x_global = x_raw + PROTEIN_OFFSET 

        # 2. 截断逻辑
        num_res = x_global.shape[0]
        if self.pad_to_max:
            target_len = pad_to_size
        else:
            target_len = num_res # 如果不做 batch padding

        if num_res > target_len:
            x_global = x_global[:target_len]
            coords = coords[:target_len]
            num_res = target_len

        # 3. 计算 Padding 长度
        pad_size = target_len - num_res
        
        # 4. 生成 One-Hot (基于全局 ID)
        # 结果形状: [num_res, len(chemical_symbols)] (未 padding)
        atomics = self._get_atomics(x_global)

        # 5. 执行 Padding
        # Mask: 0 (False) = Real, 1 (True) = Pad
        padding_mask = torch.cat([
            torch.zeros(num_res, dtype=torch.bool),
            torch.ones(pad_size, dtype=torch.bool)
        ])

        # Pad Atomics (补 0 向量) -> [Target, Total_Vocab]
        atomics_padded = torch.cat([
            atomics,
            torch.zeros(pad_size, len(chemical_symbols), dtype=torch.float32)
        ])

        # Pad Coords -> [Target, 3]
        coords_padded = torch.cat([
            coords,
            torch.zeros(pad_size, 3, dtype=torch.float32)
        ])

        # Pad Indices (补 0 或者特殊 token，这里补 0 会指向 X，或者可以补 PROTEIN_OFFSET 之外的值)
        # 通常补 0 (Element 'X') 是安全的，因为 padding_mask 会把它盖住
        x_padded = torch.cat([
            x_global,
            torch.zeros(pad_size, dtype=torch.long)
        ])

        return TensorDict(
            {
                "atomics": atomics_padded,    # [L, Total_Vocab] 全局 One-Hot
                "coords": coords_padded,      # [L, 3] CA 坐标
                "x": x_padded,                # [L] 全局 Index (120-139)
                "padding_mask": padding_mask, # [L]
                "data_type": torch.tensor([2], dtype=torch.int32)
            }
        )

    def __getitem__(self, index: int) -> TensorDict:
        data_dict = super().__getitem__(index)
        # super返回的是包含 'coords_ca', 'x' 的字典
        
        tensor_out = self._to_tensor(
            data=data_dict,
            pad_to_size=self.max_seq_len
        )
        tensor_out["index"] = index
        return tensor_out