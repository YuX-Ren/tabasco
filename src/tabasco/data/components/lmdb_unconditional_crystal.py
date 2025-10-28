import os
import yaml
import pickle
from collections import Counter
import lmdb
from tqdm import tqdm
import torch
import numpy as np
from tensordict import TensorDict

# --- 假设这些
# --- 假设这些来自您之前的代码库 ---
# (如果您没有 BaseLMDBDataset，请告诉我，我需要重新添加它)
from tabasco.data.components.lmdb_base import BaseLMDBDataset
from tabasco.data.transforms import random_rotation, permute_atoms
from tabasco.utils import RankedLogger
# -------------------------------------

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
    'Lv', 'Ts', 'Og']

class CrystalLMDBDataset(BaseLMDBDataset):
    """
    改编自 UnconditionalLMDBDataset，用于处理预先计算好的晶体结构字典。
    它将晶体数据（atom_types, frac_coords, cell）存储在 LMDB 中。
    """

    def __init__(
        self,
        data_dir: str,  # 路径应指向您的 .pt 文件
        split: str,
        add_random_rotation: bool = True,
        add_random_permutation: bool = True,
        single_sample: bool = False,
        limit_samples: int = None,
        lmdb_dir: str = None,
    ):
        """
        初始化数据集。

        Args:
            data_dir: 指向包含 [list[dict]] 的 .pt 文件的路径。
                      每个 dict 应包含键: 'atom_types', 'frac_coords', 'cell'。
            split: 数据集拆分 (e.g., 'train', 'val')。
            add_random_rotation: （可选）随机旋转原子坐标。
            add_random_permutation: （可选）随机置换原子顺序。
            single_sample: 始终返回第一个样本（用于调试）。
            limit_samples: 限制处理的样本数量。
            lmdb_dir: 存储 LMDB 和 stats.yaml 文件的目录。
        """
        super().__init__(
            split=split,
            single_sample=single_sample,
            limit_samples=limit_samples,
            lmdb_dir=lmdb_dir,
        )
        self.split = split
        self.data_dir = data_dir  # 这是 .pt 文件的路径

        # (可选) 数据增强
        self.add_random_rotation = add_random_rotation
        self.add_random_permutation = add_random_permutation

        self.db = None
        self.keys = None

        self.num_atoms_list = [] # 用于统计

        if not (os.path.exists(self.lmdb_path)):
            logger.info(f"LMDB not found at {self.lmdb_path}. Processing data...")
            self._process()
        else:
            logger.info(f"Loading existing LMDB from {self.lmdb_path}.")
            self._load_stats()

    def _update_stats(self, crystal_data_dict: dict):
        """在 _process 期间从 Numpy 字典中提取统计数据。"""
        try:
            num_atoms = crystal_data_dict["atom_types"].shape[0]
            self.num_atoms_list.append(num_atoms)
        except Exception as e:
            logger.warning(f"Error updating stats: {str(e)[:100]}")

    def compute_stats(self):
        """计算摘要统计数据并写入磁盘。"""
        num_atoms_histogram = Counter(self.num_atoms_list)
        
        if not num_atoms_histogram:
            raise ValueError("No data processed. Check your .pt file or _process method.")

        self.max_num_atoms = max(num_atoms_histogram.keys())

        _original_rot = self.add_random_rotation
        _original_perm = self.add_random_permutation
        self.add_random_rotation = False
        self.add_random_permutation = False
        example_datapoint = self.__getitem__(0)
        self.add_random_rotation = _original_rot
        self.add_random_permutation = _original_perm

        stats_dict = {
            "num_atoms_histogram": dict(num_atoms_histogram),
            "max_num_atoms": self.max_num_atoms,
            "spatial_dim": example_datapoint["coords"].shape[1], # 应该是 3
        }

        self.stats_dict = stats_dict
        yaml.dump(
            stats_dict,
            open(os.path.join(self.lmdb_dir, f"{self.split}_stats.yaml"), "w"),
        )
        logger.info(f"Stats computed and saved to YAML. Max atoms: {self.max_num_atoms}")

    def _load_stats(self):
        """从 YAML 文件加载先前计算的统计数据。"""
        stats_path = os.path.join(self.lmdb_dir, f"{self.split}_stats.yaml")
        logger.info(f"Loading stats from {stats_path}")
        with open(stats_path, "r") as f:
            self.stats_dict = yaml.safe_load(f)

        self.max_num_atoms = self.stats_dict["max_num_atoms"]
        return

    def get_stats(self):
        """获取数据集统计数据。"""
        return self.stats_dict

    def _process(self):
        """
        从 .pt 文件创建 LMDB 和统计数据。
        这只会运行一次。
        """
        db = lmdb.open(
            self.lmdb_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # 可写
        )

        # 加载 .pt 文件，假设它是一个包含数据字典的列表
        logger.info(f"Loading data from {self.data_dir}...")
        # nosec B301 是为了 pickle。torch.load 默认使用 pickle。
        # B614 是 torch.load，这里是必要的。
        crystal_data_list = torch.load(self.data_dir, weights_only=False)  # nosec B301 B614

        if not isinstance(crystal_data_list, list):
            raise TypeError(f"Expected data from {self.data_dir} to be a list, but got {type(crystal_data_list)}")
            
        logger.info(f"Loaded {len(crystal_data_list)} data points. Processing into LMDB...")

        idx_counter = 0
        with db.begin(write=True, buffers=True) as txn:
            # 假设 crystal_data_list 是 [dict, dict, ...]
            # 每个 dict 包含 {'atom_types': np.array, 'frac_coords': np.array, 'cell': np.array}
            for i, crystal_data_dict in enumerate(tqdm(crystal_data_list, total=len(crystal_data_list))):
                crystal_data_dict = crystal_data_dict['graph_arrays']
                try:
                    # 运行统计
                    self._update_stats(crystal_data_dict)
                except Exception as e:
                    logger.warning(
                        f"Error updating stats for index {i}: {e}"
                    )
                    continue

                data_to_store = {
                    "crystal_data": crystal_data_dict,
                }

                txn.put(key=str(idx_counter).encode(), value=pickle.dumps(data_to_store))
                idx_counter += 1
                
                if self.limit_samples is not None and idx_counter >= self.limit_samples:
                    logger.info(f"Reached sample limit of {self.limit_samples}. Stopping processing.")
                    break

        db.close()
        logger.info(f"LMDB processing complete. {idx_counter} items stored.")

        if idx_counter == 0:
            logger.error("No data was successfully processed and stored in LMDB.")
            return

        self.compute_stats()
        
    def _get_atomics(self, crystal_data: dict) -> torch.Tensor:
        """Return one-hot atom-type matrix of shape `(N, n_elements)`."""
        atom_numbers = crystal_data['atom_types']
        atomics = torch.zeros(
            len(atom_numbers), len(chemical_symbols), dtype=torch.float32
        )
        for i, atom_number in enumerate(atom_numbers):
            atomics[i, atom_number] = 1.0
        return atomics

    def _get_atom_types(
        self, atomics: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> list[str]:
        """Return list of element symbols, ignoring rows flagged in
        `padding_mask` if provided."""
        if padding_mask is not None:
            atomics = atomics[~padding_mask]

        return [chemical_symbols[i] for i in torch.argmax(atomics, dim=1)]

    def _to_tensor(self, crystal_data: dict, pad_to_size: int) -> TensorDict:
        atomics = self._get_atomics(crystal_data)
        frac_coords = torch.tensor(crystal_data['frac_coords'], dtype=torch.float32)
        cell = torch.tensor(crystal_data['cell'], dtype=torch.float32)
        num_atoms = atomics.shape[0]
        if num_atoms > pad_to_size:
            logger.warning(f"Found molecule with {num_atoms} atoms, but max_num_atoms is {pad_to_size}. Truncating.")
            num_atoms = pad_to_size
            atomics = atomics[:pad_to_size]
            frac_coords = frac_coords[:pad_to_size]

        pad_size = pad_to_size - num_atoms

        padding_mask = torch.cat(
            [torch.zeros(num_atoms, dtype=torch.bool), 
             torch.ones(pad_size, dtype=torch.bool)]
        )
        
        atomics_padded = torch.cat(
            [atomics, 
             torch.zeros(pad_size, len(chemical_symbols), dtype=torch.float32)]
        )
        
        frac_coords_padded = torch.cat(
            [frac_coords, 
             torch.zeros(pad_size, 3, dtype=torch.float32)]
        )

        return TensorDict(
            {
                "atomics": atomics_padded,   # [N_pad] (原子序数)
                "coords": frac_coords_padded, # [N_pad, 3]
                "lattices": cell,                      # [3, 3] (晶胞矩阵)
                "padding_mask": padding_mask,      # [N_pad] (True/False)
            }
        )

    def __getitem__(self, index: int) -> TensorDict:
        """
        返回晶体的张量表示，准备好作为模型输入。
        """
        data_dict = super().__getitem__(index)
        
        crystal_data = data_dict["crystal_data"]

        data_tensor = self._to_tensor(
            crystal_data=crystal_data,
            pad_to_size=self.max_num_atoms,
        )

        if "id" in data_dict:
            data_tensor["index"] = data_dict["id"]

        # 3. (可选) 应用实时数据增强
        # 注意: 您的 random_rotation 和 permute_atoms 函数需要
        # 能够正确处理包含 'frac_coords', 'cell', 'padding_mask' 的 TensorDict。
        # 特别是，旋转 'frac_coords'（分数坐标）在物理上可能没有意义，
        # 除非您先将其转换为笛卡尔坐标，旋转，然后再转换回来。
        # 您可能只想旋转笛卡尔坐标（如果已计算）或根本不旋转。
        
        # if self.add_random_rotation:
        #     data_tensor = random_rotation(data_tensor)

        # if self.add_random_permutation:
        #     data_tensor = permute_atoms(data_tensor)

        return data_tensor