from typing import Optional, List, Iterator

from lightning import LightningDataModule
from tabasco.data.utils import TensorDictCollator
from tabasco.data.components.lmdb_unconditional import UnconditionalLMDBDataset
from tabasco.data.components.lmdb_unconditional_crystal import CrystalLMDBDataset
from torch.utils.data import DataLoader
from tabasco.utils import RankedLogger
from torch.utils.data import DataLoader, ConcatDataset, BatchSampler, Sampler
import random

log = RankedLogger(__name__, rank_zero_only=True)

class MixedBatchSampler(BatchSampler):
    def __init__(self, mol_len: int, crystal_len: int, batch_size: int, 
                 drop_last: bool = False, shuffle: bool = True,
                 num_replicas: int = 1, rank: int = 0):
        """
        Args:
            num_replicas (int): 分布式训练的总进程数 (World Size)。
            rank (int): 当前进程的 ID (Global Rank)。
        """
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # 1. 生成所有索引
        mol_indices = list(range(mol_len))
        crystal_indices = list(range(mol_len, mol_len + crystal_len))
        
        # 2. DDP 切分 (Sharding)
        # 这一步至关重要：每个 GPU 只取属于它那一部分的数据
        if num_replicas > 1:
            # 保证每个 GPU 分到的数据量尽量均匀
            # 分子数据切分
            num_samples_mol = int(math.ceil(len(mol_indices) * 1.0 / num_replicas))
            total_size_mol = num_samples_mol * num_replicas
            # 如果需要补齐数据防止越界（可选，这里简单处理，直接切片）
            # 简单切片方式：[rank::num_replicas] (步长切片)
            mol_indices = mol_indices[rank::num_replicas]
            
            # 晶体数据切分
            crystal_indices = crystal_indices[rank::num_replicas]

        self.mol_indices_local = mol_indices
        self.crystal_indices_local = crystal_indices
        
        # 计算当前 GPU 上的 batches
        self.batches = self._generate_batches()

    def _generate_batches(self):
        mol_batches = self._create_batches(self.mol_indices_local)
        crystal_batches = self._create_batches(self.crystal_indices_local)
        batches = mol_batches + crystal_batches
        
        if self.shuffle:
            random.shuffle(batches)
        return batches

    def _create_batches(self, indices: List[int]) -> List[List[int]]:
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        # 每个 epoch 重新生成并打乱顺序
        self.batches = self._generate_batches()
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        return len(self.batches)

class LmdbDataModule(LightningDataModule):
    """PyTorch Lightning `DataModule` for unconditional ligand generation."""

    def __init__(
        self,
        data_dir: str,
        mol_lmdb_dir: str,
        crystal_lmdb_dir: str,
        add_random_rotation: bool = False,
        add_random_permutation: bool = False,
        reorder_to_smiles_order: bool = False,
        remove_hydrogens: bool = True,
        batch_size: int = 256,
        num_workers: int = 0,
        val_data_dir: Optional[str] = None,
        test_data_dir: Optional[str] = None,
        train_materials: bool = False,
        train_molecules: bool = True,
        mol_data_dir: Optional[str] = None,
        crystal_data_dir: Optional[str] = None,
        mol_val_data_dir: Optional[str] = None,
        crystal_val_data_dir: Optional[str] = None,
        mol_test_data_dir: Optional[str] = None,
        crystal_test_data_dir: Optional[str] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mol_data_dir = mol_data_dir
        self.crystal_data_dir = crystal_data_dir
        self.mol_val_data_dir = mol_val_data_dir
        self.crystal_val_data_dir = crystal_val_data_dir
        self.mol_test_data_dir = mol_test_data_dir
        self.crystal_test_data_dir = crystal_test_data_dir
        self.mol_lmdb_dir = mol_lmdb_dir
        self.crystal_lmdb_dir = crystal_lmdb_dir
        self.train_molecules = train_molecules
        self.train_materials = train_materials
        self.crystal_dataset_kwargs = {
            "add_random_rotation": add_random_rotation,
            "add_random_permutation": add_random_permutation,
        }
        self.mol_dataset_kwargs = {
            "add_random_rotation": add_random_rotation,
            "add_random_permutation": add_random_permutation,
            "reorder_to_smiles_order": reorder_to_smiles_order,
            "remove_hydrogens": remove_hydrogens,
        }
        self.mol_train_len = 0
        self.crystal_train_len = 0
        self.mol_val_len = 0
        self.crystal_val_len = 0
        self.mol_test_len = 0
        self.crystal_test_len = 0
        """Args:
            data_dir: Path to the training set .pt file produced by preprocessing.
            lmdb_dir: Directory where LMDB files and stats are stored.
            add_random_rotation: Apply random rotations inside each dataset item.
            add_random_permutation: Randomly permute heavy-atom order in each item.
            reorder_to_smiles_order: Re-index atoms to canonical SMILES before tensorization.
            remove_hydrogens: Strip explicit hydrogens before conversion to tensors.
            batch_size: Number of molecules per batch.
            num_workers: DataLoader worker count.
            val_data_dir: Optional path to a separate validation set; if None, a train/val split is created.
            test_data_dir: Optional path to a held-out test set.
        """

    def prepare_data(self):
        """Create LMDB files if they are missing (handled lazily by dataset)."""
        return

    def setup(self, stage: Optional[str] = None):
        """
        Instantiate train/val/test datasets.
        Loads molecules and/or materials and combines them using ConcatDataset.
        """
        train_datasets = []
        val_datasets = []
        test_datasets = []

        # --- 1. 加载分子数据集 ---
        if self.train_molecules:
            if self.mol_data_dir:
                log.info(f"Loading molecule train dataset from: {self.mol_data_dir}")
                mol_train_ds = UnconditionalLMDBDataset(
                    data_dir=self.mol_data_dir,
                    split="train",
                    lmdb_dir=self.mol_lmdb_dir,
                    **self.mol_dataset_kwargs,
                )
                train_datasets.append(mol_train_ds)
                self.mol_train_len = len(mol_train_ds) # <--- 存储长度

            if self.mol_val_data_dir:
                # (val 和 test 的逻辑保持不变，继续使用 ConcatDataset)
                log.info(f"Loading molecule val dataset from: {self.mol_val_data_dir}")
                mol_val_ds = UnconditionalLMDBDataset(
                    data_dir=self.mol_val_data_dir,
                    split="val",
                    lmdb_dir=self.mol_lmdb_dir,
                    **self.mol_dataset_kwargs,
                )
                val_datasets.append(mol_val_ds)
                self.mol_val_len = len(mol_val_ds) # <--- 存储长度
            if self.mol_test_data_dir:
                log.info(f"Loading molecule test dataset from: {self.mol_test_data_dir}")
                mol_test_ds = UnconditionalLMDBDataset(
                    data_dir=self.mol_test_data_dir,
                    split="test",
                    lmdb_dir=self.mol_lmdb_dir,
                    **self.mol_dataset_kwargs,
                )
                test_datasets.append(mol_test_ds)
                self.mol_test_len = len(mol_test_ds) # <--- 存储长度
        # --- 2. 加载材料 (晶体) 数据集 ---
        if self.train_materials:
            if self.crystal_data_dir:
                log.info(f"Loading material train dataset from: {self.crystal_data_dir}")
                crystal_train_ds = CrystalLMDBDataset(
                    data_dir=self.crystal_data_dir,
                    split="train",
                    lmdb_dir=self.crystal_lmdb_dir,
                    **self.crystal_dataset_kwargs,
                )
                train_datasets.append(crystal_train_ds)
                self.crystal_train_len = len(crystal_train_ds) # <--- 存储长度

            if self.crystal_val_data_dir:
                log.info(f"Loading material val dataset from: {self.crystal_val_data_dir}")
                crystal_val_ds = CrystalLMDBDataset(
                    data_dir=self.crystal_val_data_dir,
                    split="val",
                    lmdb_dir=self.crystal_lmdb_dir,
                    **self.crystal_dataset_kwargs,
                )
                val_datasets.append(crystal_val_ds)
                self.crystal_val_len = len(crystal_val_ds) # <--- 存储长度
            if self.crystal_test_data_dir:
                log.info(f"Loading material test dataset from: {self.crystal_test_data_dir}")
                crystal_test_ds = CrystalLMDBDataset(
                    data_dir=self.crystal_test_data_dir,
                    split="test",
                    lmdb_dir=self.crystal_lmdb_dir,
                    **self.crystal_dataset_kwargs,
                )
                test_datasets.append(crystal_test_ds)
                self.crystal_test_len = len(crystal_test_ds) # <--- 存储长度
        # --- 3. 组合数据集 ---
        if not train_datasets:
            raise ValueError("No training datasets loaded. Set 'train_molecules' or 'train_materials' to True and provide valid data paths.")
        
        # 我们仍然使用 ConcatDataset，因为 MixedBatchSampler 依赖于连续的索引
        self.train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
        self.val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
        self.test_dataset = ConcatDataset(test_datasets) if len(test_datasets) > 1 else test_datasets[0]

    def get_dataset_stats(self):
        # (此方法保持不变)
        if self.train_dataset is None:
            raise RuntimeError("Run setup() before calling get_dataset_stats()")
        if isinstance(self.train_dataset, ConcatDataset):
            first_dataset = self.train_dataset.datasets[0]
            log.warning(f"Returning stats from the first dataset only: {type(first_dataset).__name__}")
            return first_dataset.get_stats()
        return self.train_dataset.get_stats()

    def train_dataloader(self):
        """
        Return the training `DataLoader`.
        使用 MixedBatchSampler 来随机抽取纯批次。
        """
        if not self.train_dataset:
            raise RuntimeError("No training dataset configured.")
            
        # 确保 setup() 已经运行并设置了长度
        if self.mol_train_len == 0 and self.crystal_train_len == 0:
             log.warning("Dataset lengths are zero. Forcing setup().")
             self.setup()

        # 1. 创建自定义的 BatchSampler
        sampler = MixedBatchSampler(
            mol_len=self.mol_train_len,
            crystal_len=self.crystal_train_len,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
        )

        # 2. 创建 DataLoader
        # 注意：当提供了 batch_sampler 时，
        # batch_size, shuffle, drop_last 必须为 None (或默认值)。
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler, # <--- 使用自定义的 sampler
            num_workers=self.num_workers,
            collate_fn=TensorDictCollator(),
        )

    def val_dataloader(self):
        """Return the validation `DataLoader`."""
        sampler = MixedBatchSampler(
            mol_len=self.mol_val_len,
            crystal_len=self.crystal_val_len,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=False,
        )
        return DataLoader(
            self.val_dataset,
            batch_sampler=sampler, # <--- 使用自定义的 sampler
            num_workers=self.num_workers,
            collate_fn=TensorDictCollator(),
        )

    def test_dataloader(self):
        """Return the test `DataLoader` (falls back to validation set when absent)."""
        sampler = MixedBatchSampler(
            mol_len=self.mol_test_len,
            crystal_len=self.crystal_test_len,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=False,
        )
        return DataLoader(
            self.test_dataset,
            batch_sampler=sampler, # <--- 使用自定义的 sampler
            num_workers=self.num_workers,
            collate_fn=TensorDictCollator(),
        )
