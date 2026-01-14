from typing import Optional, List, Iterator

from lightning import LightningDataModule
from tabasco.data.utils import TensorDictCollator
from tabasco.data.components.lmdb_unconditional import UnconditionalLMDBDataset
from tabasco.data.components.lmdb_unconditional_crystal import CrystalLMDBDataset
from torch.utils.data import DataLoader
from tabasco.utils import RankedLogger
from torch.utils.data import DataLoader, ConcatDataset, BatchSampler, Sampler
import random, math
import torch
import numpy as np
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

class TrueMixedBatchSampler(Sampler):
    def __init__(self, mol_len: int, crystal_len: int, batch_size: int, 
                 num_replicas: int = 1, rank: int = 0, shuffle: bool = True):
        """
        构造混合 Batch：
        每个 Batch 包含 batch_size // 2 个分子 和 batch_size // 2 个晶体。
        """
        self.mol_len = mol_len
        self.crystal_len = crystal_len
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        
        # 确保 batch_size 是偶数以便 50/50 分配
        assert batch_size % 2 == 0, "Batch size must be even for 50/50 split."
        self.mol_batch_size = batch_size // 2
        self.crystal_batch_size = batch_size // 2

        # 计算当前 Rank 应该分到的总样本数（用于 __len__）
        # 这里以较大的数据集为基准，较小的循环采样
        self.max_len = max(mol_len, crystal_len)
        self.num_samples = int(math.ceil(self.max_len / self.num_replicas))
        self.num_batches = int(math.ceil(self.num_samples / (self.batch_size // 2)))

    def __iter__(self):
        # 1. 生成全局索引
        mol_indices = torch.arange(self.mol_len)
        # 晶体索引要在 Dataset 中偏移 mol_len
        crystal_indices = torch.arange(self.mol_len, self.mol_len + self.crystal_len)

        # 2. DDP 切分 (Subsampling)
        # 确定 deterministic 的种子，保证每个 epoch 不同但各卡同步
        g = torch.Generator()
        g.manual_seed(self.rank + 0) # 这里的 seed 可以加 epoch 偏移如果是在 set_epoch 调用中

        if self.shuffle:
            mol_indices = mol_indices[torch.randperm(self.mol_len, generator=g)]
            crystal_indices = crystal_indices[torch.randperm(self.crystal_len, generator=g)]

        # 简单的 DDP 切分：直接按 rank 取余是不够随机的，最好是 chunk
        # 这里为了简化，假设已经 shuffle 过了，直接切片
        # 注意：为了混合，我们不对“总池子”切分，而是让每个 Rank 都遍历自己的那部分
        # 更好的策略：每个 Rank 负责 Dataset 的一部分
        
        mol_indices_local = mol_indices[self.rank::self.num_replicas]
        crystal_indices_local = crystal_indices[self.rank::self.num_replicas]

        # 3. 处理长度不一致：循环较短的那个
        max_len_local = max(len(mol_indices_local), len(crystal_indices_local))
        
        def infinite_iterator(indices):
            while True:
                for idx in indices:
                    yield idx
        
        mol_iter = infinite_iterator(mol_indices_local)
        crystal_iter = infinite_iterator(crystal_indices_local)

        # 4. 生成 Batches
        # 计算当前卡需要产出多少个 batch
        # 我们以覆盖所有数据为目标
        num_batches_local = max(
            int(math.ceil(len(mol_indices_local) / self.mol_batch_size)),
            int(math.ceil(len(crystal_indices_local) / self.crystal_batch_size))
        )

        for _ in range(num_batches_local):
            batch = []
            # 取分子
            for _ in range(self.mol_batch_size):
                batch.append(next(mol_iter).item())
            # 取晶体
            for _ in range(self.crystal_batch_size):
                batch.append(next(crystal_iter).item())
            
            # (可选) 在 Batch 内部再次 Shuffle，打乱分子和晶体的顺序
            # 这样进入模型时不是前一半分子后一半晶体
            np.random.shuffle(batch)
            
            yield batch

    def __len__(self):
        # 估算长度
        mol_local = math.ceil(self.mol_len / self.num_replicas)
        crys_local = math.ceil(self.crystal_len / self.num_replicas)
        return max(
            math.ceil(mol_local / self.mol_batch_size),
            math.ceil(crys_local / self.crystal_batch_size)
        )

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
        sampler = TrueMixedBatchSampler(
            mol_len=self.mol_train_len,
            crystal_len=self.crystal_train_len,
            batch_size=self.batch_size,
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
        sampler = TrueMixedBatchSampler(
            mol_len=self.mol_val_len,
            crystal_len=self.crystal_val_len,
            batch_size=self.batch_size,
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
        sampler = TrueMixedBatchSampler(
            mol_len=self.mol_test_len,
            crystal_len=self.crystal_test_len,
            batch_size=self.batch_size,
            shuffle=False,
        )
        return DataLoader(
            self.test_dataset,
            batch_sampler=sampler, # <--- 使用自定义的 sampler
            num_workers=self.num_workers,
            collate_fn=TensorDictCollator(),
        )
