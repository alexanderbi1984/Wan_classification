import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from .syracuse import SyracuseDataModule
from .biovid import BioVidDataModule
from .combined_task_wrapper import CombinedTaskDatasetWrapper

class MultimodalDataModule(pl.LightningDataModule):
    """
    LightningDataModule for multimodal, multi-task experiments combining Syracuse and BioVid datasets.
    Wraps each dataset with CombinedTaskDatasetWrapper and provides unified DataLoaders.
    """
    def __init__(
        self,
        syracuse_cfg: dict,
        biovid_cfg: dict,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        mode: str = "train",
        fold_idx: int = 0,
        **kwargs
    ):
        super().__init__()
        self.syracuse_cfg = syracuse_cfg
        self.biovid_cfg = biovid_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.mode = mode
        self.fold_idx = fold_idx
        self.syracuse_dm = None
        self.biovid_dm = None
        self._train_ds = None
        self._val_ds = None
        self._test_ds = None

    def setup(self, stage=None):
        # Instantiate underlying DataModules
        self.syracuse_dm = SyracuseDataModule(
            meta_path=self.syracuse_cfg.get("meta_path"),
            task=self.syracuse_cfg.get("task", "classification"),
            thresholds=self.syracuse_cfg.get("thresholds", None),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            cv_fold=self.syracuse_cfg.get("cv_fold", self.fold_idx),
            seed=self.seed,
            balanced_sampling=self.syracuse_cfg.get("balanced_sampling", False),
            temporal_pooling=self.syracuse_cfg.get("temporal_pooling", "none"),
            flatten=self.syracuse_cfg.get("flatten", False)
        )
        self.biovid_dm = BioVidDataModule(
            features_path=self.biovid_cfg.get("features_path"),
            meta_path=self.biovid_cfg.get("meta_path"),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            split_ratio=self.biovid_cfg.get("split_ratio", 0.8),
            seed=self.seed,
            temporal_pooling=self.biovid_cfg.get("temporal_pooling", "none"),
            flatten=self.biovid_cfg.get("flatten", False)
        )
        self.syracuse_dm.setup(stage)
        self.biovid_dm.setup(stage)
        # Wrap and combine datasets for multimodal multitask
        self._train_ds = ConcatDataset([
            CombinedTaskDatasetWrapper(self.syracuse_dm.train_dataset, task_name='pain_level', multimodal=True),
            CombinedTaskDatasetWrapper(self.biovid_dm.train_dataset, task_name='stimulus', multimodal=True)
        ])
        # Validation: Syracuse only, multimodal wrapped
        if self.syracuse_dm.val_dataset and len(self.syracuse_dm.val_dataset) > 0:
            self._val_ds = CombinedTaskDatasetWrapper(self.syracuse_dm.val_dataset, task_name='pain_level', multimodal=True)
        else:
            self._val_ds = None
        # Test: Syracuse only, multimodal wrapped (can be extended)
        if hasattr(self.syracuse_dm, 'test_dataset') and self.syracuse_dm.test_dataset is not None:
            self._test_ds = CombinedTaskDatasetWrapper(self.syracuse_dm.test_dataset, task_name='pain_level', multimodal=True)
        else:
            self._test_ds = None

    def train_dataloader(self):
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        if self._val_ds is not None:
            return DataLoader(
                self._val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )
        return None

    def test_dataloader(self):
        if self._test_ds is not None:
            return DataLoader(
                self._test_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )
        return None

    @property
    def train_dataset(self):
        return self._train_ds

    @property
    def val_dataset(self):
        return self._val_ds

    @property
    def test_dataset(self):
        return self._test_ds 