import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from .multimodal_datasets import SyracuseMultimodalDataModule, BioVidMultimodalDataset

class MultimodalDataModule(pl.LightningDataModule):
    """
    LightningDataModule for multimodal, multi-task experiments combining Syracuse and BioVid datasets.
    Uses SyracuseMultimodalDataset and BioVidMultimodalDataset to provide unified DataLoaders.
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
        self._train_ds = None
        self._val_ds = None
        self._test_ds = None

    def setup(self, stage=None):
        # Syracuse multimodal dataset
        syr_dataset = SyracuseMultimodalDataset(
            vae_feature_dir=self.syracuse_cfg["vae_feature_dir"],
            xdit_feature_dir=self.syracuse_cfg["xdit_feature_dir"],
            meta_path=self.syracuse_cfg["meta_path"],
            task=self.syracuse_cfg.get("task", "classification"),
            thresholds=self.syracuse_cfg.get("thresholds", None),
            temporal_pooling=self.syracuse_cfg.get("temporal_pooling", "none"),
            flatten=self.syracuse_cfg.get("flatten", False)
        )
        # BioVid multimodal dataset
        biovid_dataset = BioVidMultimodalDataset(
            vae_feature_dir=self.biovid_cfg["vae_feature_dir"],
            xdit_feature_dir=self.biovid_cfg["xdit_feature_dir"],
            meta_path=self.biovid_cfg["meta_path"],
            temporal_pooling=self.biovid_cfg.get("temporal_pooling", "none"),
            flatten=self.biovid_cfg.get("flatten", False)
        )
        # For now, use all data for train/val/test (can be split as needed)
        self._train_ds = ConcatDataset([syr_dataset, biovid_dataset])
        self._val_ds = syr_dataset  # Optionally, use only Syracuse for validation
        self._test_ds = syr_dataset # Optionally, use only Syracuse for test

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