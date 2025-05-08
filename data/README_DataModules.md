# Data Modules: BioVid & Syracuse

This README describes the BioVid and Syracuse PyTorch Lightning DataModules in this project.

## 1. BioVidDataModule (`biovid.py`)

### Purpose
- Loads BioVid data features and metadata for classification.
- Subject-wise splitting of train/val for robust, leak-free evaluation.

### Key Features
- **Input:**
  - `features_path`: Folder containing npy feature files (per clip).
  - `meta_path`: `meta.json` file with clip metadata, including `subject_id`, `source`, and `5_class_label`.
- **Splitting:**
  - All available `subject_id` values are shuffled with a deterministic random seed.
  - Split into training and validation based on `split_ratio` (default: 80% train, 20% val).
  - No subject crosses between splits.
- **Note:** By the design of the BioVid experiment, class distributions are inherently balanced across subjects/sessions, so explicit class balancing in the data split is not needed.
- **Arguments:**
  - `batch_size`, `num_workers`: DataLoader settings.
  - `split_ratio`: Ratio of train subjects (0-1).
  - `seed`: Controls subject shuffle and reproducibility.
  - `use_video_fallback`: If True, would (in the future) support falling back to loading from video if npy is missing (not implemented).
- **Usage:**
  ```python
  dm = BioVidDataModule(features_path, meta_path)
  dm.setup()
  print(dm)
  train_loader = dm.train_dataloader()
  val_loader = dm.val_dataloader()
  ```
- **Output:**
  - Each batch provides (feature tensor, subject_id, class_label) for training.

---

## 2. SyracuseDataModule (`syracuse.py`)

### Purpose
- Handles Syracuse dataset, supporting both pain classification (with YAML-configured thresholds) and regression.
- Provides robust train/val splits with controlled stratification or pseudo-stratification.

### Key Features
- **Input:**
  - `meta_path`: `meta.json` with metadata for each data sample (pain level, source, video/clip ids).
  - `config_path`: YAML with settings (`task`, thresholds for classification, etc.).
- **Splitting:**
  - By default, 3-fold cross-validation on video_id (using only `syracuse_original` as the base for split).
  - For classification: stratified split by label (per thresholds in config).
  - For regression: always uses 5 uniform bins (0–2–4–6–8–10) on pain_level for pseudo-stratification; split labels are NOT used for the model, only for balancing.
  - For training set: includes both `syracuse_original` and `syracuse_aug` clips of the relevant videos. Val always uses only originals.
- **Arguments:**
  - `cv_fold`: 0/1/2 for fold selection; ≥3 means all data is training.
  - `batch_size`, `num_workers`, `balanced_sampling`, `seed` as in standard PyTorch Lightning.
  - `transform`: Optional transform applied to feature arrays.
- **Modes:**
  - `classification` (using YAML-configured thresholds)
  - `regression` (pain_level continuous, split by uniform bins for balance; model receives real value)
- **Usage:**
  ```python
  dm = SyracuseDataModule(meta_path, config_path, cv_fold=0)
  dm.setup()
  print(dm)
  train_loader = dm.train_dataloader()
  val_loader = dm.val_dataloader()
  ```
- **Output:**
  - Each batch: (feature tensor, pain_level, [class_label], video_id, clip_id)

---

## 3. Comparison Table

| Feature         | BioVidDataModule            | SyracuseDataModule                   |
|----------------|----------------------------|--------------------------------------|
| Task           | Classification only         | Classification or Regression         |
| Train/Val Split| Subject-wise, random       | Video-wise, stratified/pseudo-strat. |
| Data split     | By subject_id              | By video_id (originals for val only) |
| Data sources   | biovid only                | syracuse_original, syracuse_aug      |
| Splits config  | split_ratio, seed          | cv_fold (0,1,2:folds; >=3:all train) |
| Labeling       | 5-class labels (meta)      | Configured thresholds or real value  |

---

For in-depth options and code, see comments and docstrings in each module. 