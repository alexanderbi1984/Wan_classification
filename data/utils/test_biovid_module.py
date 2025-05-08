import sys
from biovid import BioVidDataModule

if __name__ == "__main__":
    try:
        dm = BioVidDataModule(
            features_path="/home/nbi/marlin/wan_features/",
            meta_path="/home/nbi/marlin/wan_features/meta.json",
            batch_size=64,
            num_workers=2,
            split_ratio=0.8,
            seed=42
        )
        dm.setup()
        print(dm)
        loader = dm.train_dataloader()
        print("Fetching a batch...")
        batch = next(iter(loader))
        arr, subject_id, label = batch
        print("Batch arr shape:", arr.shape)
        print("Batch subject_ids:", subject_id)
        print("Batch labels shape:", label.shape)
        print("Batch unique labels:", set(label.cpu().numpy().tolist()))
    except Exception as e:
        print(f"Error encountered: {e}", file=sys.stderr)
        sys.exit(1) 