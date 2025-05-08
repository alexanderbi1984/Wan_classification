import sys
from syracuse import SyracuseDataModule
import json

if __name__ == "__main__":
    try:
        dm = SyracuseDataModule(
            meta_path="/home/nbi/marlin/wan_features/meta.json",
            config_path="config_pain/5class.yaml",
            batch_size=64,
            num_workers=2,
            cv_fold=0,
            seed=42,
            balanced_sampling=True,
        )
        dm.setup()
        print(dm)

        print("\nTrain batch:")
        train_iter = iter(dm.train_dataloader())
        sample = next(train_iter)
        arr, pain_level, class_label, video_id, clip_id = sample
        print("  arr shape:", arr.shape)
        print("  pain_level shape:", pain_level.shape)
        print("  class_label shape:", class_label.shape)
        print("  video_ids:", video_id)
        print("  clip_ids:", clip_id)
        print("  unique class labels:", set(class_label.cpu().numpy().tolist()))

        print("\nVal batch:")
        val_iter = iter(dm.val_dataloader())
        sample = next(val_iter)
        arr, pain_level, class_label, video_id, clip_id = sample
        print("  arr shape:", arr.shape)
        print("  pain_level shape:", pain_level.shape)
        print("  class_label shape:", class_label.shape)
        print("  video_ids:", video_id)
        print("  clip_ids:", clip_id)
        print("  unique class labels:", set(class_label.cpu().numpy().tolist()))

        with open(dm.meta_path, 'r') as f:
            meta = json.load(f)
        all_orig_video_ids = {v['video_id'] for v in meta.values() if v.get('source') == 'syracuse_original' and v.get('pain_level') is not None}
        intersection = dm.val_vids & all_orig_video_ids
        print(f"[DEBUG] All original video_ids in meta: {sorted(all_orig_video_ids)[:10]} ... total={len(all_orig_video_ids)}")
        print(f"[DEBUG] val_vids: {sorted(dm.val_vids)[:10]} ... total={len(dm.val_vids)}")
        print(f"[DEBUG] Intersection in validation: {sorted(intersection)[:10]} ... total={len(intersection)}")
        if len(intersection) == 0:
            print('[WARN] None of the val_vids exist as originals in your meta!')

    except Exception as e:
        print(f"Error encountered: {e}", file=sys.stderr)
        sys.exit(1) 