import os
import re
import json

features_path = "/home/nbi/marlin/wan_features/"
out_json = os.path.join(features_path, "meta.json")

# Map GT codes to 5-class labels
GT_TO_LABEL = {
    'BL1': 0,
    'PA1': 1,
    'PA2': 2,
    'PA3': 3,
    'PA4': 4,
}

meta_dict = {}
# Allow 'm' and 'w' for gender (man/woman)
biovid_pattern = re.compile(r"^(\d{6})_([mw])_(\d+)-([A-Z]{2}\d)(?:-(\d+))?_aligned\.npy$")
# IMG subtype 1: IMG_0003_2_aligned_clip_002.npy (augmented)
img_aug_pattern = re.compile(r'^(IMG_\d{4})_\d+_aligned_(clip_\d+)\.npy$')
# IMG subtype 2: IMG_\d{4}_clip_\d+_aligned\.npy (original)
img_orig_pattern = re.compile(r'^(IMG_\d{4})_(clip_\d+)_aligned\.npy$')

unmatched = []

files = os.listdir(features_path)
for fname in files:
    if not fname.endswith('.npy'):
        continue
    m = biovid_pattern.match(fname)
    if m:
        subject_id, gender, age, ground_truth = m.group(1), m.group(2), m.group(3), m.group(4)
        label = GT_TO_LABEL.get(ground_truth, -1)
        entry = {
            "subject_id": subject_id,
            "gender": gender,
            "age": int(age),
            "ground_truth": ground_truth,
            "5_class_label": label,
            "source": "biovid"
        }
        meta_dict[fname] = entry
        continue
    m_aug = img_aug_pattern.match(fname)
    if m_aug:
        video_id, clip_id = m_aug.group(1), m_aug.group(2)
        entry = {
            "video_id": video_id,
            "clip_id": clip_id,
            "source": "syracuse_aug"
        }
        meta_dict[fname] = entry
        continue
    m_orig = img_orig_pattern.match(fname)
    if m_orig:
        video_id, clip_id = m_orig.group(1), m_orig.group(2)
        entry = {
            "video_id": video_id,
            "clip_id": clip_id,
            "source": "syracuse_original"
        }
        meta_dict[fname] = entry
        continue
    unmatched.append(fname)

with open(out_json, "w") as f:
    json.dump(meta_dict, f, indent=2)

if unmatched:
    print("\n[DEBUG] Files with no pattern match:")
    for u in unmatched[:50]:
        print(u)
    if len(unmatched) > 50:
        print(f"... ({len(unmatched)-50} more not shown)")

print(f"Meta json written to: {out_json}") 