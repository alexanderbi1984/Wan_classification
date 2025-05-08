import os
import json
from collections import Counter

meta_path = "/home/nbi/marlin/wan_features/meta.json"
clips_json_path = "/home/nbi/marlin/wan_features/clips_json.json"

# Load both JSONs
with open(meta_path, 'r') as f:
    meta = json.load(f)
with open(clips_json_path, 'r') as f:
    clips = json.load(f)

n_valid = 0
pain_levels = []

for fname, clipinfo in clips.items():
    meta_info = clipinfo.get('meta_info', {})
    pain_level = meta_info.get('pain_level')
    if pain_level is None:
        continue
    # Only update meta if this file is in meta
    if fname in meta:
        meta[fname]['pain_level'] = pain_level
        # Optionally add other fields from meta_info if desired
        n_valid += 1
        pain_levels.append(pain_level)
    else:
        # (Optionally add new syracuse entry here, but skipping for now)
        continue

# Save updated meta.json
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)

# Print stats
print(f"Number of syracuse clips with valid pain level: {n_valid}")
if pain_levels:
    dist = Counter(pain_levels)
    print("Pain level distribution:")
    for k in sorted(dist):
        print(f"  {k}: {dist[k]}")
else:
    print("No valid pain levels found.") 