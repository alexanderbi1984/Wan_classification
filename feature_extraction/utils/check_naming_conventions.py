import os
import re
from collections import defaultdict, Counter

folder = "/home/nbi/marlin/wan_features/"

prefix_counter = Counter()
example_files = {}
img_subtypes = Counter()
img_examples = {}

for fname in os.listdir(folder):
    path = os.path.join(folder, fname)
    if not os.path.isfile(path):
        continue
    prefix = fname.split('_', 1)[0]
    # Special handling for IMG prefix
    if prefix == 'IMG':
        name_body = os.path.splitext(fname)[0]
        last_part = name_body.split('_')[-1]
        if last_part.isdigit():
            key = 'IMG_num'
        elif last_part == 'aligned':
            key = 'IMG_aligned'
        else:
            key = f'IMG_{last_part}'
        img_subtypes[key] += 1
        if key not in img_examples:
            img_examples[key] = fname
        prefix_counter['IMG'] += 1
        if 'IMG' not in example_files:
            example_files['IMG'] = fname
        continue
    elif prefix.isdigit():
        key = 'number'
    else:
        key = prefix
    prefix_counter[key] += 1
    if key not in example_files:
        example_files[key] = fname

print("File count by prefix type (before first underscore):")
for key, count in prefix_counter.most_common():
    print(f"  {key:15s}: {count:4d} files; Example: {example_files[key]}")

if img_subtypes:
    print("\nIMG file subtypes (by last part before extension):")
    for key, count in img_subtypes.most_common():
        print(f"  {key:15s}: {count:4d} files; Example: {img_examples[key]}") 