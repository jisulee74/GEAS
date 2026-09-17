import re

# Load embedded data
with open('src/js/data/embedded-data.js', 'r', encoding='utf-8') as f:
    text = f.read()

import json
# Let's extract sample object from each table
for tnum in [1, 2, 3, 4, 5, 6]:
    key = f'"table_{tnum}":'
    idx = text.find(key)
    if idx == -1:
        print(f"Table {tnum} NOT FOUND!")
        continue
    obj_start = text.find('{', idx)
    obj_end = text.find('}', obj_start)
    obj_str = text[obj_start:obj_end+1]
    data = json.loads(obj_str)
    print(f"\n--- TABLE {tnum} COLUMNS ({len(data.keys())} cols) ---")
    print(", ".join(sorted(data.keys())))
