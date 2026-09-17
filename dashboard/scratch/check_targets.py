import json

with open('src/js/data/embedded-data.js', 'r', encoding='utf-8') as f:
    text = f.read()

# Check Table 2 targets
t2_idx = text.find('"table_2":')
print("table_2 found:", t2_idx != -1)
if t2_idx != -1:
    end = text.find(']', t2_idx)
    # Parse a few rows
    chunk = text[t2_idx:end+1]
    # Let's extract targets
    import re
    targets = set(re.findall(r'"target":\s*"([^"]+)"', chunk))
    print("Table 2 targets:", targets)

# Check Table 6 targets
t6_idx = text.find('"table_6":')
print("table_6 found:", t6_idx != -1)
if t6_idx != -1:
    end6 = text.find(']', t6_idx)
    chunk6 = text[t6_idx:end6+1]
    targets6 = set(re.findall(r'"target":\s*"([^"]+)"', chunk6))
    horizons6 = set(re.findall(r'"horizon":\s*"([^"]+)"', chunk6))
    print("Table 6 targets:", targets6)
    print("Table 6 horizons:", horizons6)
