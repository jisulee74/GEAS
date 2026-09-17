with open('src/js/data/embedded-data.js', 'r', encoding='utf-8') as f:
    text = f.read()

idx = text.find('"table_5":')
end = text.find(']', idx)
chunk = text[idx:idx+800]
print("Table 5 snippet:")
print(chunk)
