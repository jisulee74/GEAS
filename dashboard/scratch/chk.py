with open('src/js/data/embedded-data.js', 'r', encoding='utf-8') as f:
    t = f.read()
idx = t.find('"table_6":')
print('table_6 idx:', idx)
if idx != -1:
    print(t[idx:idx+400])
