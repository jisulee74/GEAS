import urllib.request
import json

resp = urllib.request.urlopen('http://localhost:8080/src/js/data/embedded-data.js')
text = resp.read().decode('utf-8')
d = json.loads(text[text.find('{'):text.rfind('}')+1])
print('Table 6 count in embedded data:', len(d['tables']['table_6']))

resp2 = urllib.request.urlopen('http://localhost:8080/index.html')
h = resp2.read().decode('utf-8')
print('Has drift tab in index.html:', 'data-tab="drift"' in h)
print('Has panel in index.html:', 'id="tab-panel-drift"' in h)
