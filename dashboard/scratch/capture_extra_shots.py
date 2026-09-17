import subprocess, time, json, urllib.request, socket, os, base64, struct, sys
sys.stdout.reconfigure(encoding='utf-8')
from verify_interactive_qc_charts import MinimalWS

edge_path = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
proc = subprocess.Popen([edge_path, '--headless=new', '--remote-debugging-port=9222', '--disable-gpu', '--window-size=1440,1200', 'http://localhost:8080/'])
time.sleep(3)
try:
    tabs = json.loads(urllib.request.urlopen('http://localhost:9222/json').read().decode('utf-8'))
    tab = next(t for t in tabs if '8080' in t.get('url', ''))
    ws = MinimalWS(tab['webSocketDebuggerUrl'])
    def eval_js(exp):
        res = ws.call('Runtime.evaluate', {'expression': exp, 'returnByValue': True})
        return res['result']['result'].get('value') if res and 'result' in res and 'result' in res['result'] else res
    ws.call('Network.enable')
    ws.call('Network.setCacheDisabled', {'cacheDisabled': True})
    ws.call('Page.navigate', {'url': 'http://localhost:8080/'})
    time.sleep(3.5)
    artifact_dir = r'C:\Users\10-64\.gemini\antigravity-ide\brain\042f002e-c04f-43f8-bf78-5f2b1dc74206'
    
    # Overview screenshot
    eval_js('window.switchExperiment("quality")')
    eval_js('window.switchQualityTab("quality-overview")')
    time.sleep(1.2)
    s1 = ws.call('Page.captureScreenshot', {'format': 'png'})
    with open(os.path.join(artifact_dir, 'screenshot_qc_overview_live.png'), 'wb') as f:
        f.write(base64.b64decode(s1['result']['data']))
        
    # Resource screenshot
    eval_js('window.switchQualityTab("quality-resource")')
    time.sleep(1.2)
    s2 = ws.call('Page.captureScreenshot', {'format': 'png'})
    with open(os.path.join(artifact_dir, 'screenshot_qc_resource_live.png'), 'wb') as f:
        f.write(base64.b64decode(s2['result']['data']))
    print('Screenshots captured successfully!')
finally:
    proc.terminate()
