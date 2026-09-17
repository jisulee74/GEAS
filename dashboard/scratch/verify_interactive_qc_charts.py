import subprocess, time, json, urllib.request, socket, os, base64, struct, sys
sys.stdout.reconfigure(encoding='utf-8')

class MinimalWS:
    def __init__(self, ws_url):
        url = ws_url.replace('ws://', '')
        host_port, path = url.split('/', 1)
        host, port = host_port.split(':')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, int(port)))
        key = base64.b64encode(os.urandom(16)).decode('ascii')
        req = f'GET /{path} HTTP/1.1\r\nHost: {host_port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n'
        self.sock.sendall(req.encode('ascii'))
        resp = self.sock.recv(4096).decode('latin1')
        self.msg_id = 0

    def call(self, method, params=None):
        self.msg_id += 1
        payload = json.dumps({'id': self.msg_id, 'method': method, 'params': params or {}}).encode('utf-8')
        length = len(payload)
        header = bytearray([0x81])
        if length <= 125: header.append(0x80 | length)
        elif length <= 65535: header.append(0x80 | 126); header.extend(struct.pack('!H', length))
        else: header.append(0x80 | 127); header.extend(struct.pack('!Q', length))
        mask = os.urandom(4)
        header.extend(mask)
        self.sock.sendall(header + bytearray(b ^ mask[i % 4] for i, b in enumerate(payload)))
        while True:
            data = self.sock.recv(2)
            if not data: return None
            b1, b2 = data[0], data[1]
            plen = b2 & 0x7f
            if plen == 126: plen = struct.unpack('!H', self.sock.recv(2))[0]
            elif plen == 127: plen = struct.unpack('!Q', self.sock.recv(8))[0]
            body = b''
            while len(body) < plen: body += self.sock.recv(plen - len(body))
            msg = json.loads(body.decode('utf-8', errors='ignore'))
            if msg.get('id') == self.msg_id: return msg

edge_path = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
proc = subprocess.Popen([
    edge_path,
    '--headless=new',
    '--remote-debugging-port=9222',
    '--disable-gpu',
    '--window-size=1440,1200',
    'http://localhost:8080/'
])
time.sleep(3)

try:
    tabs = json.loads(urllib.request.urlopen('http://localhost:9222/json').read().decode('utf-8'))
    tab = next(t for t in tabs if '8080' in t.get('url', ''))
    ws = MinimalWS(tab['webSocketDebuggerUrl'])
    
    def eval_js(exp):
        res = ws.call('Runtime.evaluate', {'expression': exp, 'returnByValue': True})
        return res['result']['result'].get('value') if res and 'result' in res and 'result' in res['result'] else res

    # Setup console and error capture
    eval_js('''
        window.__capturedErrors = [];
        window.addEventListener('error', e => window.__capturedErrors.push(e.message));
        window.addEventListener('unhandledrejection', e => window.__capturedErrors.push(e.reason ? (e.reason.message || String(e.reason)) : 'unhandledrejection'));
    ''')

    ws.call('Network.enable')
    ws.call('Network.setCacheDisabled', {'cacheDisabled': True})
    ws.call('Page.navigate', {'url': 'http://localhost:8080/'})
    time.sleep(3.5)

    print("=== 1. VERIFYING TRANSITION MODEL SECTION (NON-REGRESSION) ===")
    eval_js('window.switchExperiment("transition")')
    eval_js('window.switchTransitionTab("setup")')
    time.sleep(1.0)
    setup_status = eval_js('''
        (() => {
            const cards = document.querySelectorAll('.setup-cat-box');
            const tooltips = document.querySelectorAll('.setup-item-with-tooltip');
            return {
                cardCount: cards.length,
                tooltipCount: tooltips.length
            };
        })()
    ''')
    print("Transition Model Setup Status:", json.dumps(setup_status, indent=2))

    eval_js('window.switchTransitionTab("ranking")')
    time.sleep(1.0)
    ranking_status = eval_js('''
        (() => {
            const tableRows = document.querySelectorAll('#tab-panel-ranking tbody tr');
            return {
                leaderboardRowCount: tableRows.length
            };
        })()
    ''')
    print("Transition Model Ranking Status:", json.dumps(ranking_status, indent=2))

    print("\n=== 2. VERIFYING QUALITY OVERVIEW CHARTS ===")
    eval_js('window.switchExperiment("quality")')
    eval_js('window.switchQualityTab("quality-overview")')
    time.sleep(1.5)
    overview_status = eval_js('''
        (() => {
            const reconCanvas = document.getElementById('overview-reconstruction-chart');
            const clsCanvas = document.getElementById('overview-classification-chart');
            return {
                reconCanvas: reconCanvas ? { width: reconCanvas.width, height: reconCanvas.height } : null,
                clsCanvas: clsCanvas ? { width: clsCanvas.width, height: clsCanvas.height } : null
            };
        })()
    ''')
    print("Overview Charts Status:", json.dumps(overview_status, indent=2))

    print("\n=== 3. VERIFYING RECONSTRUCTION SECTION & HEATMAP ===")
    eval_js('window.switchQualityTab("quality-reconstruction")')
    time.sleep(2.0)
    recon_status = eval_js('''
        (() => {
            const heatmapContainer = document.getElementById('reconstruction-interactive-heatmap');
            const table = heatmapContainer ? heatmapContainer.querySelector('table') : null;
            const rows = table ? table.querySelectorAll('tbody tr') : [];
            const unitBadges = table ? table.querySelectorAll('.qc-unit-badge') : [];
            const globalChart = document.getElementById('recon-global-chart');
            return {
                tablePresent: !!table,
                rowCount: rows.length,
                unitBadgeCount: unitBadges.length,
                firstRowText: rows[0] ? rows[0].innerText.replace(/\\n/g, ' | ') : null,
                globalChartCanvas: globalChart ? { width: globalChart.width, height: globalChart.height } : null
            };
        })()
    ''')
    print("Reconstruction Section Status:", json.dumps(recon_status, indent=2))

    # Test toggling to MAE
    eval_js('document.getElementById("btn-heatmap-mae")?.click()')
    time.sleep(0.8)
    mae_status = eval_js('''
        (() => {
            const heatmapContainer = document.getElementById('reconstruction-interactive-heatmap');
            const rows = heatmapContainer ? heatmapContainer.querySelectorAll('tbody tr') : [];
            return {
                firstRowTextAfterMae: rows[0] ? rows[0].innerText.replace(/\\n/g, ' | ') : null
            };
        })()
    ''')
    print("Reconstruction Heatmap after MAE Click:", json.dumps(mae_status, indent=2))

    print("\n=== 4. VERIFYING DETECTION SECTION (PR CURVE & ROC OPERATING POINT) ===")
    eval_js('window.switchQualityTab("quality-detection")')
    time.sleep(2.0)
    detection_status = eval_js('''
        (() => {
            const prCanvas = document.getElementById('qc-pr-curve-chart');
            const rocCanvas = document.getElementById('qc-roc-operating-point-chart');
            const globalCanvas = document.getElementById('qc-global-detection-chart');
            const varSelect = document.getElementById('select-pr-variable');
            return {
                prCanvas: prCanvas ? { width: prCanvas.width, height: prCanvas.height } : null,
                rocCanvas: rocCanvas ? { width: rocCanvas.width, height: rocCanvas.height } : null,
                globalCanvas: globalCanvas ? { width: globalCanvas.width, height: globalCanvas.height } : null,
                varSelectOptions: varSelect ? Array.from(varSelect.options).map(o => o.value) : []
            };
        })()
    ''')
    print("Detection Section Status:", json.dumps(detection_status, indent=2))

    # Change variable
    eval_js('''
        (() => {
            const sel = document.getElementById('select-pr-variable');
            if (sel) {
                sel.value = 'in_temp';
                sel.dispatchEvent(new Event('change'));
            }
        })()
    ''')
    time.sleep(1.0)
    print("Detection variable changed to in_temp successfully.")

    print("\n=== 5. VERIFYING HPO & THRESHOLD CALIBRATION SECTION ===")
    eval_js('window.switchQualityTab("quality-hpo")')
    time.sleep(2.0)
    hpo_status = eval_js('''
        (() => {
            const hpoCanvas = document.getElementById('qc-hpo-progress-chart');
            const lossCanvas = document.getElementById('qc-training-loss-chart');
            const threshCanvas = document.getElementById('qc-threshold-curve-chart');
            const zoomBtn = document.getElementById('btn-threshold-selected');
            return {
                hpoCanvas: hpoCanvas ? { width: hpoCanvas.width, height: hpoCanvas.height } : null,
                lossCanvas: lossCanvas ? { width: lossCanvas.width, height: lossCanvas.height } : null,
                threshCanvas: threshCanvas ? { width: threshCanvas.width, height: threshCanvas.height } : null,
                zoomBtnPresent: !!zoomBtn
            };
        })()
    ''')
    print("HPO Section Status:", json.dumps(hpo_status, indent=2))

    # Test clicking Zoom button
    eval_js('document.getElementById("btn-threshold-selected")?.click()')
    time.sleep(1.0)
    print("Clicked Zoom button (0~4x best).")

    print("\n=== 6. VERIFYING COMPUTATIONAL RESOURCE EFFICIENCY SECTION ===")
    eval_js('window.switchQualityTab("quality-resource")')
    time.sleep(2.0)
    resource_status = eval_js('''
        (() => {
            const effCanvas = document.getElementById('resource-interactive-efficiency-chart');
            const table = document.querySelector('#resource-benchmark-table-container table');
            return {
                effCanvas: effCanvas ? { width: effCanvas.width, height: effCanvas.height } : null,
                tableRows: table ? table.querySelectorAll('tbody tr').length : 0
            };
        })()
    ''')
    print("Resource Section Status:", json.dumps(resource_status, indent=2))

    # Test switching metric to Peak Memory
    eval_js('document.getElementById("btn-metric-memory")?.click()')
    time.sleep(1.0)
    print("Switched resource metric to Peak RAM.")

    # Screenshot capturing
    artifact_dir = r'C:\Users\10-64\.gemini\antigravity-ide\brain\042f002e-c04f-43f8-bf78-5f2b1dc74206'
    
    # 1. Capture HPO & Threshold curves screenshot
    eval_js('window.switchQualityTab("quality-hpo")')
    time.sleep(1.0)
    screen1 = ws.call('Page.captureScreenshot', {'format': 'png'})
    with open(os.path.join(artifact_dir, 'screenshot_qc_hpo_threshold_live.png'), 'wb') as f:
        f.write(base64.b64decode(screen1['result']['data']))
    print("Saved screenshot_qc_hpo_threshold_live.png")

    # 2. Capture Reconstruction Heatmap screenshot
    eval_js('window.switchQualityTab("quality-reconstruction")')
    time.sleep(1.0)
    screen2 = ws.call('Page.captureScreenshot', {'format': 'png'})
    with open(os.path.join(artifact_dir, 'screenshot_qc_reconstruction_live.png'), 'wb') as f:
        f.write(base64.b64decode(screen2['result']['data']))
    print("Saved screenshot_qc_reconstruction_live.png")

    # 3. Capture Detection PR / ROC screenshot
    eval_js('window.switchQualityTab("quality-detection")')
    time.sleep(1.0)
    screen3 = ws.call('Page.captureScreenshot', {'format': 'png'})
    with open(os.path.join(artifact_dir, 'screenshot_qc_detection_live.png'), 'wb') as f:
        f.write(base64.b64decode(screen3['result']['data']))
    print("Saved screenshot_qc_detection_live.png")

    # Check for errors
    captured_errors = eval_js('window.__capturedErrors')
    print("\n=== CONSOLE ERRORS CAPTURED ===")
    print("Captured errors:", captured_errors)

finally:
    proc.terminate()
