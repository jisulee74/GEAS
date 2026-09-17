import subprocess, time, json, urllib.request, socket, os, base64, struct

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
proc = subprocess.Popen([edge_path, '--headless=new', '--remote-debugging-port=9222', '--disable-gpu', 'http://localhost:8080/'])
time.sleep(3)

results = []

try:
    tabs = json.loads(urllib.request.urlopen('http://localhost:9222/json').read().decode('utf-8'))
    tab = next(t for t in tabs if '8080' in t.get('url', ''))
    ws = MinimalWS(tab['webSocketDebuggerUrl'])
    
    def eval_js(exp):
        res = ws.call('Runtime.evaluate', {'expression': exp, 'returnByValue': True})
        if res and 'result' in res and 'result' in res['result']:
            return res['result']['result'].get('value')
        return res

    # Switch to transition
    eval_js('window.switchExperiment("transition")')

    # Test 1: Test clicking each sub-tab button directly via DOM click
    for tab_id in ['overview', 'setup', 'ranking', 'onestep', 'safety', 'drift', 'resource', 'ablation']:
        btn_clicked = eval_js(f'''
            (() => {{
                const btn = document.querySelector('#transition-tabs-nav .tab-btn[data-tab="{tab_id}"]');
                if (!btn) return "Button not found";
                btn.click();
                const panel = document.getElementById("tab-panel-{tab_id}");
                const active = panel && panel.classList.contains("active");
                const rowCount = panel ? panel.querySelectorAll("tbody tr").length : 0;
                return {{ active, rowCount, panelTextLen: panel ? panel.innerText.length : 0 }};
            }})()
        ''')
        results.append(f"Sub-tab click '{tab_id}': {btn_clicked}")

    # Test 2: Filter interactions on Ranking tab
    eval_js('document.querySelector(\'#transition-tabs-nav .tab-btn[data-tab="ranking"]\').click()')
    filter_test = eval_js('''
        (() => {{
            const cropSel = document.getElementById("filter-crop");
            const variantSel = document.getElementById("filter-variant");
            const modelSel = document.getElementById("filter-model");
            const eligSel = document.getElementById("filter-eligibility");
            const baseChk = document.getElementById("filter-baseline");
            const resetBtn = document.getElementById("filter-reset-btn");

            // Select strawberry
            cropSel.value = "strawberry";
            cropSel.dispatchEvent(new Event("change"));
            const strawRowCount = document.querySelectorAll("#tab-panel-ranking tbody tr").length;

            // Select with_quality_flags
            variantSel.value = "with_quality_flags";
            variantSel.dispatchEvent(new Event("change"));
            const variantRowCount = document.querySelectorAll("#tab-panel-ranking tbody tr").length;

            // Select lightgbm
            modelSel.value = "lightgbm";
            modelSel.dispatchEvent(new Event("change"));
            const lgbmRowCount = document.querySelectorAll("#tab-panel-ranking tbody tr").length;

            // Reset filters
            resetBtn.click();
            const resetRowCount = document.querySelectorAll("#tab-panel-ranking tbody tr").length;

            return {
                strawRowCount,
                variantRowCount,
                lgbmRowCount,
                resetRowCount,
                cropVal: cropSel.value,
                modelVal: modelSel.value
            };
        }})()
    ''')
    results.append(f"Filter test: {filter_test}")

    # Test 3: One-step controls
    eval_js('document.querySelector(\'#transition-tabs-nav .tab-btn[data-tab="onestep"]\').click()')
    onestep_test = eval_js('''
        (() => {{
            const tempBtn = document.querySelector('#tab-panel-onestep [data-target="indoor_temperature"]');
            const humBtn = document.querySelector('#tab-panel-onestep [data-target="indoor_humidity"]');
            const co2Btn = document.querySelector('#tab-panel-onestep [data-target="indoor_co2"]');
            const allBtn = document.querySelector('#tab-panel-onestep [data-target="all"]');
            const metricSel = document.getElementById("onestep-metric-select");

            tempBtn?.click();
            const tempRows = document.querySelectorAll("#tab-panel-onestep tbody tr").length;

            humBtn?.click();
            const humRows = document.querySelectorAll("#tab-panel-onestep tbody tr").length;

            co2Btn?.click();
            const co2Rows = document.querySelectorAll("#tab-panel-onestep tbody tr").length;

            allBtn?.click();
            const allRows = document.querySelectorAll("#tab-panel-onestep tbody tr").length;

            metricSel.value = "r2";
            metricSel.dispatchEvent(new Event("change"));

            return { tempRows, humRows, co2Rows, allRows };
        }})()
    ''')
    results.append(f"OneStep controls test: {onestep_test}")

    # Test 4: Drift controls
    eval_js('document.querySelector(\'#transition-tabs-nav .tab-btn[data-tab="drift"]\').click()')
    drift_test = eval_js('''
        (() => {{
            const h60 = document.querySelector('#tab-panel-drift [data-horizon="60min"]');
            const h30 = document.querySelector('#tab-panel-drift [data-horizon="30min"]');
            const h15 = document.querySelector('#tab-panel-drift [data-horizon="15min"]');
            const hall = document.querySelector('#tab-panel-drift [data-horizon="all"]');

            h60?.click();
            const r60 = document.querySelectorAll("#tab-panel-drift tbody tr").length;
            h30?.click();
            const r30 = document.querySelectorAll("#tab-panel-drift tbody tr").length;
            h15?.click();
            const r15 = document.querySelectorAll("#tab-panel-drift tbody tr").length;
            hall?.click();
            const rall = document.querySelectorAll("#tab-panel-drift tbody tr").length;

            return { r60, r30, r15, rall };
        }})()
    ''')
    results.append(f"Drift controls test: {drift_test}")

    # Test 5: Check console errors collected
    console_errors = eval_js('''
        window._consoleErrors || []
    ''')
    results.append(f"Console errors: {console_errors}")

    for r in results:
        print(r)

finally:
    proc.terminate()
