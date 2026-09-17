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

    # Disable cache and reload to ensure latest CSS
    ws.call('Network.enable')
    ws.call('Network.setCacheDisabled', {'cacheDisabled': True})
    ws.call('Page.navigate', {'url': 'http://localhost:8080/'})
    time.sleep(3.0)

    # 1. Switch to Quality Overview
    eval_js('window.switchExperiment("quality")')
    eval_js('window.switchQualityTab("quality-overview")')
    time.sleep(0.5)

    # 2. Check styles
    styles_info = eval_js('''
        (() => {
            const headerCard = document.querySelector('#container-exp-quality .section-header-card');
            const metricsGrid = document.querySelector('#container-exp-quality .metrics-grid');
            const metricCard = document.querySelector('#container-exp-quality .metric-card');
            const workflowGrid = document.querySelector('#container-exp-quality .pipeline-workflow-grid');
            const contentCard = document.querySelector('#container-exp-quality .content-card');

            return {
                headerCard: headerCard ? {
                    display: getComputedStyle(headerCard).display,
                    bg: getComputedStyle(headerCard).backgroundColor,
                    border: getComputedStyle(headerCard).border,
                    borderRadius: getComputedStyle(headerCard).borderRadius,
                    padding: getComputedStyle(headerCard).padding,
                    boxShadow: getComputedStyle(headerCard).boxShadow
                } : null,
                metricsGrid: metricsGrid ? {
                    display: getComputedStyle(metricsGrid).display,
                    gridCols: getComputedStyle(metricsGrid).gridTemplateColumns,
                    gap: getComputedStyle(metricsGrid).gap
                } : null,
                metricCard: metricCard ? {
                    display: getComputedStyle(metricCard).display,
                    bg: getComputedStyle(metricCard).backgroundColor,
                    border: getComputedStyle(metricCard).border,
                    padding: getComputedStyle(metricCard).padding,
                    borderRadius: getComputedStyle(metricCard).borderRadius
                } : null,
                workflowGrid: workflowGrid ? {
                    display: getComputedStyle(workflowGrid).display,
                    gridCols: getComputedStyle(workflowGrid).gridTemplateColumns
                } : null,
                contentCard: contentCard ? {
                    display: getComputedStyle(contentCard).display,
                    bg: getComputedStyle(contentCard).backgroundColor,
                    padding: getComputedStyle(contentCard).padding
                } : null
            };
        })()
    ''')
    print("Quality Overview Styles:", json.dumps(styles_info, indent=2))

    # 3. Take Screenshot
    snap = ws.call('Page.captureScreenshot', {'format': 'png'})
    if snap and 'result' in snap and 'data' in snap['result']:
        img_data = base64.b64decode(snap['result']['data'])
        out_path = r'C:\Users\10-64\.gemini\antigravity-ide\brain\042f002e-c04f-43f8-bf78-5f2b1dc74206\screenshot_quality_overview_fixed.png'
        with open(out_path, 'wb') as f:
            f.write(img_data)
        print("Saved screenshot to:", out_path)

    # 4. Check all 7 QC tabs to ensure no broken layout
    qtabs = ['quality-overview', 'quality-setup', 'quality-reconstruction', 'quality-detection', 'quality-hpo', 'quality-resource', 'quality-deployment']
    for qt in qtabs:
        eval_js(f'window.switchQualityTab("{qt}")')
        time.sleep(0.3)
        check = eval_js(f'''
            (() => {{
                const p = document.getElementById("tab-panel-{qt}");
                const headerCard = p?.querySelector(".section-header-card");
                const cards = p?.querySelectorAll(".content-card, .metric-card, .pipeline-step-card").length;
                return {{
                    hasHeader: Boolean(headerCard),
                    cardCount: cards,
                    bg: headerCard ? getComputedStyle(headerCard).backgroundColor : null
                }};
            }})()
        ''')
        print(f"Tab '{qt}' check:", check)

    # 5. Regression Check: Switch to Transition Model Selection and check
    print("\n--- Regression Check on Transition Model ---")
    eval_js('window.switchExperiment("transition")')
    eval_js('window.switchTransitionTab("ranking")')
    time.sleep(0.5)

    trans_check = eval_js('''
        (() => {
            const rankingPanel = document.getElementById("tab-panel-ranking");
            const rows = rankingPanel ? rankingPanel.querySelectorAll("tbody tr").length : 0;
            const canvases = rankingPanel ? rankingPanel.querySelectorAll("canvas").length : 0;
            const t1Count = window.dataService ? window.dataService.getTable1().length : 0;
            return { rows, canvases, t1Count };
        })()
    ''')
    print("Transition Ranking Check:", trans_check)

finally:
    proc.terminate()
