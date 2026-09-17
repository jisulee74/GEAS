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

    # Track console messages
    console_errors = []
    ws.call('Console.enable')

    # Hard reload with cache disabled
    ws.call('Network.enable')
    ws.call('Network.setCacheDisabled', {'cacheDisabled': True})
    ws.call('Page.navigate', {'url': 'http://localhost:8080/'})
    time.sleep(3.0)

    # 1. Switch to Quality Overview
    eval_js('window.switchExperiment("quality")')
    eval_js('window.switchQualityTab("quality-overview")')
    time.sleep(0.5)

    # 2. Inspect Pipeline Workflow Diagram
    pipeline_info = eval_js('''
        (() => {
            const heading = document.querySelector('#container-exp-quality .content-card .card-title');
            const workflowGrid = document.querySelector('#container-exp-quality .pipeline-workflow-grid');
            if (!workflowGrid) return { error: "workflowGrid not found" };

            const cards = Array.from(workflowGrid.querySelectorAll('.pipeline-step-card'));
            const cardData = cards.map(c => ({
                phase: c.querySelector('.step-num')?.textContent?.trim(),
                title: c.querySelector('.step-title')?.textContent?.trim(),
                items: Array.from(c.querySelectorAll('.step-list li')).map(li => li.textContent.trim())
            }));

            const style = getComputedStyle(workflowGrid);
            return {
                headingText: heading ? heading.textContent.trim() : null,
                cardCount: cards.length,
                gridDisplay: style.display,
                gridTemplateColumns: style.gridTemplateColumns,
                cards: cardData
            };
        })()
    ''')
    print("=== DESKTOP PIPELINE INFO ===")
    print(json.dumps(pipeline_info, indent=2, ensure_ascii=False))

    # Take desktop screenshot
    snap = ws.call('Page.captureScreenshot', {'format': 'png'})
    if snap and 'result' in snap and 'data' in snap['result']:
        img_data = base64.b64decode(snap['result']['data'])
        out_path = r'C:\Users\10-64\.gemini\antigravity-ide\brain\042f002e-c04f-43f8-bf78-5f2b1dc74206\screenshot_pipeline_6steps_desktop.png'
        with open(out_path, 'wb') as f:
            f.write(img_data)
        print("Desktop screenshot saved to:", out_path)

    # 3. Mobile / Small Screen Emulation (480px width)
    ws.call('Emulation.setDeviceMetricsOverride', {
        'width': 480,
        'height': 900,
        'deviceScaleFactor': 2,
        'mobile': True
    })
    time.sleep(0.5)

    mobile_info = eval_js('''
        (() => {
            const workflowGrid = document.querySelector('#container-exp-quality .pipeline-workflow-grid');
            if (!workflowGrid) return { error: "workflowGrid not found" };
            const style = getComputedStyle(workflowGrid);
            const cards = Array.from(workflowGrid.querySelectorAll('.pipeline-step-card'));
            return {
                windowInnerWidth: window.innerWidth,
                gridDisplay: style.display,
                gridTemplateColumns: style.gridTemplateColumns,
                cardRects: cards.map(c => {
                    const r = c.getBoundingClientRect();
                    return { width: Math.round(r.width), height: Math.round(r.height), top: Math.round(r.top) };
                })
            };
        })()
    ''')
    print("\n=== MOBILE PIPELINE INFO ===")
    print(json.dumps(mobile_info, indent=2, ensure_ascii=False))

    snap_mob = ws.call('Page.captureScreenshot', {'format': 'png'})
    if snap_mob and 'result' in snap_mob and 'data' in snap_mob['result']:
        img_mob_data = base64.b64decode(snap_mob['result']['data'])
        out_mob_path = r'C:\Users\10-64\.gemini\antigravity-ide\brain\042f002e-c04f-43f8-bf78-5f2b1dc74206\screenshot_pipeline_6steps_mobile.png'
        with open(out_mob_path, 'wb') as f:
            f.write(img_mob_data)
        print("Mobile screenshot saved to:", out_mob_path)

    # Reset Emulation
    ws.call('Emulation.clearDeviceMetricsOverride')
    time.sleep(0.5)

    # 4. Repeated Switching & Regression Test between Quality and Transition Model
    print("\n=== REGRESSION & INTERACTION TEST ===")
    for loop in range(3):
        eval_js('window.switchExperiment("transition")')
        eval_js('window.switchTransitionTab("setup")')
        time.sleep(0.2)
        eval_js('window.switchTransitionTab("ranking")')
        time.sleep(0.2)
        eval_js('window.switchExperiment("quality")')
        eval_js('window.switchQualityTab("quality-overview")')
        time.sleep(0.2)

    # Check transition page status
    eval_js('window.switchExperiment("transition")')
    eval_js('window.switchTransitionTab("ranking")')
    time.sleep(0.4)
    t1_check = eval_js('''
        (() => {
            const rankingPanel = document.getElementById("tab-panel-ranking");
            const rows = rankingPanel ? rankingPanel.querySelectorAll("tbody tr").length : 0;
            const t1 = window.dataService ? window.dataService.getTable1().length : 0;
            return { table1RowsDisplayed: rows, table1DataCount: t1 };
        })()
    ''')
    print("Transition Ranking Status:", t1_check)

    # Check setup 12 cards and tooltips
    eval_js('window.switchTransitionTab("setup")')
    time.sleep(0.4)
    setup_check = eval_js('''
        (() => {
            const cards = document.querySelectorAll('#container-exp-transition .setup-cat-box').length;
            const tooltips = document.querySelectorAll('#container-exp-transition .setup-item-with-tooltip').length;
            return { cards12Count: cards, formulaTooltipsCount: tooltips };
        })()
    ''')
    print("Transition Setup Status:", setup_check)

    # Check console errors
    errors = eval_js('window.__consoleErrors || []')
    print("Any JS errors recorded:", errors)

finally:
    proc.terminate()
