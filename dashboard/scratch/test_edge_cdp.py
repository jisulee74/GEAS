import subprocess, time, json, urllib.request, socket, os, base64, hashlib, struct

# Helper for minimal raw WebSocket client over standard library socket
class MinimalWS:
    def __init__(self, ws_url):
        # ws_url: ws://127.0.0.1:9222/devtools/page/XYZ
        url = ws_url.replace("ws://", "")
        host_port, path = url.split("/", 1)
        host, port = host_port.split(":")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, int(port)))
        
        # Handshake
        key = base64.b64encode(os.urandom(16)).decode('ascii')
        req = (
            f"GET /{path} HTTP/1.1\r\n"
            f"Host: {host_port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(req.encode('ascii'))
        resp = self.sock.recv(4096).decode('latin1')
        if "101" not in resp:
            raise Exception("Handshake failed: " + resp)
        self.msg_id = 0

    def send(self, method, params=None):
        self.msg_id += 1
        payload = json.dumps({"id": self.msg_id, "method": method, "params": params or {}}).encode('utf-8')
        length = len(payload)
        header = bytearray([0x81]) # FIN + text
        if length <= 125:
            header.append(0x80 | length) # masked
        elif length <= 65535:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        
        mask = os.urandom(4)
        header.extend(mask)
        masked_payload = bytearray(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(header + masked_payload)
        return self.msg_id

    def recv(self, timeout=5.0):
        self.sock.settimeout(timeout)
        data = self.sock.recv(2)
        if not data:
            return None
        b1, b2 = data[0], data[1]
        opcode = b1 & 0x0f
        payload_len = b2 & 0x7f
        if payload_len == 126:
            payload_len = struct.unpack("!H", self.sock.recv(2))[0]
        elif payload_len == 127:
            payload_len = struct.unpack("!Q", self.sock.recv(8))[0]
        
        body = b""
        while len(body) < payload_len:
            chunk = self.sock.recv(payload_len - len(body))
            if not chunk:
                break
            body += chunk
        return json.loads(body.decode('utf-8', errors='ignore'))

    def call(self, method, params=None, timeout=5.0):
        req_id = self.send(method, params)
        start = time.time()
        while time.time() - start < timeout:
            res = self.recv(timeout=timeout)
            if res and res.get("id") == req_id:
                return res
        return None

    def close(self):
        try:
            self.sock.close()
        except:
            pass

# Find Edge
edge_paths = [
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    r'C:\Program Files\Microsoft\Edge\Application\msedge.exe'
]
edge_path = next(p for p in edge_paths if os.path.exists(p))

proc = subprocess.Popen([
    edge_path,
    '--headless=new',
    '--remote-debugging-port=9222',
    '--disable-gpu',
    '--no-sandbox',
    'http://localhost:8080/'
])
time.sleep(3)

try:
    tabs = json.loads(urllib.request.urlopen('http://localhost:9222/json').read().decode('utf-8'))
    geas_tab = next(t for t in tabs if '8080' in t.get('url', ''))
    print("Found GEAS tab:", geas_tab['url'])
    ws_url = geas_tab['webSocketDebuggerUrl']
    
    ws = MinimalWS(ws_url)
    
    # 1. Enable Runtime and Console
    ws.send("Runtime.enable")
    ws.send("Console.enable")
    time.sleep(1)

    # Helper to evaluate JS
    def eval_js(expression):
        res = ws.call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
        if res and "result" in res and "result" in res["result"]:
            return res["result"]["result"].get("value")
        return res

    # Check title and current state
    print("Document title:", eval_js("document.title"))
    
    # Check dataService and qualityDataLoader
    data_check = eval_js("""
        ({
            hasDataService: typeof window.dataService !== 'undefined',
            t1Count: window.dataService ? window.dataService.getTable1().length : null,
            t2Count: window.dataService ? window.dataService.getTable2().length : null,
            t3Count: window.dataService ? window.dataService.getTable3().length : null,
            t4Count: window.dataService ? window.dataService.getTable4().length : null,
            t5Count: window.dataService ? window.dataService.getTable5().length : null,
            t6Count: window.dataService ? window.dataService.getTable6().length : null,
            hasQualityLoader: typeof window.qualityDataLoader !== 'undefined',
            qualityValCount: window.qualityDataLoader ? window.qualityDataLoader.validationResults.length : null,
            filterStoreState: window.filterStore ? window.filterStore.getState() : null,
            cropSelectOptions: Array.from(document.querySelectorAll('#filter-crop option')).map(o => ({val: o.value, text: o.text}))
        })
    """)
    print("Initial Data Check:", json.dumps(data_check, indent=2, ensure_ascii=False))

    # Switch to transition experiment
    print("\n--- Clicking Switch to Transition Model Selection ---")
    eval_js("window.switchExperiment('transition')")
    time.sleep(1)

    active_exp = eval_js("window.app.currentExperiment")
    print("Active Experiment:", active_exp)

    # Check each sub-tab
    sub_tabs = ['overview', 'setup', 'ranking', 'onestep', 'safety', 'drift', 'resource', 'ablation']
    for tab in sub_tabs:
        print(f"\n--- Checking Transition Sub-Tab: {tab} ---")
        eval_js(f"window.switchTransitionTab('{tab}')")
        time.sleep(0.5)

        tab_status = eval_js(f"""
            (() => {{
                const panel = document.getElementById('tab-panel-{tab}');
                if (!panel) return {{ error: 'Panel not found' }};
                
                const html = panel.innerHTML;
                const rows = panel.querySelectorAll('tbody tr');
                const rowCount = rows.length;
                const firstRowText = rowCount > 0 ? rows[0].innerText.slice(0, 100) : '';
                const charts = Array.from(panel.querySelectorAll('canvas')).map(c => ({{
                    id: c.id,
                    width: c.width,
                    height: c.height
                }}));
                const kpis = Array.from(panel.querySelectorAll('.kpi-card')).map(k => ({{
                    title: k.querySelector('.kpi-title')?.innerText,
                    value: k.querySelector('.kpi-value')?.innerText
                }}));
                const hasNoDataText = html.includes('No matching records') || html.includes('No available') || html.includes('no available') || html.includes('자료 없음') || html.includes('NaN');
                
                return {{
                    rowCount,
                    firstRowText,
                    charts,
                    kpis,
                    hasNoDataText,
                    panelInnerSnippet: panel.innerText.slice(0, 200).replace(/\\n+/g, ' ')
                }};
            }})()
        """)
        print(f"Tab '{tab}' status:", json.dumps(tab_status, indent=2, ensure_ascii=True))

    ws.close()
finally:
    proc.terminate()
