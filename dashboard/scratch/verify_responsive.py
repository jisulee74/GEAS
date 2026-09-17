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

# Test tablet (800px)
proc_tablet = subprocess.Popen([edge_path, '--headless=new', '--remote-debugging-port=9223', '--disable-gpu', '--window-size=800,900', 'http://localhost:8080/'])
time.sleep(2)
try:
    tabs = json.loads(urllib.request.urlopen('http://localhost:9223/json').read().decode('utf-8'))
    ws = MinimalWS(tabs[0]['webSocketDebuggerUrl'])
    ws.call('Network.enable')
    ws.call('Network.setCacheDisabled', {'cacheDisabled': True})
    ws.call('Page.navigate', {'url': 'http://localhost:8080/'})
    time.sleep(2)
    def eval_js(exp):
        res = ws.call('Runtime.evaluate', {'expression': exp, 'returnByValue': True})
        return res['result']['result'].get('value') if res and 'result' in res and 'result' in res['result'] else res
    eval_js('window.switchExperiment("quality")')
    eval_js('window.switchQualityTab("quality-overview")')
    tablet_cols = eval_js('getComputedStyle(document.querySelector("#container-exp-quality .metrics-grid")).gridTemplateColumns')
    print("Tablet (800px) columns:", tablet_cols, "Count:", len(tablet_cols.split()))
finally:
    proc_tablet.terminate()

# Test mobile (480px)
proc_mobile = subprocess.Popen([edge_path, '--headless=new', '--remote-debugging-port=9224', '--disable-gpu', '--window-size=480,900', 'http://localhost:8080/'])
time.sleep(2)
try:
    tabs = json.loads(urllib.request.urlopen('http://localhost:9224/json').read().decode('utf-8'))
    ws = MinimalWS(tabs[0]['webSocketDebuggerUrl'])
    ws.call('Network.enable')
    ws.call('Network.setCacheDisabled', {'cacheDisabled': True})
    ws.call('Page.navigate', {'url': 'http://localhost:8080/'})
    time.sleep(2)
    def eval_js(exp):
        res = ws.call('Runtime.evaluate', {'expression': exp, 'returnByValue': True})
        return res['result']['result'].get('value') if res and 'result' in res and 'result' in res['result'] else res
    eval_js('window.switchExperiment("quality")')
    eval_js('window.switchQualityTab("quality-overview")')
    mobile_cols = eval_js('getComputedStyle(document.querySelector("#container-exp-quality .metrics-grid")).gridTemplateColumns')
    print("Mobile (480px) columns:", mobile_cols, "Count:", len(mobile_cols.split()))
finally:
    proc_mobile.terminate()
