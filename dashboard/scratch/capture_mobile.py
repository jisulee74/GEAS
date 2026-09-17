import subprocess, time, json, urllib.request, socket, os, base64

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
        elif length <= 65535:
            import struct
            header.append(0x80 | 126); header.extend(struct.pack('!H', length))
        else:
            import struct
            header.append(0x80 | 127); header.extend(struct.pack('!Q', length))
        mask = os.urandom(4)
        header.extend(mask)
        self.sock.sendall(header + bytearray(b ^ mask[i % 4] for i, b in enumerate(payload)))
        while True:
            data = self.sock.recv(2)
            if not data: return None
            b1, b2 = data[0], data[1]
            plen = b2 & 0x7f
            if plen == 126:
                import struct
                plen = struct.unpack('!H', self.sock.recv(2))[0]
            elif plen == 127:
                import struct
                plen = struct.unpack('!Q', self.sock.recv(8))[0]
            body = b''
            while len(body) < plen: body += self.sock.recv(plen - len(body))
            msg = json.loads(body.decode('utf-8', errors='ignore'))
            if msg.get('id') == self.msg_id: return msg

edge_path = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
proc = subprocess.Popen([edge_path, '--headless=new', '--remote-debugging-port=9222', '--disable-gpu', 'http://localhost:8080/'])
time.sleep(3)

try:
    tabs = json.loads(urllib.request.urlopen('http://localhost:9222/json').read().decode('utf-8'))
    tab = next(t for t in tabs if '8080' in t.get('url', ''))
    ws = MinimalWS(tab['webSocketDebuggerUrl'])
    ws.call('Network.enable')
    ws.call('Network.setCacheDisabled', {'cacheDisabled': True})
    ws.call('Page.navigate', {'url': 'http://localhost:8080/'})
    time.sleep(3)

    def eval_js(exp):
        res = ws.call('Runtime.evaluate', {'expression': exp, 'returnByValue': True})
        return res['result']['result'].get('value') if res and 'result' in res and 'result' in res['result'] else res

    eval_js('window.switchExperiment("quality")')
    eval_js('window.switchQualityTab("quality-overview")')
    time.sleep(0.5)

    # Mobile screenshot (480px)
    ws.call('Emulation.setDeviceMetricsOverride', {'width': 480, 'height': 900, 'deviceScaleFactor': 1, 'mobile': True})
    time.sleep(0.5)
    snap_mob = ws.call('Page.captureScreenshot', {'format': 'png'})
    if snap_mob and 'result' in snap_mob and 'data' in snap_mob['result']:
        img_data = base64.b64decode(snap_mob['result']['data'])
        out_path = r'C:\Users\10-64\.gemini\antigravity-ide\brain\042f002e-c04f-43f8-bf78-5f2b1dc74206\screenshot_qc_mobile_fixed.png'
        with open(out_path, 'wb') as f:
            f.write(img_data)
        print("Saved mobile screenshot to:", out_path)

finally:
    proc.terminate()
