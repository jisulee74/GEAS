import subprocess
import time
import json
import urllib.request
import socket
import os
import sys
import base64
from urllib.parse import urlparse

sys.stdout.reconfigure(encoding='utf-8')

proc = subprocess.Popen([
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    '--headless=new',
    '--remote-debugging-port=9237',
    '--disable-gpu',
    'about:blank'
])
time.sleep(2)
try:
    req = urllib.request.Request('http://localhost:9237/json/new', method='PUT')
    with urllib.request.urlopen(req) as r:
        target = json.loads(r.read())
    ws_url = target['webSocketDebuggerUrl']
    parsed = urlparse(ws_url)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((parsed.hostname, parsed.port))
    sec_key = base64.b64encode(os.urandom(16)).decode()
    handshake = f'GET {parsed.path} HTTP/1.1\r\nHost: {parsed.hostname}:{parsed.port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {sec_key}\r\nSec-WebSocket-Version: 13\r\n\r\n'
    s.sendall(handshake.encode())
    s.recv(4096)

    def send_cdp(id_val, method, params=None):
        msg = json.dumps({'id': id_val, 'method': method, 'params': params or {}})
        payload = msg.encode('utf-8')
        length = len(payload)
        header = bytearray([0x81])
        mask = os.urandom(4)
        if length < 126: header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header.extend(length.to_bytes(2, 'big'))
        else:
            header.append(0x80 | 127)
            header.extend(length.to_bytes(8, 'big'))
        header.extend(mask)
        masked = bytearray(b ^ mask[i % 4] for i, b in enumerate(payload))
        s.sendall(header + masked)

    def read_frame():
        b1, b2 = s.recv(2)
        opcode = b1 & 0x0F
        masked = (b2 & 0x80) != 0
        length = b2 & 0x7F
        if length == 126: length = int.from_bytes(s.recv(2), 'big')
        elif length == 127: length = int.from_bytes(s.recv(8), 'big')
        mask = s.recv(4) if masked else None
        data = b''
        while len(data) < length:
            chunk = s.recv(length - len(data))
            if not chunk: break
            data += chunk
        if masked: data = bytearray(b ^ mask[i % 4] for i, b in enumerate(data))
        return json.loads(data.decode('utf-8', errors='replace'))

    def call_eval_async(expr):
        cid = int(time.time() * 1000) % 100000
        send_cdp(cid, 'Runtime.evaluate', {'expression': expr, 'awaitPromise': True, 'returnByValue': True})
        while True:
            f = read_frame()
            if f.get('id') == cid:
                return f.get('result', {}).get('result', {}).get('value')

    send_cdp(1, 'Runtime.enable')
    send_cdp(2, 'Page.navigate', {'url': 'http://localhost:8080/'})
    time.sleep(3.0)

    res = call_eval_async("""
        (async () => {
            const { dataService } = await import('./src/js/core/data-loader.js?v=20260911_v22');
            const { filterStore } = await import('./src/js/core/filter-store.js?v=20260911_v22');
            const rawT1 = dataService.getTable1();
            const filteredT1 = filterStore.filterRows(rawT1);
            const rawT2 = dataService.getTable2();
            const rawT3 = dataService.getTable3();
            return {
                isLoaded: dataService.isLoaded,
                rawT1Len: rawT1 ? rawT1.length : 0,
                filteredT1Len: filteredT1 ? filteredT1.length : 0,
                rawT2Len: rawT2 ? rawT2.length : 0,
                rawT3Len: rawT3 ? rawT3.length : 0,
                filterState: filterStore.getState()
            };
        })()
    """)
    print("Async inspect result:", json.dumps(res, indent=2))
finally:
    proc.terminate()
