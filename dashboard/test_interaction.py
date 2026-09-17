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

# Start Edge with remote debugging
proc = subprocess.Popen([
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    '--headless=new',
    '--remote-debugging-port=9224',
    '--disable-gpu',
    'about:blank'
])

time.sleep(2)

try:
    req = urllib.request.Request('http://localhost:9224/json/new?http://localhost:8080/', method='PUT')
    with urllib.request.urlopen(req) as r:
        target = json.loads(r.read())
    
    ws_url = target['webSocketDebuggerUrl']
    parsed = urlparse(ws_url)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((parsed.hostname, parsed.port))

    sec_key = base64.b64encode(os.urandom(16)).decode()
    handshake = (
        f"GET {parsed.path} HTTP/1.1\r\n"
        f"Host: {parsed.hostname}:{parsed.port}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {sec_key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n\r\n"
    )
    s.sendall(handshake.encode())
    s.recv(4096)

    def send_cdp(id_val, method, params=None):
        msg = json.dumps({"id": id_val, "method": method, "params": params or {}})
        payload = msg.encode('utf-8')
        length = len(payload)
        header = bytearray([0x81])
        mask = os.urandom(4)
        if length < 126:
            header.append(0x80 | length)
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
        if length == 126:
            length = int.from_bytes(s.recv(2), 'big')
        elif length == 127:
            length = int.from_bytes(s.recv(8), 'big')
        mask = s.recv(4) if masked else None
        data = b''
        while len(data) < length:
            chunk = s.recv(length - len(data))
            if not chunk: break
            data += chunk
        if masked:
            data = bytearray(b ^ mask[i % 4] for i, b in enumerate(data))
        return json.loads(data.decode('utf-8', errors='replace'))

    def call_eval(expr):
        call_id = int(time.time() * 1000) % 100000
        send_cdp(call_id, "Runtime.evaluate", {"expression": expr, "returnByValue": True})
        while True:
            f = read_frame()
            if f.get('id') == call_id:
                return f.get('result', {}).get('result', {}).get('value')

    send_cdp(1, "Runtime.enable")
    time.sleep(1.5)

    # 1. Check title and active tab
    active_tab = call_eval("document.querySelector('.tab-btn.active')?.dataset?.tab")
    print("Initial Active Tab:", active_tab)

    # 2. Click ranking tab
    call_eval("document.querySelector('[data-tab=\"ranking\"]').click()")
    time.sleep(0.5)
    active_tab = call_eval("document.querySelector('.tab-btn.active')?.dataset?.tab")
    print("After click ranking tab:", active_tab)

    # Check ranking section contents
    table1_present = call_eval("Boolean(document.querySelector('#table-1-container table'))")
    ranking_card_text = call_eval("document.querySelector('.ranking-metrics-guide-card')?.innerText")
    print("Table 1 present:", table1_present)
    print("Ranking metrics guide card present:", bool(ranking_card_text))
    if ranking_card_text:
        print("Ranking card summary:", ranking_card_text[:150].replace('\n', ' '))

    # 3. Click setup tab
    call_eval("document.querySelector('[data-tab=\"setup\"]').click()")
    time.sleep(0.5)
    active_tab = call_eval("document.querySelector('.tab-btn.active')?.dataset?.tab")
    print("After click setup tab:", active_tab)
    setup_text = call_eval("document.getElementById('tab-panel-setup')?.innerText")
    print("Setup physical bounds present (−60~80°C):", "−60~80°C" in setup_text or "-60~80°C" in setup_text)
    print("Setup Action History present:", "직전 실행 Action" in setup_text)
    print("Setup Control Condition present:", "현재 적용할 Action" in setup_text)

    # 4. Click filter reset button
    call_eval("document.getElementById('filter-reset-btn').click()")
    print("Filter reset button clicked successfully.")

    # 5. Check other tabs: onestep, safety, drift, resource, ablation
    for tab in ['onestep', 'safety', 'drift', 'resource', 'ablation', 'overview']:
        call_eval(f"document.querySelector('[data-tab=\"{tab}\"]').click()")
        tab_active = call_eval(f"document.getElementById('tab-panel-{tab}').classList.contains('active')")
        print(f"Tab '{tab}' switched successfully: {tab_active}")

finally:
    proc.terminate()
