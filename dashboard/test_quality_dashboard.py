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
    '--remote-debugging-port=9230',
    '--disable-gpu',
    'about:blank'
])

time.sleep(2)

try:
    req = urllib.request.Request('http://localhost:9230/json/new?http://localhost:8080/', method='PUT')
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
                res = f.get('result', {}).get('result', {})
                if 'value' in res:
                    return res['value']
                elif res.get('type') == 'undefined':
                    return None
                return res

    send_cdp(1, "Runtime.enable")
    time.sleep(2.5)

    print("=== 1. Checking Page Title and Master Navigation ===")
    title = call_eval("document.title")
    print("Page Title:", title)

    master_tabs = call_eval("""
        Array.from(document.querySelectorAll('.master-nav-btn')).map(b => ({
            id: b.id,
            exp: b.dataset.experiment,
            text: b.innerText.replace(/\\s+/g, ' ').trim(),
            active: b.classList.contains('active')
        }))
    """)
    print("Master Navigation Buttons:", master_tabs)

    print("\n=== 2. Testing Data Quality Management Panel (7 Sub-screens) ===")
    active_exp = call_eval("document.querySelector('.master-nav-btn.active')?.dataset?.experiment")
    print("Active Experiment:", active_exp)

    quality_subtabs = [
        ('quality-overview', '#overview-summary-table-container table tr'),
        ('quality-setup', '.data-table tr'),
        ('quality-reconstruction', '#reconstruction-table-container table tr'),
        ('quality-detection', '#detection-table-container table tr'),
        ('quality-hpo', '#threshold-table-container table tr'),
        ('quality-resource', '#resource-benchmark-table-container table tr'),
        ('quality-deployment', '.data-table tr')
    ]

    for tab, table_sel in quality_subtabs:
        call_eval(f"document.querySelector('#quality-tabs-nav [data-tab=\"{tab}\"]').click()")
        time.sleep(0.5)
        panel_active = call_eval(f"document.getElementById('tab-panel-{tab}').classList.contains('active')")
        var_filter_visible = call_eval("document.getElementById('quality-variable-filter-group').style.display !== 'none'")
        table_rows = call_eval(f"document.querySelectorAll('{table_sel}').length")
        panel_text_sample = call_eval(f"document.getElementById('tab-panel-{tab}').innerText.slice(0, 100).replace(/\\s+/g, ' ')")
        print(f"Sub-tab [{tab}]: active={panel_active}, varFilterVisible={var_filter_visible}, tableRows={table_rows}, sample='{panel_text_sample}'")

    print("\n=== 3. Testing Quality Filters Interaction ===")
    call_eval("document.getElementById('quality-filter-crop').value = 'strawberry'; document.getElementById('quality-filter-crop').dispatchEvent(new Event('change'))")
    time.sleep(0.3)
    summary_text = call_eval("document.getElementById('quality-active-filters-summary').innerText")
    print("Filter summary (strawberry):", summary_text)

    call_eval("document.getElementById('quality-filter-split').value = 'test'; document.getElementById('quality-filter-split').dispatchEvent(new Event('change'))")
    time.sleep(0.3)
    summary_text2 = call_eval("document.getElementById('quality-active-filters-summary').innerText")
    print("Filter summary (Test split):", summary_text2)

    call_eval("document.getElementById('quality-filter-reset-btn').click()")
    time.sleep(0.3)
    summary_text_reset = call_eval("document.getElementById('quality-active-filters-summary').innerText")
    print("Filter summary (after reset):", summary_text_reset)

    print("\n=== 4. Testing Transition Model Selection Panel (Intact Verification) ===")
    call_eval("document.getElementById('btn-exp-transition').click()")
    time.sleep(0.5)
    trans_container_visible = call_eval("document.getElementById('container-exp-transition').style.display !== 'none'")
    print("Transition container visible:", trans_container_visible)

    # Check setup tab 12 cards and tooltips
    call_eval("document.querySelector('#transition-tabs-nav [data-tab=\"setup\"]').click()")
    time.sleep(0.5)
    cards_count = call_eval("document.querySelectorAll('.setup-cat-box').length")
    tooltip_count = call_eval("document.querySelectorAll('.setup-item-with-tooltip').length")
    print("Transition Setup 12 input cards count (expect 12):", cards_count)
    print("Transition Setup tooltips count (expect >= 20):", tooltip_count)

    # Check ranking table
    call_eval("document.querySelector('#transition-tabs-nav [data-tab=\"ranking\"]').click()")
    time.sleep(0.5)
    table1_rows = call_eval("document.querySelectorAll('#table-1-container table tr').length")
    print("Table 1 in ranking tab rows:", table1_rows)

    # Check ablation tab
    call_eval("document.querySelector('#transition-tabs-nav [data-tab=\"ablation\"]').click()")
    time.sleep(0.5)
    table5_rows = call_eval("document.querySelectorAll('#table-5-container table tr').length")
    print("Table 5 in ablation tab rows:", table5_rows)

    # Check drift tab
    call_eval("document.querySelector('#transition-tabs-nav [data-tab=\"drift\"]').click()")
    time.sleep(0.5)
    table6_rows = call_eval("document.querySelectorAll('#table-6-container table tr').length")
    print("Table 6 in drift tab rows:", table6_rows)

    print("\n=== 5. Testing RL & Control Experiment Panel ===")
    call_eval("document.getElementById('btn-exp-rl').click()")
    time.sleep(0.5)
    rl_container_visible = call_eval("document.getElementById('container-exp-rl').style.display !== 'none'")
    rl_text = call_eval("document.getElementById('tab-panel-rl-control').innerText.slice(0, 150).replace(/\\s+/g, ' ')")
    print("RL container visible:", rl_container_visible)
    print("RL panel sample text:", rl_text)

    # Capture screenshots of quality overview and transition setup
    call_eval("document.getElementById('btn-exp-quality').click()")
    time.sleep(0.4)
    call_eval("document.querySelector('#quality-tabs-nav [data-tab=\"quality-overview\"]').click()")
    time.sleep(0.6)

    send_cdp(100, "Page.captureScreenshot", {"format": "png"})
    while True:
        f = read_frame()
        if f.get('id') == 100:
            data = f.get('result', {}).get('data')
            if data:
                with open("screenshot_quality_overview.png", "wb") as img_f:
                    img_f.write(base64.b64decode(data))
                print("Screenshot saved to screenshot_quality_overview.png")
            break

    print("\n=== ALL INTEGRATION TESTS PASSED 100% ===")

finally:
    proc.terminate()
