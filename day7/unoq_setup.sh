#!/bin/bash
# One-time setup ON THE UNO Q (run it there over ssh): a Python environment
# with the LEGO Bluetooth library and MQTT, and a Bluetooth sanity check.
#   scp -r day7 arduino@<board-ip>:~/sumo && ssh arduino@<board-ip> 'bash ~/sumo/unoq_setup.sh'
set -e
cd "$(dirname "$0")"
echo "== Bluetooth on this board"
ls /sys/class/bluetooth 2>/dev/null || echo "no Bluetooth adapter visible to Linux!"
systemctl is-active bluetooth 2>/dev/null | sed 's/^/bluetooth service: /' || true
echo "== Python environment (~/sumo/.venv)"
if ! python3 -m venv .venv 2>/dev/null; then
  echo "python3-venv missing: installing (needs sudo)"; sudo apt-get install -y python3-venv; python3 -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q legoeducation paho-mqtt numpy
echo "== quick BLE scan for LEGO devices (switch the hub on first)"
.venv/bin/python - <<'PY'
import asyncio
from bleak import BleakScanner
async def go():
    devs = await BleakScanner.discover(timeout=6.0, return_adv=True)
    lego = [(d.name or adv.local_name) for d, adv in devs.values()
            if "0000fd02-0000-1000-8000-00805f9b34fb" in adv.service_uuids or 919 in adv.manufacturer_data]
    print(f"{len(devs)} BLE devices seen; LEGO: {lego or 'none'}")
asyncio.run(go())
PY
echo "== done. run the brain with:   ~/sumo/.venv/bin/python ~/sumo/sumo.py --headless"
