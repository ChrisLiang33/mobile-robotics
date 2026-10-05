#!/bin/bash
# Drive a USB-connected Arduino UNO Q from the terminal -- no App Lab window needed.
# Uses adb (App Lab installs a copy; so does `brew install --cask android-platform-tools`).
#
#   ./unoq.sh status        board name, WiFi, IP address, app state
#   ./unoq.sh shell         interactive shell on the board (type `exit` to leave)
#   ./unoq.sh wifi          list the WiFi networks the board can see
#   ./unoq.sh deploy        copy mqtt-minifig-car to the board and (re)start it
#   ./unoq.sh logs          follow the app's print() output (Ctrl-C to leave)
#   ./unoq.sh stop          stop the app
#   ./unoq.sh boot on|off   start the app automatically when the board powers up
#   ./unoq.sh run '<cmd>'   run one command on the board
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
APP="mqtt-minifig-car"
REMOTE_DIR="/home/arduino/ArduinoApps"
REMOTE_APP="$REMOTE_DIR/$APP"

ADB="$(command -v adb || true)"
if [ -z "$ADB" ]; then
  ADB="$(find "$HOME/Library/Application Support/arduino_applab_workspace" -type f -name adb 2>/dev/null | head -1 || true)"
fi
if [ -z "$ADB" ]; then
  echo "adb not found. Install Arduino App Lab, or run: brew install --cask android-platform-tools" >&2
  exit 1
fi
if [ "$("$ADB" get-state 2>/dev/null || true)" != "device" ]; then
  echo "No UNO Q found over USB. Use a data cable (not charge-only) and give the board ~30 s to boot." >&2
  exit 1
fi

case "${1:-}" in
  status)
    "$ADB" shell '
      echo "board:    $(hostname)"
      echo "wifi:     $(nmcli -t -f DEVICE,STATE,CONNECTION dev | grep "^wlan0:" | cut -d: -f2-)"
      echo "ip:       $(hostname -I | tr " " "\n" | grep -v -e "^172\.17\." -e "^$" | tr "\n" " ")"
      echo "internet: $( (ping -c 1 -W 3 test.mosquitto.org >/dev/null 2>&1 && echo yes) || echo no)"
      echo "boot app: $(arduino-app-cli properties get default 2>&1 | tail -1)"
      echo "our apps:"; arduino-app-cli app list 2>/dev/null | grep -E "^ID|^user:" || true
    ' ;;
  shell) exec "$ADB" shell ;;
  wifi)  "$ADB" shell 'nmcli device wifi rescan 2>/dev/null; sleep 3; nmcli -f IN-USE,SSID,SECURITY,SIGNAL device wifi list' ;;
  deploy)
    find "$HERE/$APP" -name "__pycache__" -type d -prune -exec rm -rf {} +
    "$ADB" push "$HERE/$APP" "$REMOTE_DIR/" | tail -1
    "$ADB" shell "arduino-app-cli app restart $REMOTE_APP"
    echo "Started. './unoq.sh logs' shows its output." ;;
  logs)  exec "$ADB" shell "arduino-app-cli app logs $REMOTE_APP --follow" ;;
  stop)  "$ADB" shell "arduino-app-cli app stop $REMOTE_APP" ;;
  boot)
    case "${2:-}" in
      on)  "$ADB" shell "arduino-app-cli properties set default $REMOTE_APP" ;;
      off) "$ADB" shell "arduino-app-cli properties set default none" ;;
      *)   echo "usage: $0 boot on|off" >&2; exit 1 ;;
    esac ;;
  run)   shift; "$ADB" shell "$*" ;;
  *)     sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
