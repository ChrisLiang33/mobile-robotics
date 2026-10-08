#!/bin/bash
# Drive an Arduino UNO Q from the terminal -- no App Lab window needed.
#
# Over USB (default):  uses adb (App Lab installs a copy).
# Over WiFi:           set UNOQ_HOST to the board's IP, e.g.
#                        UNOQ_HOST=10.5.13.215 ./unoq.sh deploy
#                      (run `ssh-copy-id arduino@<ip>` once to skip passwords)
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
APP="${UNOQ_APP:-pingpong-board}"
REMOTE_DIR="/home/arduino/ArduinoApps"
REMOTE_APP="$REMOTE_DIR/$APP"

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; }
case "${1:-}" in ""|-h|--help|help) usage; exit 0 ;; esac

if [ -n "${UNOQ_HOST:-}" ]; then
  TARGET="arduino@$UNOQ_HOST"
  SSH_OPTS=(-o ConnectTimeout=8)
  remote()      { ssh "${SSH_OPTS[@]}" "$TARGET" "$1"; }
  remote_tty()  { exec ssh -t "${SSH_OPTS[@]}" "$TARGET" "$@"; }
  push_dir()    { scp -q -r "${SSH_OPTS[@]}" "$1" "$TARGET:$2"; }
  push_file()   { scp -q "${SSH_OPTS[@]}" "$1" "$TARGET:$2"; }
else
  ADB="$(command -v adb || true)"
  if [ -z "$ADB" ]; then
    ADB="$(find "$HOME/Library/Application Support/arduino_applab_workspace" -type f -name adb 2>/dev/null | head -1 || true)"
  fi
  if [ -z "$ADB" ]; then
    echo "adb not found. Install Arduino App Lab, or run: brew install --cask android-platform-tools" >&2
    echo "(or reach the board over WiFi: UNOQ_HOST=<board-ip> $0 ...)" >&2
    exit 1
  fi
  if [ "$("$ADB" get-state 2>/dev/null || true)" != "device" ]; then
    echo "No UNO Q found over USB. Use a data cable (not charge-only) and give the board ~30 s to boot." >&2
    echo "(or reach it over WiFi: UNOQ_HOST=<board-ip> $0 ...)" >&2
    exit 1
  fi
  remote()      { "$ADB" shell "$1"; }
  remote_tty()  { exec "$ADB" shell "$@"; }
  push_dir()    { "$ADB" push "$1" "$2" | tail -1; }
  push_file()   { "$ADB" push "$1" "$2" | tail -1; }
fi

case "${1:-}" in
  status)
    remote '
      echo "board:    $(hostname)   (App CLI $(arduino-app-cli version 2>/dev/null | sed -n "s/.*CLI version //p"))"
      echo "wifi:     $(nmcli -t -f DEVICE,STATE,CONNECTION dev | grep "^wlan0:" | cut -d: -f2-)"
      echo "ip:       $(hostname -I | tr " " "\n" | grep -v -e "^172\." -e "^$" | tr "\n" " ")"
      echo "internet: $( (ping -c 1 -W 3 test.mosquitto.org >/dev/null 2>&1 && echo yes) || echo no)"
      echo "boot app: $(arduino-app-cli properties get default 2>&1 | tail -1)"
      echo "user apps:"; arduino-app-cli app list 2>/dev/null | grep -E "^user:" | awk "{printf \"  %-28s %s\n\", \$1, \$(NF-1)}"
    ' ;;
  shell) remote_tty ;;
  wifi)  remote 'nmcli device wifi rescan 2>/dev/null; sleep 3; nmcli -f IN-USE,SSID,SECURITY,SIGNAL device wifi list' ;;
  deploy)
    find "$HERE/$APP" -name "__pycache__" -type d -prune -exec rm -rf {} +
    push_dir "$HERE/$APP" "$REMOTE_DIR/"
    # Factory-image boards list the bridge support libraries in their own
    # examples' sketch.yaml; those boards need our file to do the same.
    if remote 'grep -qs MsgPack "$HOME"/.local/share/arduino-app-cli/examples/blink/sketch/sketch.yaml && echo old || echo new' | grep -q old; then
      echo "factory-image board: installing sketch.old-image.yaml"
      push_file "$HERE/sketch.old-image.yaml" "$REMOTE_APP/sketch/sketch.yaml"
    fi
    # One app runs at a time: stop any other running user app first.
    remote 'for id in $(arduino-app-cli app list 2>/dev/null | awk "/^user:/ && / running / {print \$1}"); do
              [ "$id" = "user:'"$APP"'" ] || { echo "stopping $id"; arduino-app-cli app stop "$HOME/ArduinoApps/${id#user:}" >/dev/null 2>&1 || true; }
            done'
    remote "arduino-app-cli app restart $REMOTE_APP 2>&1 | tail -2"
    echo "Deployed. '$0 logs' shows its output." ;;
  logs)  remote_tty "arduino-app-cli app logs $REMOTE_APP --follow" ;;
  stop)  remote "arduino-app-cli app stop $REMOTE_APP 2>&1 | tail -1" ;;
  boot)
    case "${2:-}" in
      on)  remote "arduino-app-cli properties set default $REMOTE_APP" ;;
      off) remote "arduino-app-cli properties set default none" ;;
      *)   echo "usage: $0 boot on|off" >&2; exit 1 ;;
    esac ;;
  run)   shift; remote "$*" ;;
  *)     usage; exit 1 ;;
esac
