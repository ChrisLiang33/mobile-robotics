"""
Step 4 -- live tracker: webcam -> our YOLO model -> minifig position over MQTT.

    python track_minifig.py                 # uses models/minifig.pt
    python track_minifig.py --no-mqtt       # vision only
    python track_minifig.py --model yolov8n.pt --class-name person
                                            # plumbing test with the stock COCO
                                            # model before ours is trained

Every frame: run the detector, keep the most confident green_minifig box,
draw it (green box, BLUE dot on the centroid -- same color as the UNO Q's
LEDs), and publish to MQTT_TOPIC (see common.py) the JSON the UNO Q app reads:
    {"x": cx, "y": cy, "w": W, "h": H, "bw": bw, "bh": bh, "conf": 0.93, "found": true}
If nothing is detected:
    {"found": false, "w": W, "h": H}
so the UNO Q can stop the motors immediately instead of waiting for a timeout.
The corner of the window shows a 13x8 preview of what the LED matrix should
display, so the laptop side can be debugged without the board.
"""

import argparse
import json
import time

import cv2

from common import MODEL_PATH, MQTT_TOPIC, open_camera

BLUE, GREEN, YELLOW, RED, WHITE = (255, 0, 0), (0, 255, 0), (0, 220, 255), (0, 0, 255), (255, 255, 255)
LED_COLS, LED_ROWS = 13, 8
DEADBAND = 0.08     # mirrors the UNO Q policy, for the on-screen hint only


def best_detection(result, class_id):
    """(x1, y1, x2, y2, conf) of the most confident box of class_id, or None."""
    best = None
    for b in result.boxes:
        if int(b.cls[0]) != class_id:
            continue
        conf = float(b.conf[0])
        if best is None or conf > best[4]:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            best = (x1, y1, x2, y2, conf)
    return best


def draw_led_preview(frame, col, row, fresh):
    """Mini 13x8 grid in the bottom-right corner = what the UNO Q matrix shows."""
    cell, pad = 14, 12
    h, w = frame.shape[:2]
    x0, y0 = w - pad - LED_COLS * cell, h - pad - LED_ROWS * cell
    cv2.rectangle(frame, (x0 - 4, y0 - 4), (x0 + LED_COLS * cell + 4, y0 + LED_ROWS * cell + 4), (40, 40, 40), -1)
    half = 1 if fresh else 0
    for r in range(LED_ROWS):
        for c in range(LED_COLS):
            lit = col is not None and abs(c - col) <= half and abs(r - row) <= half
            cv2.rectangle(frame, (x0 + c * cell + 2, y0 + r * cell + 2),
                          (x0 + (c + 1) * cell - 2, y0 + (r + 1) * cell - 2),
                          BLUE if lit else (90, 90, 90), -1)
    cv2.putText(frame, "UNO Q matrix", (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--class-name", default="green_minifig", help="which class to track")
    ap.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--topic", default=MQTT_TOPIC)
    ap.add_argument("--hz", type=float, default=15.0, help="max MQTT publish rate")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--no-mqtt", action="store_true")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (testing)")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(args.model)
    names = {v: k for k, v in model.names.items()}
    if args.class_name not in names:
        raise SystemExit(f"class '{args.class_name}' not in model classes {list(names)}")
    class_id = names[args.class_name]
    print(f"model {args.model} on {device}, tracking class '{args.class_name}' (id {class_id})")

    client = None
    if not args.no_mqtt:
        from mqttlib import MQTTClient
        client = MQTTClient()
        client.connect()
        print(f"publishing to {args.topic} on {client.broker}")

    cap = open_camera(args.camera)
    last_pub = 0.0
    last_seen = 0.0
    last_led = (None, None)
    n = 0
    fps_t, fps = time.monotonic(), 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            n += 1
            now = time.monotonic()
            H, W = frame.shape[:2]
            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=device, verbose=False)[0]
            det = best_detection(result, class_id)

            cv2.line(frame, (W // 2, 0), (W // 2, H), YELLOW, 1)
            band = int(DEADBAND * W / 2)
            cv2.rectangle(frame, (W // 2 - band, 0), (W // 2 + band, H), YELLOW, 1)

            if det is not None:
                x1, y1, x2, y2, conf = det
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), GREEN, 2)
                cv2.circle(frame, (int(cx), int(cy)), 9, BLUE, -1)
                cv2.putText(frame, f"{args.class_name} {conf:.2f}  ({cx:.0f}, {cy:.0f})",
                            (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)
                err = (cx - W / 2) / (W / 2)
                # err > 0: the minifig is right of center, so the car has to carry it LEFT
                hint = "STOP (centered)" if abs(err) < DEADBAND else \
                    ("move it LEFT, toward center" if err > 0 else "move it RIGHT, toward center")
                cv2.putText(frame, f"err {err:+.2f}   UNO Q should: {hint}",
                            (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
                col = min(LED_COLS - 1, max(0, int(cx / W * LED_COLS)))
                row = min(LED_ROWS - 1, max(0, int(cy / H * LED_ROWS)))
                last_led, last_seen = (col, row), now
                payload = {"x": round(cx), "y": round(cy), "w": W, "h": H,
                           "bw": round(x2 - x1), "bh": round(y2 - y1),
                           "conf": round(conf, 3), "found": True}
            else:
                cv2.putText(frame, "NO MINIFIG DETECTED -> UNO Q stops", (10, H - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)
                payload = {"found": False, "w": W, "h": H}

            if client is not None and now - last_pub >= 1.0 / args.hz:
                client.publish(args.topic, json.dumps(payload))
                last_pub = now

            draw_led_preview(frame, *last_led, fresh=(now - last_seen) < 0.75)
            if n % 10 == 0:
                fps, fps_t = 10 / (now - fps_t), now
            cv2.putText(frame, f"{fps:.0f} fps   model: {args.model.split('/')[-1]}   q: quit",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2, cv2.LINE_AA)
            cv2.imshow("minifig tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")) or (args.frames and n >= args.frames):
                break
    finally:
        if client is not None:
            client.publish(args.topic, json.dumps({"found": False, "w": 0, "h": 0}))
            client.disconnect()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
