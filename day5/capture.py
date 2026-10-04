"""
Step 1 -- capture training images of the green minifig from the webcam.

    python capture.py            # SPACE saves a frame to data/raw/
    python capture.py --auto 0.5 # 'a' toggles auto-save every 0.5 s

Keys:  SPACE save   a toggle auto-save   q quit

What to capture (the model only learns what it has seen):
  * the minifig everywhere in the frame -- left, right, top, bottom, center
  * near and far (big and small), tilted, partly hidden by a hand
  * different backgrounds and lighting, on the robot and off it
  * frames with NO minifig, and with OTHER-colored minifigs -- label.py
    marks those "nothing", which is what teaches the model "green only"
Aim for ~150-300 frames.  Frames are saved at SAVE_WIDTH px wide.
"""

import argparse
import os
import time

import cv2

from common import RAW_DIR, open_camera

SAVE_WIDTH = 640


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", type=float, default=0.5,
                    help="seconds between auto-saves when 'a' mode is on")
    ap.add_argument("--camera", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    existing = [f for f in os.listdir(RAW_DIR) if f.endswith(".jpg")]
    n = len(existing)
    print(f"{n} images already in {RAW_DIR}")

    cap = open_camera(args.camera)
    auto = False
    last_auto = 0.0

    def save(frame):
        nonlocal n
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (SAVE_WIDTH, int(h * SAVE_WIDTH / w)))
        path = os.path.join(RAW_DIR, f"img_{n:04d}.jpg")
        cv2.imwrite(path, small, [cv2.IMWRITE_JPEG_QUALITY, 92])
        n += 1
        print(f"saved {path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            now = time.monotonic()
            if auto and now - last_auto >= args.auto:
                save(frame)
                last_auto = now

            hud = frame.copy()
            cv2.putText(hud, f"saved: {n}   auto-save: {'ON' if auto else 'off'} (a)   SPACE=save  q=quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            if auto:
                cv2.circle(hud, (frame.shape[1] - 30, 30), 12, (0, 0, 255), -1)
            cv2.imshow("capture", hud)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                save(frame)
            if key == ord("a"):
                auto = not auto
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"{n} images in {RAW_DIR}")


if __name__ == "__main__":
    main()
