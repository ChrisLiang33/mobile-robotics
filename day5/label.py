"""
Step 2 -- label the captured images (semi-automatic, human-verified).

    python label.py                    # every image in data/raw with no label yet
    python label.py --redo             # revisit everything
    python label.py --only 20 113 176  # revisit just these photo numbers

For each image the tool PROPOSES a box around the biggest green blob (plain
HSV thresholding -- classical CV, no neural net, no cloud service). You then
accept it, redraw it, or reject it. Nothing is written without a keypress,
so every label in the dataset is one a human checked.

Keys:
  SPACE / ENTER  accept the box shown (yellow = proposal, cyan = yours)
  mouse drag     draw the box yourself (replaces the proposal)
  x              NO green minifig here -> writes an EMPTY label
                 (negative example: background, other-colored minifig...)
  s              skip for now        d  delete this image from the dataset
  m              toggle green-mask view (tune GREEN_LO/HI if proposals are off)
  b              back one image      q  quit

Labels are YOLO format: "0 cx cy w h", all normalized 0-1, in data/labels/.
"""

import argparse
import os

import cv2
import numpy as np

from common import RAW_DIR, LABEL_DIR

# HSV range for the minifig's plastic (OpenCV hue is 0-180): a NARROW band of
# SATURATED green. Measured on our photos, minifig pixels sit at hue 66-76 with
# saturation around 215, while the car's green jumper wires are duller
# (saturation around 115) and spread over hue 48-83. A loose "anything green"
# range (35-85, sat 70+) put boxes on the wires and stretched boxes over wire
# loops; this one doesn't. Press m in the tool to see the mask, and widen the
# range if your minifig isn't being found in your lighting.
GREEN_LO = np.array([62, 120, 50])
GREEN_HI = np.array([79, 255, 255])
MIN_AREA = 70       # px^2 -- ignore specks
PAD = 7             # px added around the blob so the box covers edges/limbs
SCALE = 2           # show images this many times larger (boxes are small)

YELLOW, CYAN, WHITE = (0, 255, 255), (255, 255, 0), (255, 255, 255)


def green_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))   # rejoin a figure split by a finger
    return mask


def propose_box(img):
    """Bounding box (x1, y1, x2, y2) of the largest minifig-green blob, or None."""
    mask = green_mask(img)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA:
        return None
    x, y, w, h = cv2.boundingRect(c)
    H, W = img.shape[:2]
    return (max(0, x - PAD), max(0, y - PAD), min(W - 1, x + w + PAD), min(H - 1, y + h + PAD))


def write_label(name, box, shape):
    os.makedirs(LABEL_DIR, exist_ok=True)
    path = os.path.join(LABEL_DIR, name + ".txt")
    with open(path, "w") as f:
        if box is not None:
            H, W = shape[:2]
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            bw, bh = (x2 - x1) / W, (y2 - y1) / H
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", action="store_true", help="revisit already-labeled images too")
    ap.add_argument("--only", type=int, nargs="+", metavar="N",
                    help="revisit only these photo numbers (e.g. 20 for img_0020.jpg)")
    args = ap.parse_args()

    images = sorted(f for f in os.listdir(RAW_DIR) if f.lower().endswith(".jpg"))
    if args.only:
        wanted = {f"img_{n:04d}.jpg" for n in args.only}
        missing = sorted(wanted - set(images))
        if missing:
            print("not found:", ", ".join(missing))
        images = [f for f in images if f in wanted]
    elif not args.redo:
        images = [f for f in images
                  if not os.path.exists(os.path.join(LABEL_DIR, os.path.splitext(f)[0] + ".txt"))]
    if not images:
        print("Nothing to label (use --redo to revisit).")
        return
    print(f"{len(images)} images to label")

    state = {"box": None, "manual": False, "drag": None, "proposal": None, "size": (0, 0)}

    def on_mouse(event, x, y, flags, _):
        x, y = x // SCALE, y // SCALE          # window pixels -> image pixels
        if event == cv2.EVENT_LBUTTONDOWN:
            state["drag"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"] is not None:
            x0, y0 = state["drag"]
            state["box"] = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
            state["manual"] = True
        elif event == cv2.EVENT_LBUTTONUP and state["drag"] is not None:
            x0, y0 = state["drag"]
            state["drag"] = None
            if abs(x - x0) < 4 or abs(y - y0) < 4:      # a click, not a drag
                state["box"], state["manual"] = state["proposal"], False
                return
            W, H = state["size"]
            state["box"] = (max(0, min(x0, x)), max(0, min(y0, y)), min(W - 1, max(x0, x)), min(H - 1, max(y0, y)))
            state["manual"] = True

    cv2.namedWindow("label")
    cv2.setMouseCallback("label", on_mouse)

    i = 0
    show_mask = False
    done = skipped = negatives = 0
    while 0 <= i < len(images):
        fname = images[i]
        name = os.path.splitext(fname)[0]
        img = cv2.imread(os.path.join(RAW_DIR, fname))
        if img is None:
            i += 1
            continue
        state["proposal"] = propose_box(img)
        state["box"], state["manual"], state["drag"] = state["proposal"], False, None
        state["size"] = (img.shape[1], img.shape[0])

        while True:
            view = cv2.cvtColor(green_mask(img), cv2.COLOR_GRAY2BGR) if show_mask else img.copy()
            view = cv2.resize(view, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)
            if state["box"] is not None:
                x1, y1, x2, y2 = (v * SCALE for v in state["box"])
                cv2.rectangle(view, (x1, y1), (x2, y2), CYAN if state["manual"] else YELLOW, 2)
            status = ("box: " + ("yours" if state["manual"] else "proposal")) \
                if state["box"] is not None else "no green found -> x for negative, or draw"
            for color, thick in ((0, 0, 0), 4), (WHITE, 1):      # outlined text stays readable
                cv2.putText(view, f"[{i + 1}/{len(images)}] {fname}   {status}",
                            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thick + 1, cv2.LINE_AA)
                cv2.putText(view, "SPACE accept   drag = draw   x none   s skip   d delete   m mask   b back   q quit",
                            (10, view.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thick, cv2.LINE_AA)
            cv2.imshow("label", view)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord(" "), 13) and state["box"] is not None:
                write_label(name, state["box"], img.shape); done += 1; i += 1; break
            if key == ord("x"):
                write_label(name, None, img.shape); negatives += 1; i += 1; break
            if key == ord("s"):
                skipped += 1; i += 1; break
            if key == ord("d"):
                os.remove(os.path.join(RAW_DIR, fname))
                lp = os.path.join(LABEL_DIR, name + ".txt")
                if os.path.exists(lp):
                    os.remove(lp)
                images.pop(i); break
            if key == ord("m"):
                show_mask = not show_mask
            if key == ord("b"):
                i = max(0, i - 1); break
            if key in (27, ord("q")):
                i = len(images); break

    cv2.destroyAllWindows()
    total = len([f for f in os.listdir(LABEL_DIR) if f.endswith(".txt")]) if os.path.isdir(LABEL_DIR) else 0
    print(f"labeled {done} (+{negatives} negatives), skipped {skipped}; {total} labels total in {LABEL_DIR}")


if __name__ == "__main__":
    main()
