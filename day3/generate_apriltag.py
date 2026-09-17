"""
Generate a printable AprilTag (family tag36h11) as a PNG.

Run:
    python generate_apriltag.py             # tag id 0, 70 mm, 300 dpi
    python generate_apriltag.py --id 3      # a different tag id

Print the PNG at 100% scale (no "fit to page") so the black square comes
out at the requested physical size, then tape it to a tower on the car.
Keep the white margin around the tag -- the detector needs a quiet zone.
"""

import argparse

import cv2
import numpy as np

MM_PER_INCH = 25.4


def main():
    p = argparse.ArgumentParser(description="Generate a printable tag36h11 AprilTag PNG")
    p.add_argument("--id", type=int, default=0, help="tag id within the 36h11 family (default 0)")
    p.add_argument("--size-mm", type=float, default=70.0, help="printed size of the black square in mm (default 70)")
    p.add_argument("--dpi", type=int, default=300, help="print resolution the pixel size is computed for (default 300)")
    args = p.parse_args()

    tag_px = int(round(args.size_mm / MM_PER_INCH * args.dpi))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag = cv2.aruco.generateImageMarker(dictionary, args.id, tag_px)

    # White quiet zone (~20% of the tag on every side) plus a label strip.
    margin = tag_px // 5
    label_h = margin
    page = np.full((tag_px + 2 * margin + label_h, tag_px + 2 * margin), 255, dtype=np.uint8)
    page[margin:margin + tag_px, margin:margin + tag_px] = tag
    cv2.putText(
        page,
        f"tag36h11  id={args.id}  {args.size_mm:g}mm @ {args.dpi}dpi",
        (margin, tag_px + 2 * margin + label_h // 2),
        cv2.FONT_HERSHEY_SIMPLEX, tag_px / 900.0, 0, max(1, tag_px // 400), cv2.LINE_AA,
    )

    out = f"tag36h11_id{args.id}.png"
    cv2.imwrite(out, page)
    print(f"Wrote {out} ({page.shape[1]}x{page.shape[0]} px).")
    print(f"Print at 100% scale for a {args.size_mm:g} mm tag. Keep the white border.")


if __name__ == "__main__":
    main()
