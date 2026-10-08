"""Print sheet of the game's AprilTags: 0 = start, 1/2/3 = level. 300 dpi, 60 mm each."""
import cv2
import numpy as np

MM, DPI = 60, 300
px = int(MM / 25.4 * DPI)
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
labels = {0: "START", 1: "LEVEL 1 - easy", 2: "LEVEL 2 - medium", 3: "LEVEL 3 - pro"}
tiles = []
for i, text in labels.items():
    tag = cv2.aruco.generateImageMarker(d, i, px)
    m = px // 5
    tile = np.full((px + 2 * m + m, px + 2 * m), 255, np.uint8)
    tile[m:m + px, m:m + px] = tag
    cv2.putText(tile, f"tag {i}: {text}", (m, px + 2 * m + m // 2), cv2.FONT_HERSHEY_SIMPLEX, px / 700, 0, 3, cv2.LINE_AA)
    tiles.append(tile)
sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
cv2.imwrite("tags_sheet.png", sheet)
print("wrote tags_sheet.png -- print at 100% (each tag 60 mm), cut into four cards")
