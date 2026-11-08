import cv2
import numpy as np

VIDEO = "data/videos/clip01_junction.mp4"
v = cv2.VideoCapture(VIDEO)
fps = v.get(cv2.CAP_PROP_FPS)

tiles = []
for t in range(0, 35, 2):  # one frame every 2 seconds
    v.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, f = v.read()
    if not ok:
        continue
    f = cv2.resize(f, (480, 270))
    cv2.putText(f, f"{t}s", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
    tiles.append(f)

while len(tiles) % 4:
    tiles.append(np.zeros((270, 480, 3), np.uint8))

rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
sheet = np.vstack(rows)
cv2.imwrite("outputs/frames/contact_sheet.jpg", sheet)
print("saved", sheet.shape)