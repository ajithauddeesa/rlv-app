import json
import cv2

data = json.load(open("outputs/tracks_clip01.json"))
cap = cv2.VideoCapture("data/videos/clip01_junction.mp4")
for tid in ["16", "18", "21", "22"]:
    pts = data["tracks"][tid]
    p = max(pts, key=lambda q: q["bbox"][3] - q["bbox"][1])   # frame where the bike is biggest
    cap.set(cv2.CAP_PROP_POS_FRAMES, p["frame"])
    ok, f = cap.read()
    x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
    crop = f[max(0, y1 - 20):y2 + 20, max(0, x1 - 20):x2 + 20]
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f"outputs/frames/diag_track{tid}.jpg", crop)
    print(tid, "frame", p["frame"], "bike box size", x2 - x1, "x", y2 - y1)