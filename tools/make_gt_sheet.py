import csv
import json
import os

import cv2
import numpy as np

from app.stopline import StopLine

VIDEO = "data/videos/sample_short.mp4"
TRACKS = "outputs/tracks.json"
LINE_FILE = "data/stopline.json"
GT = "data/ground_truth/gt_events.csv"
W, H, SW = 576, 324, 504

os.makedirs("outputs/gt", exist_ok=True)
os.makedirs("data/ground_truth", exist_ok=True)
line = json.load(open(LINE_FILE)) if os.path.exists(LINE_FILE) else [0.0, 1000.0, 1230.0, 1000.0]
sl = StopLine(*line, approach_sign=1)
data = json.load(open(TRACKS))
cap = cv2.VideoCapture(VIDEO)


def grab(n):
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, f = cap.read()
    return f if ok else np.zeros((1080, 1920, 3), np.uint8)


def bottom(p):
    b = p["bbox"]
    return (b[0] + b[2]) / 2, b[3]


def annotate(f, p):
    x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
    cv2.line(f, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (0, 0, 255), 6)
    cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 255), 4)
    return cv2.resize(f, (W, H))


tids = sorted(data["tracks"], key=int)
for tid in tids:
    pts = data["tracks"][tid]
    first = pts[0]
    after = next((p for p in pts if sl.signed_distance(*bottom(p)) <= 0), None)
    second = after or pts[-1]
    f1 = annotate(grab(first["frame"]), first)
    fr2 = grab(second["frame"])
    sig = cv2.resize(fr2[0:450, 1000:1700], (SW, H))
    f2 = annotate(fr2, second)
    tile = np.hstack([f1, f2, sig])
    head = np.zeros((40, tile.shape[1], 3), np.uint8)
    where = "first frame at/over the line" if after else "LAST frame (never reached the line)"
    label = f"track {tid} | left: first frame t={first['t']} | middle: {where} t={second['t']} | right: signal then"
    cv2.putText(head, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(f"outputs/gt/track_{tid}.jpg", np.vstack([head, tile]))

if not os.path.exists(GT):
    with open(GT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["track_id", "is_motorcycle", "crossed", "true_signal", "true_plate",
                    "plate_on_correct_bike", "notes"])
        for tid in tids:
            w.writerow([tid, "", "", "", "", "", ""])
    print("created", GT)
print(len(tids), "images written to outputs/gt/")