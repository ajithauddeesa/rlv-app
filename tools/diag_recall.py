import json
import cv2
from ultralytics import YOLO

model = YOLO("models/plate_detector.pt")
data = json.load(open("outputs/tracks_clip01.json"))
cap = cv2.VideoCapture("data/videos/clip01_junction.mp4")
CFGS = {
    "conf0.15": dict(conf=0.15),
    "conf0.03": dict(conf=0.03),
    "conf0.03+imgsz1280": dict(conf=0.03, imgsz=1280),
    "conf0.03+augment": dict(conf=0.03, augment=True),
}
for tid in ["16", "18", "21", "22", "20", "1", "5"]:
    pts = data["tracks"][tid]
    big = sorted(pts, key=lambda q: (q["bbox"][2] - q["bbox"][0]) * (q["bbox"][3] - q["bbox"][1]),
                 reverse=True)[:2]
    for p in big:
        cap.set(cv2.CAP_PROP_POS_FRAMES, p["frame"])
        ok, f = cap.read()
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        crop = f[max(0, y1 - 15):y2 + 15, max(0, x1 - 15):x2 + 15]
        for name, kw in CFGS.items():
            r = model.predict(crop, verbose=False, **kw)[0]
            found = [(round(float(b.conf), 2), [int(v) for v in b.xyxy[0]]) for b in r.boxes][:3]
            print(f"track {tid:>2} f{p['frame']:>4} {name:<20} {found}")
            if name == "conf0.03":
                cv2.imwrite(f"outputs/frames/recall_t{tid}_f{p['frame']}.jpg", r.plot())