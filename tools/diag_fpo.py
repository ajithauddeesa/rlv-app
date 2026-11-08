import json
import cv2
from fast_plate_ocr import LicensePlateRecognizer

m = LicensePlateRecognizer("cct-s-v2-global-model")
plates = json.load(open("outputs/plates_clip01.json"))["plates"]

for tid in ["7", "13", "14", "15"]:
    dets = sorted(plates[tid], key=lambda d: d["plate_bbox"][2] - d["plate_bbox"][0], reverse=True)[:6]
    for d in dets:
        img = cv2.imread(d["crop"])
        w = d["plate_bbox"][2] - d["plate_bbox"][0]
        try:
            pred = m.run(img, return_confidence=True)[0]
        except Exception as e:
            pred = f"ERROR {e}"
        print(f"track {tid:>2} f{d['frame']:>4} w={w:>3}  {pred}")