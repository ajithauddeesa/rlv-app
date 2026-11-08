import json
import cv2
import numpy as np
import easyocr

ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
reader = easyocr.Reader(["en"], gpu=False)
plates = json.load(open("outputs/plates_clip01.json"))["plates"]


def variants(img):
    up = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
    return {
        "raw": img,
        "color4x": cv2.copyMakeBorder(up, 20, 20, 20, 20, cv2.BORDER_REPLICATE),
        "gray4x": gray,
        "sharp4x": sharp,
    }


def read(img):
    out = reader.readtext(img, allowlist=ALLOW, detail=1)
    if not out:
        return "-", 0.0
    return "".join(o[1] for o in out).replace(" ", ""), round(sum(o[2] for o in out) / len(out), 2)


for tid in sorted(plates, key=int):
    dets = sorted(plates[tid], key=lambda d: d["plate_bbox"][2] - d["plate_bbox"][0], reverse=True)[:3]
    for k, d in enumerate(dets):
        img = cv2.imread(d["crop"])
        v = variants(img)
        row = {name: read(im) for name, im in v.items()}
        w = d["plate_bbox"][2] - d["plate_bbox"][0]
        print(f"track {tid:>3} frame {d['frame']:>4} w={w:>3}  " +
              "  ".join(f"{n}={t}({c})" for n, (t, c) in row.items()))
        if k == 0:
            tiles = []
            for name, im in v.items():
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im
                tiles.append(cv2.resize(im, (400, 130)))
            cv2.imwrite(f"outputs/frames/diag_ocr_track{tid}.jpg", np.vstack(tiles))
            