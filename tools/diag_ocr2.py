import json
import cv2
import easyocr

ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
reader = easyocr.Reader(["en"], gpu=False)
plates = json.load(open("outputs/plates_clip01.json"))["plates"]

CONFIGS = {
    "default": dict(),
    "beam": dict(decoder="beamsearch", beamWidth=8),
    "lowthr": dict(text_threshold=0.3, low_text=0.2, link_threshold=0.2),
    "contrast": dict(mag_ratio=2.0, contrast_ths=0.05, adjust_contrast=0.7),
    "wide": dict(width_ths=1.5, text_threshold=0.3, low_text=0.2, decoder="beamsearch"),
}


def read(img, **kw):
    out = reader.readtext(img, allowlist=ALLOW, detail=1, **kw)
    out = sorted(out, key=lambda o: o[0][0][0])
    if not out:
        return "-"
    return "".join(o[1] for o in out).replace(" ", "") + f"({sum(o[2] for o in out) / len(out):.2f})"


for tid in ["7", "13", "14", "15"]:
    dets = sorted(plates[tid], key=lambda d: d["plate_bbox"][2] - d["plate_bbox"][0], reverse=True)[:4]
    for d in dets:
        img = cv2.imread(d["crop"])
        img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        row = "  ".join(f"{n}={read(gray, **kw)}" for n, kw in CONFIGS.items())
        print(f"track {tid:>2} f{d['frame']:>4}  {row}")