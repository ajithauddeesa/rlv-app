import json
from collections import Counter, defaultdict

ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MIN_W = 24      # skip plate crops narrower than this (pixels)
MIN_DET = 0.30  # skip weak plate detections


def fuse_readings(readings, min_reads=3, min_conf=0.25, agree=0.6):
    """Combine per-frame readings [{'text','conf'}, ...] into one plate string.
    Never guesses: disagreeing positions become '?', too little evidence becomes 'unreadable'."""
    n_total = len(readings)
    usable = [(r["text"], r["conf"]) for r in readings
              if r["text"] and len(r["text"]) >= 4 and r["conf"] >= min_conf]

    def unreadable():
        return {"text": "unreadable", "status": "unreadable", "confidence": 0.0,
                "n_used": len(usable), "n_total": n_total}

    if len(usable) < min_reads:
        return unreadable()

    length_weight = Counter()
    for t, c in usable:
        length_weight[len(t)] += c
    L = length_weight.most_common(1)[0][0]
    same = [(t, c) for t, c in usable if len(t) == L]
    if len(same) < min_reads:
        return unreadable()

    chars, shares = [], []
    for i in range(L):
        votes = defaultdict(float)
        for t, c in same:
            votes[t[i]] += c
        ch, w = max(votes.items(), key=lambda kv: kv[1])
        share = w / sum(votes.values())
        shares.append(share)
        chars.append(ch if share >= agree else "?")

    text = "".join(chars)
    mean_conf = sum(c for _, c in same) / len(same)
    return {"text": text, "status": "conflict" if "?" in text else "ok",
            "confidence": round(mean_conf * sum(shares) / L, 3),
            "n_used": len(same), "n_total": n_total}


def preprocess(img):
    import cv2
    h = img.shape[0]
    s = max(1.0, 96.0 / h)
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(g)
    return cv2.copyMakeBorder(g, 10, 10, 10, 10, cv2.BORDER_REPLICATE)


def read_crop(reader, img):
    g = preprocess(img)
    h, w = g.shape[:2]
    out = reader.recognize(g, horizontal_list=[[0, w, 0, h]], free_list=[], allowlist=ALLOW)
    if not out:
        return "", 0.0
    text = "".join(o[1] for o in out).replace(" ", "").upper()
    conf = sum(o[2] for o in out) / len(out)
    return text, float(conf)


def run(plates_json="outputs/plates_clip01.json", out_json="outputs/ocr_clip01.json"):
    import cv2
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)
    plates = json.load(open(plates_json))["plates"]
    results = {}
    for tid in sorted(plates, key=int):
        reads = []
        for d in plates[tid]:
            width = d["plate_bbox"][2] - d["plate_bbox"][0]
            if width < MIN_W or d["det_conf"] < MIN_DET:
                continue
            img = cv2.imread(d["crop"])
            if img is None:
                continue
            text, conf = read_crop(reader, img)
            reads.append({"frame": d["frame"], "t": d["t"], "text": text,
                          "conf": round(conf, 3), "crop": d["crop"]})
        fused = fuse_readings(reads)
        fused["reads"] = reads
        results[tid] = fused
        sample = [r["text"] for r in reads[:6]]
        print(f"track {tid:>3} {fused['status']:<10} {fused['text']:<10} conf={fused['confidence']:<6} "
              f"used={fused['n_used']}/{fused['n_total']}  reads={sample}")
    json.dump(results, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    run()