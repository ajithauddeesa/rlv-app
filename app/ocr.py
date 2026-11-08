import json
from collections import Counter, defaultdict

MIN_W = 24      # skip plate crops narrower than this (pixels)
MIN_DET = 0.30  # skip weak plate detections


def fuse_readings(readings, min_reads=3, min_conf=0.4, agree=0.6):
    """Combine per-frame readings [{'text','conf','char_probs'?}, ...] into one plate string.
    Never guesses: disagreeing positions become '?', too little evidence becomes 'unreadable'."""
    n_total = len(readings)
    usable = [r for r in readings
              if r.get("text") and len(r["text"]) >= 4 and r["conf"] >= min_conf]

    def unreadable():
        return {"text": "unreadable", "status": "unreadable", "confidence": 0.0,
                "n_used": len(usable), "n_total": n_total}

    if len(usable) < min_reads:
        return unreadable()

    length_weight = Counter()
    for r in usable:
        length_weight[len(r["text"])] += r["conf"]
    L = length_weight.most_common(1)[0][0]
    same = [r for r in usable if len(r["text"]) == L]
    if len(same) < min_reads:
        return unreadable()

    chars, shares = [], []
    for i in range(L):
        votes = defaultdict(float)
        for r in same:
            probs = r.get("char_probs")
            w = probs[i] if probs is not None and i < len(probs) else r["conf"]
            votes[r["text"][i]] += w
        ch, w = max(votes.items(), key=lambda kv: kv[1])
        share = w / sum(votes.values())
        shares.append(share)
        chars.append(ch if share >= agree else "?")

    text = "".join(chars)
    mean_conf = sum(r["conf"] for r in same) / len(same)
    return {"text": text, "status": "conflict" if "?" in text else "ok",
            "confidence": round(mean_conf * sum(shares) / L, 3),
            "n_used": len(same), "n_total": n_total}


def run(plates_json="outputs/plates_clip01.json", out_json="outputs/ocr_clip01.json"):
    import cv2
    from fast_plate_ocr import LicensePlateRecognizer
    model = LicensePlateRecognizer("cct-s-v2-global-model")
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
            pred = model.run(img, return_confidence=True)[0]
            text = pred.plate.replace("_", "").strip()
            probs = [float(p) for p in pred.char_probs][:len(text)]
            conf = sum(probs) / len(probs) if probs else 0.0
            reads.append({"frame": d["frame"], "t": d["t"], "text": text,
                          "conf": round(conf, 3), "char_probs": [round(p, 3) for p in probs],
                          "crop": d["crop"]})
        fused = fuse_readings(reads)
        fused["reads"] = reads
        results[tid] = fused
        print(f"track {tid:>3} {fused['status']:<10} {fused['text']:<10} conf={fused['confidence']:<6} "
              f"used={fused['n_used']}/{fused['n_total']}  reads={[r['text'] for r in reads[:6]]}")
    json.dump(results, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    run()