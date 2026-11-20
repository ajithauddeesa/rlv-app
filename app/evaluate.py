import csv
import json
from collections import Counter


def yes(v):
    return (v or "").strip().lower() in ("yes", "y", "1", "true")


def pct(a, b):
    return "n/a" if not b else f"{100 * a / b:.0f}% ({a}/{b})"


def score_plate(pred, truth):
    """truth may contain '?' for characters a human cannot read; those positions are skipped."""
    if pred in ("", "unreadable"):
        return {"outcome": "abstained", "committed": 0, "correct": 0}
    if len(pred) != len(truth):
        return {"outcome": "length_mismatch", "committed": len(pred.replace("?", "")), "correct": 0}
    committed = correct = 0
    for p, t in zip(pred, truth):
        if p == "?" or t == "?":
            continue
        committed += 1
        correct += int(p == t)
    if pred == truth:
        outcome = "exact"
    elif committed and correct == committed:
        outcome = "partial_all_correct"
    else:
        outcome = "has_wrong_chars"
    return {"outcome": outcome, "committed": committed, "correct": correct}


def evaluate(gt_path="data/ground_truth/gt_events.csv", ev_path="outputs/events.json"):
    ev = json.load(open(ev_path))
    events = {e["track_id"]: e for e in ev["events"]}
    rows = [r for r in csv.DictReader(open(gt_path, newline=""))
            if (r.get("is_motorcycle") or "").strip()]

    tp = fp = fn = tn = unc = 0
    sig_n = sig_ok = 0
    cls = Counter()
    plate_out = Counter()
    committed_total = correct_total = 0
    ill_total = ill_commit = 0
    m_yes = m_no = 0
    plate_detail = []

    for r in rows:
        tid = r["track_id"].strip()
        e = events.get(int(tid)) if tid.isdigit() else None
        truth_cross = yes(r["is_motorcycle"]) and yes(r["crossed"])
        pred_cross = bool(e and e["crossing_status"] == "crossed")
        if truth_cross and pred_cross:
            tp += 1
        elif truth_cross:
            fn += 1
            if e and e["crossing_status"] == "uncertain":
                unc += 1
        elif pred_cross:
            fp += 1
        else:
            tn += 1

        if truth_cross and pred_cross:
            ts = r["true_signal"].strip().upper()
            sig_n += 1
            sig_ok += int(e["signal_state"] == ts)
            truth_status = "suspected_violation" if ts == "RED" else "not_violation"
            cls[(truth_status, e["status"])] += 1

        truth_plate = (r.get("true_plate") or "").strip().upper()
        if e and truth_plate:
            pred = e["plate_text"]
            if truth_plate == "NONE":
                ill_total += 1
                ill_commit += int(pred not in ("", "unreadable"))
            else:
                s = score_plate(pred, truth_plate)
                plate_out[s["outcome"]] += 1
                committed_total += s["committed"]
                correct_total += s["correct"]
                plate_detail.append((tid, truth_plate, pred, s["outcome"]))

        pm = (r.get("plate_on_correct_bike") or "").strip().lower()
        m_yes += int(pm == "yes")
        m_no += int(pm == "no")

    print("=== Event detection (stop-line crossing) ===")
    print("labelled tracks:", len(rows), " TP", tp, " FP", fp, " FN", fn, " TN", tn,
          f" (of the FN, {unc} were flagged 'uncertain')")
    print("precision:", pct(tp, tp + fp), "  recall:", pct(tp, tp + fn))
    print("\n=== Signal phase at crossing ===")
    print("correct phase:", pct(sig_ok, sig_n))
    print("\n=== Event classification (correctly detected crossings) ===")
    for (t, p), n in sorted(cls.items()):
        print(f"  truth {t:<20} -> predicted {p:<22} x{n}")
    print("\n=== Plate OCR (tracks with a human-readable truth) ===")
    for tid, t, p, o in plate_detail:
        print(f"  track {tid:>3}: truth {t:<10} predicted {p:<10} {o}")
    print("  outcomes:", dict(plate_out))
    print("  committed characters correct:", pct(correct_total, committed_total))
    print("  plates humans could not read but the app committed text for:", pct(ill_commit, ill_total))
    print("\n=== Plate-to-motorcycle matching ===")
    print("matching errors:", pct(m_no, m_yes + m_no))
    ri = ev.get("run_info")
    if ri:
        print("\n=== Speed ===")
        print(ri["processing_fps"], "fps,", ri["seconds_per_video_second"], "s per video second on",
              ri["device"].upper(), "| stages (s):", ri["stage_seconds"])

    json.dump({"tp": tp, "fp": fp, "fn": fn, "tn": tn, "signal_correct": [sig_ok, sig_n],
               "classification": {f"{t}->{p}": n for (t, p), n in cls.items()},
               "plate_outcomes": dict(plate_out), "matching_errors": [m_no, m_yes + m_no],
               "run_info": ri}, open("outputs/evaluation.json", "w"), indent=2)


if __name__ == "__main__":
    evaluate()