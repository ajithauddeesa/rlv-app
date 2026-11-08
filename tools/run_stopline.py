import json
from app.stopline import StopLine, detect_crossing
from app.signal_phase import Signal

data = json.load(open("outputs/tracks_clip01.json"))
line = StopLine(0, 1000, 1230, 1000, approach_sign=1)
sig = Signal.from_json("data/signals/clip01_signal.json")

for tid, pts in data["tracks"].items():
    points = [(p["t"], (p["bbox"][0] + p["bbox"][2]) / 2, p["bbox"][3]) for p in pts]
    r = detect_crossing(points, line)
    if r["status"] == "crossed":
        ph = sig.lookup(r["time"])
        print(f"track {tid:>3}  CROSSED t={r['time']:>7}  {ph['state']:<7} margin={ph['margin_s']}  "
              f"near={ph['near_transition']}  conf={r['confidence']}")
    else:
        print(f"track {tid:>3}  {r['status']:<11} {r['reason']}")