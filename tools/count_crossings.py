import json

TRACKS = "outputs/tracks_clip01.json"
LINE_Y = 1000  # must match LINE in draw_line.py

data = json.load(open(TRACKS))
crossed = 0
for tid, pts in data["tracks"].items():
    ys = [p["bbox"][3] for p in pts]
    first_t, last_t = pts[0]["t"], pts[-1]["t"]
    side = "CROSSES" if ys[0] > LINE_Y > ys[-1] else "no"
    crossed += side == "CROSSES"
    print(f"track {tid:>3}  frames={len(pts):>4}  t={first_t:>5}-{last_t:>5}  y: {ys[0]:.0f} -> {ys[-1]:.0f}  {side}")
print("tracks crossing line:", crossed, "of", len(data["tracks"]))