import math
from dataclasses import dataclass


@dataclass
class StopLine:
    x1: float
    y1: float
    x2: float
    y2: float
    approach_sign: int = 1  # sign of the signed distance on the approach side

    def length(self):
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    def signed_distance(self, x, y):
        return ((self.x2 - self.x1) * (y - self.y1)
                - (self.y2 - self.y1) * (x - self.x1)) / self.length()

    def along(self, x, y):
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        return ((x - self.x1) * dx + (y - self.y1) * dy) / (dx * dx + dy * dy)


def _result(status, reason, time=None, confidence=0.0):
    return {"status": status, "reason": reason, "time": time, "confidence": confidence}


def detect_crossing(points, line, dead_band=4.0, min_frames=3,
                    near_px=25.0, max_gap_s=0.5):
    """points: list of (t, x, y), where (x, y) is the bike's bottom-centre.
    Returns status: crossed | not_crossed | uncertain."""
    if len(points) < 2:
        return _result("not_crossed", "too_short")

    # s > 0 means the approach ("before") side, s < 0 the "after" side
    s = [line.approach_sign * line.signed_distance(x, y) for _, x, y in points]
    side = [0 if abs(v) < dead_band else (1 if v > 0 else -1) for v in s]
    seq = [v for v in side if v != 0]
    if not seq:
        return _result("uncertain", "lost_near_line")

    flips = sum(1 for a, b in zip(seq, seq[1:]) if a != b)

    if flips == 0:
        if seq[0] == -1:
            return _result("not_crossed", "started_after_line")
        if s[-1] < near_px:
            return _result("uncertain", "lost_near_line")
        return _result("not_crossed", "never_reached_line")

    if flips > 1:
        return _result("uncertain", "jitter_at_line")

    if seq[0] == -1:
        return _result("not_crossed", "wrong_direction")

    i0 = max(i for i, v in enumerate(side) if v == 1)
    i1 = min(i for i, v in enumerate(side) if v == -1)
    n_before = side.count(1)
    n_after = side.count(-1)
    if n_before < min_frames or n_after < min_frames:
        return _result("uncertain", "too_few_frames")

    t0, x0, y0 = points[i0]
    t1, x1, y1 = points[i1]
    if t1 - t0 > max_gap_s:
        return _result("uncertain", "gap_at_crossing")

    frac = s[i0] / (s[i0] - s[i1])
    t_cross = t0 + frac * (t1 - t0)
    xc = x0 + frac * (x1 - x0)
    yc = y0 + frac * (y1 - y0)
    if not (-0.02 <= line.along(xc, yc) <= 1.02):
        return _result("not_crossed", "outside_line_span")

    conf = round(min(1.0, min(n_before, n_after) / 8), 2)
    return _result("crossed", "ok", round(t_cross, 3), conf)