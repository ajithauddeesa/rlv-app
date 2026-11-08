import json


class Signal:
    def __init__(self, phases, clock_offset_s=0.0, uncertainty_s=0.25, approach="main"):
        self.phases = sorted(phases, key=lambda p: p["start"])
        self.clock_offset_s = clock_offset_s
        self.uncertainty_s = uncertainty_s
        self.approach = approach

    @classmethod
    def from_json(cls, path):
        d = json.load(open(path))
        return cls(d["phases"], d.get("clock_offset_s", 0.0),
                   d.get("transition_uncertainty_s", 0.25), d.get("approach", "main"))

    def lookup(self, video_t):
        st = video_t + self.clock_offset_s
        for p in self.phases:
            if p["start"] <= st < p["end"]:
                margin = min(st - p["start"], p["end"] - st)
                return {"state": p["state"], "signal_time": round(st, 3),
                        "margin_s": round(margin, 3),
                        "near_transition": margin < self.uncertainty_s}
        return {"state": "UNKNOWN", "signal_time": round(st, 3),
                "margin_s": None, "near_transition": False}