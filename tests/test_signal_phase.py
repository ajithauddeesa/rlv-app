from app.signal_phase import Signal

PHASES = [
    {"start": 0.0, "end": 26.25, "state": "GREEN"},
    {"start": 26.25, "end": 29.25, "state": "YELLOW"},
    {"start": 29.25, "end": 35.0, "state": "RED"},
]


def test_basic_states():
    s = Signal(PHASES)
    assert s.lookup(10.0)["state"] == "GREEN"
    assert s.lookup(27.5)["state"] == "YELLOW"
    assert s.lookup(31.0)["state"] == "RED"
    assert not s.lookup(10.0)["near_transition"]


def test_near_transition_flagged():
    assert Signal(PHASES).lookup(26.3)["near_transition"]


def test_clock_offset():
    assert Signal(PHASES, clock_offset_s=2.0).lookup(24.5)["state"] == "YELLOW"


def test_gap_and_outside_coverage_unknown():
    gappy = [{"start": 0, "end": 10, "state": "GREEN"},
             {"start": 20, "end": 30, "state": "RED"}]
    assert Signal(gappy).lookup(15)["state"] == "UNKNOWN"
    assert Signal(PHASES).lookup(40)["state"] == "UNKNOWN"