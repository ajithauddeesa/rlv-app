import pytest
from app.stopline import StopLine, detect_crossing

LINE = StopLine(0, 1000, 1230, 1000, approach_sign=1)


def ramp(a, b, n):
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def track(ys, x=500, fps=30):
    return [(i / fps, x, y) for i, y in enumerate(ys)]


def test_clean_crossing_time_interpolated():
    r = detect_crossing(track(ramp(1080, 900, 40)), LINE)
    assert r["status"] == "crossed"
    assert r["time"] == pytest.approx(17.333 / 30, abs=0.01)


def test_wrong_direction():
    r = detect_crossing(track(ramp(900, 1080, 40)), LINE)
    assert (r["status"], r["reason"]) == ("not_crossed", "wrong_direction")


def test_never_reaches_line():
    r = detect_crossing(track(ramp(1080, 1050, 30)), LINE)
    assert (r["status"], r["reason"]) == ("not_crossed", "never_reached_line")


def test_lost_near_line_is_uncertain():
    r = detect_crossing(track(ramp(1080, 1005, 30)), LINE)
    assert (r["status"], r["reason"]) == ("uncertain", "lost_near_line")


def test_jitter_is_uncertain():
    ys = ramp(1080, 1010, 20) + [995, 1010, 995, 1010, 995] + ramp(990, 900, 20)
    r = detect_crossing(track(ys), LINE)
    assert (r["status"], r["reason"]) == ("uncertain", "jitter_at_line")


def test_too_few_frames_before_line():
    ys = [1020, 1010] + ramp(990, 900, 10)
    r = detect_crossing(track(ys), LINE)
    assert (r["status"], r["reason"]) == ("uncertain", "too_few_frames")


def test_outside_line_span():
    r = detect_crossing(track(ramp(1080, 900, 40), x=1500), LINE)
    assert (r["status"], r["reason"]) == ("not_crossed", "outside_line_span")


def test_time_gap_at_crossing_is_uncertain():
    before = [(i / 30, 500, 1050 - i) for i in range(5)]
    after = [(1.5 + i / 30, 500, 950 - i) for i in range(5)]
    r = detect_crossing(before + after, LINE)
    assert (r["status"], r["reason"]) == ("uncertain", "gap_at_crossing")