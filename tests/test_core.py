import numpy as np

from retina.geom import Grid, Rect
from retina.perception import Perception, grows


def make(**kw) -> Perception:
    return Perception(Grid(3840, 1080, 32, -1920, 0), uia=None, **kw)


def test_components_are_in_screen_coordinates():
    g = Grid(3840, 1080, 32, -1920, 0)
    m = g.empty()
    m[0, 0] = m[0, 1] = True          # top-left of the left monitor
    m[10, 60] = True                  # first column of the main monitor
    rects = sorted(g.components(m), key=lambda r: r.x)
    assert rects == [Rect(-1920, 0, 64, 32), Rect(0, 320, 32, 32)]


def test_crop_clips_to_screen_and_maps_origin():
    g = Grid(3840, 1080, 32, -1920, 0)
    frame = np.zeros((1080, 3840, 4), np.uint8)
    frame[0, 1920] = 255              # pixel at screen x=0
    crop, r = g.crop(frame, Rect(-8, -8, 16, 16))
    assert r == Rect(-8, 0, 16, 8)
    assert crop.shape[:2] == (8, 16) and crop[0, 8, 0] == 255
    assert g.crop(frame, Rect(5000, 0, 10, 10))[1] is None


def test_cell_lookup_round_trips():
    g = Grid(3840, 1080, 32, -1920, 0)
    assert g.cell_at(*g.cell_center(3, 70)) == (3, 70)
    assert g.cell_at(-1921, 0) is None


def test_grows_means_typing_or_deleting_at_the_end():
    assert grows("Hel", "Hello") and grows("Hello", "Hel")
    assert not grows("36%", "37%") and not grows("", "a") and not grows("a", "a")


def test_typing_is_throttled_and_sends_only_the_added_part():
    p = make(typing_interval=1.0)
    ident = ("val", 1, "edit", "Message")
    assert p._typed(ident, "edit", "", "H", "", now=10.0) == "~ edit = «H»"
    assert p._typed(ident, "edit", "H", "He", "", now=10.3) is None        # throttled
    assert p._typed(ident, "edit", "He", "Hello", "", now=10.6) is None
    assert p._flush_typed(now=10.9) == []                                   # still inside the interval
    flushed = p._flush_typed(now=11.1)
    assert [line for _, line in flushed] == ["~ edit += «ello»"]
    assert flushed[0][0] == 10.3                                            # latency counts from first change


def test_tickers_get_muted_after_repeated_changes():
    p = make(ticker_hits=3, ticker_window=10.0)
    ident = ("clock",)
    assert [p._tick(ident, t) for t in (1.0, 2.0, 3.0, 4.0)] == ["no", "no", "new", "muted"]
    assert p._tick(ident, 20.0) == "no"                                     # quiet for long: unmuted


def test_monitor_names_follow_position():
    mons = [{"left": 0, "top": 0, "width": 1920, "height": 1080, "is_primary": True},
            {"left": -1920, "top": 0, "width": 1920, "height": 1080}]
    p = make(monitors=mons)
    assert p.where(Rect(-1900, 10, 800, 600)) == " [left monitor]"
    assert p.where(Rect(-8, -8, 1936, 1048)) == " [main monitor]"
    assert p.where(Rect(-32000, -32000, 160, 28)) == " [off-screen]"


def test_focus_flicker_back_and_forth_costs_nothing():
    p = make(typing_interval=1.0)
    ident = ("focus", 1)
    assert p._settle(ident, "focus → edit «Message»", "", now=1.0) == "focus → edit «Message»"
    assert p._settle(ident, "focus → btn «Send»", "", now=1.2) is None
    assert p._settle(ident, "focus → edit «Message»", "", now=1.4) is None
    assert p._flush_typed(now=2.5) == []                       # ended where it was reported
    assert p._settle(ident, "focus → btn «Send»", "", now=3.0) == "focus → btn «Send»"
