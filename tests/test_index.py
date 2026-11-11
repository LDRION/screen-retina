from retina import index
from retina.geom import Grid, Rect
from retina.index import fold, similarity
from retina.perception import Perception, Win
from retina.uia import El


def el(role, name, x, y, w=80, h=24, value="", state=""):
    return El(role, name, value, state, False, Rect(x, y, w, h))


def make_per(targets, text=(), words=None, fg=1):
    p = Perception(Grid(3840, 1080, 32, -1920, 0), uia=None)
    p.fg = fg
    p.wins[1] = Win(1, "chrome.exe", "Mail", Rect(0, 0, 1920, 1080), list(targets), 0.0, list(targets))
    p.wins[2] = Win(2, "explorer.exe", "", Rect(0, 1040, 1920, 40), [], 0.0, [el("btn", "Пуск", 0, 1040)])
    p.text = list(text)
    p.words = words or {}
    return p


def test_fold_ignores_case_yo_and_punctuation():
    assert fold("Ещё раз!") == fold("еще  РАЗ") == "еще раз"


def test_similarity_orders_exact_prefix_contains_and_typos():
    assert similarity("Отправить", "отправить") == 1.0
    assert 0.85 < similarity("Отправить", "Отправить сейчас") < 1.0
    assert 0.75 < similarity("Карты", "Почта Карты Маркет") < 0.9
    assert similarity("кнопка Войти", "Войти") > 0.7
    assert similarity("Отправить", "Отправигь") > 0.7          # OCR misread
    assert similarity("Отправить", "Удалить") == 0.0
    assert similarity("в", "кнопка в меню") > 0 and similarity("Войти", "в") == 0.0


def test_find_prefers_buttons_over_text_and_reports_rect_centre():
    p = make_per([el("text", "Войти", 100, 100), el("btn", "Войти", 500, 300)])
    r = index.find(index.collect(p), "войти")
    assert r["matches"][0]["role"] == "btn"
    assert r["matches"][0]["center"] == [540, 312]


def test_duplicates_stay_separate_and_are_flagged_ambiguous():
    p = make_per([el("btn", "Удалить", 100, 100), el("btn", "Удалить", 100, 200)])
    r = index.find(index.collect(p), "Удалить")
    assert len(r["matches"]) == 2 and r["ambiguous"]


def test_link_and_its_inner_text_are_one_target():
    p = make_per([el("link", "Профиль", 100, 100), el("text", "Профиль", 104, 102, 70, 20)])
    r = index.find(index.collect(p), "Профиль")
    assert len(r["matches"]) == 1 and r["matches"][0]["role"] == "link"


def test_ocr_match_clicks_the_word_not_the_glued_line():
    line = (1, Rect(10, 10, 300, 20), "Почта Карты Маркет")
    words = {line: ((Rect(10, 10, 60, 20), "Почта"), (Rect(110, 10, 60, 20), "Карты"),
                    (Rect(210, 10, 100, 20), "Маркет"))}
    p = make_per([], text=[line], words=words)
    m = index.find(index.collect(p), "Карты")["matches"][0]
    assert m["source"] == "ocr" and m["rect"] == [110, 10, 60, 20]


def test_hidden_targets_are_dropped_and_scope_and_area_filter():
    p = make_per([el("btn", "Сохранить", 100, 100)])
    covered = index.collect(p, visible_fn=lambda x, y: 99)        # another window on top
    assert index.find(covered, "Сохранить")["matches"] == []
    assert index.find(covered, "Сохранить", hidden=True)["matches"]
    assert [t.name for t in index.collect(p, scope="fg")] == ["Сохранить"]
    assert [t.name for t in index.collect(p, area=Rect(0, 1000, 1920, 80))] == ["Пуск"]


def test_value_matches_fields_and_role_filter_applies():
    p = make_per([el("edit", "Поиск", 100, 100, value="screen retina"), el("btn", "Найти", 300, 100)])
    t = index.collect(p)
    assert index.find(t, "screen retina")["matches"][0]["role"] == "edit"
    assert index.find(t, "Поиск", role="btn")["matches"] == []


def test_settle_waits_for_a_fresh_frame_quiet_pixels_and_reread_windows():
    s = index.settle_state
    assert s(10.0, 10.5, t_step=9.9, t_changed=9.0, dirty=False, quiet=0.3, expect=False) == "wait"
    assert s(10.0, 10.5, t_step=10.4, t_changed=9.0, dirty=False, quiet=0.3, expect=False) == "settled"
    assert s(10.0, 10.5, t_step=10.4, t_changed=10.3, dirty=False, quiet=0.3, expect=False) == "wait"
    assert s(10.0, 10.5, t_step=10.4, t_changed=9.0, dirty=True, quiet=0.3, expect=False) == "wait"
    assert s(10.0, 10.5, t_step=10.4, t_changed=9.0, dirty=False, quiet=0.3, expect=True) == "wait"
    assert s(10.0, 10.9, t_step=10.8, t_changed=10.5, dirty=False, quiet=0.3, expect=True) == "settled"


def test_perception_keeps_ocr_word_boxes_and_two_letter_buttons(monkeypatch):
    import numpy as np
    from retina import perception as P

    class FakeOcr:
        def read(self, crop, band):
            return [(Rect(10, 10, 300, 20), "Почта Карты Маркет",
                     ((Rect(10, 10, 60, 20), "Почта"), (Rect(110, 10, 60, 20), "Карты"),
                      (Rect(210, 10, 100, 20), "Маркет"))),
                    (Rect(400, 10, 30, 20), "OK", ((Rect(400, 10, 30, 20), "OK"),))]

    p = Perception(Grid(1920, 1080, 32, 0, 0), uia=None, ocr=FakeOcr())
    p.wins[1] = Win(1, "app.exe", "App", Rect(0, 0, 1920, 1080), [], 0.0)
    monkeypatch.setattr(P.winapi, "root_window_at", lambda x, y: 1)
    p._read_text(np.zeros((1080, 1920, 4), np.uint8), Rect(0, 0, 1920, 64), 1, Rect(0, 0, 1920, 64), now=1.0)
    targets = index.collect(p)
    assert index.find(targets, "Карты")["matches"][0]["rect"] == [110, 10, 60, 20]
    assert index.find(targets, "OK")["matches"][0]["center"] == [415, 20]


def test_last_change_is_tracked_per_area():
    p = Perception(Grid(1920, 1080, 32, 0, 0), uia=None)
    changed = p.grid.empty()
    changed[0, 0] = True                                          # top-left corner moved at t=5
    p.t_cell[changed] = 5.0
    p.t_changed = 5.0
    assert p.last_change() == 5.0
    assert p.last_change(Rect(0, 0, 100, 100)) == 5.0
    assert p.last_change(Rect(1000, 500, 300, 300)) == 0.0        # a quiet window elsewhere
    assert p.last_change(Rect(5000, 0, 10, 10)) == 0.0            # off-screen


def test_ocr_splits_glued_labels_and_penalises_prose():
    from retina.index import segments
    menu = ((Rect(10, 10, 50, 20), "Почта"), (Rect(120, 10, 50, 20), "Карты"), (Rect(230, 10, 60, 20), "Маркет"))
    assert [len(s) for s in segments(menu)] == [1, 1, 1]
    words = "Нашёл опасный случай клик по Search прямо в логе".split()
    prose = tuple((Rect(10 + 60 * i, 50, 50, 20), w) for i, w in enumerate(words))
    label = ((Rect(400, 300, 70, 24), "Search"),)
    p = make_per([], text=[(1, Rect(10, 50, 540, 20), " ".join(words)), (1, Rect(400, 300, 70, 24), "Search")],
                 words={(1, Rect(10, 50, 540, 20), " ".join(words)): prose,
                        (1, Rect(400, 300, 70, 24), "Search"): label})
    ms = index.find(index.collect(p), "Search")["matches"]
    assert ms[0]["rect"] == [400, 300, 70, 24]                    # отдельная подпись
    assert all(m["score"] < 0.8 for m in ms[1:])                  # слово в предложении — слабое


def test_exclude_skips_the_agents_own_windows():
    p = make_per([el("btn", "Отправить", 100, 100)])
    p.wins[3] = Win(3, "python.exe", "AI Vision — Лаунчер", Rect(0, 0, 900, 900), [], 0.0,
                    [el("text", "Отправить", 50, 50)])
    names = [(t.process, t.name) for t in index.collect(p, exclude=["AI Vision"])]
    assert ("python.exe", "Отправить") not in names and ("chrome.exe", "Отправить") in names


class _FakeUia:
    """UI Automation element as UiaReader._element sees it (cached properties only)."""
    def __init__(self, ctype, name="", help_text=None, value=None):
        from types import SimpleNamespace
        self.CachedControlType, self.CachedName = ctype, name
        self.CachedHasKeyboardFocus = False
        self.CachedBoundingRectangle = SimpleNamespace(left=10, top=20, right=42, bottom=52)
        self._props = {30013: help_text, 30045: value}

    def GetCachedPropertyValue(self, pid):
        return self._props.get(pid)


def test_icon_buttons_stay_click_targets_and_take_their_tooltip_as_name():
    from retina.uia import UiaReader
    read = lambda e: UiaReader._element(None, e)
    assert read(_FakeUia(50000, "", help_text="Настройки")).name == "Настройки"     # имя из подсказки
    icon = read(_FakeUia(50000, ""))                                           # глиф шрифта-иконок
    assert icon is not None and icon.name == "" and icon.rect == Rect(10, 20, 32, 32)
    assert read(_FakeUia(50020, "")) is None                                         # пустой текст — не цель
    assert read(_FakeUia(50000, "Сохранить", help_text="Ctrl+S")).name == "Сохранить"


def test_runtime_exclude_drops_windows_by_title_prefix(monkeypatch):
    from retina import perception as P
    monkeypatch.setattr(P.winapi, "find_window", lambda t: 0)
    p = make_per([el("btn", "Отправить", 100, 100)])
    p.wins[3] = Win(3, "python.exe", "AI Vision — Лаунчер", Rect(0, 0, 900, 900), [], 0.0,
                    [el("text", "клик «Отправить»", 50, 50)])
    p.wins[4] = Win(4, "Code.exe", "notes about AI Vision - VS Code", Rect(0, 0, 900, 900), [], 0.0,
                    [el("btn", "Run", 50, 50)])
    p.text = [(3, Rect(10, 10, 200, 20), "лог лаунчера")]
    p.exclude(["AI Vision", "AI"])                                  # слишком короткий префикс — игнор
    assert p.exclude_prefixes == ["AI Vision"]
    assert 3 not in p.wins and 3 in p.own and p.text == []
    assert 4 in p.wins                                               # только начало заголовка
    assert {t.process for t in index.collect(p.view)} == {"chrome.exe", "explorer.exe", "Code.exe"}


def test_events_since_and_exclude_over_http(monkeypatch):
    import json
    import urllib.request
    from collections import deque
    from retina import perception as P
    from retina.server import serve
    monkeypatch.setattr(P.winapi, "find_window", lambda t: 0)
    p = make_per([el("btn", "OK", 10, 10)])
    journal = deque([(1, "window", "window: chrome.exe «Почта»"), (2, "ui", "+ btn «Отправить»"),
                     (3, "text", "+ «Сообщение отправлено»")])
    serve(p, None, None, None, 18765, deque(["[12:00:00] old"]), lambda: {}, journal)
    get = lambda path, data=None: json.loads(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:18765" + path, data=data, headers={"Content-Type": "application/json"}),
        timeout=5).read().decode("utf-8"))
    assert get("/events") == {"events": ["[12:00:00] old"]}                  # старый формат цел
    r = get("/events?since=1")
    assert [e["text"] for e in r["events"]] == ["+ btn «Отправить»", "+ «Сообщение отправлено»"] and r["last"] == 3
    assert get("/events?since=3")["events"] == []
    assert get("/exclude", json.dumps({"prefixes": ["AI Vision"]}).encode())["prefixes"] == ["AI Vision"]
