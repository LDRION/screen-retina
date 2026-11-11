"""Layer 0b: structure. Reads what the computer already knows via UI Automation.

One cross-process call (FindAllBuildCache) fetches every visible element of a window
with the properties we need, which keeps a snapshot in the tens of milliseconds.
"""
import unicodedata
from dataclasses import dataclass, replace

import comtypes
import comtypes.client

from .geom import Rect

comtypes.client.GetModule("UIAutomationCore.dll")
from comtypes.gen import UIAutomationClient as UIA  # noqa: E402

ROLES = {
    50000: "btn", 50001: "calendar", 50002: "check", 50003: "combo", 50004: "edit",
    50005: "link", 50006: "img", 50007: "item", 50008: "list", 50009: "menu",
    50010: "menubar", 50011: "menuitem", 50012: "progress", 50013: "radio",
    50014: "scrollbar", 50015: "slider", 50016: "spinner", 50017: "status", 50018: "tabs",
    50019: "tab", 50020: "text", 50021: "toolbar", 50022: "tooltip", 50023: "tree",
    50024: "treeitem", 50025: "custom", 50026: "group", 50027: "thumb", 50028: "grid",
    50029: "cell", 50030: "doc", 50031: "splitbtn", 50032: "window", 50033: "pane",
    50034: "header", 50035: "headeritem", 50036: "table", 50037: "titlebar",
    50038: "separator", 50039: "zoom", 50040: "appbar",
}
SKIP_ROLES = {"scrollbar", "thumb", "separator", "tooltip", "titlebar"}
# Roles whose on-screen area is text we already have, so OCR can skip it.
TEXT_ROLES = {"btn", "check", "combo", "edit", "link", "item", "menuitem", "radio", "status",
              "tab", "text", "treeitem", "cell", "headeritem", "splitbtn"}
RANGE_ROLES = {"slider", "progress", "spinner"}
# Roles a person clicks: kept as click targets even without a label (an icon button),
# so a caller can name them by tooltip or by look.
CLICK_ROLES = {"btn", "splitbtn", "link", "menuitem", "tab", "check", "radio", "item", "treeitem"}

P_VALUE = 30045           # Value.Value
P_RANGE = 30047           # RangeValue.Value
P_SELECTED = 30079        # SelectionItem.IsSelected
P_TOGGLE = 30086          # Toggle.ToggleState
P_PASSWORD = 30019        # IsPassword
P_HELP = 30013            # HelpText: usually the tooltip of an icon button


@dataclass(frozen=True)
class El:
    role: str
    name: str
    value: str
    state: str
    focused: bool
    rect: Rect

    @property
    def key(self) -> tuple:
        """Identity for diffing; position and focus are deliberately left out."""
        return self.role, self.name, self.value, self.state

    def render(self, limit: int = 100) -> str:
        s = f"{self.role} «{clip(self.name, limit)}»" if self.name else self.role
        if self.value:
            # URLs are long and mostly noise past the path start.
            s += f" = «{clip(self.value, 60 if '://' in self.value else limit)}»"
        if self.state:
            s += f" [{self.state}]"
        return s


def clean(s: str) -> str:
    """Collapse whitespace and drop invisible format characters (LRM marks and the like)."""
    return " ".join("".join(ch for ch in s if unicodedata.category(ch) != "Cf").split())


def clip(s: str, n: int) -> str:
    s = clean(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _plain(v):
    return v if isinstance(v, (str, int, float, bool)) else None


class UiaReader:
    def __init__(self, max_elements: int = 600):
        self.uia = comtypes.client.CreateObject(UIA.CUIAutomation, interface=UIA.IUIAutomation)
        self.max_elements = max_elements
        self.cache = self.uia.CreateCacheRequest()
        for pid in (UIA.UIA_NamePropertyId, UIA.UIA_ControlTypePropertyId,
                    UIA.UIA_BoundingRectanglePropertyId, UIA.UIA_HasKeyboardFocusPropertyId,
                    P_VALUE, P_RANGE, P_SELECTED, P_TOGGLE, P_PASSWORD, P_HELP):
            self.cache.AddProperty(pid)
        self.cond = self.uia.CreateAndCondition(
            self.uia.ControlViewCondition,
            self.uia.CreatePropertyCondition(UIA.UIA_IsOffscreenPropertyId, False))

    def snapshot(self, hwnd: int, window_title: str = "") -> list[El]:
        return self.read(hwnd, window_title)[0]

    def read(self, hwnd: int, window_title: str = "") -> tuple[list[El], list[El]]:
        """(elements for the event log, click targets).

        The log wants each label once; clicking needs every copy with its own rect
        (three «Delete» buttons are three targets), so targets skip the dedupe.
        """
        try:
            root = self.uia.ElementFromHandle(hwnd)
            arr = root.FindAllBuildCache(UIA.TreeScope_Descendants, self.cond, self.cache)
        except comtypes.COMError:
            return [], []
        out: list[El] = []
        targets: list[El] = []
        seen: set[tuple] = set()
        values: set[str] = set()
        recent: list[str] = []
        for i in range(min(arr.Length, self.max_elements)):
            try:
                el = self._element(arr.GetElement(i))
            except comtypes.COMError:
                continue
            if el is None:
                continue
            if el.rect.area and el.role not in ("pane", "window", "doc", "group", "custom"):
                targets.append(el)
            # Panes and documents often repeat the window title; the title is already known.
            if len(el.name) >= 10 and el.name in window_title:
                el = replace(el, name="")
            if not el.name and (not el.value or el.value in values):
                continue
            # Same role and label twice (e.g. a link and its inner anchor): keep the first.
            if el.key in seen or (el.name and (el.role, el.name) in seen):
                continue
            # A text or image repeating the label next to it adds nothing.
            if el.role in ("text", "img") and not el.value and el.name in recent:
                continue
            seen.add(el.key)
            seen.add((el.role, el.name))
            if el.value:
                values.add(el.value)
            recent = (recent + [el.name])[-4:]
            out.append(el)
        return out, targets

    def _element(self, e) -> El | None:
        role = ROLES.get(e.CachedControlType, "?")
        if role in SKIP_ROLES:
            return None
        name = clean(e.CachedName or "")
        if not any(ch.isalnum() for ch in name):   # icon-font glyphs and punctuation only
            name = ""
        if not name:
            # An icon button without a name often carries its tooltip as help text.
            hlp = _plain(e.GetCachedPropertyValue(P_HELP))
            hlp = clean(hlp) if isinstance(hlp, str) else ""
            if any(ch.isalnum() for ch in hlp) and len(hlp) <= 80:
                name = hlp
        value = ""
        if not e.GetCachedPropertyValue(P_PASSWORD):
            v = _plain(e.GetCachedPropertyValue(P_VALUE))
            # Chrome reports RangeValue 0 on everything, so only trust it where a range exists.
            if (v is None or v == "") and role in RANGE_ROLES:
                v = _plain(e.GetCachedPropertyValue(P_RANGE))
                if isinstance(v, float):
                    v = f"{v:g}"
            value = clean(str(v)) if v not in (None, "") else ""
        else:
            value = "***"
        if value == name:
            value = ""
        if not name and not value and role not in CLICK_ROLES:
            return None
        state = ""
        if _plain(e.GetCachedPropertyValue(P_SELECTED)) is True:
            state = "selected"
        toggle = _plain(e.GetCachedPropertyValue(P_TOGGLE))
        if toggle == 1:
            state = "on"
        elif toggle == 0 and role in ("check", "radio"):
            state = "off"
        r = e.CachedBoundingRectangle
        rect = Rect(r.left, r.top, max(r.right - r.left, 0), max(r.bottom - r.top, 0))
        return El(role, name, value, state, bool(e.CachedHasKeyboardFocus), rect)
