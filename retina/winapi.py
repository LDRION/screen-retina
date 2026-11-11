"""Small Win32 helpers: DPI awareness, foreground window, titles, process names."""
import ctypes
import os
from ctypes import wintypes

from .geom import Rect

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

user32.GetForegroundWindow.restype = wintypes.HWND
user32.WindowFromPoint.argtypes = [wintypes.POINT]
user32.WindowFromPoint.restype = wintypes.HWND
user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
user32.GetAncestor.restype = wintypes.HWND
kernel32.GetConsoleWindow.restype = wintypes.HWND
user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
user32.FindWindowW.restype = wintypes.HWND

GA_ROOT = 2
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def set_dpi_aware() -> None:
    """Make capture and UI Automation agree on physical pixel coordinates."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except OSError:
        user32.SetProcessDPIAware()


def foreground() -> int:
    return user32.GetForegroundWindow() or 0


def console_window() -> int:
    return kernel32.GetConsoleWindow() or 0


def find_window(window_title: str) -> int:
    return user32.FindWindowW(None, window_title) or 0


def root_window_at(x: int, y: int) -> int:
    h = user32.WindowFromPoint(wintypes.POINT(x, y))
    return (user32.GetAncestor(h, GA_ROOT) or 0) if h else 0


def title(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(wintypes.HWND(hwnd), buf, 512)
    return buf.value


def rect(hwnd: int) -> Rect:
    r = wintypes.RECT()
    user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(r))
    return Rect(r.left, r.top, r.right - r.left, r.bottom - r.top)


def process_name(hwnd: int) -> str:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not h:
        return "?"
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return os.path.basename(buf.value)
        return "?"
    finally:
        kernel32.CloseHandle(h)
