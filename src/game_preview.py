"""Windows fullscreen presentation that leaves focus and input with the game."""
import logging
import win32api
import win32con
import win32gui

logger = logging.getLogger(__name__)


class GamePreview:
    def __init__(self, title, target_hwnd, preview_hwnd=None):
        self.target = target_hwnd
        self.hwnd = preview_hwnd or win32gui.FindWindow(None, title)
        if not self.hwnd:
            raise RuntimeError("Preview window was not created.")
        style = win32gui.GetWindowLong(self.hwnd, win32con.GWL_EXSTYLE)
        win32gui.SetWindowLong(self.hwnd, win32con.GWL_EXSTYLE,
                              style | win32con.WS_EX_LAYERED |
                              win32con.WS_EX_TRANSPARENT | win32con.WS_EX_NOACTIVATE)
        win32gui.SetLayeredWindowAttributes(self.hwnd, 0, 255, win32con.LWA_ALPHA)
        # The preview must never become its own capture source.
        self.visible = None
        try:
            win32gui.SetForegroundWindow(self.target)
        except Exception:
            logger.info("Select the game to show the fullscreen preview.")
        self.update()

    def update(self):
        if not win32gui.IsWindow(self.target):
            return False
        active = (win32gui.GetForegroundWindow() == self.target and
                  not win32gui.IsIconic(self.target))
        if active != self.visible:
            if active:
                monitor = win32api.MonitorFromWindow(self.target, win32con.MONITOR_DEFAULTTONEAREST)
                left, top, right, bottom = win32api.GetMonitorInfo(monitor)["Monitor"]
                win32gui.SetWindowPos(self.hwnd, win32con.HWND_TOPMOST,
                                      left, top, right-left, bottom-top,
                                      win32con.SWP_NOACTIVATE | win32con.SWP_SHOWWINDOW)
            else:
                win32gui.ShowWindow(self.hwnd, win32con.SW_HIDE)
            self.visible = active
        return True

    @staticmethod
    def stop_requested():
        return all(win32api.GetAsyncKeyState(key) & 0x8000
                   for key in (win32con.VK_CONTROL, win32con.VK_MENU, ord("Q")))
