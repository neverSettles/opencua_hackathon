"""Kernel-backed computer adapter for tzafon's CuaRunner.

The vendored `cua_runner.py` calls a small surface on the `computer` argument:
  - screenshot() / get_screenshot_url(result)
  - click(x,y) / double_click(x,y) / right_click(x,y) / drag(x1,y1,x2,y2)
  - type(text) / hotkey(keys) / key_down(key) / key_up(key)
  - scroll(dx, dy, x, y) (keyword args)
  - navigate(url)
  - wait(seconds)
  - id (attribute)
  - context manager for cleanup

This adapter implements that surface against a Kernel browser session, plus a
Playwright-over-CDP page reference for the only thing Kernel doesn't natively
support: in-tab navigation by URL.

Coordinates passed in are already pixel-space (CuaRunner denormalizes 0-999 to
display_width/display_height before calling these methods).
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any

from PIL import Image

try:  # Playwright is optional - only needed for `navigate(url)`
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None  # type: ignore[assignment]


class KernelComputerAdapter:
    """Wraps a Kernel browser session in the interface CuaRunner expects."""

    def __init__(
        self,
        kernel: Any,
        *,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        stealth: bool = True,
        screenshot_dir: Path | None = None,
        verbose: bool = True,
    ) -> None:
        self._kernel = kernel
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.stealth = stealth
        self._screenshot_dir = screenshot_dir
        self._verbose = verbose

        self._browser: Any | None = None
        self._pw_ctx = None
        self._pw_chromium = None
        self._page: Any = None
        self._screenshot_counter = 0

    # ------------------------------------------------------------------ id
    @property
    def id(self) -> str | None:
        return self._browser.session_id if self._browser is not None else None

    @property
    def live_view_url(self) -> str | None:
        return getattr(self._browser, "browser_live_view_url", None) if self._browser else None

    # ------------------------------------------------------------------ ctx
    def __enter__(self) -> "KernelComputerAdapter":
        self._browser = self._kernel.browsers.create(
            stealth=self.stealth,
            headless=False,
            timeout_seconds=1500,
            viewport={"width": self.viewport_width, "height": self.viewport_height},
        )
        if self._verbose:
            print(f"[kernel] session_id={self._browser.session_id}")
            if self._browser.browser_live_view_url:
                print(f"[kernel] live_view={self._browser.browser_live_view_url}")

        if sync_playwright is not None:
            self._pw_ctx = sync_playwright().start()
            try:
                self._pw_chromium = self._pw_ctx.chromium.connect_over_cdp(self._browser.cdp_ws_url)
                contexts = self._pw_chromium.contexts
                ctx = contexts[0] if contexts else self._pw_chromium.new_context()
                self._page = ctx.pages[0] if ctx.pages else ctx.new_page()
            except Exception as e:
                if self._verbose:
                    print(f"[kernel] (warn) Playwright CDP attach failed: {e}")
                self._page = None
        return self

    def __exit__(self, *exc) -> None:
        try:
            if self._pw_chromium is not None:
                self._pw_chromium.close()
        except Exception:
            pass
        try:
            if self._pw_ctx is not None:
                self._pw_ctx.stop()
        except Exception:
            pass
        try:
            if self._browser is not None:
                self._kernel.browsers.delete_by_id(self._browser.session_id)
        except Exception as e:
            if self._verbose:
                print(f"[kernel] (cleanup) browser delete error: {e}")

    # --------------------------------------------------------- screenshots
    def screenshot(self) -> bytes:
        self._ensure()
        resp = self._kernel.browsers.computer.capture_screenshot(self._browser.session_id)
        raw = resp.read() if hasattr(resp, "read") else bytes(resp)
        return raw

    def get_screenshot_url(self, screenshot: bytes) -> str:
        img = Image.open(io.BytesIO(screenshot))
        if img.size != (self.viewport_width, self.viewport_height):
            img = img.resize((self.viewport_width, self.viewport_height), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()
        if self._screenshot_dir is not None:
            self._screenshot_counter += 1
            (self._screenshot_dir / f"step_{self._screenshot_counter:03d}.png").write_bytes(png)
        return f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}"

    # --------------------------------------------------------------- mouse
    def click(self, x: int, y: int) -> None:
        self._ensure()
        self._kernel.browsers.computer.click_mouse(self._browser.session_id, x=x, y=y)

    def double_click(self, x: int, y: int) -> None:
        self._ensure()
        self._kernel.browsers.computer.click_mouse(
            self._browser.session_id, x=x, y=y, num_clicks=2
        )

    def right_click(self, x: int, y: int) -> None:
        self._ensure()
        self._kernel.browsers.computer.click_mouse(
            self._browser.session_id, x=x, y=y, button="right"
        )

    def drag(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self._ensure()
        self._kernel.browsers.computer.drag_mouse(
            self._browser.session_id, path=[{"x": x1, "y": y1}, {"x": x2, "y": y2}]
        )

    def scroll(self, *, dx: int = 0, dy: int = 0, x: int = 640, y: int = 360) -> None:
        self._ensure()
        self._kernel.browsers.computer.scroll(
            self._browser.session_id, x=x, y=y, delta_x=dx, delta_y=dy
        )

    # ----------------------------------------------------------- keyboard
    def type(self, text: str) -> None:  # noqa: A003 - mirrors upstream API
        self._ensure()
        self._kernel.browsers.computer.type_text(self._browser.session_id, text=text)

    def hotkey(self, keys) -> None:
        self._ensure()
        keys_list = list(keys) if not isinstance(keys, str) else [keys]
        self._kernel.browsers.computer.press_key(self._browser.session_id, keys=keys_list)

    def key_down(self, key: str) -> None:
        self._ensure()
        self._kernel.browsers.computer.press_key(self._browser.session_id, keys=[key])

    def key_up(self, key: str) -> None:
        # Kernel's API doesn't separate down/up; press_key handles full press.
        # If we already emitted a key_down via press_key, emit a no-op here.
        return None

    # --------------------------------------------------------------- nav
    def navigate(self, url: str) -> None:
        if self._page is None:
            raise RuntimeError(
                "navigate() requires Playwright over CDP, which failed to attach. "
                "Have the model use clicks + URL-bar typing instead."
            )
        self._page.goto(url, wait_until="domcontentloaded", timeout=60_000)

    # --------------------------------------------------------------- misc
    def wait(self, seconds: float) -> None:
        time.sleep(seconds)

    # ------------------------------------------------------------ private
    def _ensure(self) -> None:
        if self._browser is None:
            raise RuntimeError("KernelComputerAdapter used outside of `with` block.")
