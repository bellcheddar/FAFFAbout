#!/usr/bin/env python
"""screenshot.py: capture the running app for the README, and prove the capture is real.

Drives Chrome over the DevTools protocol in a REAL-TIME session rather than with
--virtual-time-budget, which pauses requestAnimationFrame and tends to fire the capture
before an in-flight fetch has resolved: the page then screenshots as an empty shell that
looks exactly like a broken app.

It waits for a DOM condition that only a completed forecast produces, not for a timeout,
and then checks the PNG is not uniform before writing it. A capture with no Screen
Recording grant still exits 0 and still writes a plausible PNG of the right dimensions in
which every pixel is black, and that will be believed unless something checks.

Usage
  .venv/bin/python scripts/screenshot.py --url "http://127.0.0.1:8009/?q=P0A6Y8&host=arctic&tag=his"
  .venv/bin/python scripts/screenshot.py --out docs/screenshots/rig.png --width 1400
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import time
from pathlib import Path

import requests
import websocket  # websocket-client

ROOT = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9222


class CDP:
    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(ws_url, timeout=60, suppress_origin=True)
        self.n = 0

    def send(self, method: str, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def evaluate(self, expr: str):
        r = self.send("Runtime.evaluate", expression=expr, returnByValue=True)
        return r.get("result", {}).get("value")

    def close(self):
        try:
            self.ws.close()
        except Exception:  # noqa: BLE001
            pass


def launch(width: int, height: int, profile: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
         "--remote-allow-origins=*",
         f"--window-size={width},{height}", f"--user-data-dir={profile}",
         "--hide-scrollbars", "--force-device-scale-factor=2",
         "--no-first-run", "--no-default-browser-check", "--disable-gpu", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for_devtools(timeout: float = 30) -> str:
    end = time.time() + timeout
    while time.time() < end:
        try:
            tabs = requests.get(f"http://127.0.0.1:{PORT}/json", timeout=2).json()
            for t in tabs:
                if t.get("type") == "page" and t.get("webSocketDebuggerUrl"):
                    return t["webSocketDebuggerUrl"]
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.3)
    raise RuntimeError("Chrome never exposed a DevTools page target")


def verify(png: bytes, label: str) -> None:
    """Refuse to ship a uniform image. Check a band that should never be flat."""
    from io import BytesIO

    from PIL import Image
    im = Image.open(BytesIO(png)).convert("L")
    lo, hi = im.getextrema()
    if hi == lo:
        raise SystemExit(f"{label}: the capture is uniform ({lo}); it is not the app")
    w, h = im.size
    band = im.crop((0, int(h * 0.25), w, int(h * 0.6)))
    blo, bhi = band.getextrema()
    if bhi - blo < 25:
        raise SystemExit(f"{label}: the main panel is nearly flat ({blo}-{bhi}); "
                         "the forecast probably did not render")
    print(f"  {label}: {w}x{h}, luminance {lo}-{hi}, panel {blo}-{bhi}  ok")


def downscale(png: bytes, width: int) -> bytes:
    from io import BytesIO

    from PIL import Image
    im = Image.open(BytesIO(png)).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = BytesIO()
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8009/?q=P0A6Y8&host=arctic&tag=his")
    ap.add_argument("--out", default="docs/screenshots/rig.png")
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=1150)
    ap.add_argument("--ready", default="document.querySelectorAll('.gate').length >= 9 "
                                       "&& document.querySelector('.tbl') !== null")
    ap.add_argument("--timeout", type=float, default=90)
    ap.add_argument("--pre", default="", help="JS to run once ready, before capturing")
    args = ap.parse_args()

    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    profile = ROOT / "data" / "chrome-profile"
    profile.mkdir(parents=True, exist_ok=True)

    proc = launch(args.width, args.height, profile)
    try:
        cdp = CDP(wait_for_devtools())
        cdp.send("Page.enable")
        cdp.send("Runtime.enable")
        cdp.send("Emulation.setDeviceMetricsOverride", width=args.width, height=args.height,
                 deviceScaleFactor=2, mobile=False)
        print(f"navigating to {args.url}")
        cdp.send("Page.navigate", url=args.url)

        end = time.time() + args.timeout
        ready = False
        while time.time() < end:
            try:
                if cdp.evaluate(args.ready):
                    ready = True
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)
        if not ready:
            err = cdp.evaluate("document.querySelector('.err') && document.querySelector('.err').textContent")
            raise SystemExit(f"the forecast never rendered within {args.timeout}s"
                             + (f"; page says: {err}" if err else ""))
        if args.pre:
            cdp.evaluate(args.pre)
            time.sleep(1.5)
        # let fonts settle so the capture is not mid-swap
        time.sleep(1.5)
        print("  forecast rendered:",
              cdp.evaluate("document.getElementById('flowNote').textContent"))

        shot = cdp.send("Page.captureScreenshot", format="png", captureBeyondViewport=True)
        png = base64.b64decode(shot["data"])
        verify(png, out.name)
        # Captured at device-scale 2 for sharpness, committed at the house width: a 3x
        # Retina grab is several megabytes and GitHub will not thank you for it.
        png = downscale(png, args.width)
        verify(png, out.name + " (downscaled)")
        out.write_bytes(png)
        print(f"wrote {out} ({len(png)/1000:.0f} kB)")
        cdp.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
