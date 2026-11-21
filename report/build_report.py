"""Render report/report.html to report/report.pdf with a headless Chromium browser.

Usage (from the project root):
    python report/build_report.py

Equations are typeset by MathJax (loaded from a CDN), so an internet connection
is needed while building. The browser runs headless with a throwaway profile, so
no window opens and an already-running browser is not affected.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "report.html"
OUTPUT = HERE / "report.pdf"

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser", "msedge",
]


def find_browser():
    for candidate in BROWSERS:
        path = Path(candidate)
        if path.is_file():
            return str(path)
        found = shutil.which(candidate)
        if found:
            return found
    sys.exit("No Chrome/Chromium/Edge found; install one or edit BROWSERS.")


def main():
    browser = find_browser()
    with tempfile.TemporaryDirectory() as profile:
        cmd = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={profile}",
            "--no-pdf-header-footer",
            "--virtual-time-budget=30000",  # give MathJax time to typeset
            f"--print-to-pdf={OUTPUT}",
            SOURCE.as_uri(),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=180)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
