#!/usr/bin/env python3
"""
Watch README.md, render to docs/index.html with Python-Markdown (raw HTML/JS allowed),
and serve docs/ at http://localhost:9000.

Deps:
  pip install markdown watchdog
"""

import sys
import argparse
import time
import threading
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from functools import partial
from typing import Optional
from pygments.formatters import HtmlFormatter


import markdown
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

# ---- config ----
ROOT = Path(__file__).resolve().parent.parent
MD_ROOT = ROOT / "blog"
SRC_MD = MD_ROOT / "main.md"
OUT_DIR = ROOT / "docs"
OUT_HTML = OUT_DIR / "index.html"
PORT = 9000
TITLE = "README"

PYGMENTS_STYLE = "friendly"  # pick any: "default", "friendly", "monokai", ...
PYGMENTS_CSS = HtmlFormatter(style=PYGMENTS_STYLE).get_style_defs(".highlight")

# Basic HTML shell — keeps your raw <script> blocks intact
HTML_SHELL = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Peralta 🌐🤖🛜</title>
  <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🔥</text></svg>">
  <style>
    /* minimal readable defaults; tweak as you like */
    html, body {{ margin:0; padding:0; }}
    body {{ font: 16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,"Apple Color Emoji","Segoe UI Emoji"; padding: 24px; }}
    main {{ max-width: 1100px; margin: 0 auto; }}
    pre {{ background:#f6f8fa; padding:12px; overflow:auto; border-radius:6px; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
    table {{ border-collapse: collapse; }}
    td, th {{ border: 1px solid #ddd; padding: 6px 10px; }}
    .toc {{ background:#fafafa; padding:12px; border:1px solid #eee; border-radius:6px; }}
    /* legend overlay if you embed Plotly or other widgets is up to your snippet */
    {pygments_css}
  </style>
</head>
<body>
  <main>
  {content}
  </main>
<script id="MathJax-script" async
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</body>
</html>
"""


# Debounce writes to avoid double-firing from some editors
class Debouncer:
    def __init__(self, wait_sec: float = 0.2):
        self.wait = wait_sec
        self._last = 0.0

    def ready(self) -> bool:
        now = time.time()
        if now - self._last >= self.wait:
            self._last = now
            return True
        return False


def render_markdown(src: Path, dst: Path, title: Optional[str] = None) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source Markdown: {src}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    text = src.read_text(encoding="utf-8")
    math_1k_plot = Path("docs/assets/math_1k.html").read_text()
    math_10k_plot = Path("docs/assets/math_10k.html").read_text()
    text = text.replace("<math_1k_plot />", math_1k_plot)
    text = text.replace("<math_10k_plot />", math_10k_plot)

    # Python-Markdown keeps raw HTML/JS by default (no safe_mode here)
    html_body = markdown.markdown(
        text,
        extensions=[
            "extra",  # tables, etc.
            "toc",  # table of contents
            "fenced_code",  # ``` blocks
            "attr_list",  # {#id .class}
            "codehilite",  # code blocks
            "pymdownx.arithmatex",
            "admonition",
            # add more pymdownx.* if you want
        ],
        extension_configs={
            "codehilite": {
                "guess_lang": False,
                "pygments_style": "default",  # or 'friendly', 'monokai', etc.
                "noclasses": False,  # use CSS classes (preferred)
                "css_class": "highlight",  # 👈 match CDN CSS selectors
            },
            "pymdownx.arithmatex": {"generic": True},
            "toc": {
                "permalink": "¶",  # adds a clickable link symbol
                "permalink_class": "headerlink",
                "permalink_title": "Link to this section",
            },
        },
        output_format="html",
    )

    html_full = HTML_SHELL.format(
        title=title or TITLE,
        content=html_body,
        pygments_css=PYGMENTS_CSS,
    )
    dst.write_text(html_full, encoding="utf-8")
    print(f"[build] Wrote {dst.relative_to(ROOT)}  ({time.strftime('%H:%M:%S')})")


class ReadmeHandler(FileSystemEventHandler):
    def __init__(self, debouncer: Debouncer):
        super().__init__()
        self.debouncer = debouncer

    def on_any_event(self, event: FileSystemEvent) -> None:
        # Rebuild on modify/create/move of README.md
        if event.is_directory:
            return
        p = Path(str(event.src_path))
        if p.resolve() == SRC_MD.resolve():
            if self.debouncer.ready():
                try:
                    render_markdown(SRC_MD, OUT_HTML, TITLE)
                except Exception as e:
                    print(f"[build][error] {e}", file=sys.stderr)


def serve_docs(directory: Path, port: int) -> None:
    handler_cls = partial(SimpleHTTPRequestHandler, directory=str(directory))
    httpd = ThreadingHTTPServer(("localhost", port), handler_cls)
    print(f"[serve] http://localhost:{port} (serving {directory.relative_to(ROOT)}/)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        print("[serve] stopped")


def main(write_only: bool = False) -> int:
    # initial build
    render_markdown(SRC_MD, OUT_HTML, TITLE)

    if write_only:
        return 0

    # start server thread
    t = threading.Thread(target=serve_docs, args=(OUT_DIR, PORT), daemon=True)
    t.start()

    # watch blog
    observer = Observer()
    debouncer = Debouncer(0.2)
    handler = ReadmeHandler(debouncer)
    observer.schedule(handler, str(MD_ROOT), recursive=False)  # watch project root only
    observer.start()
    print("[watch] Watching README.md for changes. Ctrl-C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[watch] stopping…")
        observer.stop()
    observer.join()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build docs from README.md; serve and watch by default."
    )
    parser.add_argument(
        "-w",
        "--write",
        action="store_true",
        help="Write docs once and exit (no server, no watcher)",
    )
    args = parser.parse_args()
    sys.exit(main(write_only=args.write))
