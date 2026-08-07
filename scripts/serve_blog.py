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

PYGMENTS_STYLE = "monokai"  # dark code blocks on a white page
PYGMENTS_CSS = HtmlFormatter(style=PYGMENTS_STYLE).get_style_defs(".highlight")

# HTML shell — keeps raw HTML/JS from the markdown intact.
# Tokens __CONTENT__ and __PYGMENTS_CSS__ are substituted at render time
# (plain .replace(), so CSS braces need no escaping).
HTML_SHELL = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Peralta 🌐🤖🛜</title>
  <meta name="description" content="Distributed RLVF over Wifi." />
  <meta property="og:site_name" content="peralta.jarrodkahn.com" />
  <meta property="og:type" content="article" />
  <meta property="og:title" content="Peralta — Distributed RLVF Over Wifi" />
  <meta property="og:description" content="Distributed RL with verifiable reward on ~500 lines of code and $500 of heterogeneous hardware." />
  <meta property="og:url" content="https://peralta.jarrodkahn.com/" />
  <meta property="og:image" content="https://peralta.jarrodkahn.com/assets/rack.jpg" />
  <meta name="twitter:card" content="summary_large_image" />
  <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🔥</text></svg>">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #ffffff;
      --ink: #0e0f12;
      --muted: #6d727c;
      --faint: #9ba1ab;
      --line: #e7e8ec;
      --code-bg: #0e0f12;
      --accent: #2757ff;
      --pop-orange: #ff5c00;
      --pop-green: #00b368;
      --pop-magenta: #e02a8f;
      --pop-gold: #d99a06;
      --mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
      --sans: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    }

    * { box-sizing: border-box; }

    html {
      -webkit-font-smoothing: antialiased;
      scroll-behavior: smooth;
    }

    html, body { margin: 0; padding: 0; }

    body {
      background: var(--bg);
      color: var(--ink);
      font-family: var(--sans);
      font-size: 16px;
      line-height: 1.7;
      padding: 0 1.25rem 5rem 1.25rem;
    }

    ::selection { background: var(--accent); color: #fff; }

    /* ---------- top bar ---------- */

    .topbar {
      max-width: 48rem;
      margin: 0 auto;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      padding: 0.85rem 0;
      border-bottom: 1px solid var(--line);
      font-family: var(--mono);
      font-size: 0.8rem;
    }

    .topbar .crumb { color: var(--muted); }
    .topbar .crumb b { color: var(--ink); font-weight: 600; }
    .topbar .crumb .tilde { color: var(--accent); }
    .topbar nav { display: flex; gap: 1.25rem; }
    .topbar a { color: var(--muted); text-decoration: none; }
    .topbar a:hover { color: var(--accent); }

    main {
      max-width: 48rem;
      margin: 0 auto;
    }

    /* ---------- typography ---------- */

    a {
      color: var(--accent);
      text-decoration: none;
    }

    a:hover { text-decoration: underline; text-underline-offset: 3px; }

    h1, h2, h3, h4, h5, h6 {
      line-height: 1.25;
      letter-spacing: -0.015em;
      font-weight: 650;
    }

    h1 {
      font-size: 2.3rem;
      letter-spacing: -0.03em;
      margin: 3.5rem 0 1.25rem 0;
    }

    h1 .subtitle {
      display: block;
      font-family: var(--mono);
      font-size: 0.82rem;
      font-weight: 600;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--accent);
      margin-top: 0.9rem;
    }

    main { counter-reset: sec; }

    h2 {
      font-size: 1.4rem;
      margin: 3rem 0 1rem 0;
      padding-bottom: 0.45em;
      border-bottom: 1px solid var(--line);
      counter-increment: sec;
    }

    h2::before {
      content: counter(sec, decimal-leading-zero);
      font-family: var(--mono);
      font-size: 0.75em;
      font-weight: 600;
      color: var(--accent);
      margin-right: 0.75ch;
    }

    h3 { font-size: 1.12rem; margin: 2.25rem 0 0.6rem 0; }
    h4 { font-size: 1rem; margin: 1.9rem 0 0.5rem 0; }

    p { margin: 0.85em 0; }
    li { margin: 0.35em 0; }

    strong { font-weight: 650; }

    /* ---------- header permalinks ---------- */

    .headerlink {
      visibility: hidden;
      margin-left: 0.4ch;
      color: var(--faint);
      font-weight: 400;
      text-decoration: none;
    }

    h2:hover .headerlink, h3:hover .headerlink,
    h4:hover .headerlink, h5:hover .headerlink, h6:hover .headerlink {
      visibility: visible;
    }

    .headerlink:hover { color: var(--accent); text-decoration: none; }
    h1 .headerlink { display: none; }

    /* ---------- code ---------- */

    code {
      font-family: var(--mono);
      font-size: 0.85em;
    }

    p code, li code, td code {
      background: #f2f3f5;
      border-radius: 4px;
      padding: 0.1em 0.35em;
    }

    .highlight {
      background: var(--code-bg);
      border-radius: 10px;
      margin: 1.25rem 0;
    }

    .highlight pre {
      background: transparent;
      margin: 0;
      padding: 1rem 1.2rem;
      overflow-x: auto;
      font-size: 0.82rem;
      line-height: 1.7;
    }

    /* ---------- tables ---------- */

    table {
      border-collapse: collapse;
      width: 100%;
      margin: 1.25rem 0;
      font-size: 0.9rem;
    }

    th {
      font-family: var(--mono);
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      text-align: left;
      border-bottom: 2px solid var(--ink);
      padding: 0.5em 0.9em;
    }

    td {
      border-bottom: 1px solid var(--line);
      padding: 0.55em 0.9em;
    }

    /* ---------- admonitions ---------- */

    .admonition {
      border: 1px solid var(--line);
      border-left: 3px solid var(--accent);
      border-radius: 8px;
      background: #fafbfc;
      padding: 0.75em 1.1em;
      margin: 1.25em 0;
      font-size: 0.95rem;
    }

    .admonition-title {
      font-family: var(--mono);
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
      margin: 0 0 0.35em 0;
    }

    .admonition.warning { border-left-color: var(--pop-orange); }
    .admonition.note    { border-left-color: var(--accent); }
    .admonition.tip     { border-left-color: var(--pop-green); }

    /* ---------- figures & captions ---------- */

    img {
      max-width: 100%;
      border-radius: 10px;
    }

    p.comment {
      font-family: var(--mono);
      font-size: 0.76rem;
      line-height: 1.7;
      color: var(--muted);
      text-align: center;
      max-width: 36rem;
      margin: 0.9rem auto 2.25rem auto;
    }

    .plotly-graph-div {
      margin: 1.5rem 0;
    }

    /* ---------- toc ---------- */

    .toc {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fafbfc;
      padding: 0.9rem 1.2rem;
      font-size: 0.9rem;
    }

    /* ---------- footer ---------- */

    footer {
      max-width: 48rem;
      margin: 4rem auto 0 auto;
      padding-top: 1.1rem;
      border-top: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.5rem;
      font-family: var(--mono);
      font-size: 0.74rem;
      color: var(--faint);
    }

    footer a { color: var(--faint); }
    footer a:hover { color: var(--accent); }

    /* ---------- pygments (monokai) on dark blocks ---------- */
    __PYGMENTS_CSS__

    /* keep our own dark, slightly cooler code background */
    .highlight { background: var(--code-bg); }
  </style>
</head>
<body>
  <div class="topbar">
    <span class="crumb"><span class="tilde">~</span>/<b>peralta</b></span>
    <nav>
      <a href="https://github.com/kahnvex/peralta">github</a>
      <a href="https://jarrodkahn.com">jarrodkahn.com</a>
    </nav>
  </div>
  <main>
  __CONTENT__
  </main>
  <footer>
    <span>© jarrod kahn</span>
    <span>markdown → html · plotly · mathjax</span>
  </footer>
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

    html_full = HTML_SHELL.replace("__PYGMENTS_CSS__", PYGMENTS_CSS).replace(
        "__CONTENT__", html_body
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
