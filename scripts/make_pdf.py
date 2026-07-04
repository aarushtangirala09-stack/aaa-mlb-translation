"""Convert Paper_Draft.md to Paper_Draft.pdf via headless Chrome.

Chrome is the only PDF engine present on this system. The pipeline is:
  markdown → HTML (with academic CSS) → temporary .html file → Chrome
  headless --print-to-pdf → Paper_Draft.pdf.

Chrome's PDF engine handles tables, page breaks, and font rendering
better than any pure-Python option would on this machine.
"""

from __future__ import annotations

import base64
import re
import subprocess
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
MD_IN = ROOT / "Paper_Draft.md"
PDF_OUT = ROOT / "Paper_Draft.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS_STYLE = """
@page {
    size: Letter;
    margin: 0.8in 0.9in;
}
body {
    font-family: "Times New Roman", Georgia, serif;
    font-size: 11pt;
    line-height: 1.45;
    color: #111;
    max-width: 100%;
}
h1 {
    font-size: 20pt;
    margin: 0 0 0.4em 0;
    line-height: 1.2;
    text-align: center;
}
h2 {
    font-size: 14pt;
    margin: 1.4em 0 0.4em 0;
    border-bottom: 1px solid #999;
    padding-bottom: 4px;
    page-break-after: avoid;
}
h3 {
    font-size: 12pt;
    margin: 1.1em 0 0.3em 0;
    page-break-after: avoid;
}
p {
    margin: 0 0 0.6em 0;
    text-align: justify;
    hyphens: auto;
}
strong { font-weight: 700; }
em { font-style: italic; }
code {
    font-family: "Menlo", "Consolas", monospace;
    font-size: 10pt;
    background: #f4f4f4;
    padding: 1px 4px;
    border-radius: 2px;
}
pre {
    font-family: "Menlo", "Consolas", monospace;
    font-size: 9.5pt;
    background: #f7f7f7;
    padding: 10px 12px;
    border-left: 3px solid #bbb;
    overflow-x: auto;
    page-break-inside: avoid;
    white-space: pre-wrap;
}
table {
    border-collapse: collapse;
    margin: 0.6em 0 1em 0;
    font-size: 10pt;
    width: 100%;
    page-break-inside: avoid;
}
th, td {
    border: 1px solid #999;
    padding: 5px 8px;
    text-align: left;
    vertical-align: top;
}
th {
    background: #eee;
    font-weight: 700;
}
hr {
    border: none;
    border-top: 1px solid #bbb;
    margin: 1.4em 0;
}
blockquote {
    border-left: 3px solid #999;
    padding-left: 12px;
    color: #333;
    font-style: italic;
    margin: 0.6em 0;
}
ul, ol { margin: 0.4em 0 0.7em 1.5em; padding: 0; }
li { margin-bottom: 0.2em; }
a { color: #003366; text-decoration: none; }

figure {
    margin: 1em auto;
    text-align: center;
    page-break-inside: avoid;
}
figure img {
    max-width: 95%;
    height: auto;
}
figcaption {
    font-size: 9.5pt;
    color: #333;
    margin-top: 6px;
    text-align: left;
    padding: 0 0.5em;
    line-height: 1.35;
}
.formula {
    text-align: center;
    font-family: "Cambria Math", "Times New Roman", serif;
    font-size: 12pt;
    margin: 0.9em 0;
    page-break-inside: avoid;
}
.formula-tall { font-size: 11pt; }
.overbar {
    border-top: 1px solid #111;
    padding: 0 1px;
}
sub, sup { font-size: 75%; line-height: 0; }
"""


def inline_images(html: str) -> str:
    """Replace <img src="figures/*.png"> with base64 data URIs so Chrome
    never has to resolve local file paths (which sometimes silently drops
    images inside <figure> blocks near page boundaries)."""

    def repl(m: re.Match) -> str:
        src = m.group(1)
        path = ROOT / src
        if not path.exists():
            return m.group(0)
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'src="data:image/png;base64,{data}"'

    return re.sub(r'src="(figures/[^"]+\.png)"', repl, html)


def main() -> None:
    text = MD_IN.read_text()
    html_body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "md_in_html", "attr_list"],
    )
    html_body = inline_images(html_body)
    html_doc = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Paper Draft</title>
<style>{CSS_STYLE}</style>
</head>
<body>
{html_body}
</body></html>"""

    with tempfile.NamedTemporaryFile(suffix=".html", mode="w",
                                     dir=ROOT, delete=False,
                                     encoding="utf-8") as tf:
        tf.write(html_doc)
        html_path = Path(tf.name)

    try:
        cmd = [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            f"--print-to-pdf={PDF_OUT}",
            f"file://{html_path}",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if proc.returncode != 0 or not PDF_OUT.exists():
            print("Chrome stderr:", proc.stderr[-2000:])
            raise SystemExit(f"Chrome exited {proc.returncode}")
    finally:
        html_path.unlink(missing_ok=True)

    size_kb = PDF_OUT.stat().st_size / 1024
    print(f"Wrote {PDF_OUT.relative_to(ROOT)}  ({size_kb:,.0f} KB)")


if __name__ == "__main__":
    main()
