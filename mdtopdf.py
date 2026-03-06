#!/usr/bin/env python3
"""
Convert Markdown file to PDF with embedded local images.

Usage:
  python scripts/mdtopdf.py input.md
  python scripts/mdtopdf.py input.md -o output.pdf
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Markdown to PDF and keep images."
    )
    parser.add_argument("input_md", help="Path to input markdown file")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output PDF file (default: same name as markdown)",
    )
    parser.add_argument(
        "--title",
        default="Markdown Report",
        help="Document title in HTML metadata",
    )
    return parser


def _rewrite_local_image_links(html: str, md_dir: Path) -> str:
    """
    Convert local <img src="..."> to absolute file:// URIs.
    External links (http/https/data/file) are kept untouched.
    """
    pattern = re.compile(r'(<img[^>]+src=")([^"]+)(")')

    def replacer(match: re.Match[str]) -> str:
        prefix, src, suffix = match.groups()
        src = src.strip()
        if src.startswith(("http://", "https://", "data:", "file://")):
            return match.group(0)
        abs_path = (md_dir / src).resolve()
        return f'{prefix}{abs_path.as_uri()}{suffix}'

    return pattern.sub(replacer, html)


def md_to_pdf(input_md: Path, output_pdf: Path, title: str) -> None:
    try:
        import markdown
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'markdown'. Install with: pip install markdown"
        ) from exc

    try:
        from weasyprint import HTML
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'weasyprint'. Install with: pip install weasyprint"
        ) from exc

    text = input_md.read_text(encoding="utf-8")
    body_html = markdown.markdown(
        text,
        extensions=["extra", "tables", "fenced_code", "toc"],
        output_format="html5",
    )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    @page {{
      size: A4;
      margin: 18mm 16mm;
    }}
    body {{
      font-family: "DejaVu Sans", Arial, sans-serif;
      line-height: 1.5;
      font-size: 11pt;
      color: #111;
    }}
    h1, h2, h3 {{
      margin-top: 1.2em;
      margin-bottom: 0.5em;
      line-height: 1.25;
    }}
    img {{
      max-width: 100%;
      height: auto;
      display: block;
      margin: 10px auto;
      page-break-inside: avoid;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 10px 0;
      font-size: 10pt;
    }}
    th, td {{
      border: 1px solid #999;
      padding: 6px 8px;
      vertical-align: top;
    }}
    code {{
      background: #f2f2f2;
      padding: 1px 4px;
      border-radius: 3px;
      font-family: "DejaVu Sans Mono", monospace;
    }}
    pre code {{
      display: block;
      padding: 10px;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
    }}
  </style>
</head>
<body>
{body_html}
</body>
</html>
"""

    html = _rewrite_local_image_links(html, input_md.parent)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(input_md.parent.resolve())).write_pdf(str(output_pdf))


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    input_md = Path(args.input_md).resolve()
    if not input_md.exists():
        print(f"Input markdown not found: {input_md}", file=sys.stderr)
        return 1

    output_pdf = Path(args.output).resolve() if args.output else input_md.with_suffix(".pdf")

    try:
        md_to_pdf(input_md, output_pdf, args.title)
    except Exception as exc:  # keep CLI UX simple
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 2

    print(f"PDF created: {output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
