"""Re-render the doc pictures where headless Chrome will not run.

The pictures are normally made with the Chrome command written at the top of
each HTML file, and that stays the way to do it. On machines where Chrome
crashes, such as a Cray login node, this gives the same PNG through WeasyPrint.

recipe_status and step3_figure render this way. pipeline_flowchart draws its
connectors as SVG strokes with arrow markers, and WeasyPrint fills them in as
solid black bars, so that one still needs Chrome.

One WeasyPrint trap worth knowing when editing those pages: text-anchor is
honoured as an SVG attribute but ignored as a CSS property, and a chart whose
labels rely on the CSS form silently stacks every label on top of its bars.

Two things have to be patched for WeasyPrint, and they are patched here rather
than in the HTML so the Chrome command keeps working unchanged:
    flex `gap`   unsupported, so the same spacing is written as margins
    weight 500   Liberation Sans has no Medium and 500 falls back to Regular,
                 so the headings Chrome draws in Medium are asked for in Bold

Usage:
    uv run --with weasyprint --with pypdfium2 --with pillow python scripts/render_docs_png.py
"""

from __future__ import annotations

from pathlib import Path

import PIL.Image
import PIL.ImageChops
import pypdfium2
import weasyprint

DOCS = Path(__file__).resolve().parents[1] / "docs"
PAGE_WIDTH = 740  # the --window-size width in every file's Chrome command
PADDING = 24  # .page padding, kept as the bottom margin after trimming
SCALE = 2  # --force-device-scale-factor=2

SHIM = weasyprint.CSS(string=f"""
    @page {{ size: {PAGE_WIDTH}px 4000px; margin: 0 }}
    .legend > * + * {{ margin-left: 16px }}
    .step > .num {{ margin-right: 12px }}
    h1, .title, .num {{ font-weight: 700 !important }}
""")


def render(name: str) -> None:
    source = DOCS / f"{name}.html"
    pdf = weasyprint.HTML(filename=str(source)).write_pdf(stylesheets=[SHIM])
    page = pypdfium2.PdfDocument(pdf)[0]
    # WeasyPrint lays out in points, so scale back up to CSS pixels, then to 2x.
    image = page.render(scale=PAGE_WIDTH * SCALE / page.get_size()[0]).to_pil().convert("RGB")

    background = PIL.Image.new("RGB", image.size, image.getpixel((5, 5)))
    ink = PIL.ImageChops.difference(image, background).convert("L").point(lambda v: 255 if v > 8 else 0)
    bottom = ink.getbbox()[3] + PADDING * SCALE
    image.crop((0, 0, image.width, min(image.height, bottom))).save(DOCS / f"{name}.png")
    print(f"{name}.png  {image.width}x{bottom}")


if __name__ == "__main__":
    for picture in ("recipe_status", "step3_figure"):
        render(picture)
