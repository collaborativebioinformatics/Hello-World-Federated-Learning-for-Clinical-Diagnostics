"""Re-render the doc pictures.

Each picture under docs/ is drawn from the HTML file of the same name by headless
Chrome, 740 CSS pixels wide at device scale 2, the same window as the Chrome
command written at the top of each HTML file. The window is opened taller than
any picture needs and the PNG is trimmed under the last line plus the page
padding, so the height in that command does not have to be kept up to date.

    uv run --with pillow python scripts/render_docs_png.py                          # every picture
    uv run --with pillow python scripts/render_docs_png.py pipeline_flowchart_built  # one or more names

On machines where Chrome will not run, such as a Cray login node, --weasyprint
gives the same PNG through WeasyPrint for recipe_status, step3_figure and
step4_figure. The two flowcharts draw their connectors as SVG strokes with arrow
markers, and WeasyPrint fills them in as solid black bars, so those still need Chrome.

    uv run --with weasyprint --with pypdfium2 --with pillow python scripts/render_docs_png.py --weasyprint

One WeasyPrint trap worth knowing when editing those pages: text-anchor is
honoured as an SVG attribute but ignored as a CSS property, and a chart whose
labels rely on the CSS form silently stacks every label on top of its bars.

Two things have to be patched for WeasyPrint, and they are patched here rather
than in the HTML so the Chrome command keeps working unchanged:
    flex `gap`   unsupported, so the same spacing is written as margins
    weight 500   Liberation Sans has no Medium and 500 falls back to Regular,
                 so the headings Chrome draws in Medium are asked for in Bold
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import PIL.Image
import PIL.ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_manuscript import run_browser  # noqa: E402  the same browser search as the manuscript build

DOCS = Path(__file__).resolve().parents[1] / "docs"
PAGE_WIDTH = 740  # the --window-size width in every file's Chrome command
PADDING = 24  # .page padding, kept as the bottom margin after trimming
SCALE = 2  # --force-device-scale-factor=2
TALL = 5000  # a window taller than any picture; the PNG is trimmed to the drawing

PICTURES = ("pipeline_flowchart_built", "recipe_status", "pipeline_flowchart", "step3_figure", "step4_figure")
WEASYPRINT_PICTURES = ("recipe_status", "step3_figure", "step4_figure")

SHIM = """
    @page {{ size: {width}px 4000px; margin: 0 }}
    .legend > * + * {{ margin-left: 16px }}
    .step > .num {{ margin-right: 12px }}
    h1, .title, .num {{ font-weight: 700 !important }}
"""


def trim_and_save(image: PIL.Image.Image, name: str) -> None:
    """Cut the blank page under the drawing, keeping the page padding."""
    background = PIL.Image.new("RGB", image.size, image.getpixel((5, 5)))
    ink = PIL.ImageChops.difference(image, background).convert("L").point(lambda v: 255 if v > 8 else 0)
    bottom = ink.getbbox()[3] + PADDING * SCALE
    image.crop((0, 0, image.width, min(image.height, bottom))).save(DOCS / f"{name}.png")
    print(f"{name}.png  {image.width}x{min(image.height, bottom)}")


def render_chrome(name: str) -> None:
    source = DOCS / f"{name}.html"
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        shot = Path(folder) / "shot.png"
        run_browser("--hide-scrollbars", f"--force-device-scale-factor={SCALE}", f"--window-size={PAGE_WIDTH},{TALL}",
                    f"--user-data-dir={Path(folder) / 'profile'}", f"--screenshot={shot}", source.as_uri())
        image = PIL.Image.open(shot).convert("RGB")
        image.load()
    trim_and_save(image, name)


def render_weasyprint(name: str) -> None:
    import pypdfium2
    import weasyprint

    source = DOCS / f"{name}.html"
    pdf = weasyprint.HTML(filename=str(source)).write_pdf(stylesheets=[weasyprint.CSS(string=SHIM.format(width=PAGE_WIDTH))])
    page = pypdfium2.PdfDocument(pdf)[0]
    # WeasyPrint lays out in points, so scale back up to CSS pixels, then to 2x.
    image = page.render(scale=PAGE_WIDTH * SCALE / page.get_size()[0]).to_pil().convert("RGB")
    trim_and_save(image, name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-render the pictures under docs/ from their HTML")
    parser.add_argument("names", nargs="*", help=f"pictures to render; default all of {', '.join(PICTURES)}")
    parser.add_argument("--weasyprint", action="store_true", help="draw with WeasyPrint instead of Chrome")
    args = parser.parse_args()

    names = args.names or (list(WEASYPRINT_PICTURES) if args.weasyprint else list(PICTURES))
    for name in names:
        if not (DOCS / f"{name}.html").exists():
            print(f"no docs/{name}.html", file=sys.stderr)
            return 1
        if args.weasyprint:
            if name not in WEASYPRINT_PICTURES:
                print(f"{name} needs Chrome: WeasyPrint fills its arrow markers in as black bars", file=sys.stderr)
                return 1
            render_weasyprint(name)
        else:
            render_chrome(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
