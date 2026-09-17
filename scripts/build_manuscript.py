"""Draw Figure 2 of the manuscript from the results files, then print Manuscript.md to PDF.

Figure 2 is drawn from the numbers the scripts wrote, so it cannot drift from a run:
    data/results_checks.json    written by scripts/03_check_results.py

Writes  docs/manuscript_figure2.html and .png
        Manuscript.pdf

Both the PNG and the PDF are made by headless Chrome or Edge, the same way as the
other pictures in docs/.

Usage:
    uv run --with markdown python scripts/build_manuscript.py
    uv run --with markdown python scripts/build_manuscript.py --figure-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CHECKS = ROOT / "data" / "results_checks.json"
MANUSCRIPT = ROOT / "Manuscript.md"

BROWSERS = [
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser", "chrome",
]

# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------
# One row per source of frequency evidence, in the order of Table 3.
SOURCES = [
    ("no_frequency", "None, scores alone"),
    ("public", "Public reference"),
    ("own_hospital", "Own hospital"),
    ("federated_query", "Federated query"),
    ("public_plus_query", "Public reference with federated query"),
    ("ceiling", "Ceiling"),
]
FEDERATED = {"federated_query", "public_plus_query"}  # drawn in the accent colour
SIMULATED = {"own_hospital", "federated_query", "public_plus_query"}  # these depend on the simulated patients

ACCENT, MUTED, INK, FAINT, GRID = "#2f6db5", "#a9b4c2", "#1d1d1b", "#6b6b68", "#e3e1da"
WIDTH, LABELS, PLOT_LEFT, PLOT_RIGHT = 800, 262, 274, 610  # the right margin holds the value and the spread
ROW, BAR = 34, 14
AUC_AXIS = (0.95, 1.00, [0.95, 0.96, 0.97, 0.98, 0.99, 1.00])
COUNT_AXIS = (0, 16, [0, 4, 8, 12, 16])


def scale(value: float, axis: tuple) -> float:
    low, high, _ = axis
    return PLOT_LEFT + (value - low) / (high - low) * (PLOT_RIGHT - PLOT_LEFT)


def panel_frame(top: int, letter: str, title: str, axis: tuple, tick_format: str) -> list[str]:
    """Panel title, row labels, grid lines and tick labels."""
    bottom = top + 30 + ROW * len(SOURCES)
    parts = [f'<text x="0" y="{top + 14}" class="panel"><tspan class="letter">{letter}</tspan>  {title}</text>']
    for tick in axis[2]:
        x = scale(tick, axis)
        parts.append(f'<line x1="{x:.1f}" y1="{top + 28}" x2="{x:.1f}" y2="{bottom}" stroke="{GRID}"/>')
        parts.append(f'<text x="{x:.1f}" y="{bottom + 16}" class="tick" text-anchor="middle">{tick:{tick_format}}</text>')
    for index, (_, label) in enumerate(SOURCES):
        parts.append(f'<text x="{LABELS}" y="{top + 30 + ROW * index + 21}" class="label" text-anchor="end">{label}</text>')
    return parts


def colour_of(source: str) -> tuple[str, str]:
    """Fill and outline. The ceiling is hollow because no hospital could use it."""
    if source == "ceiling":
        return "#ffffff", ACCENT
    return (ACCENT, ACCENT) if source in FEDERATED else (MUTED, MUTED)


def figure_svg(results: dict) -> tuple[str, int]:
    settings, cohorts = results["settings"], results["fresh_cohorts"]
    top_a, top_b = 0, 30 + ROW * len(SOURCES) + 56
    height = top_b + 30 + ROW * len(SOURCES) + 30
    parts = panel_frame(top_a, "A", "Ranking: AUC on the 2,373 test variants", AUC_AXIS, ".2f")
    parts += panel_frame(top_b, "B", "Decisions: false positives among the 183 population-discordant benign test variants",
                         COUNT_AXIS, "d")

    for index, (source, _) in enumerate(SOURCES):
        fill, outline = colour_of(source)

        # A: a dot, because the axis does not start at zero and a bar would mislead
        y = top_a + 30 + ROW * index + 17
        auc = settings[source]["auc_all"]
        parts.append(f'<circle cx="{scale(auc, AUC_AXIS):.1f}" cy="{y}" r="5.5" fill="{fill}" stroke="{outline}" stroke-width="1.6"/>')
        parts.append(f'<text x="{PLOT_RIGHT + 14}" y="{y + 4}" class="value">{auc:.3f}</text>')

        # B: a bar from zero for the reported build
        y = top_b + 30 + ROW * index + 8
        count = settings[source]["false_positives_discordant"]
        width = scale(count, COUNT_AXIS) - PLOT_LEFT
        parts.append(f'<rect x="{PLOT_LEFT}" y="{y}" width="{width:.1f}" height="{BAR}" rx="2" fill="{fill}" stroke="{outline}" stroke-width="1.6"/>')
        parts.append(f'<text x="{PLOT_RIGHT + 14}" y="{y + 12}" class="value">{count}</text>')

        # and, under it, the spread over the re-simulated cohorts
        if source in SIMULATED:
            spread = cohorts[source]
            low, high, median = (scale(spread[k], COUNT_AXIS) for k in ("lowest", "highest", "median"))
            line = y + BAR + 7
            parts.append(f'<line x1="{low:.1f}" y1="{line}" x2="{high:.1f}" y2="{line}" stroke="{INK}" stroke-width="1.4"/>')
            for x in (low, high):
                parts.append(f'<line x1="{x:.1f}" y1="{line - 4}" x2="{x:.1f}" y2="{line + 4}" stroke="{INK}" stroke-width="1.4"/>')
            parts.append(f'<path d="M{median:.1f} {line - 5} l5 5 l-5 5 l-5 -5 z" fill="{INK}"/>')
            parts.append(f'<text x="{PLOT_RIGHT + 40}" y="{y + 12}" class="spread">'
                         f'median {spread["median"]}, {spread["lowest"]} to {spread["highest"]}</text>')

    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}">' + "".join(parts) + "</svg>", height


def figure_html(results: dict) -> tuple[str, int]:
    svg, height = figure_svg(results)
    page = f"""<!DOCTYPE html>
<!-- Figure 2 of Manuscript.md. Written by scripts/build_manuscript.py from data/results_checks.json: rerun it, do not edit. -->
<html lang="en"><head><meta charset="utf-8"><title>Figure 2</title>
<style>
  body{{margin:0;padding:20px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK}}}
  .panel{{font-size:14px;font-weight:600}} .letter{{font-weight:700}}
  .label{{font-size:13px}} .value{{font-size:12.5px;font-weight:600}}
  .tick{{font-size:11px;fill:{FAINT}}} .spread{{font-size:11px;fill:{FAINT}}}
</style></head><body>{svg}</body></html>
"""
    return page, height + 40


# ---------------------------------------------------------------------------
# The PDF
# ---------------------------------------------------------------------------
PRINT_STYLE = """
  @page { size: A4; margin: 22mm 20mm 22mm 20mm; @bottom-center { content: counter(page); font: 9pt Georgia, serif; color: #555 } }
  body { font: 10.5pt/1.5 Georgia, "Times New Roman", serif; color: #111; }
  h1 { font-size: 17pt; line-height: 1.25; margin: 0 0 10pt; }
  h2 { font-size: 12.5pt; margin: 18pt 0 5pt; }
  h3 { font-size: 11pt; margin: 13pt 0 3pt; }
  h2, h3 { break-after: avoid; }
  p { margin: 0 0 7pt; text-align: justify; hyphens: auto; }
  sup { font-size: 7pt; }
  table { border-collapse: collapse; width: 100%; margin: 4pt 0 12pt; font: 8.8pt/1.35 "Helvetica Neue", Arial, sans-serif; break-inside: avoid; }
  th { text-align: left; border-top: 1px solid #111; border-bottom: 1px solid #111; padding: 3pt 5pt; }
  td { border-bottom: 0.5px solid #ccc; padding: 3pt 5pt; vertical-align: top; }
  img { display: block; max-width: 100%; max-height: 205mm; margin: 8pt auto 5pt; }
  code { font: 8.8pt Consolas, Menlo, monospace; }
  pre { background: #f5f5f2; padding: 6pt 8pt; border-radius: 3px; font: 8.5pt/1.4 Consolas, Menlo, monospace; white-space: pre-wrap; break-inside: avoid; }
  ol, ul { margin: 0 0 7pt; padding-left: 16pt; }
  h2#references + ol, h2#references ~ p { font-size: 9pt; text-align: left; }
"""


def manuscript_html() -> str:
    import markdown  # supplied by `uv run --with markdown`, so it stays out of the project's dependencies

    body = markdown.markdown(MANUSCRIPT.read_text(encoding="utf-8"), extensions=["tables", "fenced_code", "toc"])
    return (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><base href="{ROOT.as_uri()}/">'
            f"<title>Manuscript</title><style>{PRINT_STYLE}</style></head><body>{body}</body></html>")


# ---------------------------------------------------------------------------
def find_browser() -> str:
    for candidate in BROWSERS:
        found = candidate if Path(candidate).exists() else shutil.which(candidate)
        if found:
            return str(found)
    raise SystemExit("No Chrome, Chromium or Edge found. Install one, or add its path to BROWSERS.")


def run_browser(*arguments: str) -> None:
    subprocess.run([find_browser(), "--headless=new", "--disable-gpu", *arguments],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw Figure 2 and print the manuscript to PDF")
    parser.add_argument("--figure-only", action="store_true", help="skip the PDF")
    args = parser.parse_args()

    if not CHECKS.exists():
        print("data/results_checks.json is missing. Run scripts/03_check_results.py first.", file=sys.stderr)
        return 1

    page, height = figure_html(json.loads(CHECKS.read_text(encoding="utf-8")))
    source = DOCS / "manuscript_figure2.html"
    source.write_text(page, encoding="utf-8", newline="\n")
    run_browser("--hide-scrollbars", "--force-device-scale-factor=2", f"--window-size={WIDTH + 40},{height}",
                f"--screenshot={DOCS / 'manuscript_figure2.png'}", source.as_uri())
    print("wrote docs/manuscript_figure2.png")

    if not args.figure_only:
        with tempfile.TemporaryDirectory() as folder:
            printable = Path(folder) / "manuscript.html"
            printable.write_text(manuscript_html(), encoding="utf-8")
            run_browser("--no-pdf-header-footer", f"--print-to-pdf={ROOT / 'Manuscript.pdf'}", printable.as_uri())
        print("wrote Manuscript.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
