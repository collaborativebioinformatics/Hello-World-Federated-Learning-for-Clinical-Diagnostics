"""Draw Figures 2, 3 and 4 of the manuscript from the results files, then print Manuscript.md to PDF.

The figures are drawn from the numbers the scripts wrote, so they cannot drift from a run:
    Figure 2   data/results_checks.json         written by scripts/03_check_results.py
    Figure 3   data/results_local.json, data/cancer/results_local.json, data/all/results_local.json
               written by scripts/03_train_local.py for each disease area, with the panel counts
               from config/panel_sets.json and config/all_panel_versions.json
    Figure 4   docs/quarterly_replay.json        written by scripts/05_quarterly_replay.py

Writes  docs/manuscript_figure2.html and .png, and the same for figures 3 and 4
        Manuscript.pdf

Figure 2 needs its file. Figures 3 and 4 are skipped with a note when a file of theirs
is missing, and the committed PNG stands.

Both the PNGs and the PDF are made by headless Chrome or Edge, the same way as the
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
REPLAY = DOCS / "quarterly_replay.json"
MANUSCRIPT = ROOT / "Manuscript.md"

BROWSERS = [
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser", "chrome",
]

# ---------------------------------------------------------------------------
# Shared drawing
# ---------------------------------------------------------------------------
ACCENT, MUTED, INK, FAINT, GRID = "#2f6db5", "#a9b4c2", "#1d1d1b", "#6b6b68", "#e3e1da"
WIDTH, LABELS, PLOT_LEFT, PLOT_RIGHT = 800, 262, 274, 610  # the right margin holds the value and the spread
ROW, BAR = 34, 14
FEDERATED = {"federated_query", "public_plus_query", "federated"}  # drawn in the accent colour


def colour_of(source: str) -> tuple[str, str]:
    """Fill and outline. The ceiling is hollow because no hospital could use it."""
    if source == "ceiling":
        return "#ffffff", ACCENT
    return (ACCENT, ACCENT) if source in FEDERATED else (MUTED, MUTED)


def page(number: int, sources: str, svg: str, height: int) -> tuple[str, int]:
    """The HTML around a figure's SVG, and the window height that shows it."""
    html = f"""<!DOCTYPE html>
<!-- Figure {number} of Manuscript.md. Written by scripts/build_manuscript.py from {sources}: rerun it, do not edit. -->
<html lang="en"><head><meta charset="utf-8"><title>Figure {number}</title>
<style>
  body{{margin:0;padding:20px;background:#fff;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;color:{INK}}}
  .panel{{font-size:14px;font-weight:600}} .letter{{font-weight:700}}
  .label{{font-size:13px}} .value{{font-size:12.5px;font-weight:600}}
  .tick{{font-size:11px;fill:{FAINT}}} .spread{{font-size:11px;fill:{FAINT}}} .key{{font-size:11.5px;fill:{FAINT}}}
</style></head><body>{svg}</body></html>
"""
    return html, height + 40


# ---------------------------------------------------------------------------
# Figure 2: the frequency source, ranking and decisions
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
SIMULATED = {"own_hospital", "federated_query", "public_plus_query"}  # these depend on the simulated patients
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


def figure2_svg(results: dict) -> tuple[str, int]:
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


def figure2() -> tuple[str, int]:
    return page(2, "data/results_checks.json", *figure2_svg(json.loads(CHECKS.read_text(encoding="utf-8"))))


# ---------------------------------------------------------------------------
# Figure 3: false positives by disease area
# ---------------------------------------------------------------------------
AREAS = [  # set in config/panel_sets.json, its data folder, the panel title, and a qualifier for the run
    ("cardiac", ROOT / "data", "Cardiac", ""),
    ("cancer", ROOT / "data" / "cancer", "Inherited cancer", ""),
    ("all", ROOT / "data" / "all", "All signed-off panels", ", first run"),  # steps 2 and 3 only, no NVFlare run yet
]
AREA_SOURCES = [  # keys of false_alarm_counts in results_local.json, in the order of Table 6
    ("scores_only", "None, scores alone"),
    ("public", "Public reference"),
    ("own_hospital", "Own hospital"),
    ("federated_query", "Federated query"),
    ("ceiling", "Ceiling"),
]
PANEL_LEFT, PANEL_STEP, PANEL_WIDTH, LABELS3, GRID_TOP = 140, 228, 150, 130, 48  # three panels side by side


def panel_count(area: str) -> int:
    sets = json.loads((ROOT / "config" / "panel_sets.json").read_text(encoding="utf-8"))
    if sets[area].get("panels"):
        return len(sets[area]["panels"])
    # the all-panels set has no hand-written list; step 0 records what it took from the signed-off list
    return json.loads((ROOT / "config" / "all_panel_versions.json").read_text(encoding="utf-8"))["panel_count"]


def bar_axis(highest: int) -> tuple[int, list[int]]:
    """A top with room for the value beside the bar, ticks at 0, half and the top."""
    for top in (16, 40, 80, 160, 300, 600, 1000, 2000, 4000, 8000):
        if highest <= top:
            return top, [0, top // 2, top]
    raise ValueError(f"{highest} false positives is off the chart")


def figure3_svg(areas: list[tuple[str, str, int, int, dict]]) -> tuple[str, int]:
    """areas: title, qualifier, panels, population-discordant benign test rows, false alarms by source."""
    bottom = GRID_TOP + 2 + ROW * len(AREA_SOURCES)
    height = bottom + 30
    parts = [f'<text x="{LABELS3}" y="{GRID_TOP + 23 + ROW * index}" class="label" text-anchor="end">{label}</text>'
             for index, (_, label) in enumerate(AREA_SOURCES)]
    for letter, (title, qualifier, panels, rows, counts), offset in zip("ABC", areas, range(0, PANEL_STEP * 3, PANEL_STEP)):
        left = PANEL_LEFT + offset
        top, ticks = bar_axis(max(counts[source] for source, _ in AREA_SOURCES))
        parts.append(f'<text x="{left}" y="14" class="panel"><tspan class="letter">{letter}</tspan>  {title}</text>')
        parts.append(f'<text x="{left}" y="30" class="key">{panels} panels, of {rows:,} variants{qualifier}</text>')
        for tick in ticks:
            x = left + tick / top * PANEL_WIDTH
            parts.append(f'<line x1="{x:.1f}" y1="{GRID_TOP}" x2="{x:.1f}" y2="{bottom}" stroke="{GRID}"/>')
            parts.append(f'<text x="{x:.1f}" y="{bottom + 16}" class="tick" text-anchor="middle">{tick}</text>')
        for index, (source, _) in enumerate(AREA_SOURCES):
            fill, outline = colour_of(source)
            y = GRID_TOP + 10 + ROW * index
            width = counts[source] / top * PANEL_WIDTH
            parts.append(f'<rect x="{left}" y="{y}" width="{width:.1f}" height="{BAR}" rx="2" fill="{fill}" stroke="{outline}" stroke-width="1.6"/>')
            parts.append(f'<text x="{left + width + 8:.1f}" y="{y + 12}" class="value">{counts[source]}</text>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}">' + "".join(parts) + "</svg>", height


def figure3() -> tuple[str, int] | None:
    areas, files = [], []
    for area, folder, title, qualifier in AREAS:
        results_file = folder / "results_local.json"
        if not results_file.exists():
            print(f"Figure 3 skipped: {results_file.relative_to(ROOT)} is missing. Run scripts/03_train_local.py --panel {area}.")
            return None
        results = json.loads(results_file.read_text(encoding="utf-8"))
        areas.append((title, qualifier, panel_count(area), results["discordant_benign_rows"], results["false_alarm_counts"]))
        files.append(results_file.relative_to(ROOT).as_posix())
    return page(3, ", ".join(files), *figure3_svg(areas))


# ---------------------------------------------------------------------------
# Figure 4: the quarterly replay of the cardiac area
# ---------------------------------------------------------------------------
REPLAY_AREA = "cardiac"
PLOT_X, PLOT_W, BASE, RISE, PANEL_B = 34, 218, 226, 160, 420  # the plot inside a panel, and where panel B starts


def line_axis(highest: int) -> tuple[int, list[int]]:
    for top in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
        if highest <= top:
            return top, [0, top // 2, top]
    raise ValueError(f"{highest} is off the chart")


def line_panel(x0: int, letter: str, title: str, key: str, quarters: list[str], series: list[tuple]) -> list[str]:
    """One panel of lines over the quarters. series: legend name, label name, values, colour, hollow."""
    top, ticks = line_axis(max(max(values) for _, _, values, _, _ in series))
    left, right = x0 + PLOT_X, x0 + PLOT_X + PLOT_W
    y_of = lambda value: BASE - value / top * RISE  # noqa: E731
    xs = [left + index * PLOT_W / (len(quarters) - 1) for index in range(len(quarters))]
    parts = [f'<text x="{x0}" y="14" class="panel"><tspan class="letter">{letter}</tspan>  {title}</text>',
             f'<text x="{x0}" y="30" class="key">{key}</text>']
    x = x0 + 5
    for legend, _, _, colour, hollow in series:
        ring = 'fill="none"' if hollow else f'fill="{colour}"'
        parts.append(f'<circle cx="{x}" cy="48" r="4.5" {ring} stroke="{colour}" stroke-width="1.6"/>')
        parts.append(f'<text x="{x + 9}" y="52" class="key">{legend}</text>')
        x += 9 + 6 * len(legend) + 22  # about 6 px a character at this size, then a gap
    for tick in ticks:
        parts.append(f'<line x1="{left}" y1="{y_of(tick):.1f}" x2="{right}" y2="{y_of(tick):.1f}" stroke="{GRID}"/>')
        parts.append(f'<text x="{left - 6}" y="{y_of(tick) + 4:.1f}" class="tick" text-anchor="end">{tick}</text>')
    for x, quarter in zip(xs, quarters):
        parts.append(f'<text x="{x:.1f}" y="{BASE + 16}" class="tick" text-anchor="middle">{quarter}</text>')
    for _, _, values, colour, hollow in series:
        points = " ".join(f"{x:.1f},{y_of(value):.1f}" for x, value in zip(xs, values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="{1.4 if hollow else 2}" '
                     f'stroke-opacity="{0.55 if hollow else 1}" stroke-linejoin="round" stroke-linecap="round"/>')
    for _, _, values, colour, hollow in series:
        for x, value in zip(xs, values):
            if hollow:
                parts.append(f'<circle cx="{x:.1f}" cy="{y_of(value):.1f}" r="6.5" fill="none" stroke="{colour}" stroke-width="1.6"/>')
            else:
                parts.append(f'<circle cx="{x:.1f}" cy="{y_of(value):.1f}" r="6.5" fill="#ffffff"/>')
                parts.append(f'<circle cx="{x:.1f}" cy="{y_of(value):.1f}" r="4.5" fill="{colour}" stroke="{colour}" stroke-width="1.6"/>')
    # the last value, named at the right; a hollow series sharing its end with another moves apart from it
    labels = [[y_of(values[-1]) + 4, f"{label} {values[-1]}", hollow] for _, label, values, _, hollow in series]
    for mine in labels:
        for other in labels:
            if mine is not other and mine[2] and not other[2] and abs(mine[0] - other[0]) < 12:
                mine[0], other[0] = mine[0] - 8, other[0] + 10
    for y, text, _ in labels:
        parts.append(f'<text x="{right + 12}" y="{y:.1f}" class="value">{text}</text>')
    return parts


def figure4_svg(replay: dict) -> tuple[str, int]:
    quarters = replay["areas"][REPLAY_AREA]["quarters"]
    names = [quarter["quarter"] for quarter in quarters]
    alarms = lambda source: [quarter["model"]["false_alarms"][source] for quarter in quarters]  # noqa: E731
    changed = lambda site: [quarter["query"]["changed_calls_per_site"][site] for quarter in quarters]  # noqa: E731
    parts = line_panel(0, "A", "False positives by quarter",
                       f'pooled model refitted each quarter, of {quarters[0]["model"]["discordant_rows"]} discordant benign test variants',
                       names, [("public reference", "public reference", alarms("public"), MUTED, False),
                               ("federated query", "federated query", alarms("federated"), ACCENT, False),
                               ("ceiling", "ceiling", alarms("ceiling"), ACCENT, True)])
    parts += line_panel(PANEL_B, "B", "Readings changed by the count query",
                        f'for a patient at each hospital, of {quarters[0]["query"]["variants"]:,} missense variants',
                        names, [("patient at Oslo", "Oslo", changed("site_oslo"), ACCENT, False),
                                ("at Karachi", "Karachi", changed("site_karachi"), INK, False),
                                ("at Lagos", "Lagos", changed("site_lagos"), MUTED, False)])
    height = BASE + 30
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}">' + "".join(parts) + "</svg>", height


def figure4() -> tuple[str, int] | None:
    if not REPLAY.exists():
        print(f"Figure 4 skipped: {REPLAY.relative_to(ROOT)} is missing. Run scripts/05_quarterly_replay.py.")
        return None
    return page(4, "docs/quarterly_replay.json", *figure4_svg(json.loads(REPLAY.read_text(encoding="utf-8"))))


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


def write_figure(number: int, html: str, height: int) -> None:
    source = DOCS / f"manuscript_figure{number}.html"
    source.write_text(html, encoding="utf-8", newline="\n")
    run_browser("--hide-scrollbars", "--force-device-scale-factor=2", f"--window-size={WIDTH + 40},{height}",
                f"--screenshot={DOCS / f'manuscript_figure{number}.png'}", source.as_uri())
    print(f"wrote docs/manuscript_figure{number}.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw Figures 2 to 4 and print the manuscript to PDF")
    parser.add_argument("--figure-only", action="store_true", help="skip the PDF")
    args = parser.parse_args()

    if not CHECKS.exists():
        print("data/results_checks.json is missing. Run scripts/03_check_results.py first.", file=sys.stderr)
        return 1

    write_figure(2, *figure2())
    for number, figure in ((3, figure3), (4, figure4)):
        drawn = figure()
        if drawn:
            write_figure(number, *drawn)

    if not args.figure_only:
        with tempfile.TemporaryDirectory() as folder:
            printable = Path(folder) / "manuscript.html"
            printable.write_text(manuscript_html(), encoding="utf-8")
            run_browser("--no-pdf-header-footer", f"--print-to-pdf={ROOT / 'Manuscript.pdf'}", printable.as_uri())
        print("wrote Manuscript.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
