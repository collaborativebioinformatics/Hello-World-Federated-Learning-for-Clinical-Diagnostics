"""Step 6, checked on screen: drive the patient query without a terminal, and picture it.

Every scenario runs the real app through Textual's headless test harness, presses the
keys a clinician would press, and checks what ends up on the screen. Four of them are
also exported as pictures for the docs, the way the README picture was made: the
screen as SVG, then headless Chrome or Edge turns the SVG into a PNG.

    start            the heart area opens on the examples, with the model line and rule 1's line
    search           typing narrows the list; the arrow keys move through it and the hospitals answer
    move the patient F3 moves the patient to the next hospital and back round
    hiding           F2 shows the counts under 5, and says so
    area             the dropdown switches to inherited cancer: its genes, its examples, its two-copy line
    help             F1 and ? open the help overlay, any key closes it
    one deletion     the MYBPC3 deletion is found by either of its ids
    small window     80 x 24 still shows the call and the chart

Reads   data/ (the heart area) and data/cancer/ (skipped when it is not built)
Writes  docs/patient_query_tui.png, docs/patient_query_tui_cancer.png,
        docs/patient_query_tui_help.png, docs/patient_query_tui_other_types.png

Usage:
    uv run python scripts/06_query_tui_check.py               # check, and write the pictures into docs/
    uv run python scripts/06_query_tui_check.py --out DIR     # the pictures somewhere else
    uv run python scripts/06_query_tui_check.py --no-png      # SVG only, on a machine without Chrome or Edge
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import io
import shutil
import subprocess
import sys
import time
from pathlib import Path

from rich.console import Console
from textual.widgets import Input, Select, Static

import variant_spelling
from hospital_query import DEFAULT_AREA, available_areas, examples, set_area

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# The app lives in a file whose name starts with a digit, so it is loaded by path.
_spec = importlib.util.spec_from_file_location("query_tui", Path(__file__).with_name("06_query_tui.py"))
tui = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tui)

BROWSERS = [
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser", "chrome",
]
WIDE, WIDE_PIXELS = (120, 38), (1482, 980)  # the size of docs/patient_query_tui_other_types.png, and its pixels
README_SIZE, README_PIXELS = (118, 32), (1226, 700)  # docs/patient_query_tui.png: 118 x 32, drawn 1226 pixels wide


class Check:
    """One scenario: what it did, what it found, and the pictures it took."""

    def __init__(self, out: Path, png: bool) -> None:
        self.out, self.png = out, png
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.pictures: list[str] = []

    async def run(self, name: str, scenario, size=(118, 36), area: str = DEFAULT_AREA) -> None:
        set_area(DEFAULT_AREA)
        app = tui.PatientQuery(area)
        started = time.time()
        try:
            async with app.run_test(size=size) as pilot:
                await pilot.pause()
                await scenario(app, pilot)
                await pilot.pause()
        except AssertionError as problem:
            self.failed.append(f"{name}: {problem}")
            print(f"  FAIL  {name}: {problem}")
            return
        self.passed.append(name)
        print(f"  ok    {name}  ({time.time() - started:.1f}s)")

    def picture(self, app, name: str, pixels: tuple[int, int] = WIDE_PIXELS) -> None:
        """The screen as SVG, then as PNG of the given size when a browser is there. The SVG scales to fit."""
        svg = self.out / f"{name}.svg"
        svg.write_text(app.export_screenshot(title="Patient query"), encoding="utf-8")
        self.pictures.append(svg.name)
        if not self.png:
            return
        browser = find_browser()
        if not browser:
            print("  no Chrome or Edge found: SVG only")
            self.png = False
            return
        png = self.out / f"{name}.png"
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={pixels[0]},{pixels[1]}",
                        f"--screenshot={png}", svg.as_uri()], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        svg.unlink()
        self.pictures[-1] = png.name


def find_browser() -> str | None:
    for candidate in BROWSERS:
        found = candidate if Path(candidate).exists() else shutil.which(candidate)
        if found:
            return str(found)
    return None


def on_screen(app, selector: str) -> str:
    """The plain text a widget shows, rendered wide so that nothing wraps."""
    widget = app.query_one(selector, Static)
    content = widget.content if hasattr(widget, "content") else widget.renderable
    if isinstance(content, str):
        return content
    console = Console(width=300, file=io.StringIO(), record=True, force_terminal=False, color_system=None)
    console.print(content)
    return console.export_text()


# ---------------------------------------------------------------------------
# The scenarios
# ---------------------------------------------------------------------------
async def start(app, pilot, check: Check):
    assert app.matches == examples(), f"the list should open on the examples, got {app.matches}"
    assert app.result and app.result.name == "DSP N1526K", "the first example should be answered straight away"
    verdict = on_screen(app, "#verdict")
    assert "line 0.1%" in verdict, "rule 1's line is missing from the verdict panel"
    assert "trained model puts this at" in verdict, "the model line is missing from the verdict panel"
    assert "◇ missense" in verdict or "missense" in verdict, "the mutation type is missing"
    assert "Oslo" in app.query_one("#verdict").border_title and "20,000 patients" in app.query_one("#verdict").border_title
    chart = on_screen(app, "#evidence")
    assert "public database" in chart and "Lagos" in chart


async def readme_picture(app, pilot, check: Check):
    """The README picture: the variants whose call changes once the other hospitals answer, missense only."""
    app.query_one("#show", Select).value = "changed"
    app.query_one("#kind", Select).value = "missense"
    await pilot.pause()
    assert app.matches and app.result, "the changed list should have a first row"
    check.picture(app, "patient_query_tui", README_PIXELS)


async def search(app, pilot, check: Check):
    await pilot.click(Input)
    await pilot.press(*"n1526")
    await pilot.pause()
    assert "DSP N1526K" in app.matches, f"typing n1526 should find DSP N1526K, list is {app.matches[:5]}"
    assert on_screen(app, "#counter").endswith("variant") or "variants" in on_screen(app, "#counter")
    before = app.result.name
    await pilot.press("down")
    await pilot.pause()
    assert len(app.matches) == 1 or app.result.name != before, "the down arrow should move to the next variant"
    await pilot.press("ctrl+a", "delete")  # the search box takes ctrl+a
    app.query_one(Input).value = ""
    await pilot.pause()
    assert app.matches == examples(), "clearing the search should bring the examples back"


async def move_the_patient(app, pilot, check: Check):
    first = app.patient_at
    await pilot.press("f3")
    await pilot.pause()
    assert app.patient_at != first and app.result.patient_at == app.patient_at
    assert "Karachi" in app.query_one("#verdict").border_title
    assert "4,000 patients" in app.query_one("#verdict").border_title
    for _ in range(len(app.sites) - 1):
        await pilot.press("f3")
    await pilot.pause()
    assert app.patient_at == first, "F3 should come back round to the first hospital"


async def hiding(app, pilot, check: Check):
    hidden_before = any(a.healthy.hidden or a.sick.hidden for a in app.result.answers)
    await pilot.press("f2")
    await pilot.pause()
    assert app.min_count == 0
    assert "NOT hidden" in on_screen(app, "#what-travelled")
    assert not any(a.healthy.hidden or a.sick.hidden for a in app.result.answers), "no count should be hidden after F2"
    assert hidden_before, "DSP N1526K at Oslo should have had a hidden count to reveal"
    await pilot.press("f2")
    await pilot.pause()
    assert app.min_count == 5 and "NOT hidden" not in on_screen(app, "#what-travelled")


async def area(app, pilot, check: Check):
    app.query_one("#area", Select).value = "cancer"
    await pilot.pause()
    assert app.area == "cancer", "the area dropdown should switch the area"
    assert app.matches == examples(), f"the cancer examples should be listed, got {app.matches}"
    genes = {value for _, value in app.query_one("#gene", Select)._options}
    assert "MUTYH" in genes and "DSP" not in genes, "the gene dropdown should hold the cancer genes"
    assert app.result and app.result.area == "cancer"
    check.picture(app, "patient_query_tui_cancer")
    app.query_one("#gene", Select).value = "MUTYH"
    await pilot.pause()
    assert all(name.startswith("MUTYH") for name in app.matches)
    assert "line 1%" in on_screen(app, "#verdict"), "MUTYH needs two bad copies, so the line should be 1%"
    assert "both copies" in on_screen(app, "#verdict")
    app.query_one("#area", Select).value = "cardiac"
    await pilot.pause()
    assert app.area == "cardiac" and app.matches == examples(), "switching back should restore the heart examples"


async def help_overlay(app, pilot, check: Check):
    await pilot.press("f1")
    await pilot.pause()
    assert isinstance(app.screen, tui.Help), "F1 should open the help overlay"
    text = on_screen(app.screen, "Static")
    assert "F3" in text and "both copies" in text and "20,000" in text
    check.picture(app, "patient_query_tui_help")
    await pilot.press("escape")
    await pilot.pause()
    assert not isinstance(app.screen, tui.Help), "Escape should close the help"
    await pilot.press("question_mark")
    await pilot.pause()
    assert isinstance(app.screen, tui.Help), "? should open the help overlay too"
    await pilot.press("space")
    await pilot.pause()
    assert not isinstance(app.screen, tui.Help), "any key should close the help"


async def one_deletion(app, pilot, check: Check):
    canonical = variant_spelling.MYBPC3_AS_CLINVAR_AND_GNOMAD
    app.query_one("#show", Select).value = "all"
    for spelled in (variant_spelling.MYBPC3_AS_DBSNP, canonical):
        app.query_one(Input).value = spelled
        await pilot.pause()
        assert len(app.matches) == 1, f"{spelled} should list exactly one variant, got {app.matches}"
        assert app.result.variant_id == canonical, f"{spelled} should resolve to {canonical}"
        karachi = next(a for a in app.result.answers if a.site == "site_karachi")
        assert karachi.healthy.frequency > 0.03, "Karachi should report about 3% healthy carriers"
        assert app.result.after.call == "LIKELY HARMLESS"
        if spelled == variant_spelling.MYBPC3_AS_DBSNP:
            check.picture(app, "patient_query_tui_other_types")


async def small_window(app, pilot, check: Check):
    assert app.short and app.screen.has_class("narrow"), "80 x 24 should be both narrow and short"
    assert app.result and "line 0.1%" in on_screen(app, "#verdict")
    chart = on_screen(app, "#evidence")
    assert "Lagos" in chart and "15.2%" in chart, "the chart should still show the decisive number"
    assert app.chart_width() <= 76, "the chart should be drawn within the panel"


# ---------------------------------------------------------------------------
async def main_async(out: Path, png: bool) -> int:
    print(f"patient query on screen: headless checks, pictures into {out}")
    check = Check(out, png)
    scenarios = [
        ("start", start, (118, 36), DEFAULT_AREA),
        ("README picture", readme_picture, README_SIZE, DEFAULT_AREA),
        ("search and select", search, (118, 36), DEFAULT_AREA),
        ("move the patient", move_the_patient, (118, 36), DEFAULT_AREA),
        ("hiding counts under 5", hiding, (118, 36), DEFAULT_AREA),
        ("help overlay", help_overlay, WIDE, DEFAULT_AREA),
        ("the MYBPC3 deletion by either id", one_deletion, WIDE, DEFAULT_AREA),
        ("80 x 24 window", small_window, (80, 24), DEFAULT_AREA),
    ]
    if "cancer" in available_areas():
        scenarios.insert(5, ("area switch to cancer", area, WIDE, DEFAULT_AREA))
    else:
        print("  skip  area switch: data/cancer/ is not built")
    for name, scenario, size, start_area in scenarios:
        await check.run(name, lambda app, pilot, s=scenario: s(app, pilot, check), size, start_area)

    print(f"\n{len(check.passed)} passed, {len(check.failed)} failed. Pictures: {', '.join(check.pictures) or 'none'}")
    return 1 if check.failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Drive the patient query headlessly and picture it")
    parser.add_argument("--out", type=Path, default=DOCS, help="where the pictures go (default: docs/)")
    parser.add_argument("--no-png", action="store_true", help="write SVG only")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=True)
    return asyncio.run(main_async(args.out, not args.no_png))


if __name__ == "__main__":
    raise SystemExit(main())
