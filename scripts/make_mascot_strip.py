"""Draw Tally, the count courier, in its three verdict moods as one picture for the README.

Reads the frames from scripts/mascot.py, so the picture can never drift from the
screen. Writes docs/tally_moods.html and renders docs/tally_moods.png with the
same headless Chrome the other pictures use.

Usage:
    uv run --with pillow python scripts/make_mascot_strip.py
"""

from __future__ import annotations

import html
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mascot  # noqa: E402

CALLS = [("FREQUENCY SAYS NOTHING", "cannot tell", "#ffa500"), ("LIKELY HARMLESS", "likely harmless", "#9ece3a"), ("KEEP FLAGGED", "keep flagged", "#ff3d7f")]
NUMBERS = {"FREQUENCY SAYS NOTHING": "0", "LIKELY HARMLESS": "15.2%", "KEEP FLAGGED": "1.5%"}
OUT_HTML = ROOT / "docs" / "tally_moods.html"
OUT_PNG = ROOT / "docs" / "tally_moods.png"


def verdict_tick(call: str) -> int:
    """The first tick at which Tally shows the mood for this call."""
    wanted = mascot.VERDICT_MOOD[call]
    for since in range(0, 40):
        if mascot.pose(since, call)[0] == wanted:
            return since
    raise RuntimeError(f"no verdict pose for {call}")


def colour_of(style: str, palette: dict[str, str]) -> str:
    value = palette.get(style, "")
    for word in str(value).split():
        if word.startswith("#"):
            return word
    return "#e6e6e6"


def column(call: str, caption: str, colour: str) -> str:
    since = verdict_tick(call)
    rows = mascot.lines(since, call, NUMBERS[call])
    palette = mascot.styles(call)
    text = []
    for row in rows:
        # keep the drawing only; the caption below says the words
        chars = [(ch, style) for piece, style in row for ch in piece][-mascot.ART_WIDTH:]
        text.append("".join(f'<span style="color:{colour_of(style, palette)}">{html.escape(ch)}</span>' for ch, style in chars))
    art = "\n".join(text)
    return f'<div class="tally"><pre>{art}</pre><div class="caption" style="color:{colour}">{caption}</div></div>'


def main() -> None:
    columns = "\n".join(column(*c) for c in CALLS)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Tally</title>
<style>
body{{margin:0;background:#1b1b1b;font-family:"Cascadia Mono","Fira Code",Consolas,Menlo,monospace;color:#e6e6e6}}
.strip{{display:flex;justify-content:space-around;padding:22px 26px 14px}}
.tally{{text-align:center}}
pre{{margin:0;font-size:22px;line-height:1.2;white-space:pre}}
.caption{{margin-top:8px;font-size:15px;font-weight:700;letter-spacing:.02em}}
.name{{padding:0 26px 18px;color:#8a8a8a;font-size:13px;text-align:center}}
</style></head><body><div class="strip">{columns}</div>
<div class="name">Tally, the count courier: carries counts between hospitals, never a patient record</div></body></html>"""
    OUT_HTML.write_text(page, encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location("render_docs_png", ROOT / "scripts" / "render_docs_png.py")
    render = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(render)
    render.render_chrome("tally_moods")
    print(f"wrote {OUT_PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
