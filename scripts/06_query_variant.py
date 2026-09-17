"""Step 6: ask every hospital about one variant, and print what comes back.

Usage:
    uv run python scripts/06_query_variant.py "DSP N1526K"
    uv run python scripts/06_query_variant.py "TTR V142I" --patient-at site_lagos
    uv run python scripts/06_query_variant.py "DSP N1526K" --min-count 0   # no small-count hiding
    uv run python scripts/06_query_variant.py "DSP N1526K" --json          # for other programs

The interactive version is scripts/06_query_tui.py. Both use scripts/hospital_query.py,
which explains what runs inside each hospital and what runs where the patient is.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hospital_query import DEFAULT_MIN_COUNT, QueryResult, Reading, examples, query

CALL_COLOURS = {"LIKELY HARMLESS": "green", "KEEP FLAGGED": "red", "FREQUENCY SAYS NOTHING": "yellow"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask every hospital how many of its patients carry a variant")
    parser.add_argument("variant", nargs="?", help='a name like "DSP N1526K", or a variant id')
    parser.add_argument("--patient-at", help="the hospital the patient is at (default: the first one)")
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT, help="hospitals hide counts below this")
    parser.add_argument("--json", action="store_true", help="print the result as data")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a narrower encoding
    console = Console()

    if not args.variant:
        console.print("Give a variant name. Good ones to try: " + ", ".join(f'"{name}"' for name in examples()))
        return 1
    try:
        result = query(args.variant, args.patient_at, args.min_count)
    except (LookupError, ValueError, FileNotFoundError) as problem:
        console.print(f"[red]{problem}[/red]")
        return 1

    if args.json:
        print(json.dumps(asdict(result), indent=2))
    else:
        show(console, result)
    return 0


def show(console: Console, result: QueryResult) -> None:
    console.print()
    console.print(f"[bold]{result.name}[/bold]   [dim]{result.variant_id}[/dim]")
    console.print(f"patient at {result.patient_at} · public database (Europeans only): {result.public_frequency:.2%}")
    console.print()

    table = Table(title="what each hospital answered", title_justify="left", title_style="dim")
    table.add_column("hospital")
    table.add_column("healthy carriers", justify="right")
    table.add_column("sick carriers", justify="right")
    for answer in result.answers:
        style = "bold" if answer.site == result.patient_at else None
        table.add_row(answer.site, answer.healthy.text(), answer.sick.text(), style=style)
    console.print(table)

    judged = [f"{a.site} ({a.verdict}, {a.stars} stars)" for a in result.answers if a.verdict]
    console.print("already judged by: " + (", ".join(judged) or "no hospital"))

    console.print(reading_panel(f"{result.patient_at} alone, with the public database", result.before))
    console.print(reading_panel("all hospitals together", result.after))
    console.print(f"model: {result.model}")
    console.print("[dim]only counts left each hospital[/dim]")
    console.print()


def reading_panel(title: str, reading: Reading) -> Panel:
    colour = CALL_COLOURS[reading.call]
    lines = [f"[bold {colour}]{reading.call}[/bold {colour}]"] + [f"  {reason}" for reason in reading.reasons]
    return Panel("\n".join(lines), title=title, title_align="left", border_style=colour)


if __name__ == "__main__":
    raise SystemExit(main())
