"""Step 6: ask every hospital about one variant, and print what comes back.

Usage:
    uv run python scripts/06_query_variant.py "DSP N1526K"
    uv run python scripts/06_query_variant.py "TTR V142I" --patient-at site_lagos
    uv run python scripts/06_query_variant.py "DSP N1526K" --min-count 0   # no small-count hiding
    uv run python scripts/06_query_variant.py "DSP N1526K" --json          # for other programs
    uv run python scripts/06_query_variant.py "MUTYH G63D" --panel cancer  # another disease area

Any small mutation can be asked about once data/other_types/ is built, by name or by DNA change
in any valid spelling. The MYBPC3 deletion filed under two ids shows why the spelling matters:

    uv run python scripts/06_query_variant.py "chr11:g.47332282_47332306del" --patient-at site_karachi
    uv run python scripts/06_query_variant.py "chr11:g.47332282_47332306del" --patient-at site_karachi --no-spelling-fix

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

import variant_spelling
from hospital_query import (
    DEFAULT_AREA, DEFAULT_MIN_COUNT, MISSENSE, QueryResult, Reading, area_title, available_areas, build_command,
    examples, query, set_area,
)

CALL_COLOURS = {"LIKELY HARMLESS": "green", "KEEP FLAGGED": "red", "FREQUENCY SAYS NOTHING": "yellow"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask every hospital how many of its patients carry a variant")
    parser.add_argument("variant", nargs="?", help='a name like "DSP N1526K", or a variant id')
    parser.add_argument("--panel", default=DEFAULT_AREA, help="disease area: cardiac (default) or any area built into data/<area>/")
    parser.add_argument("--patient-at", help="the hospital the patient is at (default: the first one)")
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT, help="hospitals hide counts below this")
    parser.add_argument("--json", action="store_true", help="print the result as data")
    parser.add_argument("--no-spelling-fix", action="store_true",
                        help="send the text as typed, and let each hospital match it against its own lab's spelling only")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a narrower encoding
    console = Console()

    if args.panel not in available_areas():
        built = ", ".join(available_areas()) or "none"
        console.print(f"[red]The {area_title(args.panel)} area is not built. Built areas: {built}. To build it:[/red]\n{build_command(args.panel)}")
        return 1
    set_area(args.panel)

    if not args.variant:
        console.print("Give a variant name. Good ones to try: " + ", ".join(f'"{name}"' for name in examples()))
        return 1
    try:
        result = query(args.variant, args.patient_at, args.min_count, spelling_fix=not args.no_spelling_fix)
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
    console.print(f"{area_title(result.area)} · patient at {result.patient_at} · public database (Europeans only): {result.public_frequency:.2%}")
    if result.mutation_type != MISSENSE:
        console.print(f"mutation type: {result.mutation_type.replace('_', ' ')}" + spellings_line(result))
    if not result.spelling_fix:
        console.print(f"[yellow]spelling fix off: every hospital was asked for '{result.asked_as}' and matched it "
                      "against its own lab's spelling only[/yellow]")
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
    console.print(f"rule 1's {result.line_reason}")

    console.print(reading_panel(f"{result.patient_at} alone, with the public database", result.before))
    console.print(reading_panel("all hospitals together", result.after))
    console.print(f"model: {result.model}")
    if result.same_class:
        console.print(same_class_lines(result))
    console.print("[dim]only counts left each hospital[/dim]")
    console.print()


def spellings_line(result: QueryResult) -> str:
    """' · one change, 8 valid spellings' for an insertion or deletion that can slide; nothing otherwise."""
    try:
        ways = variant_spelling.every_spelling(result.variant_id)
    except (variant_spelling.SpellingError, variant_spelling.NoSequence):
        return ""
    return f" · one change, {len(ways)} valid spellings" if len(ways) > 1 else ""


def same_class_lines(result: QueryResult) -> str:
    """Carriers of any protein-cutting, frameshift or splice-site change in the gene, per hospital."""
    lines = [f"any protein-cutting, frameshift or splice-site change in {result.gene}, whichever it is:"]
    for answer in result.same_class:
        lines.append(f"  {answer.site:<14} healthy {answer.healthy.text():<28} sick {answer.sick.text()}")
    return "\n".join(lines)


def reading_panel(title: str, reading: Reading) -> Panel:
    colour = CALL_COLOURS[reading.call]
    lines = [f"[bold {colour}]{reading.call}[/bold {colour}]"] + [f"  {reason}" for reason in reading.reasons]
    return Panel("\n".join(lines), title=title, title_align="left", border_style=colour)


if __name__ == "__main__":
    raise SystemExit(main())
