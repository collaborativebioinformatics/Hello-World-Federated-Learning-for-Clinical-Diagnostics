"""How the patient query looks: the palette, and functions that turn a result into styled text.

There is no logic about variants in here (that is hospital_query.py) and no screen
handling (that is 06_query_tui.py). Every function takes data and returns Rich text.

Design rules this file holds itself to:

    Show, don't tell.   A frequency is a bar, a call is a coloured badge. Words are for
                        names and for the one fact behind the call.
    One shared axis.    Every source is a row of the same chart, so bars line up and the
                        eye compares them directly. The axis is labelled where it matters:
                        on the "too common to cause a rare disease" tick. No legend needed.
    One job per colour. green / red / amber = a call, nothing else. Blue = where the
                        patient is. Greys = hierarchy: what decides the call is brightest.
    Fit the window.     The chart is given the width it has and sizes its bars to it;
                        when it is too narrow for bars, the numbers remain.
"""

from __future__ import annotations

import re
from math import log10
from typing import NamedTuple

from rich.console import Group
from rich.table import Table
from rich.text import Text

from hospital_query import MISSENSE, MODEL_NOT_CONNECTED, TOO_COMMON, Count, HospitalAnswer, QueryResult, is_common, is_piling_up, site_population

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
PATIENT_HERE = "#5fafff"
BRIGHT = "white"  # the numbers and letters that decide the call
PLAIN = "grey78"  # readable, but not the point
LABEL = "grey58"  # names of things
FAINT = "grey39"  # separators, units
TRACK = "grey23"  # the empty part of a bar


class CallStyle(NamedTuple):
    colour: str
    label: str  # short enough for a badge
    css_class: str  # the verdict panel's border takes this class


CALLS = {
    "LIKELY HARMLESS": CallStyle("green", "LIKELY HARMLESS", "harmless"),
    "KEEP FLAGGED": CallStyle("red", "KEEP FLAGGED", "flagged"),
    "FREQUENCY SAYS NOTHING": CallStyle("yellow", "CANNOT TELL", "cannot-tell"),
}

POPULATIONS = {"nfe": "European", "sas": "South Asian", "afr": "African"}
MAX_STARS = 4

# ---------------------------------------------------------------------------
# The frequency axis: log scale, 0.001% on the left to 100% on the right
# ---------------------------------------------------------------------------
SCALE_LOW, SCALE_HIGH = -5.0, 0.0  # log10 of frequency
MIN_BAR, MAX_BAR = 12, 44  # narrower than this a bar says nothing; wider adds nothing

PERCENT_WIDTH, GAP = 7, 2


class ChartLayout(NamedTuple):
    """How the evidence chart spends the width it is given."""

    source_width: int
    lab_width: int
    healthy_bar: int
    sick_bar: int
    compact: bool  # short source names, lab verdict without its stars


def chart_layout(chart_width: int) -> ChartLayout:
    """As the window narrows: first the names and the stars shrink, then the 'sick' bar
    goes, then the 'healthy' bar. The percentages always stay."""
    for compact in (False, True):
        source_width, lab_width = (14, 12) if compact else (28, 17)
        room = chart_width - (source_width + 2 * PERCENT_WIDTH + lab_width + 5 * GAP)
        if room >= 2 * MIN_BAR:
            bar = min(room // 2, MAX_BAR)
            return ChartLayout(source_width, lab_width, bar, bar, compact)
    if room >= MIN_BAR:
        return ChartLayout(source_width, lab_width, min(room, MAX_BAR), 0, compact=True)
    return ChartLayout(source_width, lab_width, 0, 0, compact=True)


def position(frequency: float, width: int) -> int:
    """Where a frequency sits on a bar of this width, in characters from the left."""
    if frequency <= 0:
        return 0
    clamped = min(max(log10(frequency), SCALE_LOW), SCALE_HIGH)
    return round((clamped - SCALE_LOW) / (SCALE_HIGH - SCALE_LOW) * width)


def bar(frequency: float | None, colour: str, width: int) -> Text:
    """A frequency as a bar. None means the hospital hid the count: something is there, it will not say how much."""
    if width == 0:
        return Text()
    if frequency is None:
        return Text("┄┄┄", style=LABEL) + Text(" " * (width - 3))
    filled, tick = position(frequency, width), position(TOO_COMMON, width)
    drawn = Text()
    for i in range(width):
        if i == tick:
            drawn.append("┃", f"bold {BRIGHT}" if i < filled else LABEL)
        elif i < filled:
            drawn.append("━", colour)
        else:
            drawn.append("─", TRACK)
    return drawn


def axis(width: int) -> Text:
    """The axis header: 'rare' and 'common' at the ends, the threshold labelled on its tick."""
    if width == 0:
        return Text()
    tick_label = f"┃{TOO_COMMON:.1%}"
    tick = position(TOO_COMMON, width)
    cells = [" "] * width
    cells[tick:tick + len(tick_label)] = tick_label
    if tick >= 5:
        cells[0:4] = "rare"
    if width - (tick + len(tick_label)) >= 8:
        cells[width - 6:width] = "common"
    return Text("".join(cells[:width]), style=LABEL)


def percent(frequency: float) -> str:
    if frequency == 0:
        return "0"
    if frequency >= 0.01:
        return f"{frequency:.1%}"
    return f"{frequency:.2%}" if frequency >= 0.0001 else f"{frequency:.3%}"


# ---------------------------------------------------------------------------
# Naming things
# ---------------------------------------------------------------------------
def short(site: str) -> str:
    """site_oslo -> Oslo"""
    return site.removeprefix("site_").capitalize()


def with_short_names(sentence: str, sites: list[str]) -> str:
    for site in sites:
        sentence = sentence.replace(site, short(site))
    return sentence


def emphasise_numbers(sentence: str) -> Text:
    """Quiet sentence, bright numbers: '15.2%' and '5x' are what the reader is looking for."""
    text = Text(sentence, style=PLAIN)
    text.highlight_regex(r"\d[\d.,]*(%|x)", f"bold {BRIGHT}")
    return text


def badge(call: str) -> Text:
    style = CALLS[call]
    return Text(f" {style.label} ", style=f"bold black on {style.colour}")


def stars(how_many: int) -> str:
    """ClinVar's review rating, drawn as a score out of four."""
    return "★" * how_many + "☆" * (MAX_STARS - how_many)


def list_row(name: str, call: str) -> Text:
    """A row of the variant list: a dot in the call's colour, the gene quiet, the change clear."""
    gene, _, change = name.partition(" ")
    return Text.assemble(("● ", CALLS[call].colour), (gene + " ", LABEL), (change, PLAIN))


def variant_title(name: str, variant_id: str) -> Text:
    """'BAG3 C151R' as: the gene, the amino-acid change as before -> after,
    then the DNA address in parts: chromosome, position, letter before -> after."""
    protein_change = re.fullmatch(r"(\S+) ([A-Z*])(\d+)([A-Z*]+|fs|=)", name)
    if not protein_change:
        return Text.assemble((name, f"bold {BRIGHT}"), ("   " + variant_id, FAINT))
    gene, before, place, after = protein_change.groups()

    title = Text.assemble((gene, f"bold {BRIGHT}"), "  ", (before, PLAIN), (place, LABEL), (after, f"bold {BRIGHT}"))
    dna_change = re.fullmatch(r"(chr\w+):g\.(\d+)([ACGT]+)>([ACGT]+)", variant_id)
    if dna_change:
        chromosome, base, was, now = dna_change.groups()
        title.append_text(Text.assemble(
            ("      " + chromosome, LABEL), (" : ", FAINT), (f"{int(base):,}", LABEL),
            ("   " + was, PLAIN), (" → ", FAINT), (now, f"bold {BRIGHT}"),
        ))
    else:
        title.append("      " + variant_id, FAINT)
    return title


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------
def verdict_title(result: QueryResult) -> str:
    return f"a patient at [b {PATIENT_HERE}]{short(result.patient_at)}[/] carries this variant: is it harmful?"


def verdict_panel(result: QueryResult, sites: list[str]) -> Group:
    lines = [
        variant_title(result.name, result.variant_id),
        Text(),
        Text.assemble(
            (f"{short(result.patient_at)} alone  ", LABEL), badge(result.before.call), ("   ──▶   ", FAINT),
            (f"all {len(sites)} hospitals  ", LABEL), badge(result.after.call),
        ),
        Text(),
        emphasise_numbers(with_short_names(result.after.headline, sites)),
    ]
    if result.mutation_type != MISSENSE:
        lines.append(type_line(result))
    if result.model != MODEL_NOT_CONNECTED:
        lines.append(Text(f"trained model says: {result.model}", style=PLAIN))
    return Group(*lines)


def type_line(result: QueryResult) -> Text:
    """'frameshift · starts out presumed harmful' for the other mutation types. The call above can overrule it."""
    line = Text.assemble((result.mutation_type.replace("_", " "), PLAIN))
    if result.presumption:
        line.append_text(Text.assemble(("  ·  ", FAINT), (result.presumption.split(",")[0], LABEL)))
    if not result.spelling_fix:
        line.append_text(Text.assemble(("  ·  ", FAINT), ("spelling fix off", "bold yellow")))
    return line


# ---------------------------------------------------------------------------
# The evidence chart: one row per source, every bar on the same axis
# ---------------------------------------------------------------------------
def evidence_chart(result: QueryResult, chart_width: int) -> Table:
    layout = chart_layout(chart_width)

    chart = Table.grid(padding=(0, GAP))
    chart.add_column(width=layout.source_width, no_wrap=True)
    chart.add_column(width=layout.healthy_bar or None, no_wrap=True)
    chart.add_column(width=PERCENT_WIDTH, justify="right", no_wrap=True)
    chart.add_column(width=layout.sick_bar or None, no_wrap=True)
    chart.add_column(width=PERCENT_WIDTH, justify="right", no_wrap=True)
    chart.add_column(width=layout.lab_width, no_wrap=True)

    # Two header rows: what each column is, then the axis its bars are drawn on.
    # With no room for a bar, the name sits over the percentage instead.
    over_healthy_bar, over_healthy_number = ("healthy patients", "") if layout.healthy_bar else ("", "healthy")
    over_sick_bar, over_sick_number = ("sick patients", "") if layout.sick_bar else ("", "sick")
    lab_header = "lab" if layout.compact else "lab verdict"
    chart.add_row("", Text(over_healthy_bar, style=LABEL), Text(over_healthy_number, style=LABEL),
                  Text(over_sick_bar, style=LABEL), Text(over_sick_number, style=LABEL), Text(lab_header, style=LABEL))
    if layout.healthy_bar:
        chart.add_row("", axis(layout.healthy_bar), "", axis(layout.sick_bar), "", "")

    chart.add_row(*public_database_row(result, layout))
    for answer in result.answers:
        chart.add_row(*hospital_row(answer, answer.site == result.patient_at, layout))
    return chart


def public_database_row(result: QueryResult, layout: ChartLayout) -> list[Text]:
    decisive = result.public_frequency >= TOO_COMMON
    name = Text("  public db", style=PLAIN) if layout.compact else Text.assemble(("  public database ", PLAIN), ("Europeans", FAINT))
    return [
        name,
        bar(result.public_frequency, "green" if decisive else PLAIN, layout.healthy_bar),
        Text(percent(result.public_frequency), style=f"bold {BRIGHT}" if decisive else PLAIN),
        Text(), Text(), Text(),
    ]


def hospital_row(answer: HospitalAnswer, patient_is_here: bool, layout: ChartLayout) -> list[Text]:
    population = "" if layout.compact else " " + POPULATIONS.get(site_population(answer.site), "")
    if patient_is_here:
        source = Text.assemble(("▸ ", PATIENT_HERE), (short(answer.site), f"bold {PATIENT_HERE}"), (population, FAINT))
    else:
        source = Text.assemble(("  " + short(answer.site), PLAIN), (population, FAINT))

    if answer.verdict:
        colour = "red" if answer.verdict == "pathogenic" else "green"
        lab = Text.assemble((f" {answer.verdict} ", f"black on {colour}"))
        if not layout.compact:
            lab.append(" " + stars(answer.stars), LABEL)
    else:
        lab = Text("—", style=FAINT)

    return [source, *measured(answer.healthy, "green", is_common(answer), layout.healthy_bar),
            *measured(answer.sick, "red", is_piling_up(answer), layout.sick_bar), lab]


def measured(count: Count, colour: str, decisive: bool, width: int) -> tuple[Text, Text]:
    """A bar and its percentage. Only a number that decides the call gets colour and brightness."""
    value = f"<{count.hidden_below}" if count.hidden else percent(count.frequency)
    value_style = f"bold {colour if colour == 'red' else BRIGHT}" if decisive else PLAIN
    return bar(None if count.hidden else count.frequency, colour if decisive else PLAIN, width), Text(value, style=value_style)


# ---------------------------------------------------------------------------
# What crossed hospital walls
# ---------------------------------------------------------------------------
def what_travelled_line(result: QueryResult, min_count: int) -> Text:
    """The zero is the point."""
    others = len(result.answers) - 1
    line = Text.assemble(
        ("↑ 1 variant name", LABEL), ("   ", FAINT),
        (f"↓ counts from {others} hospitals", LABEL), ("   ", FAINT),
        ("patient records shared ", LABEL), ("0", f"bold {BRIGHT}"),
    )
    if not min_count:
        line.append("   counts under 5 are NOT hidden", "bold yellow")
    return line
