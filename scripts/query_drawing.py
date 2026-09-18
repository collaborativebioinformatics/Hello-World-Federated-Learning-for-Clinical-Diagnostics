"""How the patient query looks: the palette, and functions that turn a result into styled text.

There is no logic about variants in here (that is hospital_query.py) and no screen
handling (that is 06_query_tui.py). Every function takes data and returns Rich text.

Design rules this file holds itself to:

    Show, don't tell.   A frequency is a bar, a call is a coloured badge. Words are for
                        names and for the one fact behind the call.
    One shared axis.    Every source is a row of the same chart, so bars line up and the
                        eye compares them directly. The axis is labelled where it matters:
                        on the "too common to cause a rare disease" tick, which sits where
                        the gene's own line is. No legend needed.
    One job per colour. green / red / amber = a call, nothing else. Blue = where the
                        patient is. Greys = hierarchy: what decides the call is brightest.
    Fit the window.     The chart is given the width it has and sizes its bars to it;
                        when it is too narrow for bars, the numbers remain.
    Clinician words.    Nothing on the screen names a file, a flag or a method.

Every glyph used here is a plain text symbol from the blocks every terminal font carries.
Set the environment variable PATIENT_QUERY_PLAIN=1 to draw words instead of symbols.
"""

from __future__ import annotations

import os
import re
import sys
from math import log10
from typing import NamedTuple

from rich.console import Group
from rich.table import Table
from rich.text import Text

from hospital_query import (
    MISSENSE, TOO_COMMON, Count, HospitalAnswer, QueryResult, area_title, is_common, is_piling_up,
    site_patients, site_population,
)

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
# Glyphs: one per disease area and one per mutation type, each with its words
# ---------------------------------------------------------------------------
AREAS = {"cardiac": ("♥", "heart"), "cancer": ("◉", "inherited cancer"), "all": ("⊕", "all panels")}
KINDS = {
    "missense": ("◇", "missense"),
    "missense_no_scores": ("◇", "missense, no scores"),
    "protein_cutting": ("■", "protein cutting"),
    "frameshift": ("»", "frameshift"),
    "splice_site": ("×", "splice site"),
    "splice_region": ("~", "splice region"),
    "synonymous": ("=", "synonymous"),
    "intronic": ("○", "intronic"),
    "utr": ("◦", "untranslated region"),
    "inframe_indel": ("±", "in-frame indel"),
    "start_or_stop_lost": ("∅", "start or stop lost"),
    "other": ("?", "other"),
}


def glyphs_allowed() -> bool:
    """Symbols, unless asked for words or the terminal cannot encode them."""
    if os.environ.get("PATIENT_QUERY_PLAIN"):
        return False
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    if not sys.stdout.isatty():
        return True
    try:
        "".join(g for g, _ in [*AREAS.values(), *KINDS.values()]).encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


USE_GLYPHS = glyphs_allowed()


def area_label(area: str) -> str:
    """'♥ heart' for the dropdown. An area with no glyph of its own shows its title."""
    glyph, words = AREAS.get(area, ("◌", area_title(area).lower()))
    return f"{glyph} {words}" if USE_GLYPHS else words


def kind_glyph(mutation_type: str) -> str:
    return KINDS.get(mutation_type, ("?", ""))[0] if USE_GLYPHS else ""


def kind_words(mutation_type: str) -> str:
    return KINDS.get(mutation_type, (None, mutation_type.replace("_", " ")))[1]


def kind_label(mutation_type: str) -> str:
    """'◇ missense' for the dropdown and the verdict panel."""
    glyph = kind_glyph(mutation_type)
    return f"{glyph} {kind_words(mutation_type)}" if glyph else kind_words(mutation_type)


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


def bar(frequency: float | None, colour: str, width: int, too_common: float = TOO_COMMON) -> Text:
    """A frequency as a bar, with the gene's 'too common' line as a tick.
    None means the hospital hid the count: something is there, it will not say how much."""
    if width == 0:
        return Text()
    if frequency is None:
        return Text("┄┄┄", style=LABEL) + Text(" " * (width - 3))
    filled, tick = position(frequency, width), position(too_common, width)
    drawn = Text()
    for i in range(width):
        if i == tick:
            drawn.append("┃", f"bold {BRIGHT}" if i < filled else LABEL)
        elif i < filled:
            drawn.append("━", colour)
        else:
            drawn.append("─", TRACK)
    return drawn


def line_label(too_common: float) -> str:
    """'0.1%' or '1%': the line as it is spoken."""
    return f"{too_common:.1%}" if too_common < 0.01 else f"{too_common:.0%}"


def axis(width: int, too_common: float = TOO_COMMON) -> Text:
    """The axis header: 'rare' and 'common' at the ends, the gene's line labelled on its tick."""
    if width == 0:
        return Text()
    tick_label = f"┃{line_label(too_common)}"
    tick = position(too_common, width)
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


def hospital_words(site: str) -> str:
    """'Oslo · European · 20,000 patients'."""
    return f"{short(site)} · {POPULATIONS.get(site_population(site), site_population(site))} · {site_patients(site):,} patients"


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


def list_row(name: str, call: str, mutation_type: str = MISSENSE) -> Text:
    """A row of the variant list: a dot in the call's colour, the mutation type's glyph, the gene quiet, the change clear."""
    gene, _, change = name.partition(" ")
    glyph = kind_glyph(mutation_type)
    return Text.assemble(("● ", CALLS[call].colour), (glyph + " " if glyph else "", LABEL), (gene + " ", LABEL), (change, PLAIN))


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
    return f"patient at [b {PATIENT_HERE}]{hospital_words(result.patient_at)}[/] · is this variant harmful?"


def verdict_panel(result: QueryResult, sites: list[str], compact: bool = False) -> Group:
    """The variant, its kind, the call alone and together, and the three lines that explain it:
    the fact behind the call, rule 1's line for the gene, and the trained model's opinion."""
    blank = [] if compact else [Text()]
    lines = [
        variant_title(result.name, result.variant_id),
        kind_line(result),
        *blank,
        Text.assemble(
            (f"{short(result.patient_at)} alone  ", LABEL), badge(result.before.call), ("   ──▶   ", FAINT),
            (f"all {len(sites)} hospitals  ", LABEL), badge(result.after.call),
        ),
        *blank,
        emphasise_numbers(with_short_names(result.after.headline, sites)),
        emphasise_numbers(result.line_reason),
        model_line(result),
    ]
    if result.same_class:
        lines.append(same_class_line(result))
    return Group(*lines)


def kind_line(result: QueryResult) -> Text:
    """'■ protein cutting  ·  starts out presumed harmful'. The call above can overrule the presumption."""
    line = Text.assemble((kind_label(result.mutation_type), PLAIN))
    if result.presumption:
        line.append_text(Text.assemble(("  ·  ", FAINT), (result.presumption.split(",")[0], LABEL)))
    return line


def model_line(result: QueryResult) -> Text:
    """The trained model's opinion, its numbers bright, its call in the call's colour."""
    text = emphasise_numbers(result.model)
    if result.model_call == "likely harmful":
        text.highlight_words(["likely harmful"], "bold red")
    elif result.model_call == "likely harmless":
        text.highlight_words(["likely harmful"], PLAIN)
    return text


def same_class_line(result: QueryResult) -> Text:
    """For a change that cuts the protein, shifts the frame or breaks a splice site: carriers of ANY such change in the gene."""
    def counts(pick) -> Text:
        parts = Text()
        for answer in result.same_class:
            count = pick(answer)
            value = f"<{count.hidden_below}" if count.hidden else percent(count.frequency)
            parts.append_text(Text.assemble((short(answer.site) + " ", LABEL), (value, PLAIN), ("  ", FAINT)))
        return parts

    line = Text.assemble((f"any cutting, frameshift or splice change in {result.gene}  ", LABEL), ("healthy ", FAINT))
    line.append_text(counts(lambda a: a.healthy))
    line.append_text(Text("sick ", style=FAINT))
    line.append_text(counts(lambda a: a.sick))
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
    # With no room for a bar, the name sits over the percentage instead. A short bar gets a short name.
    healthy_words = "healthy patients" if layout.healthy_bar >= 16 else "healthy"
    sick_words = "sick patients" if layout.sick_bar >= 16 else "sick"
    over_healthy_bar, over_healthy_number = (healthy_words, "") if layout.healthy_bar else ("", "healthy")
    over_sick_bar, over_sick_number = (sick_words, "") if layout.sick_bar else ("", "sick")
    lab_header = "lab" if layout.compact else "lab verdict"
    chart.add_row("", Text(over_healthy_bar, style=LABEL), Text(over_healthy_number, style=LABEL),
                  Text(over_sick_bar, style=LABEL), Text(over_sick_number, style=LABEL), Text(lab_header, style=LABEL))
    if layout.healthy_bar:
        chart.add_row("", axis(layout.healthy_bar, result.too_common), "", axis(layout.sick_bar, result.too_common), "", "")

    chart.add_row(*public_database_row(result, layout))
    for answer in result.answers:
        chart.add_row(*hospital_row(answer, answer.site == result.patient_at, layout, result.too_common))
    return chart


def public_database_row(result: QueryResult, layout: ChartLayout) -> list[Text]:
    decisive = result.public_frequency >= result.too_common
    name = Text("  public db", style=PLAIN) if layout.compact else Text.assemble(("  public database ", PLAIN), ("Europeans", FAINT))
    return [
        name,
        bar(result.public_frequency, "green" if decisive else PLAIN, layout.healthy_bar, result.too_common),
        Text(percent(result.public_frequency), style=f"bold {BRIGHT}" if decisive else PLAIN),
        Text(), Text(), Text(),
    ]


def hospital_row(answer: HospitalAnswer, patient_is_here: bool, layout: ChartLayout, too_common: float = TOO_COMMON) -> list[Text]:
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

    return [source, *measured(answer.healthy, "green", is_common(answer, too_common), layout.healthy_bar, too_common),
            *measured(answer.sick, "red", is_piling_up(answer, too_common), layout.sick_bar, too_common), lab]


def measured(count: Count, colour: str, decisive: bool, width: int, too_common: float = TOO_COMMON) -> tuple[Text, Text]:
    """A bar and its percentage. Only a number that decides the call gets colour and brightness."""
    value = f"<{count.hidden_below}" if count.hidden else percent(count.frequency)
    value_style = f"bold {colour if colour == 'red' else BRIGHT}" if decisive else PLAIN
    return bar(None if count.hidden else count.frequency, colour if decisive else PLAIN, width, too_common), Text(value, style=value_style)


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


# ---------------------------------------------------------------------------
# The help overlay
# ---------------------------------------------------------------------------
KEYS = [
    ("↑ ↓", "move through the list; the hospitals answer as you go"),
    ("Tab", "move between the dropdowns, the search box and the list"),
    ("typing", "searches by name; a DNA change finds its variant in any spelling"),
    ("F1 or ?", "this help"),
    ("F2", "hide counts under 5, or show them: what privacy costs"),
    ("F3", "move the patient to the next hospital"),
    ("F4", "show or hide Tally, the count courier in the corner"),
    ("Esc", "quit"),
]

WHAT_YOU_SEE = [
    ("area", "which disease area's genes and hospitals are asked"),
    ("the call", "the patient's own hospital alone, then all hospitals together"),
    ("the line", "rule 1: too common among healthy people to cause disease. 0.1% when one bad copy"),
    ("", "of the gene is enough, 1% when both copies must be bad: healthy carriers are then expected"),
    ("the pile-up", "rule 2: common among the healthy yet 3x as common among the sick stays flagged"),
    ("the model", "for a missense change, how likely the trained model finds it harmful, against its cut-off"),
    ("", "pooled: trained on every hospital's verdicts put together"),
    ("", "federated: trained across the hospitals, no records shared"),
    ("the chart", "one bar per source on one axis: the public database, then each hospital's healthy and sick"),
    ("the last line", "what crossed hospital walls: one variant name out, counts back, no patient record"),
]


def help_panel(sites: list[str]) -> Group:
    hospitals = " · ".join(f"{short(site)} {POPULATIONS.get(site_population(site), '')} {site_patients(site):,}" for site in sites)
    return Group(
        Text.assemble(("The patient query   ", f"bold {BRIGHT}"),
                      ("a patient carries a variant; their hospital asks the others how many healthy and how many sick patients carry it", PLAIN)),
        Text(),
        Text("keys", style=LABEL),
        *[Text.assemble((f"  {key:<9}", f"bold {BRIGHT}"), (what, PLAIN)) for key, what in KEYS],
        Text(),
        Text("what the screen shows", style=LABEL),
        *[Text.assemble((f"  {name:<15}", f"bold {BRIGHT}"), (what, PLAIN)) for name, what in WHAT_YOU_SEE],
        Text(),
        Text.assemble(("hospitals and patients  ", LABEL), (hospitals, PLAIN)),
        Text("Counts among sick patients are simulated from the known verdict. Research prototype, not for patient care.", style=FAINT),
        Text("press any key to close", style=FAINT),
    )
