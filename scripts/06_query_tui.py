"""Step 6, interactive: a patient carries a variant. Is it harmful? Ask the other hospitals.

Usage:
    uv run python scripts/06_query_tui.py
    uv run python scripts/06_query_tui.py --panel cancer     # start on another disease area

Needs a real terminal: Windows Terminal, the VS Code terminal, or any Mac or Linux
terminal. It reflows when the window is resized: bars grow and shrink with the width,
below about 90 columns the variant picker moves above the verdict, and below 30 rows
the header goes and the panels lose their blank lines. At 80 x 24 the screen scrolls.

    area         dropdown: which disease area's genes and hospitals to ask. Only built areas are listed
    show         dropdown: examples, variants whose call CHANGES once the other
                 hospitals answer, variants kept flagged, or everything
    gene         dropdown: narrow to one gene
    kind         dropdown: missense only, one of the other mutation types, or every kind.
                 The other kinds exist once data/other_types/ is built
    type         filter by name, or paste a DNA change in any of its valid spellings
    up / down    move through the list; the hospitals answer as you go
    Tab          move between the dropdowns, the search box and the list
    F1 or ?      help: the keys, and what each part of the screen shows
    F2           hide counts under 5 or not, to see what privacy costs
    F3           move the patient to the next hospital
    F4           show or hide Tally, the count courier in the corner (on by default,
                 remembered for the session; a window under 30 rows has no room for it)
    Escape       quit

The screen, top to bottom: pick a variant; the call, with rule 1's line for the gene and
the trained model's opinion; one chart with a row per source of evidence, every bar on
the same axis; what crossed hospital walls; and Tally, bottom right, who sets off when a
query runs, comes back with the number, reacts to the call and then stands idle.

This file is only the app: widgets, keys and filtering.
    scripts/hospital_query.py   the logic, shared with 06_query_variant.py
    scripts/query_drawing.py    how everything looks
    scripts/mascot.py           Tally's frames and when each shows
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Resize
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Input, OptionList, Select, Static
from textual.widgets.option_list import Option

import mascot
from hospital_query import (
    DEFAULT_AREA, DEFAULT_MIN_COUNT, MISSENSE, QueryResult, area_title, available_areas, build_command, examples,
    list_sites, overview, query, resolve, set_area,
)
from query_drawing import (
    CALLS, FAINT, area_label, evidence_chart, help_panel, kind_label, list_row, percent, verdict_panel, verdict_title,
    what_travelled_line,
)

ANY_GENE = "any gene"
ANY_KIND = "any kind"
DEFAULT_KIND = ANY_KIND  # the full list filters fast enough to start from every kind
MAX_LISTED = 300  # rows put in the list at once; the counter says how many matched
NARROW_BELOW = 90  # columns: under this, the picker sits above the verdict instead of beside it
SHORT_BELOW = 30  # rows: under this, the header goes and the panels lose their blank lines
CHART_CHROME = 6  # the chart panel's border and padding, in columns


def filter_variants(table: pd.DataFrame, show: str, gene: str, text: str, kind: str = ANY_KIND) -> pd.DataFrame:
    """The rows the list should hold, given the three dropdowns and the search box."""
    # A DNA change pasted into the search box finds its variant, whatever spelling it came in.
    if text.startswith("chr"):
        try:
            return table[table.name == resolve(text)[1]]
        except LookupError:
            return table.iloc[:0]
    # Typing or picking a gene or a kind searches everything, not just the handful of examples.
    if show == "examples" and (text or gene != ANY_GENE or kind != ANY_KIND):
        show = "all"

    if show == "examples":
        table = table.set_index("name").loc[examples()].reset_index()
    elif show == "changed":
        table = table[table.changed].sort_values("best_frequency", ascending=False)
    elif show == "flagged":
        table = table[table.after == "KEEP FLAGGED"].sort_values("best_frequency", ascending=False)

    if gene != ANY_GENE:
        table = table[table.gene == gene]
    if kind != ANY_KIND:
        table = table[table.mutation_type == kind]
    if text:
        table = table[table.name.str.lower().str.contains(text, regex=False)]
    return table


class Help(ModalScreen):
    """The keys and what the screen shows. Any key closes it, the arrow keys scroll it first."""

    DEFAULT_CSS = """
    Help { align: center middle; background: $background 70%; }
    Help > VerticalScroll { width: 116; max-width: 100%; height: auto; max-height: 100%;
                            border: round $panel-lighten-2; padding: 0 2; background: $panel; }
    """
    SCROLL_KEYS = {"up", "down", "pageup", "pagedown", "home", "end"}

    def __init__(self, sites: list[str]) -> None:
        super().__init__()
        self.sites = sites

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static(help_panel(self.sites))

    def on_key(self, event) -> None:
        if event.key not in self.SCROLL_KEYS:
            event.stop()  # or Escape would go on to quit the app
            self.dismiss()

    def on_click(self) -> None:
        self.dismiss()


class SearchBox(Input):
    """The search box. A ? opens the help instead of being typed, since no variant name holds one."""

    def on_key(self, event) -> None:
        if event.key == "question_mark":
            event.prevent_default()
            event.stop()
            self.app.action_help()


class PatientQuery(App):
    TITLE = "Patient query"
    SUB_TITLE = "is this patient's variant harmful? ask the other hospitals"

    CSS = """
    #top { height: 15; }
    #picker { width: 34; border: round $panel-lighten-2; padding: 0 1; }
    #picker Input { border: none; height: 1; padding: 0; margin: 1 0 0 0; background: $boost; }
    #counter { height: 1; color: $text-muted; }
    #matches { border: none; background: transparent; padding: 0; height: 1fr;
               scrollbar-size-vertical: 1; scrollbar-background: $panel; scrollbar-color: $panel-lighten-3; }

    #verdict { width: 1fr; border: round $panel-lighten-2; padding: 1 2; }
    #verdict.harmless { border: round $success; }
    #verdict.flagged { border: round $error; }
    #verdict.cannot-tell { border: round $warning; }

    #evidence { height: auto; border: round $panel-lighten-2; padding: 0 2; }
    #what-travelled { height: 1; padding: 0 3; margin-top: 1; }
    #mascot-row { height: 4; padding: 0 3; margin-top: 1; align-horizontal: right; }
    #mascot { width: 40; height: 4; }
    #mascot-row.hidden { display: none; }

    Screen.narrow #top { layout: vertical; height: auto; }
    Screen.narrow #picker { width: 1fr; height: 12; }
    Screen.narrow #verdict { height: auto; padding: 0 2; }

    Screen.short Header { display: none; }
    Screen.short #top { height: 13; }
    Screen.short #picker Input { margin: 0; }
    Screen.short #verdict { padding: 0 2; }
    Screen.short #what-travelled { margin-top: 0; }
    Screen.short #mascot-row { display: none; }
    Screen.short.narrow #top { height: auto; }
    Screen.short.narrow #picker { height: 10; }
    """

    BINDINGS = [
        ("f1", "help", "Help"),
        Binding("question_mark", "help", "Help", show=False, priority=True),  # before the list's own type-to-search
        ("f2", "toggle_hiding", "Hide counts under 5"),
        ("f3", "move_patient", "Move the patient"),
        ("f4", "toggle_mascot", mascot.NAME),
        ("escape", "quit", "Quit"),
    ]

    def __init__(self, area: str = DEFAULT_AREA) -> None:
        super().__init__()
        set_area(area)
        self.area = area
        self.sites = list_sites()
        # the two settings
        self.patient_at = self.sites[0]
        self.min_count = DEFAULT_MIN_COUNT
        # the four filters
        self.show = "examples"
        self.gene = ANY_GENE
        self.kind = DEFAULT_KIND
        self.text = ""
        # what is on screen
        self.width = 118  # columns; kept up to date by on_resize
        self.short = False  # under SHORT_BELOW rows
        self.matches: list[str] = []
        self.result: QueryResult | None = None
        # Tally: ticks since the last query decide the pose; hold pins them (for pictures and checks)
        self.mascot_since = 0
        self.mascot_hold: int | None = None
        self.mascot_shown = True
        self.mascot_drawn = ""

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="top"):
            with Vertical(id="picker"):
                yield Select([(area_label(area), area) for area in available_areas()], value=self.area,
                             allow_blank=False, compact=True, id="area")
                yield Select(self.show_options(), value=self.show, allow_blank=False, compact=True, id="show")
                yield Select(self.gene_options(), value=self.gene, allow_blank=False, compact=True, id="gene")
                yield Select(self.kind_options(), value=self.kind, allow_blank=False, compact=True, id="kind")
                yield SearchBox(placeholder="search by name, e.g. N1526")
                yield Static(id="counter")
                yield OptionList(id="matches")
            yield Static(id="verdict")
        yield Static(id="evidence")
        yield Static(id="what-travelled")
        with Horizontal(id="mascot-row"):
            yield Static(id="mascot")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#picker").border_title = "pick a variant"
        self.query_one("#evidence").border_title = "how common is it?"
        self.query_one(Input).focus()  # so typing searches and the arrow keys move the list straight away
        self.refresh_list()
        self.set_interval(1 / mascot.FPS, self.mascot_tick)

    def on_resize(self, event: Resize) -> None:
        """Reflow: stack the top row when narrow, drop the header when short, and redraw the chart at its new width."""
        self.width = event.size.width
        self.short = event.size.height < SHORT_BELOW
        self.screen.set_class(self.width < NARROW_BELOW, "narrow")
        self.screen.set_class(self.short, "short")
        self.refresh_bindings()  # the F4 hint goes with the mascot when the window is short
        if self.result:
            self.call_after_refresh(self.draw, self.result)  # once the panels have their new sizes

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "toggle_mascot" and self.short:
            return False  # no room for Tally: no key for it either, and the footer stays as it was
        return True

    def chart_width(self) -> int:
        """The columns the chart really has: inside the panel's border and padding, minus any scrollbar."""
        return self.query_one("#evidence", Static).content_region.width or (self.width - CHART_CHROME)

    # ------------------------------------------------------------------ the dropdowns
    def show_options(self) -> list[tuple[str, str]]:
        """The 'show' dropdown, with live counts. They depend on where the patient is and on the hiding."""
        table = overview(self.patient_at, self.min_count)
        return [
            ("examples to try", "examples"),
            (f"changed by asking ({int(table.changed.sum()):,})", "changed"),
            (f"kept flagged ({int((table.after == 'KEEP FLAGGED').sum()):,})", "flagged"),
            (f"all variants ({len(table):,})", "all"),
        ]

    def gene_options(self) -> list[tuple[str, str]]:
        return [(gene, gene) for gene in [ANY_GENE] + sorted(overview().gene.unique())]

    def kind_options(self) -> list[tuple[str, str]]:
        kinds = [MISSENSE] + sorted(set(overview().mutation_type.unique()) - {MISSENSE})
        return [(ANY_KIND, ANY_KIND)] + [(kind_label(kind), kind) for kind in kinds]

    # ------------------------------------------------------------------ the list of variants
    def refresh_list(self) -> None:
        found = filter_variants(overview(self.patient_at, self.min_count), self.show, self.gene, self.text, self.kind)
        listed = found.head(MAX_LISTED)
        self.matches = listed.name.tolist()

        if len(found) > MAX_LISTED:
            counter = f"{MAX_LISTED} of {len(found):,} · type to narrow"
        else:
            counter = f"{len(found):,} variant{'' if len(found) == 1 else 's'}"
        self.query_one("#counter", Static).update(counter)

        options = self.query_one("#matches", OptionList)
        stay_on = options.highlighted or 0
        options.clear_options()
        options.add_options([Option(list_row(name, call, kind))
                             for name, call, kind in zip(listed.name, listed.after, listed.mutation_type, strict=True)])
        if self.matches:
            options.highlighted = min(stay_on, len(self.matches) - 1)  # this fires variant_highlighted
            if options.highlighted == min(stay_on, len(self.matches) - 1):
                self.ask()  # the same row as before fires nothing, yet the answer may have changed
        else:
            self.show_nothing_found()

    @property
    def variant(self) -> str | None:
        index = self.query_one("#matches", OptionList).highlighted
        return self.matches[index] if index is not None and index < len(self.matches) else None

    # ------------------------------------------------------------------ events
    @on(Select.Changed)
    def dropdown_changed(self, event: Select.Changed) -> None:
        if event.select.id == "area":
            self.switch_area(str(event.value))
            return
        if event.select.id == "show":
            self.show = str(event.value)
        elif event.select.id == "gene":
            self.gene = str(event.value)
        else:
            self.kind = str(event.value)
        self.refresh_list()

    @on(Input.Changed)
    def search_as_you_type(self, event: Input.Changed) -> None:
        text = event.value.strip()
        self.text = text if text.lower().startswith("chr") else text.lower()  # a DNA change keeps its letters
        self.refresh_list()

    @on(OptionList.OptionHighlighted, "#matches")
    def variant_highlighted(self) -> None:
        self.ask()

    def on_key(self, event) -> None:
        # Arrow keys move through the list even while the cursor is in the search box.
        if event.key in ("down", "up") and self.focused is self.query_one(Input):
            matches = self.query_one("#matches", OptionList)
            matches.action_cursor_down() if event.key == "down" else matches.action_cursor_up()
            event.prevent_default()

    def action_help(self) -> None:
        self.push_screen(Help(self.sites))

    def action_toggle_hiding(self) -> None:
        self.min_count = 0 if self.min_count else DEFAULT_MIN_COUNT
        self.settings_changed()

    def action_move_patient(self) -> None:
        self.patient_at = self.sites[(self.sites.index(self.patient_at) + 1) % len(self.sites)]
        self.settings_changed()

    def action_toggle_mascot(self) -> None:
        self.mascot_shown = not self.mascot_shown
        self.query_one("#mascot-row").set_class(not self.mascot_shown, "hidden")

    def settings_changed(self) -> None:
        dropdown = self.query_one("#show", Select)
        with self.prevent(Select.Changed):
            dropdown.set_options(self.show_options())
            dropdown.value = self.show
        self.refresh_list()

    def switch_area(self, area: str) -> None:
        """Another disease area: its own genes, hospitals and examples. The filters start over."""
        if area == self.area:
            return
        set_area(area)
        self.area = area
        self.sites = list_sites()
        if self.patient_at not in self.sites:
            self.patient_at = self.sites[0]
        self.gene, self.kind, self.text = ANY_GENE, DEFAULT_KIND, ""
        with self.prevent(Select.Changed, Input.Changed):
            for select_id, options, value in (("gene", self.gene_options(), self.gene), ("kind", self.kind_options(), self.kind)):
                dropdown = self.query_one(f"#{select_id}", Select)
                dropdown.set_options(options)
                dropdown.value = value
            self.query_one(Input).value = ""
        self.settings_changed()

    # ------------------------------------------------------------------ asking and drawing
    def ask(self) -> None:
        if self.variant:
            self.result = query(self.variant, self.patient_at, self.min_count)
            self.draw(self.result)

    def draw(self, result: QueryResult) -> None:
        verdict = self.query_one("#verdict", Static)
        verdict.set_classes(CALLS[result.after.call].css_class)
        verdict.border_title = verdict_title(result)
        verdict.update(verdict_panel(result, self.sites, compact=self.short))

        self.query_one("#evidence", Static).update(evidence_chart(result, self.chart_width()))
        self.query_one("#what-travelled", Static).update(what_travelled_line(result, self.min_count))
        self.restart_mascot()

    def show_nothing_found(self) -> None:
        self.result = None
        verdict = self.query_one("#verdict", Static)
        verdict.set_classes("")
        verdict.border_title = ""
        verdict.update(Text("no variant matches", style=FAINT))
        self.query_one("#evidence", Static).update("")
        self.restart_mascot()

    # ------------------------------------------------------------------ Tally, the count courier
    def restart_mascot(self) -> None:
        """A query just ran: Tally sets off again. Unless held still, for a picture or a check."""
        if self.mascot_hold is None:
            self.mascot_since = 0
        self.draw_mascot()

    def mascot_tick(self) -> None:
        if self.mascot_hold is None:
            self.mascot_since += 1
            self.draw_mascot()

    def hold_mascot(self, since: int | None) -> None:
        """Pin Tally at this many ticks after a query, so that a picture or a check sees one fixed frame;
        None lets it move again. The web demo's TERM.mascotHold() does the same."""
        self.mascot_hold = since
        if since is not None:
            self.mascot_since = since
        self.draw_mascot()

    def draw_mascot(self) -> None:
        """Only redraw when the frame changed: most ticks change nothing."""
        call = self.result.after.call if self.result else None
        number = percent(self.result.best_frequency) if self.result else ""
        text = mascot.render(self.mascot_since, call, number)
        key = f"{call}|{number}|{text.plain}|{mascot.pose(self.mascot_since, call)}"
        if key != self.mascot_drawn:
            self.mascot_drawn = key
            self.query_one("#mascot", Static).update(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="The patient query, on screen")
    parser.add_argument("--panel", default=DEFAULT_AREA, help="disease area to start on: cardiac (default) or any area built into data/<area>/")
    args = parser.parse_args()

    built = available_areas()
    if args.panel not in built:
        where = f" Built areas: {', '.join(built)}." if built else ""
        print(f"The {area_title(args.panel)} area is not built.{where} Build it first:\n{build_command(args.panel)}", file=sys.stderr)
        return 1
    try:
        PatientQuery(args.panel).run()
    except FileNotFoundError as problem:
        print(problem, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
