"""Step 6, interactive: a patient carries a variant. Is it harmful? Ask the other hospitals.

Usage:
    uv run python scripts/06_query_tui.py

Needs a real terminal: Windows Terminal, the VS Code terminal, or any Mac or Linux
terminal. It reflows when the window is resized: bars grow and shrink with the width,
and below about 90 columns the variant picker moves above the verdict.

    show         dropdown: demo examples, variants whose call CHANGES once the other
                 hospitals answer, variants kept flagged, or everything
    gene         dropdown: narrow to one gene
    kind         dropdown: missense only, one of the other mutation types, or every kind.
                 The other kinds exist once data/other_types/ is built
    type         filter by name, or paste a DNA change in any of its valid spellings
    up / down    move through the list; the hospitals answer as you go
    Tab          move between the dropdowns, the search box and the list
    F2           hide counts under 5 or not, to see what privacy costs
    F3           move the patient to the next hospital
    Escape       quit

The screen, top to bottom: pick a variant; the call; one chart with a row per source
of evidence, every bar on the same axis; what crossed hospital walls.

This file is only the app: widgets, keys and filtering.
    scripts/hospital_query.py   the logic, shared with 06_query_variant.py
    scripts/query_drawing.py    how everything looks
"""

from __future__ import annotations

import pandas as pd
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Resize
from textual.widgets import Footer, Header, Input, OptionList, Select, Static
from textual.widgets.option_list import Option

from hospital_query import DEFAULT_MIN_COUNT, MISSENSE, QueryResult, examples, list_sites, overview, query, resolve
from query_drawing import CALLS, FAINT, evidence_chart, list_row, verdict_panel, verdict_title, what_travelled_line

ANY_GENE = "any gene"
ANY_KIND = "any kind"
DEFAULT_KIND = ANY_KIND  # the full list filters fast enough to start from every kind
MAX_LISTED = 300  # rows put in the list at once; the counter says how many matched
NARROW_BELOW = 90  # columns: under this, the picker sits above the verdict instead of beside it
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


class PatientQuery(App):
    TITLE = "Patient query"
    SUB_TITLE = "is this patient's variant harmful? ask the other hospitals"

    CSS = """
    #top { height: 13; }
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

    Screen.narrow #top { layout: vertical; height: auto; }
    Screen.narrow #picker { width: 1fr; height: 10; }
    Screen.narrow #verdict { height: auto; padding: 0 2; }
    """

    BINDINGS = [
        ("f2", "toggle_hiding", "Hide counts under 5"),
        ("f3", "move_patient", "Move the patient"),
        ("escape", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
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
        self.matches: list[str] = []
        self.result: QueryResult | None = None

    # ------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        genes = [ANY_GENE] + sorted(overview().gene.unique())
        kinds = [ANY_KIND, MISSENSE] + sorted(set(overview().mutation_type.unique()) - {MISSENSE})
        yield Header()
        with Horizontal(id="top"):
            with Vertical(id="picker"):
                yield Select(self.show_options(), value=self.show, allow_blank=False, compact=True, id="show")
                yield Select([(gene, gene) for gene in genes], value=ANY_GENE, allow_blank=False, compact=True, id="gene")
                yield Select([(kind.replace("_", " "), kind) for kind in kinds], value=self.kind, allow_blank=False, compact=True, id="kind")
                yield Input(placeholder="search by name, e.g. N1526")
                yield Static(id="counter")
                yield OptionList(id="matches")
            yield Static(id="verdict")
        yield Static(id="evidence")
        yield Static(id="what-travelled")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#picker").border_title = "pick a variant"
        self.query_one("#evidence").border_title = "how common is it?"
        self.query_one(Input).focus()  # so typing searches and the arrow keys move the list straight away
        self.refresh_list()

    def on_resize(self, event: Resize) -> None:
        """Reflow: stack the top row when narrow, and redraw the chart at its new width."""
        self.width = event.size.width
        self.screen.set_class(self.width < NARROW_BELOW, "narrow")
        if self.result:
            self.draw(self.result)

    # ------------------------------------------------------------------ the list of variants
    def show_options(self) -> list[tuple[str, str]]:
        """The 'show' dropdown, with live counts. They depend on where the patient is and on the hiding."""
        table = overview(self.patient_at, self.min_count)
        return [
            ("demo examples", "examples"),
            (f"changed by asking ({int(table.changed.sum()):,})", "changed"),
            (f"kept flagged ({int((table.after == 'KEEP FLAGGED').sum()):,})", "flagged"),
            (f"all variants ({len(table):,})", "all"),
        ]

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
        options.add_options([Option(list_row(name, call)) for name, call in zip(listed.name, listed.after, strict=True)])
        if self.matches:
            options.highlighted = min(stay_on, len(self.matches) - 1)  # this fires variant_highlighted
        else:
            self.show_nothing_found()

    @property
    def variant(self) -> str | None:
        index = self.query_one("#matches", OptionList).highlighted
        return self.matches[index] if index is not None and index < len(self.matches) else None

    # ------------------------------------------------------------------ events
    @on(Select.Changed)
    def dropdown_changed(self, event: Select.Changed) -> None:
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

    def action_toggle_hiding(self) -> None:
        self.min_count = 0 if self.min_count else DEFAULT_MIN_COUNT
        self.settings_changed()

    def action_move_patient(self) -> None:
        self.patient_at = self.sites[(self.sites.index(self.patient_at) + 1) % len(self.sites)]
        self.settings_changed()

    def settings_changed(self) -> None:
        dropdown = self.query_one("#show", Select)
        dropdown.set_options(self.show_options())
        dropdown.value = self.show
        self.refresh_list()

    # ------------------------------------------------------------------ asking and drawing
    def ask(self) -> None:
        if self.variant:
            self.result = query(self.variant, self.patient_at, self.min_count)
            self.draw(self.result)

    def draw(self, result: QueryResult) -> None:
        verdict = self.query_one("#verdict", Static)
        verdict.set_classes(CALLS[result.after.call].css_class)
        verdict.border_title = verdict_title(result)
        verdict.update(verdict_panel(result, self.sites))

        self.query_one("#evidence", Static).update(evidence_chart(result, self.width - CHART_CHROME))
        self.query_one("#what-travelled", Static).update(what_travelled_line(result, self.min_count))

    def show_nothing_found(self) -> None:
        self.result = None
        verdict = self.query_one("#verdict", Static)
        verdict.set_classes("")
        verdict.border_title = ""
        verdict.update(Text("no variant matches", style=FAINT))
        self.query_one("#evidence", Static).update("")


if __name__ == "__main__":
    PatientQuery().run()
