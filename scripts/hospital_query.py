"""The patient query: ask every hospital about one variant, read what comes back.

This is the shared core. scripts/06_query_variant.py (run and print) and
scripts/06_query_tui.py (interactive screen) both call it, so they cannot disagree.
The ML side can import it too: `query("DSP N1526K").best_frequency` is the
frequency to feed a model for a patient whose own hospital knows little about the variant.

How it maps onto a real deployment:

    ask_hospital()   runs INSIDE one hospital. It reads that hospital's own
                     patient_counts.csv and returns a handful of numbers.
    query()          runs at the hospital the patient is at. It only ever sees those numbers.
    read_evidence()  two rules a clinical lab already applies by hand. No model.
    model_verdict()  the slot for the trained model. Empty until the ML side fills it.

Counts among unaffected patients are fair evidence: step 2 draws them from real
population frequencies only. Counts among affected patients were generated using
the verdict, so the pile-up rule is a demonstration of what real hospital data
would allow, and those counts must never be fed to a model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# A hospital never reports a count below this. It says "fewer than 5" instead,
# so that no single patient can be picked out. 0 switches the hiding off.
DEFAULT_MIN_COUNT = 5

# Rule 1: this common among healthy people is too common to cause a rare heart
# disease. Illustrative; clinical labs use disease-specific limits, usually stricter.
TOO_COMMON = 0.001

# Rule 2: the variant "piles up" among the sick when it is this many times more
# frequent in affected than in unaffected patients.
PILE_UP_FOLD = 3.0

MODEL_NOT_CONNECTED = "not connected yet"


@dataclass
class Count:
    carriers: int | None  # None means hidden: some, but fewer than the hospital will report
    total: int
    hidden_below: int = 0

    @property
    def hidden(self) -> bool:
        return self.carriers is None

    @property
    def frequency(self) -> float:
        return 0.0 if self.hidden else self.carriers / self.total

    def text(self) -> str:
        if self.hidden:
            return f"fewer than {self.hidden_below}"
        if not self.carriers:
            return f"none of {self.total:,}"
        return f"{self.carriers:,} of {self.total:,} ({self.frequency:.2%})"


@dataclass
class HospitalAnswer:
    """Everything that leaves a hospital. Counts and, if it has one, its own verdict."""

    site: str
    healthy: Count
    sick: Count
    verdict: str | None = None
    stars: int | None = None


@dataclass
class Reading:
    call: str  # LIKELY HARMLESS, KEEP FLAGGED or FREQUENCY SAYS NOTHING
    reasons: list[str] = field(default_factory=list)
    headline: str = ""  # the one fact behind the call, short enough for a status line


@dataclass
class QueryResult:
    name: str
    variant_id: str
    patient_at: str
    public_frequency: float
    answers: list[HospitalAnswer]
    before: Reading  # the patient's hospital alone: its own patients plus the public database
    after: Reading  # all hospitals together: the same, plus every other hospital's answer
    best_frequency: float  # highest frequency among healthy patients at any hospital
    best_site: str | None
    model: str  # the trained model's verdict, or a note that it is not connected


# ---------------------------------------------------------------------------
# Inside one hospital
# ---------------------------------------------------------------------------
def ask_hospital(site: str, variant_id: str, min_count: int = DEFAULT_MIN_COUNT) -> HospitalAnswer:
    counts = _patient_counts(site)
    row = counts.loc[variant_id]

    def reported(carriers: int, total: int) -> Count:
        return Count(None if 0 < carriers < min_count else int(carriers), int(total), min_count)

    answer = HospitalAnswer(
        site=site,
        healthy=reported(row.ac_unaffected, row.an_unaffected),
        sick=reported(row.ac_affected, row.an_affected),
    )
    verdicts = _verdicts(site)
    if variant_id in verdicts.index:
        answer.verdict = verdicts.loc[variant_id, "verdict"]
        answer.stars = int(verdicts.loc[variant_id, "stars"])
    return answer


# ---------------------------------------------------------------------------
# At the hospital the patient is at
# ---------------------------------------------------------------------------
def query(variant: str, patient_at: str | None = None, min_count: int = DEFAULT_MIN_COUNT) -> QueryResult:
    sites = list_sites()
    patient_at = patient_at or sites[0]
    if patient_at not in sites:
        raise ValueError(f"{patient_at} is not a hospital. Choose one of: {', '.join(sites)}")

    variant_id, name = resolve(variant)
    answers = [ask_hospital(site, variant_id, min_count) for site in sites]

    own_answer = next(a for a in answers if a.site == patient_at)
    most_common = max(answers, key=lambda a: a.healthy.frequency)
    public_frequency = float(_public_reference().get(variant_id, 0.0))

    return QueryResult(
        name=name,
        variant_id=variant_id,
        patient_at=patient_at,
        public_frequency=public_frequency,
        answers=answers,
        # Alone, a doctor has the public database and their own hospital's patients.
        before=read_evidence([own_answer], public_frequency),
        after=read_evidence(answers, public_frequency),
        best_frequency=most_common.healthy.frequency,
        best_site=most_common.site if most_common.healthy.frequency > 0 else None,
        model=model_verdict(variant_id, most_common.healthy.frequency),
    )


def read_evidence(answers: list[HospitalAnswer], public_frequency: float = 0.0) -> Reading:
    """Two rules, no model. Rule 2 can overrule rule 1.

    The public database counts as evidence too: it is what a doctor checks first.
    """
    reasons = []

    piles_up = [a for a in answers if is_piling_up(a)]
    for a in piles_up:
        reasons.append(
            f"piles up among the sick at {a.site}: {a.sick.frequency:.1%} of sick against "
            f"{a.healthy.frequency:.1%} of healthy patients "
            f"({a.sick.frequency / a.healthy.frequency:.0f} times as common)"
        )

    # (how common, a phrase saying where) for every source that clears the variant
    common = [(a.healthy.frequency, f"of healthy patients at {a.site}") for a in answers if is_common(a)]
    if public_frequency >= TOO_COMMON:
        common.append((public_frequency, "of people in the public database"))
    for frequency, where in common:
        reasons.append(f"common: {frequency:.1%} {where} carry it")

    if piles_up:
        worst = max(piles_up, key=lambda a: a.sick.frequency / a.healthy.frequency)
        fold = worst.sick.frequency / worst.healthy.frequency
        return Reading("KEEP FLAGGED", reasons, f"{fold:.0f}x more common in sick than healthy at {worst.site}")
    if common:
        frequency, where = max(common)
        reasons.append("too common among healthy people to cause a rare heart disease")
        return Reading("LIKELY HARMLESS", reasons, f"{frequency:.1%} {where} carry it")

    only_in_sick = [a for a in answers if a.sick.carriers and a.healthy.carriers == 0]
    for a in only_in_sick:
        reasons.append(f"seen in {a.sick.carriers} sick patients at {a.site} and in no healthy ones")
    reasons.append("rare or unseen among healthy patients, so frequency cannot clear it: the scores have to decide")
    return Reading("FREQUENCY SAYS NOTHING", reasons, "rare among healthy patients: the scores must decide")


@cache
def overview(patient_at: str | None = None, min_count: int = DEFAULT_MIN_COUNT) -> pd.DataFrame:
    """The call for every variant at once: name, gene, before, after, changed, best_frequency.

    Same two rules as read_evidence(), applied to whole columns so that a screen can
    list and filter thousands of variants instantly. One row per variant name.
    """
    sites = list_sites()
    patient_at = patient_at or sites[0]

    common, piles_up, healthy = {}, {}, {}
    for site in sites:
        counts = _patient_counts(site)
        healthy_hidden = (counts.ac_unaffected > 0) & (counts.ac_unaffected < min_count)
        sick_hidden = (counts.ac_affected > 0) & (counts.ac_affected < min_count)
        healthy[site] = counts.ac_unaffected.mask(healthy_hidden, 0) / counts.an_unaffected
        sick = counts.ac_affected.mask(sick_hidden, 0) / counts.an_affected
        common[site] = healthy[site] >= TOO_COMMON
        piles_up[site] = common[site] & ~sick_hidden & (sick >= PILE_UP_FOLD * healthy[site])

    index = _patient_counts(sites[0]).index
    common_in_public = pd.Series(_public_reference()).reindex(index).fillna(0.0) >= TOO_COMMON

    def call(among: list[str]) -> pd.Series:
        flagged = pd.concat([piles_up[s] for s in among], axis=1).any(axis=1)
        cleared = pd.concat([common[s] for s in among], axis=1).any(axis=1) | common_in_public
        calls = pd.Series("FREQUENCY SAYS NOTHING", index=index)
        calls[cleared] = "LIKELY HARMLESS"
        calls[flagged] = "KEEP FLAGGED"
        return calls

    names = _names().set_index("variant_id")
    table = pd.DataFrame({
        "name": names.name,
        "gene": names.name.str.split(" ").str[0],
        "carried": names.carried,
        "before": call([patient_at]),
        "after": call(sites),
        "best_frequency": pd.concat(healthy.values(), axis=1).max(axis=1),
    })
    table["changed"] = table.before != table.after
    # A name can belong to more than one DNA change: keep the one resolve() would pick.
    return table.sort_values("carried", ascending=False).drop_duplicates("name").sort_values("name")


def is_common(answer: HospitalAnswer) -> bool:
    """Rule 1: common among this hospital's healthy patients."""
    return answer.healthy.frequency >= TOO_COMMON


def is_piling_up(answer: HospitalAnswer) -> bool:
    """Rule 2: common among the healthy, yet several times more common among the sick."""
    if answer.sick.hidden or not is_common(answer):
        return False
    return answer.sick.frequency >= PILE_UP_FOLD * answer.healthy.frequency


def model_verdict(variant_id: str, frequency: float) -> str:
    """The slot for the trained model.

    ML side: load your model here, build the feature row for `variant_id` from
    data/variants.csv using the columns in data/columns.json, put `frequency` in
    the frequency column, and return something like "0.08 (harmless)".
    """
    return MODEL_NOT_CONNECTED


# ---------------------------------------------------------------------------
# Finding variants and reading files
# ---------------------------------------------------------------------------
def list_sites() -> list[str]:
    return list(_run()["sites"])


def site_population(site: str) -> str:
    """The gnomAD population code of a hospital's patients: nfe, sas or afr."""
    return _run()["sites"][site]["population"]


@cache
def _run() -> dict:
    run_file = DATA_DIR / "sites.json"
    if not run_file.exists():
        raise FileNotFoundError("data/sites.json is missing. Run scripts/02_simulate_hospitals.py first.")
    return json.loads(run_file.read_text(encoding="utf-8"))


def resolve(variant: str) -> tuple[str, str]:
    """Accept a name like 'DSP N1526K' or an id like 'chr6:g.7580768C>A'."""
    names = _names()
    text = variant.strip()
    exact = names[(names.name.str.lower() == text.lower()) | (names.variant_id == text)]
    if exact.empty:
        close = search(text, limit=5)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise LookupError(f"No variant called '{variant}'.{hint}")
    # A name can belong to more than one DNA change. Take the one patients carry most.
    row = exact.sort_values("carried", ascending=False).iloc[0]
    return row.variant_id, row["name"]


def search(text: str, limit: int = 12) -> list[str]:
    names = _names()
    text = text.strip().lower()
    if not text:
        return []
    starts = names[names.name.str.lower().str.startswith(text)]
    contains = names[names.name.str.lower().str.contains(text, regex=False)]
    found = pd.concat([starts, contains]).drop_duplicates("name")
    return found.name.head(limit).tolist()


def examples() -> list[str]:
    """Variants worth showing: the two demo stories plus the most one-sided ones."""
    wanted = ["DSP N1526K", "TTR V142I", "FLNC T834M", "MYH7 R403Q"]
    known = set(_names().name)
    return [name for name in wanted if name in known]


@cache
def _patient_counts(site: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / site / "patient_counts.csv").set_index("variant_id")


@cache
def _verdicts(site: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / site / "verdicts.csv").drop_duplicates("variant_id").set_index("variant_id")


@cache
def _public_reference() -> dict:
    reference = pd.read_csv(DATA_DIR / "public_reference.csv")
    return dict(zip(reference.variant_id, reference.af_public, strict=True))


@cache
def _names() -> pd.DataFrame:
    first_site = _patient_counts(list_sites()[0]).reset_index()[["variant_id", "name"]]
    carried = sum(_patient_counts(site).ac for site in list_sites())
    return first_site.assign(carried=first_site.variant_id.map(carried))
