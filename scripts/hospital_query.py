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

Every kind of small mutation is covered, where the trained model covers missense only.
Counting carriers does not depend on what a variant does to the protein. When
data/other_types/ exists (scripts/01_build_other_types.py, then 02_simulate_other_types.py)
its variants are answered for as well. Two things come with them:

    a starting presumption   from the mutation type, the way a clinical lab starts: a change that
                             cuts the protein short is presumed harmful, a silent one harmless.
                             It is a reason line of its own. The two frequency rules make the call,
                             and they can overrule it.
    one spelling             an insertion or deletion inside a repeat can be written several valid
                             ways. Whatever is typed is turned into the canonical spelling before
                             any hospital is asked (scripts/variant_spelling.py). Without that,
                             a hospital that wrote the variant another way answers zero, silently.
                             `spelling_fix=False` shows exactly that failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import pandas as pd

import variant_spelling

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OTHER_TYPES_DIR = DATA_DIR / "other_types"  # same layout as data/, for every small mutation that is not missense. Optional

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
MODEL_IS_MISSENSE_ONLY = "not used: it scores missense variants only"

# Where a lab starts before it has looked at any count, from the mutation type alone.
MISSENSE = "missense"
PRESUMED_HARMFUL = {
    "protein_cutting": "it cuts the protein short",
    "frameshift": "it shifts the reading frame, which scrambles the rest of the protein",
    "splice_site": "it breaks a splice site, so the pieces of the gene are joined wrongly",
}
PRESUMED_HARMLESS = {
    "synonymous": "it is a silent change, the protein stays the same",
    "intronic": "it sits inside an intron, outside the code for the protein",
}


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
    mutation_type: str = MISSENSE
    presumption: str = ""  # the starting presumption from the mutation type, in words. Empty when there is none
    asked_as: str = ""  # the spelling that was sent to the hospitals
    spelling_fix: bool = True  # False: hospitals matched the text against their own lab's spelling, nothing else


# ---------------------------------------------------------------------------
# Inside one hospital
# ---------------------------------------------------------------------------
def ask_hospital(site: str, variant_id: str, min_count: int = DEFAULT_MIN_COUNT, spelling_fix: bool = True) -> HospitalAnswer:
    """The hospital's own counts for one variant, however the variant was spelled.

    With `spelling_fix=False` the hospital only recognises the spelling its own lab uses. Any other
    spelling looks like a variant it has never seen, and it answers zero. That is the silent failure.
    """
    counts = _patient_counts(site)
    if not spelling_fix:
        own_spelling = counts[counts.written_as == variant_id]
        if own_spelling.empty:
            anyone = counts.iloc[0]  # only for the size of the cohort
            return HospitalAnswer(site, Count(0, int(anyone.an_unaffected), min_count), Count(0, int(anyone.an_affected), min_count))
        variant_id = own_spelling.index[0]
    elif variant_id not in counts.index:
        variant_id = canonical_spelling(variant_id) or variant_id
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
def query(variant: str, patient_at: str | None = None, min_count: int = DEFAULT_MIN_COUNT,
          spelling_fix: bool = True) -> QueryResult:
    sites = list_sites()
    patient_at = patient_at or sites[0]
    if patient_at not in sites:
        raise ValueError(f"{patient_at} is not a hospital. Choose one of: {', '.join(sites)}")

    variant_id, name = resolve(variant)
    mutation_type = mutation_type_of(variant_id)
    # With the fix, the canonical spelling travels. Without it, the text travels as the patient's hospital wrote it.
    asked_as = variant_id if spelling_fix else spelling_used_at(patient_at, variant, variant_id)
    answers = [ask_hospital(site, asked_as, min_count, spelling_fix) for site in sites]

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
        before=read_evidence([own_answer], public_frequency, mutation_type),
        after=read_evidence(answers, public_frequency, mutation_type),
        best_frequency=most_common.healthy.frequency,
        best_site=most_common.site if most_common.healthy.frequency > 0 else None,
        model=model_verdict(variant_id, most_common.healthy.frequency) if mutation_type == MISSENSE else MODEL_IS_MISSENSE_ONLY,
        mutation_type=mutation_type,
        presumption=presumption(mutation_type),
        asked_as=asked_as,
        spelling_fix=spelling_fix,
    )


def article(word: str) -> str:
    """'an intronic change', 'a frameshift change'."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def presumption(mutation_type: str) -> str:
    """Where a lab starts from the mutation type alone, before any count. Empty when the type says nothing."""
    kind = mutation_type.replace("_", " ")
    if mutation_type in PRESUMED_HARMFUL:
        return f"starts out presumed harmful, as {article(kind)} {kind} change: {PRESUMED_HARMFUL[mutation_type]}"
    if mutation_type in PRESUMED_HARMLESS:
        return f"starts out presumed harmless, as {article(kind)} {kind} change: {PRESUMED_HARMLESS[mutation_type]}"
    return ""


def read_evidence(answers: list[HospitalAnswer], public_frequency: float = 0.0, mutation_type: str = MISSENSE) -> Reading:
    """Two rules, no model. Rule 2 can overrule rule 1.

    The public database counts as evidence too: it is what a doctor checks first.

    For a mutation type other than missense the first reason is the starting presumption from
    the type. It never makes the call. The two frequency rules do, and they can overrule it.
    """
    reasons = [presumption(mutation_type)] if presumption(mutation_type) else []

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
    what_decides, in_short = what_decides_without_frequency(mutation_type)
    reasons.append(f"rare or unseen among healthy patients, so frequency cannot clear it: {what_decides}")
    return Reading("FREQUENCY SAYS NOTHING", reasons, f"rare among healthy patients: {in_short}")


def what_decides_without_frequency(mutation_type: str) -> tuple[str, str]:
    """When the counts say nothing, what is left to go on: a sentence, and the same in a few words."""
    if mutation_type == MISSENSE:
        return "the scores have to decide", "the scores must decide"
    if mutation_type in PRESUMED_HARMFUL:
        return "the presumption from its mutation type stands, harmful", "presumed harmful from its type"
    if mutation_type in PRESUMED_HARMLESS:
        return "the presumption from its mutation type stands, harmless", "presumed harmless from its type"
    return "its mutation type gives no presumption, so a lab has to weigh the rest", "its type gives no presumption"


@cache
def overview(patient_at: str | None = None, min_count: int = DEFAULT_MIN_COUNT) -> pd.DataFrame:
    """The call for every variant at once: name, gene, mutation_type, before, after, changed, best_frequency.

    Same two rules as read_evidence(), applied to whole columns so that a screen can
    list and filter thousands of variants instantly. One row per variant name.
    `mutation_type` is `missense` for the missense table, and the type column of data/other_types/ for the rest.
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
        "mutation_type": names.mutation_type,
        "carried": names.carried,
        "before": call([patient_at]),
        "after": call(sites),
        "best_frequency": pd.concat(healthy.values(), axis=1).max(axis=1),
    })
    table["changed"] = table.before != table.after
    # A name can belong to more than one DNA change: keep the one resolve() would pick.
    # A stable sort, so that two changes carried equally often are settled by file order, whatever the table size.
    return table.sort_values("carried", ascending=False, kind="stable").drop_duplicates("name").sort_values("name")


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
    """Accept a name like 'DSP N1526K' or an id like 'chr6:g.7580768C>A'.

    An insertion or deletion can be given in any of its valid spellings. It is turned
    into the canonical one first, so `chr11:g.47332282_47332306del` and
    `chr11:g.47332275_47332299del` find the same MYBPC3 deletion.
    """
    names = _names()
    text = variant.strip()
    exact = names[(names.name_lower == text.lower()) | (names.variant_id == text)]
    if exact.empty:
        exact = names[names.variant_id == canonical_spelling(text)]
    if exact.empty:
        close = search(text, limit=5)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise LookupError(f"No variant called '{variant}'.{hint}")
    # A name can belong to more than one DNA change. Take the one patients carry most.
    row = exact.sort_values("carried", ascending=False).iloc[0]
    return row.variant_id, row["name"]


def search(text: str, limit: int = 12) -> list[str]:
    """Names that start with or contain the text. A DNA change, in any valid spelling, finds its variant."""
    names = _names()
    spelled = names[names.variant_id == canonical_spelling(text)]
    text = text.strip().lower()
    if not text:
        return []
    starts = names[names.name_lower.str.startswith(text)]
    contains = names[names.name_lower.str.contains(text, regex=False)]
    found = pd.concat([spelled, starts, contains]).drop_duplicates("name")
    return found.name.head(limit).tolist()


def canonical_spelling(text: str) -> str | None:
    """The one spelling every hospital files a DNA change under. None if the text is not a DNA change in the panel genes."""
    text = text.strip()
    if text in _database_ids():  # the public database's own id. For a duplication that id is not valid HGVS
        return _database_ids()[text]
    try:
        return variant_spelling.canonical(text)
    except (variant_spelling.SpellingError, variant_spelling.NoSequence):
        return None


def spelling_used_at(site: str, typed: str, variant_id: str) -> str:
    """Without the spelling fix: what the patient's hospital sends out. An id goes as typed, a name as its own lab wrote it."""
    if variant_spelling.looks_like_an_id(typed):
        return typed.strip()
    return _patient_counts(site).written_as[variant_id]


def mutation_type_of(variant_id: str) -> str:
    return _mutation_types().get(variant_id, MISSENSE)


def examples() -> list[str]:
    """Variants worth showing: the two demo stories plus the most one-sided ones, and the deletion with two spellings."""
    wanted = ["DSP N1526K", "TTR V142I", "FLNC T834M", "MYH7 R403Q"]
    names = _names()
    known = set(names.name)
    spelled_two_ways = names.name[names.variant_id == variant_spelling.MYBPC3_AS_CLINVAR_AND_GNOMAD].tolist()
    return [name for name in wanted if name in known] + spelled_two_ways


def _both_tables(file: str, columns: list[str] | None = None) -> pd.DataFrame:
    """One file of the missense build, with the same file of data/other_types/ under it when that exists."""
    found = [DATA_DIR / file]
    if (OTHER_TYPES_DIR / file).exists():
        found.append(OTHER_TYPES_DIR / file)
    return pd.concat([pd.read_csv(path, usecols=columns) for path in found], ignore_index=True)


@cache
def _patient_counts(site: str) -> pd.DataFrame:
    counts = _both_tables(f"{site}/patient_counts.csv")
    # `written_as` is the hospital's own spelling. Only insertions and deletions can differ from the canonical one.
    written_as = counts.written_as if "written_as" in counts else pd.Series(pd.NA, index=counts.index)
    return counts.assign(written_as=written_as.fillna(counts.variant_id)).set_index("variant_id")


@cache
def _verdicts(site: str) -> pd.DataFrame:
    verdicts = _both_tables(f"{site}/verdicts.csv", ["variant_id", "verdict", "stars"])
    return verdicts.drop_duplicates("variant_id").set_index("variant_id")


@cache
def _public_reference() -> dict:
    reference = _both_tables("public_reference.csv")
    return dict(zip(reference.variant_id, reference.af_public, strict=True))


@cache
def _names() -> pd.DataFrame:
    first_site = _patient_counts(list_sites()[0]).reset_index()[["variant_id", "name"]]
    carried = sum(_patient_counts(site).ac for site in list_sites())
    return first_site.assign(
        carried=first_site.variant_id.map(carried),
        name_lower=first_site.name.str.lower(),
        mutation_type=first_site.variant_id.map(_mutation_types()).fillna(MISSENSE),
    )


@cache
def _other_types() -> pd.DataFrame:
    """What the patient's hospital works out from the variant itself: its mutation type, and the public database's id for it."""
    table_file = OTHER_TYPES_DIR / "variants.csv"
    columns = ["variant_id", "mutation_type", "database_id"]
    return pd.read_csv(table_file, usecols=columns) if table_file.exists() else pd.DataFrame(columns=columns)


@cache
def _mutation_types() -> dict:
    return dict(zip(_other_types().variant_id, _other_types().mutation_type, strict=True))


@cache
def _database_ids() -> dict:
    return dict(zip(_other_types().database_id, _other_types().variant_id, strict=True))
