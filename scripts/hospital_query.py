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
    model_verdict()  the trained model of step 3 or step 4, given the same scores and
                     the best frequency the query found. Missense variants only.

Counts among unaffected patients are fair evidence: step 2 draws them from real
population frequencies only. Counts among affected patients were generated using
the verdict, so the pile-up rule is a demonstration of what real hospital data
would allow, and those counts must never be fed to a model.

Rule 1 has one line per gene. A variant is too common to cause a rare disease at
0.1% among healthy people when one bad copy of the gene is enough to cause it, and
at 1% when both copies must be bad, because a two-copy disease expects healthy
carriers of one copy. How each gene is inherited is read from config/<area>_gene_panel.txt.

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

Disease areas. The heart build sits at the top of data/ and is the default. Any other
area built by steps 1 and 2 with --panel lives in data/<area>/, with data/<area>/other_types/
beside it when that was built. `set_area()` switches; `available_areas()` lists what is built.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import pandas as pd

import variant_spelling

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "config"
DEFAULT_AREA = "cardiac"  # its files sit at the top of data/. Every other area is data/<area>/
OTHER_TYPES = "other_types"  # the folder, same layout as its area, for every small mutation that is not missense. Optional

# A hospital never reports a count below this. It says "fewer than 5" instead,
# so that no single patient can be picked out. 0 switches the hiding off.
DEFAULT_MIN_COUNT = 5

# Rule 1: this common among healthy people is too common to cause a rare disease.
# Which line applies depends on how the gene is inherited, as recorded per gene in
# config/<area>_gene_panel.txt. Illustrative; clinical labs use disease-specific limits.
TOO_COMMON = 0.001  # one bad copy of the gene is enough to cause disease
TOO_COMMON_TWO_COPIES = 0.01  # both copies must be bad, so healthy carriers of one copy are expected
TWO_COPY_MODES = {"BIALLELIC", "X-LINKED-BIALLELIC"}  # every other recorded mode, and a gene not in the file, means one copy is enough

# Rule 2: the variant "piles up" among the sick when it is this many times more
# frequent in affected than in unaffected patients.
PILE_UP_FOLD = 3.0

# The trained model: weights the ML side saved, read from the area's folder.
MODEL_FEDERATED_FILE = "results_federated.json"  # step 4, NVFlare. Used when it exists
MODEL_LOCAL_FILE = "results_local.json"  # step 3. Its pooled model is the fallback
MODEL_DROPPED_SCORES = {"polyphen2_hdiv"}  # step 3 leaves this score out, so the query must too
MODEL_NEUTRAL_SCORE = 0.5  # what step 3 puts in for a missing score
MODEL_FREQUENCY_FLOOR = 1e-6  # step 3's fixed frequency transform: log10, floored, rescaled to 0-1
MODEL_NOT_SCORED = "not scored: no trained model found. Run scripts/03_train_local.py"
MODEL_IS_MISSENSE_ONLY = "the trained model scores missense changes only, so it is not used here"
MODEL_NOT_CONNECTED = MODEL_NOT_SCORED  # the old name, kept for anything that imports it

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

# Variants worth showing first on the screen, per area. Anything not in the built table is skipped.
EXAMPLES = {
    "cardiac": ["DSP N1526K", "TTR V142I", "FLNC T834M", "MYH7 R403Q"],
    "cancer": ["POLD1 S173N", "PMS2 T511M", "MUTYH G63D", "RNF43 R657P"],
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
    model: str  # the trained model's verdict in a sentence, or why there is none
    mutation_type: str = MISSENSE
    presumption: str = ""  # the starting presumption from the mutation type, in words. Empty when there is none
    asked_as: str = ""  # the spelling that was sent to the hospitals
    spelling_fix: bool = True  # False: hospitals matched the text against their own lab's spelling, nothing else
    area: str = DEFAULT_AREA
    gene: str = ""
    too_common: float = TOO_COMMON  # rule 1's line for this gene
    line_reason: str = ""  # which line applied and why, from how the gene is inherited
    model_probability: float | None = None  # the trained model's probability that the variant is harmful
    model_threshold: float | None = None  # its cut-off, set by the ML side at 95% sensitivity on the training rows
    model_call: str = ""  # likely harmful, likely harmless, not scored, or not used
    model_source: str = ""  # federated or pooled
    same_class: list[HospitalAnswer] = field(default_factory=list)  # A4: carriers of ANY change of the same presumed-harmful class in this gene


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

    answer = HospitalAnswer(
        site=site,
        healthy=reported(row.ac_unaffected, row.an_unaffected, min_count),
        sick=reported(row.ac_affected, row.an_affected, min_count),
    )
    verdicts = _verdicts(site)
    if variant_id in verdicts.index:
        answer.verdict = verdicts.loc[variant_id, "verdict"]
        answer.stars = int(verdicts.loc[variant_id, "stars"])
    return answer


def reported(carriers: int, total: int, min_count: int) -> Count:
    """A count as the hospital reports it: hidden when it is above zero and below the line."""
    return Count(None if 0 < carriers < min_count else int(carriers), int(total), min_count)


def same_class_counts(gene: str, mutation_type: str, min_count: int = DEFAULT_MIN_COUNT) -> list[HospitalAnswer]:
    """For a change that cuts the protein, shifts the frame or breaks a splice site: how many patients at each
    hospital carry ANY such change in this gene. That is what a lab asks next about a variant nobody has seen.

    Empty for every other mutation type, and when data/other_types/ is not built.
    """
    if mutation_type not in PRESUMED_HARMFUL:
        return []
    answers = []
    for site in list_sites():
        sums = _same_class_sums(site)
        if gene not in sums.index:
            continue
        row = sums.loc[gene]
        answers.append(HospitalAnswer(site, reported(row.ac_unaffected, row.an_unaffected, min_count),
                                      reported(row.ac_affected, row.an_affected, min_count)))
    return answers


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
    gene = gene_of(name)
    mutation_type = mutation_type_of(variant_id)
    too_common, line_reason = inheritance_line(gene)
    # With the fix, the canonical spelling travels. Without it, the text travels as the patient's hospital wrote it.
    asked_as = variant_id if spelling_fix else spelling_used_at(patient_at, variant, variant_id)
    answers = [ask_hospital(site, asked_as, min_count, spelling_fix) for site in sites]

    own_answer = next(a for a in answers if a.site == patient_at)
    most_common = max(answers, key=lambda a: a.healthy.frequency)
    public_frequency = float(_public_reference().get(variant_id, 0.0))
    model = model_verdict(variant_id, most_common.healthy.frequency) if mutation_type == MISSENSE else ModelVerdict.not_used()

    return QueryResult(
        name=name,
        variant_id=variant_id,
        patient_at=patient_at,
        public_frequency=public_frequency,
        answers=answers,
        # Alone, a doctor has the public database and their own hospital's patients.
        before=read_evidence([own_answer], public_frequency, mutation_type, too_common),
        after=read_evidence(answers, public_frequency, mutation_type, too_common),
        best_frequency=most_common.healthy.frequency,
        best_site=most_common.site if most_common.healthy.frequency > 0 else None,
        model=model.text,
        mutation_type=mutation_type,
        presumption=presumption(mutation_type),
        asked_as=asked_as,
        spelling_fix=spelling_fix,
        area=current_area(),
        gene=gene,
        too_common=too_common,
        line_reason=line_reason,
        model_probability=model.probability,
        model_threshold=model.threshold,
        model_call=model.call,
        model_source=model.source,
        same_class=same_class_counts(gene, mutation_type, min_count),
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


def read_evidence(answers: list[HospitalAnswer], public_frequency: float = 0.0, mutation_type: str = MISSENSE,
                  too_common: float = TOO_COMMON) -> Reading:
    """Two rules, no model. Rule 2 can overrule rule 1.

    The public database counts as evidence too: it is what a doctor checks first.

    `too_common` is rule 1's line for the gene: 0.1% when one bad copy is enough to cause
    disease, 1% when both copies must be bad. See inheritance_line().

    For a mutation type other than missense the first reason is the starting presumption from
    the type. It never makes the call. The two frequency rules do, and they can overrule it.
    """
    reasons = [presumption(mutation_type)] if presumption(mutation_type) else []

    piles_up = [a for a in answers if is_piling_up(a, too_common)]
    for a in piles_up:
        reasons.append(
            f"piles up among the sick at {a.site}: {a.sick.frequency:.1%} of sick against "
            f"{a.healthy.frequency:.1%} of healthy patients "
            f"({a.sick.frequency / a.healthy.frequency:.0f} times as common)"
        )

    # (how common, a phrase saying where) for every source that clears the variant
    common = [(a.healthy.frequency, f"of healthy patients at {a.site}") for a in answers if is_common(a, too_common)]
    if public_frequency >= too_common:
        common.append((public_frequency, "of people in the public database"))
    for frequency, where in common:
        reasons.append(f"common: {frequency:.1%} {where} carry it")

    if piles_up:
        worst = max(piles_up, key=lambda a: a.sick.frequency / a.healthy.frequency)
        fold = worst.sick.frequency / worst.healthy.frequency
        return Reading("KEEP FLAGGED", reasons, f"{fold:.0f}x more common in sick than healthy at {worst.site}")
    if common:
        frequency, where = max(common)
        if too_common >= TOO_COMMON_TWO_COPIES:
            reasons.append(f"too common among healthy people even for a two-copy disease, where healthy carriers of one copy are expected: over the {too_common:.0%} line")
        else:
            reasons.append(f"too common among healthy people to cause {disease_words()}")
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


def disease_words() -> str:
    """What rule 1 says the variant is too common to cause, in the words of the area."""
    return "a rare heart disease" if current_area() == DEFAULT_AREA else "a rare inherited disease"


@cache
def overview(patient_at: str | None = None, min_count: int = DEFAULT_MIN_COUNT) -> pd.DataFrame:
    """The call for every variant at once: name, gene, mutation_type, before, after, changed, best_frequency, too_common.

    Same two rules as read_evidence(), applied to whole columns so that a screen can
    list and filter thousands of variants instantly. One row per variant name.
    `mutation_type` is `missense` for the missense table, and the type column of data/other_types/ for the rest.
    `too_common` is rule 1's line for the row's gene.
    """
    sites = list_sites()
    patient_at = patient_at or sites[0]
    names = _names().set_index("variant_id")
    line = names.gene.map(too_common_line)

    common, piles_up, healthy = {}, {}, {}
    for site in sites:
        counts = _patient_counts(site)
        healthy_hidden = (counts.ac_unaffected > 0) & (counts.ac_unaffected < min_count)
        sick_hidden = (counts.ac_affected > 0) & (counts.ac_affected < min_count)
        healthy[site] = counts.ac_unaffected.mask(healthy_hidden, 0) / counts.an_unaffected
        sick = counts.ac_affected.mask(sick_hidden, 0) / counts.an_affected
        common[site] = healthy[site] >= line.reindex(counts.index)
        piles_up[site] = common[site] & ~sick_hidden & (sick >= PILE_UP_FOLD * healthy[site])

    index = _patient_counts(sites[0]).index
    common_in_public = pd.Series(_public_reference()).reindex(index).fillna(0.0) >= line.reindex(index)

    def call(among: list[str]) -> pd.Series:
        flagged = pd.concat([piles_up[s] for s in among], axis=1).any(axis=1)
        cleared = pd.concat([common[s] for s in among], axis=1).any(axis=1) | common_in_public
        calls = pd.Series("FREQUENCY SAYS NOTHING", index=index)
        calls[cleared] = "LIKELY HARMLESS"
        calls[flagged] = "KEEP FLAGGED"
        return calls

    table = pd.DataFrame({
        "name": names.name,
        "gene": names.gene,
        "mutation_type": names.mutation_type,
        "carried": names.carried,
        "before": call([patient_at]),
        "after": call(sites),
        "best_frequency": pd.concat(healthy.values(), axis=1).max(axis=1),
        "too_common": line,
    })
    table["changed"] = table.before != table.after
    # A name can belong to more than one DNA change: keep the one resolve() would pick.
    # A stable sort, so that two changes carried equally often are settled by file order, whatever the table size.
    return table.sort_values("carried", ascending=False, kind="stable").drop_duplicates("name").sort_values("name")


def is_common(answer: HospitalAnswer, too_common: float = TOO_COMMON) -> bool:
    """Rule 1: common among this hospital's healthy patients."""
    return answer.healthy.frequency >= too_common


def is_piling_up(answer: HospitalAnswer, too_common: float = TOO_COMMON) -> bool:
    """Rule 2: common among the healthy, yet several times more common among the sick."""
    if answer.sick.hidden or not is_common(answer, too_common):
        return False
    return answer.sick.frequency >= PILE_UP_FOLD * answer.healthy.frequency


# ---------------------------------------------------------------------------
# How the gene is inherited: which line rule 1 uses
# ---------------------------------------------------------------------------
def gene_of(name: str) -> str:
    """'DSP N1526K' -> 'DSP'."""
    return name.partition(" ")[0]


def gene_modes(gene: str) -> list[str]:
    """The modes of inheritance recorded for a gene in config/<area>_gene_panel.txt. Empty when it is not there."""
    return _gene_modes().get(gene, [])


def needs_two_copies(gene: str) -> bool:
    """True when every recorded mode says both copies of the gene must be bad to cause disease."""
    modes = gene_modes(gene)
    return bool(modes) and all(mode in TWO_COPY_MODES for mode in modes)


def too_common_line(gene: str) -> float:
    return TOO_COMMON_TWO_COPIES if needs_two_copies(gene) else TOO_COMMON


def inheritance_line(gene: str) -> tuple[float, str]:
    """Rule 1's line for this gene, and why, in clinician words."""
    modes = gene_modes(gene)
    line = too_common_line(gene)
    at = f"{line:.1%}" if line < 0.01 else f"{line:.0%}"
    if needs_two_copies(gene):
        return line, f"line {at}: both copies of {gene} must be bad to cause disease, so healthy carriers of one copy are expected"
    if not modes:
        return line, f"line {at}: how {gene} is inherited is not recorded, so one bad copy is taken to be enough"
    if modes == ["UNKNOWN"]:
        return line, f"line {at}: how {gene} is inherited is not settled, so one bad copy is taken to be enough"
    if any(mode in TWO_COPY_MODES or mode == "BOTH" for mode in modes):
        return line, f"line {at}: one bad copy of {gene} can be enough to cause disease"
    if "MITOCHONDRIAL" in modes:
        return line, f"line {at}: {gene} is inherited through the mother's mitochondria, so one bad copy is taken to be enough"
    return line, f"line {at}: one bad copy of {gene} is enough to cause disease"


@cache
def _gene_modes() -> dict[str, list[str]]:
    """gene -> modes, from the lines `GENE  # panel ids | MODE; MODE` of the area's gene list."""
    panel_file = CONFIG_DIR / f"{current_area()}_gene_panel.txt"
    if not panel_file.exists():
        return {}
    modes = {}
    for line in panel_file.read_text(encoding="utf-8").splitlines():
        gene, _, comment = line.partition("#")
        if not gene.strip():
            continue
        _, _, inheritance = comment.partition("|")
        modes[gene.strip()] = [mode.strip() for mode in inheritance.split(";") if mode.strip()]
    return modes


# ---------------------------------------------------------------------------
# The trained model
# ---------------------------------------------------------------------------
@dataclass
class ModelVerdict:
    text: str
    probability: float | None = None
    threshold: float | None = None
    call: str = ""  # likely harmful, likely harmless, not scored, or not used
    source: str = ""  # federated or pooled

    @staticmethod
    def not_used() -> ModelVerdict:
        return ModelVerdict(MODEL_IS_MISSENSE_ONLY, call="not used")


@dataclass
class Model:
    """One logistic regression as the ML side saved it: a weight per feature, an intercept and a cut-off."""

    source: str  # federated or pooled
    features: list[str]  # the score columns, then log_frequency
    weights: list[float]
    intercept: float
    threshold: float

    def probability(self, scores: list[float], frequency: float) -> float:
        row = [MODEL_NEUTRAL_SCORE if math.isnan(s) else s for s in scores] + [log_frequency(frequency)]
        total = sum(w * x for w, x in zip(self.weights, row, strict=True)) + self.intercept
        return 1 / (1 + math.exp(-total)) if total > -700 else 0.0


def log_frequency(frequency: float) -> float:
    """Step 3's fixed transform: log10 of the frequency, floored at one in a million, rescaled to 0-1."""
    floor = math.log10(MODEL_FREQUENCY_FLOOR)
    return (math.log10(frequency + MODEL_FREQUENCY_FLOOR) - floor) / -floor


def model_verdict(variant_id: str, frequency: float) -> ModelVerdict:
    """The trained model's call for a missense variant, given the best frequency the query found.

    The feature row is built as scripts/03_train_local.py builds it: the same score columns,
    a missing score replaced by the neutral 0.5, and the frequency put through the same
    log transform. The federated model is used when step 4 saved one, else step 3's pooled model.
    """
    model = _model()
    scores = model_scores(variant_id, model.features[:-1]) if model else None
    if model is None or scores is None:
        return ModelVerdict(MODEL_NOT_SCORED, call="not scored")
    probability = model.probability(scores, frequency)
    call = "likely harmful" if probability >= model.threshold else "likely harmless"
    return ModelVerdict(model_sentence(probability, model), probability, model.threshold, call, model.source)


def model_sentence(probability: float, model: Model) -> str:
    """'the trained model puts this at 3% likely harmful, below its cut-off of 41% · pooled model'."""
    side = "above" if probability >= model.threshold else "below"
    return f"the trained model puts this at {percent_word(probability)} likely harmful, {side} its cut-off of {percent_word(model.threshold)} · {model.source} model"


def percent_word(value: float) -> str:
    return f"{value:.1%}" if 0 < value < 0.01 else f"{value:.0%}"


def model_scores(variant_id: str, score_columns: list[str]) -> list[float] | None:
    """The variant's prediction scores, in the model's order. None when the variant has no score row at all."""
    scores = _scores()
    if variant_id not in scores.index:
        return None
    row = scores.loc[variant_id, score_columns]
    if row.isna().all():
        return None
    return [float(value) for value in row]


@cache
def _model() -> Model | None:
    """The federated model when step 4 saved one, else step 3's pooled model. None when neither file exists."""
    folder = area_folder()
    federated, local = folder / MODEL_FEDERATED_FILE, folder / MODEL_LOCAL_FILE
    features = None
    if local.exists():
        saved = json.loads(local.read_text(encoding="utf-8"))
        features = list(saved["features"])
    if federated.exists():
        runs = json.loads(federated.read_text(encoding="utf-8")).get("per_run", {}).get("federated", [])
        if runs and "weights" in runs[0] and "threshold" in runs[0]:
            weights = [float(w) for w in runs[0]["weights"]]
            features = features or model_features_from_columns()
            if len(weights) == len(features) + 1:
                return Model("federated", features, weights[:-1], weights[-1], float(runs[0]["threshold"]))
    if local.exists():
        pooled = saved["training"]["pooled"]
        weights = pooled["weights"]
        return Model("pooled", features, [float(weights[f]) for f in features], float(weights["intercept"]), float(pooled["threshold"]))
    return None


def model_features_from_columns() -> list[str]:
    """Step 3's feature list, rebuilt from columns.json the way step 3 builds it."""
    columns = json.loads((area_folder() / "columns.json").read_text(encoding="utf-8"))
    return [c for c in columns["features"] if c not in MODEL_DROPPED_SCORES] + ["log_frequency"]


@cache
def _scores() -> pd.DataFrame:
    """Every variant's prediction scores, from the area's variants.csv. Empty when the table is not built."""
    table_file = area_folder() / "variants.csv"
    if not table_file.exists():
        return pd.DataFrame(columns=["variant_id"]).set_index("variant_id")
    header = pd.read_csv(table_file, nrows=0).columns
    columns = json.loads((area_folder() / "columns.json").read_text(encoding="utf-8"))
    wanted = ["variant_id"] + [c for c in columns["features"] if c in header]
    return pd.read_csv(table_file, usecols=wanted).set_index("variant_id")


# ---------------------------------------------------------------------------
# Disease areas
# ---------------------------------------------------------------------------
_area = DEFAULT_AREA


def current_area() -> str:
    return _area


def set_area(area: str) -> None:
    """Point every reader at another built area. Everything cached is dropped, so the next call reads its files."""
    global _area
    if area == _area:
        return
    _area = area
    clear_caches()


def area_folder(area: str | None = None) -> Path:
    """The heart build keeps its place at the top of data/. Every other area is data/<area>/."""
    area = area or current_area()
    return DATA_DIR if area == DEFAULT_AREA else DATA_DIR / area


def other_types_folder(area: str | None = None) -> Path:
    return area_folder(area) / OTHER_TYPES


def available_areas() -> list[str]:
    """Every area whose hospitals are built: the heart area first when it is, then data/<name>/sites.json in name order."""
    built = [DEFAULT_AREA] if (area_folder(DEFAULT_AREA) / "sites.json").exists() else []
    others = sorted(p.parent.name for p in DATA_DIR.glob("*/sites.json") if p.parent.name != OTHER_TYPES)
    return built + others


def area_title(area: str | None = None) -> str:
    """The area's name for people, from config/panel_sets.json. The folder name when there is none."""
    area = area or current_area()
    sets_file = CONFIG_DIR / "panel_sets.json"
    if sets_file.exists():
        return json.loads(sets_file.read_text(encoding="utf-8")).get(area, {}).get("title", area)
    return area


def build_command(area: str) -> str:
    """The commands that build an area's files, for a message when they are missing."""
    panel = "" if area == DEFAULT_AREA else f" --panel {area}"
    steps = [f"uv run python scripts/01_build_table.py{panel}", "uv run python scripts/02_simulate_hospitals.py"]
    if area != DEFAULT_AREA:
        steps[1] += f"    # today this step reads data/ only: point it at data/{area}/, see docs/data_contract.md"
    return "\n".join("    " + step for step in steps)


def clear_caches() -> None:
    for value in list(globals().values()):
        if hasattr(value, "cache_clear"):
            value.cache_clear()


# ---------------------------------------------------------------------------
# Finding variants and reading files
# ---------------------------------------------------------------------------
def list_sites() -> list[str]:
    return list(_run()["sites"])


def site_population(site: str) -> str:
    """The gnomAD population code of a hospital's patients: nfe, sas or afr."""
    return _run()["sites"][site]["population"]


def site_patients(site: str) -> int:
    """How many patients the hospital has sequenced."""
    return int(_run()["sites"][site]["patients"])


@cache
def _run() -> dict:
    run_file = area_folder() / "sites.json"
    if not run_file.exists():
        raise FileNotFoundError(
            f"{run_file.relative_to(ROOT).as_posix()} is missing: the {area_title()} area is not built. Build it first:\n"
            + build_command(current_area())
        )
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
    """Variants worth showing: the area's demo stories, and the deletion with two spellings.

    An area with no stories written down shows the variants whose call changes most clearly once the other hospitals answer.
    """
    names = _names()
    known = set(names.name)
    wanted = [name for name in EXAMPLES.get(current_area(), []) if name in known]
    if not wanted:
        table = overview()
        wanted = table[table.changed].sort_values("best_frequency", ascending=False).name.head(4).tolist()
    spelled_two_ways = names.name[names.variant_id == variant_spelling.MYBPC3_AS_CLINVAR_AND_GNOMAD].tolist()
    return wanted + [name for name in spelled_two_ways if name not in wanted]


def _both_tables(file: str, columns: list[str] | None = None) -> pd.DataFrame:
    """One file of the missense build, with the same file of other_types/ under it when that exists."""
    found = [area_folder() / file]
    if (other_types_folder() / file).exists():
        found.append(other_types_folder() / file)
    return pd.concat([pd.read_csv(path, usecols=columns) for path in found], ignore_index=True)


@cache
def _patient_counts(site: str) -> pd.DataFrame:
    counts = _both_tables(f"{site}/patient_counts.csv")
    # `written_as` is the hospital's own spelling. Only insertions and deletions can differ from the canonical one.
    written_as = counts.written_as if "written_as" in counts else pd.Series(pd.NA, index=counts.index)
    return counts.assign(written_as=written_as.fillna(counts.variant_id)).set_index("variant_id")


@cache
def _same_class_sums(site: str) -> pd.DataFrame:
    """Per gene: carriers of any protein-cutting, frameshift or splice-site change at one hospital, healthy and sick."""
    counts = _patient_counts(site)
    kinds = counts.index.map(_mutation_types()).fillna(MISSENSE)
    rows = counts[kinds.isin(list(PRESUMED_HARMFUL))]
    if rows.empty:
        return pd.DataFrame(columns=["ac_unaffected", "an_unaffected", "ac_affected", "an_affected"])
    return rows.groupby(rows.name.map(gene_of)).agg(
        ac_unaffected=("ac_unaffected", "sum"), an_unaffected=("an_unaffected", "first"),
        ac_affected=("ac_affected", "sum"), an_affected=("an_affected", "first"),
    )


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
        gene=first_site.name.map(gene_of),
        carried=first_site.variant_id.map(carried),
        name_lower=first_site.name.str.lower(),
        mutation_type=first_site.variant_id.map(_mutation_types()).fillna(MISSENSE),
    )


@cache
def _other_types() -> pd.DataFrame:
    """What the patient's hospital works out from the variant itself: its mutation type, and the public database's id for it."""
    table_file = other_types_folder() / "variants.csv"
    columns = ["variant_id", "mutation_type", "database_id"]
    return pd.read_csv(table_file, usecols=columns) if table_file.exists() else pd.DataFrame(columns=columns)


@cache
def _mutation_types() -> dict:
    return dict(zip(_other_types().variant_id, _other_types().mutation_type, strict=True))


@cache
def _database_ids() -> dict:
    return dict(zip(_other_types().database_id, _other_types().variant_id, strict=True))
