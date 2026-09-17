# Critical review

A review of the code and the approach, focused on issues that could change a
conclusion or a clinical claim. Nitpicks are omitted on purpose.

Scope reviewed: the pipeline scripts (`00`–`06`), `hospital_query.py`, the
`README.md` and `Manuscript.md` claims, and the results committed under `data/`.
The code is unusually careful — every quantitative step has a `--self-check`, the
maths passes brute-force verification, and the team's own `docs/step2_review.md`
already catches most data-side traps. So the findings below are mostly about the
**approach and its framing**, not bugs. Where an issue is already acknowledged
somewhere in the repo, this is stated, because several of them are acknowledged
in one place and over-claimed in another.

Verdict: nothing here is a coding error that produces a wrong committed number.
The concerns are (1) an end-to-end system that does not yet exist, (2) an
evaluation that is structurally blind to the approach's main clinical danger, and
(3) a headline that the README states more confidently than the evidence and the
manuscript's own limitations allow.

---

## Critical / fundamental

### 1. The federated model is never connected to the clinical tool

`model_verdict()` in [hospital_query.py:251](scripts/hospital_query.py:251)
always returns the constant `"not connected yet"`. The patient-facing query
(`06_query_variant.py`, `06_query_tui.py`) therefore reaches a verdict using
**only two hand-coded frequency rules** in
[read_evidence()](scripts/hospital_query.py:156) — the logistic-regression model
that steps 3 and 4 train, average with NVFlare, and evaluate is not used to
classify anything a user sees.

So the two halves of the project are disjoint:

- Steps 3–4 are an **offline benchmark** of a model on a locked test set.
- Step 6 is a **separate rule engine** that never calls that model.

This is honest in the code (`model_verdict` is documented as an empty slot) and
the manuscript lists step 7 as planned. But the title, the README, and the
abstract all read as an end-to-end "federated learning for clinical diagnostics"
system, and that system does not exist yet: no path takes a patient's variant,
runs the federated model, and returns a pathogenicity call. This should be stated
plainly in the README, not only inferable from a code comment.

### 2. The evaluation cannot see the approach's main clinical risk

Confirmed against the committed data:

| set | discordant benign | discordant **pathogenic** |
|---|---|---|
| test set (`data/test/variants.csv`) | 183 | **0** |
| whole table (`data/variants.csv`) | 620 | 2 (`TTR V142I`, `KCNQ1 G92A`) |

The only two population-discordant *pathogenic* variants in the entire table both
fell in the **training** pool, so the test set contains **zero** of them. Every
headline number about the frequency source — the 15 → 7 → 2 false-alarm drop in
the README, Table 3 in the manuscript — is measured **exclusively on benign
variants**. It quantifies how much frequency sharing reduces false *positives*
and says nothing measurable about false *negatives*.

That matters because the false-negative case is the dangerous one clinically
(see clinical issue 1) and it is exactly the case the design pushes on. The
manuscript acknowledges this ("The test set contains no pathogenic
population-discordant variant … TTR V142I shows that the case is real"), but the
README presents "removed most false alarms" and "harmful variants were still
caught" as if the two directions were on equal footing. They are not: one was
measured on 183 variants, the other on 0.

### 3. The simulation validates internal consistency, not the real gap

Patients are drawn from the same gnomAD per-population frequencies that also
define the `ceiling` evidence and the `federated_query` truth
([02_simulate_hospitals.py:183](scripts/02_simulate_hospitals.py:183),
[03_train_local.py:163](scripts/03_train_local.py:163)). The "European-only
public reference" is an *artificial* handicap
([02_simulate_hospitals.py:75](scripts/02_simulate_hospitals.py:75)) — gnomAD
already ships AFR and SAS columns, and those columns are used here as ground
truth. So the experiment can only show that carrier counts of 4k–20k patients
reproduce gnomAD frequencies (they do), which is close to a tautology given the
generative model. It cannot show that federation helps on populations that *real*
references cover poorly, because there is no such population in the data.

Acknowledged in the manuscript limitations, and the framing is deliberate. It is
listed here because it is the ceiling on every conclusion the project can draw,
and the README's "Asking the other hospitals removed most false alarms" reads as
an empirical result about the world rather than about a simulation of it.

---

## Clinical

### 1. The frequency feature is actively harmful for founder pathogenic variants — and the federated query makes it worse

This is the most important clinical point. The pooled model's frequency
coefficient is strongly negative (verified from `data/results_local.json`):

```
log_frequency   -3.73        # higher frequency -> pushed toward benign
```

The `federated_query` evidence takes the **maximum** frequency across the three
hospitals ([03_train_local.py:182](scripts/03_train_local.py:182)). For a variant
that is both **common in one ancestry and genuinely pathogenic** — a founder
pathogenic variant — surfacing that high frequency drives the model toward
"benign". Quantified for a `TTR V142I`-like variant (≈1.5% in African ancestry
vs ≈0.003% in the European reference):

```
frequency logit contribution:  European view  -0.93
                               federated max   -2.60     (~5x odds shift toward benign)
```

So the same count-sharing that removes false positives on benign variants would,
on a founder pathogenic variant, push it ~1.7 logits *further* toward a
false-negative than the European-only reference would — and `TTR V142I`, the
project's own showcase variant, is exactly this case (common in African
ancestry, an established cause of cardiac amyloidosis). The approach's benefit
and its danger fall on the **same underserved populations**.

Two things currently mask this:

- The **query's Rule 2** ([hospital_query.py:244](scripts/hospital_query.py:244))
  rescues `TTR V142I` via case/control enrichment — but the **model** has no
  equivalent safeguard, and the model is what steps 3–4 are about.
- The test set has zero pathogenic discordant variants (critical issue 2), so
  the harm is unmeasured.

Recommendation: adopt a disease-specific **maximum credible allele frequency**
floor (Whiffin et al., already cited as ref 15) so frequency can *clear* a
variant but never *override* independent pathogenic evidence; and state
explicitly that a cross-population frequency veto is unsafe for founder variants.

### 2. The `TTR V142I` demonstration rests on synthetic, label-derived data and a rule the model does not contain

The affected-patient counts that make Rule 2 fire are generated *from the label*
with a 5× enrichment ([02_simulate_hospitals.py:201](scripts/02_simulate_hospitals.py:201)).
The code and manuscript both say these counts "carry no evidential weight", which
is correct. But in the README's "Try it" section the `TTR V142I` story is the
emotional core of the pitch, and a reader is invited to conclude that the system
correctly keeps a founder variant flagged. What actually keeps it flagged is
(a) counts synthesized from the answer key and (b) a hand rule the classifier
does not use. This should be flagged inline where the demo is presented, not only
in the methods.

### 3. Flat frequency threshold, no principled floor

Rule 1 uses a single 0.1% cut for all diseases
([hospital_query.py:41](scripts/hospital_query.py:41)); the code comment
concedes clinical labs use disease-specific, usually stricter limits. The model
side uses a smooth slope with no floor, so a handful of carriers can move a
borderline pathogenic variant under the decision threshold — three pathogenic
variants lost their flag for this reason (`MYBPC3 D770N/R502W`, `LDLR E288K`).
Acknowledged in the manuscript; the fix is the same maximum-credible-frequency
floor as clinical issue 1.

### 4. The headline is statistically inconclusive, and the README under-states this

Verified from `data/results_local.json`: the discordant-benign improvement from
public → federated query is **5 changed calls, exact p = 0.0625** — the smallest
value a two-sided sign test can return at that count, i.e. it *cannot* reach
significance. The per-ancestry "no group left behind" claim rests on 12 AFR and
23 SAS pathogenic test rows (`docs/step2_review.md` concern 7). The manuscript
states both honestly. The README does not carry the caveat with the same weight:
"African-ancestry patients came out very slightly ahead" and the bar-chart of
15/7/2 read as findings, when the supporting test is underpowered by
construction.

---

## Privacy (fundamental to the "federated" premise)

### 1. "Only 14 numbers travel" is not, by itself, a privacy guarantee

The README and abstract lean on sharing "only the 14 numbers that define the
model" as the privacy benefit. Weight/gradient exchange in federated learning is
not private without secure aggregation or differential privacy, neither of which
is implemented; the only protection in the whole system is suppressing carrier
counts below 5 ([hospital_query.py:36](scripts/hospital_query.py:36)), and
aggregate genomic queries are a known re-identification vector. The manuscript
limitations say this clearly (ref 18); the README's framing does not, and a
reader is likely to take "federated" to mean "private". One sentence in the
README would close the gap.

---

## Code correctness

No material bug was found. The pieces most likely to hide one were checked:

- `roc_auc`, the logistic fit, and the log-frequency transform pass their
  brute-force `--self-check`, and `03_check_results.py` confirms gradient descent
  reaches the exact Newton optimum (largest coefficient gap 0.011, 1 of 9,492
  test calls differ).
- The test/train split is leak-checked: split by `gene:amino-acid position`
  ([02_simulate_hospitals.py:107](scripts/02_simulate_hospitals.py:107)) with a
  self-test proving the safety `check()` can actually fail.
- Pooled de-duplication, the FedAvg row-count weighting, and the "clients
  received the same global model" assertion are all present and correct.

One framing-level item worth a fix rather than a nit:

- The `own_hospital` evidence is hard-wired to **Oslo**
  ([03_train_local.py:180](scripts/03_train_local.py:180)). Oslo *is* the
  European population that the public reference is built from, so "a hospital
  that consults only its own patients does about as well as the public database"
  (README) is true *only for Oslo* and is close to circular. For Karachi or
  Lagos, consulting their own patients would beat the European reference
  substantially — which is the project's whole thesis. Reporting the own-hospital
  baseline only for the one site where it coincides with the public reference
  understates federation's value and slightly misleads. Consider reporting it
  per-site, or naming it "Oslo's own patients" everywhere.

---

## What is already handled well (for calibration)

So the report is not read as more alarming than it is: label leakage from
ClinVar-trained meta-predictors is deliberately avoided
([01_build_table.py:52](scripts/01_build_table.py:52)); same-position and
whole-gene leakage are both prevented and an "unseen genes" honest-AUC is
reported; the affected/unaffected count split keeps label-derived data out of the
model; reproducibility is pinned and fingerprinted; and the manuscript's
limitations section is candid about most of the fundamental issues above. The gap
this review is really pointing at is between that candid manuscript and a README
that sells the result harder than the evidence supports — plus the two
structural facts that the model is not yet wired to the tool and the evaluation
cannot see the false-negative risk.
