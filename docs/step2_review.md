# Step 2 review: answered by the ML side

> **Reviewed on 17 September 2026.** Four of the settings below changed as a
> result; the review reply is at the bottom of this page. Concerns 1, 4 and 7
> are resolved, the rest stand. `config/reference_build.json` was regenerated,
> so pull and rerun step 2 before you train on anything.

Step 2 (`scripts/02_simulate_hospitals.py`) locks the test set and creates the three hospitals. It runs and is reproducible, but the sampling choices were made on the data side. Before anyone trains on these files, we would like the people with ML experience to look at how the split is made. This page lists what it does, every choice that can be changed, and the concerns we already know about.

Background reading: [data_contract.md](data_contract.md) for the files and columns, [table_columns.md](table_columns.md) for the variant table.

## What it does, in five lines

1. Reads the variant table: 8,790 real, labelled variants.
2. Locks a test set: six whole genes no hospital gets, plus a random fifth of the rest.
3. Deals every remaining row to exactly one hospital. Rows are copied unaltered. A variant common in one population mostly lands at that population's hospital; a variant nobody has seen goes anywhere, in proportion to lab size.
4. Simulates each hospital's patient cohort from its population's real gnomAD frequencies, giving carrier counts per variant.
5. Gives each hospital row two frequencies: `af_public` (European reference) and `af_local` (its own unaffected patients).

Only the counts are invented. Verdicts, scores and population frequencies are real.

## The choices, and the question for each

All of these are constants at the top of the script.

| setting | now | question |
|---|---|---|
| `TEST_SHARE` | 0.20, stratified by verdict and by the population-discordant flag | right size? right strata? |
| `UNSEEN_GENES` | 6, picked at random once among genes with at least 10 of each verdict | better as k-fold over genes, so the score does not hinge on one draw? |
| one hospital per variant | yes, no overlap | real labs overlap. Does a clean partition matter for FedAvg, or should common variants appear at several sites? |
| population pull in the dealing | proportional to each population's share of carriers | this sets how non-IID the sites are. Do you want a dial from fully random to fully population-driven, e.g. as the control experiment? |
| `verdict_share` | Oslo 0.70, Karachi 0.15, Lagos 0.15 | are the small sites small enough to show a federated gain? Should we sweep the size? |
| cohort sizes | 20,000 / 4,000 / 4,000 patients | with cohorts this big `af_local` is almost exactly the true frequency (r = 1.00). Smaller cohorts would make local evidence noisier and the federated sum more valuable |
| `SEED` | 12, a single run | several seeds for error bars? |
| public reference | European frequency only | a deliberate pretence, see the data contract. Acceptable framing? |

## Known concerns

**1. The random part of the test set leaks. This is the one to fix first.** — **FIXED.**
39% of the `random` test rows (619 of 1,602) sat at the same gene and amino-acid position as a training row, for example two different DNA changes that both give `ACTA2 M46I`. Their scores are near-identical and they usually share a verdict. The proposed fix was the right one and is now in: `lock_test_set` splits by `gene:position` via `amino_acid_position()`, so every variant at one position lands on the same side. Leakage is now **0 of 1,596**, and the whole test set is trustworthy, not only the unseen-gene part.

One trap worth recording: the position must be read from the amino-acid change only. `name.str.extract(r"(\d+)")` picks up the digits in the *gene* name, so `COL1A1 G272D` becomes position 1 and the entire gene collapses into one group. The regex is anchored as `" [A-Za-z](\d+)[A-Za-z]$"`.

**2. The unseen-gene test is dominated by one gene.**
DMD is 411 of its 777 rows and 95% benign. The other five genes have 59 to 99 rows each. A per-gene score, or k-fold over genes, would be more honest than one pooled number.

| unseen gene | rows | pathogenic |
|---|---|---|
| DMD | 411 | 20 |
| COL5A2 | 99 | 14 |
| CACNA1C | 75 | 38 |
| TNNI3 | 69 | 44 |
| MYBPC3 | 64 | 34 |
| TNNT2 | 59 | 48 |

**3. Labels track genes.**
Three genes supply 45% of all pathogenic rows and five supply 41% of the benign ones. A model can look good by recognising the gene. This is why the unseen-gene test exists; it may also argue for leaving gene-level scores such as `mpc` and `bstatistic` out, or at least checking their importance.

**4. The hospitals differ in verdict mix, not only in population.** — **EASED, on purpose kept.**
Label skew on top of feature skew is realistic and we want to keep it. What was not acceptable was the absolute size: Karachi had 609 positives and Lagos 569, which is thin for a per-site baseline. `SITE_OVERLAP = 0.5` lets a variant be classified by more than one lab, which is what real labs do, and the small sites roughly doubled: Karachi 1,102 positives, Lagos 1,084. The skew itself barely moved.

**5. Missing scores.**
39% of training rows lack at least one of the 13 recommended scores. Almost all of that is `polyphen2_hdiv` (31% missing in the training rows; it scraped under the 30% limit on the full table). The rest are under 10%. Either drop that column or decide how to fill gaps without hospitals sharing statistics, for example a fixed constant plus a "was missing" flag.

**6. Frequency columns are mostly zero.**
`af_local` is 0 for 84% of Oslo's rows, 75% of Karachi's and 66% of Lagos's. Any log transform needs a floor, e.g. `log10(af + 1e-6)`.

**7. The population-discordant test rows are all benign.** — **AGREED, and it generalises.**
183 of them after the resplit, none pathogenic. AUC is meaningless there; the false-alarm rate under the four evidence settings is the right measure.

The same arithmetic kills the per-population AUC in README section 5, which is worth stating plainly. Labelling each test row by the population it is most common in gives 382 NFE rows / 58 pathogenic, 291 SAS / 23, 241 AFR / **12**, and 1,459 rows with no population at all because gnomAD never saw the variant in any of the three. An AUC on 12 positives carries a confidence interval wider than any federated gain we could hope to measure. Every run now prints these slice sizes so nobody quotes such a number by accident.

So: the evidence-setting comparison in the data contract is the headline, on all 2,373 rows. Per-population AUC is a secondary read on the 914 rows that have a population, always reported with its positive count. `test/variants.csv` carries a `pop` column for it.

**8. Expect the headline comparison to be flat.**
A small model on 13 scores saturates with a few hundred rows, and "common means harmless" is the same rule in every population. Single-site, federated and pooled AUC will probably land close together. The result we expect to be large is the false-alarm rate: public-only evidence against federated counts. Worth designing the experiments around that rather than being surprised by it.

**9. Counts among affected patients give the answer away.**
`ac_affected` was generated from the verdict. It is for the patient-query demo only and must never be a model input. Counts among unaffected patients are safe.

## Review reply, 17 September 2026

Short version: the split was sound, the one thing that had to change was the leak you had already found. Four changes went in.

| change | why |
|---|---|
| test split by `gene:position` | concern 1. Leakage 39% -> 0% |
| `SITE_OVERLAP = 0.5`, a new constant | concern 4. Real labs overlap, and the small sites needed the rows. Set it to `0.0` for the clean-partition control run, which is the sharper federated-vs-pooled contrast |
| `check()` runs on every build | four assertions: no test variant in a hospital file, no variant twice in one hospital, both verdicts present with at least `MIN_PATHOGENIC_PER_SITE` positives, and `af_local` really drawn from that hospital's own population |
| `pop` column on the test set, slice sizes printed | concern 7. Makes the per-population AUC's sample size impossible to miss |

Answers to the settings table, where we did not change anything:

- **`TEST_SHARE` 0.20 and the verdict strata** — right. Keep.
- **`UNSEEN_GENES` = 6, one draw** — k-fold over genes is the better instrument and concern 2 is real, but it is an evaluation-side change, not a data-side one. Step 5 can fold over the six genes it already has and report per-gene numbers; DMD alone is 411 of the 777 rows and should never be pooled with the rest.
- **Population pull dial** — `SITE_OVERLAP` is one dial; the pull itself we left alone. If step 5 wants a fully-random control, set `SITE_OVERLAP = 0.0` and the population weighting in `deal_verdicts` to uniform.
- **`verdict_share` 0.70/0.15/0.15** — keep. With overlap the small sites now have enough to train on, and the asymmetry is the point.
- **Cohort sizes** — agreed that `r = 1.00` makes `af_local` too good. Worth a sweep in step 5, but it costs nothing to leave as is until a federated gain actually needs explaining.
- **`SEED` = 12, single run** — fine for the demo. Error bars over seeds are a step 5 concern.
- **European-only public reference** — acceptable and well framed. It is the whole premise of the patient demo.

Still open, for whoever trains:

- **Concern 5, `polyphen2_hdiv` 31% missing in training rows.** Our call: drop it, or add a `was_missing` flag and fill with a fixed constant. Do not impute from statistics, since that would need hospitals to share distributions.
- **Concern 8, expect a flat headline.** Agreed, and it is the reason the evidence-setting comparison is now the headline rather than the three-arm AUC table.
- **Concern 9, `ac_affected` must never be an input.** Agreed. It lives in `patient_counts.csv`, never in `verdicts.csv`, so a model trained on `verdicts.csv` cannot reach it by accident.

## How to respond

- Tell us in the team chat, or edit this page.
- Or change the settings yourself: edit the constants at the top of `scripts/02_simulate_hospitals.py`, rerun, then run it once more with `--set-reference` and commit `config/reference_build.json`. That puts the whole team back on identical data.

Once the ML side is happy with the split, step 2 turns green in [recipe_status.png](recipe_status.png).
