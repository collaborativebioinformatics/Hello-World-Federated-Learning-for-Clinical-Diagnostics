# Step 2 is open for review: input wanted from the ML side

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

**1. The random part of the test set leaks. This is the one to fix first.**
39% of the `random` test rows (619 of 1,602) sit at the same gene and amino-acid position as a training row, for example two different DNA changes that both give `ACTA2 M46I`. Their scores are near-identical and they usually share a verdict. The `unseen_gene` part has no such overlap (0 of 777). Until fixed, trust the unseen-gene score. Proposed fix: split by gene plus position, so all variants at one position stay together.

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

**4. The hospitals differ in verdict mix, not only in population.**
Benign variants are the ones with a measurable frequency, so population-weighted dealing pulls them to the small sites: Oslo is 65% pathogenic, Karachi 52%, Lagos 48%, the test set 48%. Realistic, but it is label skew on top of feature skew.

**5. Missing scores.**
39% of training rows lack at least one of the 13 recommended scores. Almost all of that is `polyphen2_hdiv` (31% missing in the training rows; it scraped under the 30% limit on the full table). The rest are under 10%. Either drop that column or decide how to fill gaps without hospitals sharing statistics, for example a fixed constant plus a "was missing" flag.

**6. Frequency columns are mostly zero.**
`af_local` is 0 for 84% of Oslo's rows, 75% of Karachi's and 66% of Lagos's. Any log transform needs a floor, e.g. `log10(af + 1e-6)`.

**7. The population-discordant test rows are all benign.**
180 of them (110 random, 70 unseen-gene), none pathogenic. AUC is meaningless there. Use the false-alarm rate under the four evidence settings in the data contract.

**8. Expect the headline comparison to be flat.**
A small model on 13 scores saturates with a few hundred rows, and "common means harmless" is the same rule in every population. Single-site, federated and pooled AUC will probably land close together. The result we expect to be large is the false-alarm rate: public-only evidence against federated counts. Worth designing the experiments around that rather than being surprised by it.

**9. Counts among affected patients give the answer away.**
`ac_affected` was generated from the verdict. It is for the patient-query demo only and must never be a model input. Counts among unaffected patients are safe.

## How to respond

- Tell us in the team chat, or edit this page.
- Or change the settings yourself: edit the constants at the top of `scripts/02_simulate_hospitals.py`, rerun, then run it once more with `--set-reference` and commit `config/reference_build.json`. That puts the whole team back on identical data.

Once the ML side is happy with the split, step 2 turns green in [recipe_status.png](recipe_status.png).
