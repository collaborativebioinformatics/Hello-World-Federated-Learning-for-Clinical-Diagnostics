# Step 3 results: single-site and pooled baselines

The model is **logistic regression**:

```
p(pathogenic) = sigmoid( w · [12 prediction scores, log frequency] + b )
```

Fourteen weights including the intercept. Deliberately small. Weight averaging is
clean for a linear model and a heuristic for anything deeper, our sites are
non-IID on purpose, and the claim to defend is "frequency is the lever", which is
a statement about one coefficient. A stronger model would also be better at the
thing we do not want: recognising the gene, which 45% of the pathogenic rows
would let it away with ([step2_review.md](step2_review.md) concern 3).

No PyTorch. The weights are one vector of 14 floats, so FedAvg in step 4 is a
mean over three of them, and "only weights travel" is literally 14 numbers.

Reproduce with `uv run python scripts/03_train_local.py`. Every number below is
also in `data/results_local.json`.

## Setup

**Features.** The 13 recommended score columns from `columns.json`, minus
`polyphen2_hdiv` (missing for 29-32% of training rows), leaving 12. Remaining
gaps, about 13% of rows, are filled with the neutral rank score 0.5. They are not
imputed from a distribution, which would need hospitals to share one, and there
is no "was missing" flag: gaps follow the gene and the gene follows the label, so
the flag would smuggle the answer in as a feature.

Frequency is the only input that is not already a 0-to-1 rank score, and 76-85%
of `af_local` is exactly zero, so it enters as `log10(af + 1e-6)` rescaled to
0-1 by a **fixed** transform. Fixed matters: a fitted scaler is a per-site
statistic that hospitals would have to share, which is what rank scores were
chosen to avoid.

**Threshold.** Set on each model's own training rows at 95% sensitivity, then
applied unchanged to the test set. Choosing it on the test set would tune the
thing being measured.

**Evidence settings.** One trained model, four answers to "how common is this
variant", defined in [data_contract.md](data_contract.md).

## What was trained

| trained on | rows | pathogenic | training AUC | threshold |
|---|---|---|---|---|
| `site_oslo` | 4,319 | 2,811 | 0.982 | 0.616 |
| `site_karachi` | 1,817 | 1,102 | 0.986 | 0.562 |
| `site_lagos` | 1,921 | 1,084 | 0.986 | 0.561 |
| pooled | 6,417 | 3,808 | 0.984 | 0.575 |

Pooled is the three `verdicts.csv` stacked with duplicates dropped, so a variant
two labs both classified counts once.

## The finding: frequency changes the decision, not the ranking

This is the result to build the presentation on, and it is not the one the
README section 5 grid was drawn to show.

| pooled model | test AUC | false alarms on the 183 discordant benign rows |
|---|---|---|
| 12 scores, no frequency at all | 0.971 | 15 / 183 |
| 12 scores + public frequency | 0.976 | 7 / 183 |
| 12 scores + federated frequency | 0.978 | **2 / 183** |

AUC moves by 0.007 and would be invisible on any plot. False alarms fall
**sevenfold**. A ranking metric cannot see what frequency does, because frequency
does not reorder variants, it vetoes a small number of them — exactly the ones
where a hospital would otherwise have called a harmless variant disease-causing
in a patient whose ancestry it rarely sees.

## The grid

AUC, all 2,373 test rows:

| trained on | public | own hospital | federated | ceiling |
|---|---|---|---|---|
| `site_oslo` | 0.976 | 0.975 | 0.978 | 0.980 |
| `site_karachi` | 0.976 | 0.975 | 0.978 | 0.980 |
| `site_lagos` | 0.975 | 0.974 | 0.978 | 0.980 |
| pooled | 0.976 | 0.975 | 0.978 | 0.981 |

AUC, the 777 rows from genes no hospital has seen, which is the honest number:

| trained on | public | own hospital | federated | ceiling |
|---|---|---|---|---|
| `site_oslo` | 0.971 | 0.969 | 0.974 | 0.976 |
| `site_karachi` | 0.969 | 0.968 | 0.973 | 0.975 |
| `site_lagos` | 0.968 | 0.966 | 0.972 | 0.975 |
| pooled | 0.971 | 0.969 | 0.974 | 0.977 |

**Every cell is the same.** Training on 1,817 Karachi rows scores what training on
all 6,417 does, and single-site matches pooled to three decimals. Concern 8 of
the step 2 review predicted this and it is confirmed: a 14-weight model on 12
scores saturates long before 1,800 rows, and "common means harmless" is the same
rule in every population. **Do not expect FedAvg in step 4 to move these numbers,
and do not present a flat table as a result.** The federated row will land on top
of the others, which is the correct outcome and a boring one.

The AUC is high because the task is easier than it looks, not because anything
leaks. AlphaMissense alone scores 0.950 on this test set, CADD 0.906, MPC 0.917.
Our labels are star-rated ClinVar consensus calls in 97 well-studied cardiac
genes, which is the subset experts already agree on. Real VUS are much harder.
Test and training share no variant, verified in step 2.

## The headline, counted rather than rated

183 rows turns every rate into single digits, so here are the digits.

| evidence | false alarms | rate | 95% interval |
|---|---|---|---|
| public | 7 / 183 | 0.038 | 0.019 to 0.077 |
| own hospital | 11 / 183 | 0.060 | 0.034 to 0.104 |
| federated | 2 / 183 | 0.011 | 0.003 to 0.039 |
| ceiling | 2 / 183 | 0.011 | 0.003 to 0.039 |

Those intervals overlap. Quoting "a threefold drop in false alarms" from them
alone would not survive a referee. The settings are scored on the **same**
variants, so the paired comparison is the one that carries weight:

| subset | comparison | removed | introduced | exact p |
|---|---|---|---|---|
| discordant benign | public → federated | 5 | 0 | 0.063 |
| discordant benign | own hospital → federated | 9 | 0 | 0.004 |
| all benign (1,221) | public → federated | 20 | 8 | 0.036 |
| all benign (1,221) | own hospital → federated | 21 | 0 | <0.001 |

Honest reading: against **today's public reference**, federation removes false
alarms and introduces none on the pre-registered discordant subset, but at 5
events that is p = 0.063 and underpowered. On all benign rows it reaches
p = 0.036. Against a **hospital using only its own patients** it is unambiguous.

Two things are not underpowered and matter as much:

**Federation equals the ceiling.** 2 / 183 either way. Exchanging counts loses
nothing against being handed gnomAD's real per-population frequencies, which no
hospital is allowed. There is no accuracy argument left for pooling raw rows.

**Asking only your own hospital is worse than the public reference**, 11 false
alarms against 7. Oslo's cohort is 34,000 gene copies from unaffected patients;
gnomAD's European set is 113,000. A single hospital's own data is a noisier
version of what it already has, even for its own population. The gain comes from
combining, not from localness.

## The risk side

Sensitivity on all 1,152 pathogenic test rows, at the same thresholds:

| trained on | public | own hospital | federated | ceiling |
|---|---|---|---|---|
| pooled | 0.919 | 0.921 | 0.919 | 0.917 |

Flat. The false alarms are not bought by missing pathogenic variants, which is
the failure mode a frequency veto invites. Report this next to the headline; a
false-alarm reduction without it means nothing.

The gap that remains: the test set contains **no** pathogenic discordant variant,
so we cannot measure the veto misfiring on a variant that is both common and
genuinely disease-causing. Only two such variants exist in the whole table,
`TTR V142I` and `KCNQ1 G92A`, and both are in the training pool. That is biology
rather than a sampling accident, since a variant causing severe disease stays
rare unless it is late-onset, and it means the case belongs in the discussion as
a named example, not in a statistics table.

## What the model learned

Pooled weights, largest first. Negative pushes towards benign.

| feature | weight |
|---|---|
| alphamissense | 4.01 |
| **log frequency** | **−3.73** |
| mpc | 3.05 |
| cadd | 2.51 |
| sift | 1.44 |
| provean | 0.96 |
| (intercept) | −6.55 |

This is why the model is linear. Frequency is the second strongest input of
thirteen and the only large negative one: the model learned the veto from expert
verdicts, without being told the rule. On a two-layer MLP this table would be a
SHAP plot and an argument.

## Per-population AUC

Reported because README section 5 asks for it, with the positive counts that say
what it is worth.

| population | rows | pathogenic | public | own hospital | federated | ceiling |
|---|---|---|---|---|---|---|
| afr | 241 | 12 | 0.950 | 0.948 | 0.952 | 0.963 |
| nfe | 382 | 58 | 0.935 | 0.934 | 0.934 | 0.935 |
| sas | 291 | 23 | 0.917 | 0.916 | 0.926 | 0.934 |
| none | 1,459 | 1,059 | 0.976 | 0.976 | 0.976 | 0.976 |

Twelve positives in the African slice. The differences here are noise and should
not be quoted. `none` is the two thirds of test rows gnomAD never saw in any of
the three populations.

## For step 4

1. FedAvg will not beat these AUCs. Present the federated row as landing on the
   pooled row, which is the actual claim: no raw data moved and nothing was lost.
2. Score the federated model under the four evidence settings and the paired
   false-alarm test, not on AUC.
3. The single-site rows above are the baseline federation must match. They are
   already strong, so the win to argue is privacy at equal accuracy, plus the
   false-alarm result, which is about evidence at query time rather than about
   training at all.
