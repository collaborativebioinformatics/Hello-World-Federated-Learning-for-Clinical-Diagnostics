# Inherited cancer results: the second disease area through steps 2, 3 and 4

The heart build is the one every other page describes. This page runs the same
three scripts on the inherited cancer table, 41 genes and 4,107 variants, built
by steps 0 and 1 as described in [disease_areas.md](disease_areas.md). Nothing in
the model, the sampling or the evaluation was changed for it. The scripts took a
`--panel` flag, described in [data_contract.md](data_contract.md), and the whole
run was:

```
uv run python scripts/02_simulate_hospitals.py --panel cancer --set-reference
uv run python scripts/03_train_local.py --panel cancer
uv run python scripts/03_check_results.py --panel cancer
uv run python scripts/04_federated_train.py --panel cancer
```

Every number below is in `data/cancer/results_local.json`,
`data/cancer/results_checks.json` and `data/cancer/results_federated.json`. The
heart numbers quoted for comparison are from [step3_results.md](step3_results.md)
and [step4_results.md](step4_results.md).

**How it was run.** Steps 2 and 3 and the check ran on Windows with `uv run`,
as for the heart. Step 4 did not: NVFlare 2.9.0's simulator starts its
processes with `preexec_fn=os.setsid`, in `nvflare/job_config/fed_job_config.py`,
and Windows Python has no `os.setsid`, so the script stops at its first
`simulator_run` with `AttributeError: module 'os' has no attribute 'setsid'`,
before anything panel-specific happens. A retry with `--runs 1` stopped the same
way. The step 4 numbers on this page were therefore made from the same checkout
in WSL Ubuntu, Linux CPython 3.12.3, with `uv run --frozen` and the same
`uv.lock`, in a throwaway environment under `/tmp` that leaves the Windows one
alone. The five runs took 320 seconds there.

## The build

Step 2 with the same seed and settings as the heart build. Every self-check
passed, and `--self-check --panel cancer` stopped the deliberately broken build
and rebuilt an identical one. The fingerprints are in
`config/reference_build_cancer.json`.

| | population | patients | verdicts | pathogenic | benign | population-discordant |
|---|---|---|---|---|---|---|
| `site_oslo` | European | 20,000 | 1,560 | 647 | 913 | 43 |
| `site_karachi` | South Asian | 4,000 | 667 | 250 | 417 | 38 |
| `site_lagos` | African | 4,000 | 647 | 231 | 416 | 98 |

Test set: 1,829 rows, 582 random and 1,247 from the six unseen genes ATM, BRCA1,
MLH1, RET, SMAD4 and VHL. That is 44.5% of the table, against 27.0% for the
heart, because the draw of unseen genes took BRCA1, the largest gene. 2,278
training variants remain, 550 of them held by more than one hospital.

The test set has 101 population-discordant rows. 100 are benign and are the
rows every false-alarm count below is taken on. The one pathogenic row is
`MUTYH Y151C`, in a gene where two bad copies are needed for disease, so healthy
carriers are common. The pooled model calls it pathogenic under all four
evidence settings, with a probability of 0.545 against a threshold of 0.411
under the federated query. The heart test set has no such row at all.

Test rows by the population a variant is most common in: 229 European with 63
pathogenic, 101 South Asian with 13, 116 African with 10, and 1,383 that gnomAD
never saw in any of the three, with 765 pathogenic. Two of the four slices are
under the 30 positives step 2 asks for before it trusts an AUC, so the
per-population numbers below are reported with that warning attached.

## Step 3: single-site and pooled baselines

The cancer table recommends 16 score columns and step 3 drops `polyphen2_hdiv`
as it does for the heart, leaving 15 scores plus log frequency, 17 weights with
the intercept. The heart model has 12 scores and 14 weights.

| trained on | rows | pathogenic | training AUC | threshold |
|---|---|---|---|---|
| `site_oslo` | 1,560 | 647 | 0.986 | 0.451 |
| `site_karachi` | 667 | 250 | 0.987 | 0.436 |
| `site_lagos` | 647 | 231 | 0.993 | 0.557 |
| pooled | 2,278 | 870 | 0.987 | 0.411 |

### The finding, again

| pooled model | test AUC | false alarms on the 100 discordant benign rows |
|---|---|---|
| 15 scores, no frequency at all | 0.971 | 12 / 100 |
| 15 scores + public frequency | 0.975 | 8 / 100 |
| 15 scores + federated frequency | 0.978 | **3 / 100** |

The same shape as the heart result, where the counts were 15, 7 and 2 of 183.
AUC moves by 0.007 and the false alarms fall fourfold.

### The grid

AUC, all 1,829 test rows:

| trained on | public | own hospital | federated query | oracle |
|---|---|---|---|---|
| `site_oslo` | 0.975 | 0.975 | 0.977 | 0.977 |
| `site_karachi` | 0.975 | 0.975 | 0.978 | 0.978 |
| `site_lagos` | 0.973 | 0.973 | 0.976 | 0.976 |
| pooled | 0.975 | 0.975 | 0.978 | 0.978 |

AUC, the 1,247 rows from genes no hospital has seen:

| trained on | public | own hospital | federated query | oracle |
|---|---|---|---|---|
| `site_oslo` | 0.973 | 0.973 | 0.974 | 0.974 |
| `site_karachi` | 0.974 | 0.973 | 0.975 | 0.975 |
| `site_lagos` | 0.971 | 0.970 | 0.972 | 0.972 |
| pooled | 0.974 | 0.973 | 0.975 | 0.975 |

Flat, as for the heart. Karachi's 667 rows score what the pooled 2,278 do. On
this table one score alone gets most of the way: CADD scores 0.955 on the test
set, AlphaMissense 0.953 and ESM1b 0.900.

### The headline, counted

| evidence | false alarms | rate | 95% interval |
|---|---|---|---|
| public | 8 / 100 | 0.080 | 0.041 to 0.150 |
| own hospital | 9 / 100 | 0.090 | 0.048 to 0.162 |
| federated query | 3 / 100 | 0.030 | 0.010 to 0.085 |
| oracle | 3 / 100 | 0.030 | 0.010 to 0.085 |

The intervals overlap, so the paired comparison on the same variants is the one
that counts:

| subset | comparison | removed | introduced | exact p |
|---|---|---|---|---|
| discordant benign | public → federated query | 5 | 0 | 0.063 |
| discordant benign | own hospital → federated query | 6 | 0 | 0.031 |
| all benign (978) | public → federated query | 6 | 1 | 0.125 |
| all benign (978) | own hospital → federated query | 8 | 0 | 0.008 |

Against the public reference, the federated query removes five alarms and
introduces none on the discordant rows, at p = 0.063, the same five-to-nothing
count and the same p as the heart. On all benign rows the heart reached
p = 0.036 with 20 removed and 8 introduced; here it is 6 and 1, p = 0.125,
because the cancer test set has 978 benign rows against 1,221 and fewer of them
sit near the threshold. Against a hospital that only asks its own patients, the
result is clear on both subsets.

Federation again equals the oracle, 3 of 100 either way. Asking only Oslo's own
patients is again slightly worse than the public reference, 9 against 8.

### The risk side

Sensitivity of the pooled model on the 851 pathogenic test rows, same thresholds:

| trained on | public | own hospital | federated query | oracle |
|---|---|---|---|---|
| pooled | 0.912 | 0.913 | 0.911 | 0.907 |

The check script names the pathogenic variants whose flag is lost against the
public reference once the hospitals answer: `ATM E2039K`, `BRIP1 A349P` and
`CDKN2A D68H`, three of 851, while two others gain a flag. The heart run lost
three as well.

### What the model learned

Pooled weights, largest first:

| feature | weight |
|---|---|
| cadd | 3.79 |
| **log frequency** | **−3.13** |
| alphamissense | 3.01 |
| mpc | 2.61 |
| sift4g | 0.95 |
| mutationassessor | 0.93 |
| (intercept) | −7.30 |

Frequency is again the second strongest input and the only large negative one.
CADD and AlphaMissense swap places compared with the heart, where AlphaMissense
led at 4.01 and frequency stood at −3.73. Per site the frequency weight is −2.70
for Oslo, −2.80 for Karachi and −3.21 for Lagos, the same ordering as the heart,
where the two small hospitals lean on frequency hardest.

### Is the headline sturdy

`03_check_results.py --panel cancer`, the same four questions as for the heart.

- The fit. Largest gap between a step 3 coefficient and the exact optimum
  0.023, and 0 of 7,316 test calls differ between the two fits.
- Fresh cohorts. Across 200 re-drawn patient cohorts with the model held fixed,
  the federated query gives a median of 3 false alarms on the discordant rows,
  lowest 3 and highest 4, against a fixed 8 for the public reference. It beats
  the public reference in 100% of cohorts. Own hospital alone gives a median of
  9, from 7 to 10.
- Adding the answers to the public reference instead of replacing it, which is
  what a real hospital would do, gives the same 3 false alarms and introduces
  none.

Every setting side by side, pooled model:

| frequency evidence | AUC all | AUC unseen genes | false alarms, discordant | false alarms, all benign | sensitivity |
|---|---|---|---|---|---|
| no frequency | 0.971 | 0.968 | 12 | 70 | 0.912 |
| public | 0.975 | 0.974 | 8 | 64 | 0.912 |
| own hospital | 0.975 | 0.973 | 9 | 67 | 0.913 |
| federated query | 0.978 | 0.975 | 3 | 59 | 0.911 |
| public plus query | 0.977 | 0.974 | 3 | 58 | 0.908 |
| oracle | 0.978 | 0.975 | 3 | 55 | 0.907 |

### Per-population AUC, pooled model

| population | rows | pathogenic | public | own hospital | federated query | oracle |
|---|---|---|---|---|---|---|
| afr | 116 | 10 | 0.969 | 0.967 | 0.976 | 0.982 |
| nfe | 229 | 63 | 0.966 | 0.968 | 0.970 | 0.966 |
| sas | 101 | 13 | 0.879 | 0.877 | 0.894 | 0.883 |
| none | 1,383 | 765 | 0.978 | 0.978 | 0.978 | 0.978 |

Ten and thirteen positives in the African and South Asian slices. The
differences here should not be quoted.

## Step 4: federated training with NVFlare

NVFlare 2.9.0, FedAvg, 20 aggregation rounds of 150 local steps, simulator
mode, a server and three clients as separate processes. Five runs, and what
varies between them is the deal, re-drawn with `deal_verdicts` under seeds 100
to 104. The test set, the patient cohorts and the model are the same across
runs, as in the heart run.

Mean (standard deviation over the five runs), scored on the locked test set:

| | AUC, public frequency | AUC, federated count query | false alarms / 100, public | false alarms / 100, federated counts |
|---|---|---|---|---|
| Oslo only | 0.9748 (0.0004) | 0.9771 (0.0004) | 7.2 (1.1) | 3.6 (0.9) |
| **Federated, NVFlare FedAvg** | **0.9752 (0.0003)** | **0.9776 (0.0003)** | **6.4 (0.5)** | **3.0 (0.0)** |
| Everything pooled | 0.9754 (0.0000) | 0.9779 (0.0000) | 7.4 (0.5) | 3.0 (0.0) |

**Federated lands on pooled again.** The AUC gap is 0.0003, the same size as
the run-to-run standard deviation, and under the federated count query both
arms gave exactly 3 false alarms in every one of the five runs. Oslo alone gave
5, 3, 3, 4 and 3, and 8, 6, 8, 8 and 6 under the public reference, so once more
the single site is the only arm that moves with the deal.

Under the public reference the federated arm shows 6.4 false alarms against
7.4 pooled, per run 7, 6, 6, 6, 7 against 7, 7, 7, 8, 8. That is one variant
and should not be read as federation beating pooling. The two arms also use
different thresholds: the federated one is the row-weighted average of each
site's own 95% quantile, 0.439 on average, against 0.412 for pooled, so a
variant near the cut can land on different sides for reasons that have nothing
to do with the weights.

### Per population

AUC on each slice under federated count evidence, mean (sd) over the five runs:

| trained on | European NFE | South Asian SAS | African AFR | none |
|---|---|---|---|---|
| test rows | 229 | 101 | 116 | 1,383 |
| of which pathogenic | 63 | **13** | **10** | 765 |
| Oslo only | 0.9694 (0.0006) | 0.8869 (0.0013) | 0.9755 (0.0013) | 0.9774 (0.0004) |
| Federated, FedAvg | 0.9692 (0.0011) | 0.8925 (0.0020) | 0.9749 (0.0008) | 0.9779 (0.0003) |
| Everything pooled | 0.9706 (0.0003) | 0.8955 (0.0005) | 0.9762 (0.0004) | 0.9779 (0.0000) |

Oslo alone does not collapse on the populations it does not serve here either.
The South Asian slice sits lower for every arm, around 0.89, but it holds 13
pathogenic variants and the slices contain different variants, so only
within-slice, between-arm comparisons mean anything, and those are within a few
thousandths.

### Does the global model let any population down?

Same test rows, two models, so the per-run difference is paired:

| population | its own hospital | that hospital's model alone | federated | difference |
|---|---|---|---|---|
| European NFE | Oslo | 0.9694 (0.0006) | 0.9692 (0.0011) | −0.0002 (0.0016) |
| South Asian SAS | Karachi | 0.8967 (0.0052) | 0.8925 (0.0020) | −0.0042 (0.0043) |
| African AFR | Lagos | 0.9760 (0.0056) | 0.9749 (0.0008) | −0.0011 (0.0053) |

Every difference is within about one standard deviation of zero. The South
Asian one is the largest, −0.0042 on 13 positives, and points the other way
from the heart run's African slice; at this size neither is a result. What
repeats from the heart is the stabilising effect: Karachi's own model varies
by 0.0052 between deals and Lagos's by 0.0056, against 0.0020 and 0.0008 for
the global model.

### The model FedAvg arrived at

On the coefficient the project is about, mean over the five runs:

| | log frequency weight |
|---|---|
| Federated (FedAvg) | −3.05 (0.10) |
| Pooled | −3.22 (0.04) |

Per run the federated weight was −2.95, −3.02, −2.98, −3.07 and −3.21, and
the pooled one −3.19, −3.28, −3.24, −3.19 and −3.19. Step 3's pooled model on
the official deal has −3.13, and Oslo alone ranges from −2.63 to −2.99. The
veto survives averaging here as well. The averaged weights sit further from the
pooled ones than in the heart run: the largest coefficient gap is 0.25 to 0.31
of 17 weights, mean 0.275, against 0.156 of 14 for the heart. The sites are
smaller here and the model has three more weights, which is one plausible
reason, though this was not tested.

### What varies across the five runs

Hospital sizes under each deal, against the intended 0.70 / 0.15 / 0.15 split:

| seed | Oslo | Karachi | Lagos |
|---|---|---|---|
| 100 | 1,562 | 701 | 646 |
| 101 | 1,579 | 661 | 657 |
| 102 | 1,526 | 706 | 672 |
| 103 | 1,584 | 631 | 645 |
| 104 | 1,562 | 681 | 677 |

What crosses the wire is a vector of 17 floats and a row count, per site per
round, plus one threshold per site at the end. No variant, no score, no
verdict.

## What is the same and what is different from the heart

- The shape of the result is the same. A flat AUC, a false-alarm count that
  falls only when the frequency evidence comes from more than one population,
  federation landing on the oracle, and a frequency coefficient that survives
  averaging.
- The counts are smaller. 100 discordant benign rows against 183, so every rate
  is in single digits and the paired tests have the same or less power.
- The table is less balanced. 41.9% pathogenic against 56.4%, and 44.5% of it
  is held out because the unseen-gene draw included BRCA1.
- The feature list is longer, 15 scores against 12, because three columns that
  were too sparse in the heart table are filled in here. The recommended list
  is read from `columns.json` and nothing else needed to change.
- One pathogenic discordant variant exists in this test set, `MUTYH Y151C`, and
  the veto did not misfire on it. That is one variant and says nothing general.
- FedAvg lands on pooled here too, to 0.0003 in AUC and to the variant in false
  alarms under federated counts, and the frequency coefficient comes through
  the averaging at −3.05 against −3.22 pooled. The averaged weights sit a little
  further from the pooled ones than for the heart.
- The step 4 script needs a Linux Python on this laptop. That is NVFlare's
  simulator and has nothing to do with the panel.
