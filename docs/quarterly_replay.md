# Quarterly replay: the hospitals grow, the shared model is retrained each quarter

> **A simulation, replayed.** The three hospitals of step 2 are simulated. Here their cohorts and
> verdicts are made to grow in four equal steps, and the step 3 model is retrained after each one.
> Counts among sick patients were generated from the verdict and only demonstrate the query.
> The pooled fit stands in for the federated model; NVFlare was not run. Nothing here describes a real hospital.

Reproduce with `uv run python scripts/05_quarterly_replay.py`. Every number below is also in [quarterly_replay.json](quarterly_replay.json), which the live pipeline page reads. Run on 2026-09-18.

## What was done

1. Step 2 was replayed with its own functions and its own seed, 12: the same test set, the same dealing of verdicts, the same patient counts. The files this gives were compared byte for byte with the build in `data/`, and they are the same. That build is the last quarter.
2. Each hospital's final cohort was cut into nested sub-cohorts holding 25%, 50% and 75% of its patients. The carriers among the first share of gene copies were drawn without replacement from the final counts, so a carrier seen in one quarter is still there in the next, and each quarter's counts have the distribution step 2 would give a cohort of that size. Sick and healthy patients are split as step 2 splits them. Seed 12, a separate generator from the build's.
3. Each hospital's final verdicts were put in one fixed random order, and a quarter holds the first 25%, 50%, 75% or 100% of them. `af_local` in a quarter's verdict file comes from that quarter's own counts, through step 2's `hospital_view()`.
4. Every quarter was written in the step 2 layout under `data/quarterly/<area>/q1` to `q4`, and step 3's own `load()`, `evidence_columns()`, `fit()` and threshold rule were run on it unchanged. The pooled model, the three `verdicts.csv` stacked with duplicates dropped, is the federated stand-in: step 4 showed FedAvg lands on it. The scores-only model of step 3's ablation gives the `none` column. `own` is Oslo's counts alone, as in step 3.
5. The count query was applied to each quarter's counts with the two rules of `scripts/hospital_query.py`: rule 1 at the gene's line from `too_common_line()`, 0.1% among healthy patients when one bad copy is enough and 1% when both copies must be bad, rule 2 at 3 times more common among the sick, counts under 5 hidden. For a patient at each hospital the table counts the variants whose call changes once the other hospitals answer. At Q4 this was checked against `hospital_query.overview()` on the build itself.

The evidence settings are those of [data_contract.md](data_contract.md): `none` is the model without a frequency column, `public` the European-only reference, `own` the patient's hospital alone, `federated` the highest count any hospital reports, `ceiling` gnomAD's real per-population frequency, which no hospital may use.

## Heart, `cardiac`

8,790 variants, test set 2,373 rows, unseen genes CACNA1C, COL5A2, DMD, MYBPC3, TNNI3, TNNT2. Model: 12 scores plus log frequency. Files under `data/quarterly/cardiac/`. Run time about 5 s.

| hospital | population | final patients | final verdicts |
|---|---|---|---|
| `site_oslo` | nfe | 20,000 | 4,319 |
| `site_karachi` | sas | 4,000 | 1,817 |
| `site_lagos` | afr | 4,000 | 1,921 |

### What each hospital held

| quarter | share | `site_oslo` patients | verdicts (pathogenic) | `site_karachi` patients | verdicts (pathogenic) | `site_lagos` patients | verdicts (pathogenic) | pooled verdicts |
|---|---|---|---|---|---|---|---|---|
| Q1 | 25% | 5,000 | 1,080 (705) | 1,000 | 454 (286) | 1,000 | 480 (258) | 1,917 |
| Q2 | 50% | 10,000 | 2,160 (1,384) | 2,000 | 908 (555) | 2,000 | 960 (530) | 3,594 |
| Q3 | 75% | 15,000 | 3,239 (2,103) | 3,000 | 1,363 (817) | 3,000 | 1,441 (807) | 5,071 |
| Q4 | 100% | 20,000 | 4,319 (2,811) | 4,000 | 1,817 (1,102) | 4,000 | 1,921 (1,084) | 6,417 |

### The pooled model each quarter, false alarms on the 183 population-discordant benign test rows

| quarter | none | public | own | federated | ceiling | frequency weight |
|---|---|---|---|---|---|---|
| Q1 | 15 / 183 | 9 / 183 | 13 / 183 | 4 / 183 | 3 / 183 | -2.74 |
| Q2 | 13 / 183 | 7 / 183 | 10 / 183 | 3 / 183 | 3 / 183 | -3.21 |
| Q3 | 14 / 183 | 7 / 183 | 10 / 183 | 2 / 183 | 2 / 183 | -3.54 |
| Q4 | 15 / 183 | 7 / 183 | 11 / 183 | 2 / 183 | 2 / 183 | -3.73 |

AUC on all test rows, and sensitivity on the pathogenic test rows at the training threshold:

| quarter | AUC none | AUC public | AUC own | AUC federated | AUC ceiling | sens. none | sens. public | sens. own | sens. federated | sens. ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | 0.971 | 0.975 | 0.974 | 0.976 | 0.979 | 0.921 | 0.923 | 0.923 | 0.922 | 0.922 |
| Q2 | 0.971 | 0.976 | 0.975 | 0.977 | 0.980 | 0.919 | 0.916 | 0.918 | 0.917 | 0.914 |
| Q3 | 0.971 | 0.976 | 0.975 | 0.978 | 0.980 | 0.921 | 0.917 | 0.918 | 0.917 | 0.914 |
| Q4 | 0.971 | 0.976 | 0.975 | 0.978 | 0.981 | 0.922 | 0.919 | 0.921 | 0.919 | 0.917 |

Paired on the same rows, public reference to federated query: false alarms removed and introduced, with the exact p of step 3.

| quarter | removed | introduced | p |
|---|---|---|---|
| Q1 | 5 | 0 | 0.0625 |
| Q2 | 4 | 0 | 0.125 |
| Q3 | 5 | 0 | 0.0625 |
| Q4 | 5 | 0 | 0.0625 |

### The count query without a model

For a patient at each hospital, how many variants change call once the other two hospitals answer, of the variants in the table, one row per name.

| quarter | patient at `site_oslo` | patient at `site_karachi` | patient at `site_lagos` | likely harmless once all answer |
|---|---|---|---|---|
| Q1 | 306 | 257 | 82 | 653 |
| Q2 | 430 | 312 | 136 | 769 |
| Q3 | 450 | 317 | 135 | 785 |
| Q4 | 467 | 324 | 143 | 803 |

### Does Q4 land on the known numbers

- Q4 files against the build in `data/`: identical.
- False alarms of the pooled model, Q4 against `results_local.json`: the same. Step 3 has none 15, public 7, own 11, federated 2, ceiling 2; Q4 has none 15, public 7, own 11, federated 2, ceiling 2. Frequency weight -3.7288 in step 3, -3.7288 here.
- Calls changed by the query, Q4 against `hospital_query.overview()` on the build, missense rows: the same. overview() gives site_oslo 467, site_karachi 324, site_lagos 143.

## Inherited cancer, `cancer`

4,107 variants, test set 1,829 rows, unseen genes ATM, BRCA1, MLH1, RET, SMAD4, VHL. Model: 15 scores plus log frequency. Files under `data/quarterly/cancer/`. Run time about 3 s.

| hospital | population | final patients | final verdicts |
|---|---|---|---|
| `site_oslo` | nfe | 20,000 | 1,560 |
| `site_karachi` | sas | 4,000 | 667 |
| `site_lagos` | afr | 4,000 | 647 |

### What each hospital held

| quarter | share | `site_oslo` patients | verdicts (pathogenic) | `site_karachi` patients | verdicts (pathogenic) | `site_lagos` patients | verdicts (pathogenic) | pooled verdicts |
|---|---|---|---|---|---|---|---|---|
| Q1 | 25% | 5,000 | 390 (162) | 1,000 | 167 (55) | 1,000 | 162 (60) | 683 |
| Q2 | 50% | 10,000 | 780 (335) | 2,000 | 334 (115) | 2,000 | 324 (124) | 1,296 |
| Q3 | 75% | 15,000 | 1,170 (500) | 3,000 | 500 (182) | 3,000 | 485 (177) | 1,820 |
| Q4 | 100% | 20,000 | 1,560 (647) | 4,000 | 667 (250) | 4,000 | 647 (231) | 2,278 |

### The pooled model each quarter, false alarms on the 100 population-discordant benign test rows

| quarter | none | public | own | federated | ceiling | frequency weight |
|---|---|---|---|---|---|---|
| Q1 | 11 / 100 | 6 / 100 | 6 / 100 | 2 / 100 | 2 / 100 | -3.30 |
| Q2 | 11 / 100 | 6 / 100 | 6 / 100 | 2 / 100 | 2 / 100 | -3.29 |
| Q3 | 13 / 100 | 8 / 100 | 9 / 100 | 4 / 100 | 4 / 100 | -3.13 |
| Q4 | 12 / 100 | 8 / 100 | 9 / 100 | 3 / 100 | 3 / 100 | -3.13 |

AUC on all test rows, and sensitivity on the pathogenic test rows at the training threshold:

| quarter | AUC none | AUC public | AUC own | AUC federated | AUC ceiling | sens. none | sens. public | sens. own | sens. federated | sens. ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | 0.968 | 0.972 | 0.971 | 0.973 | 0.975 | 0.868 | 0.873 | 0.882 | 0.879 | 0.867 |
| Q2 | 0.970 | 0.974 | 0.974 | 0.976 | 0.977 | 0.891 | 0.886 | 0.894 | 0.891 | 0.879 |
| Q3 | 0.971 | 0.975 | 0.975 | 0.977 | 0.977 | 0.920 | 0.915 | 0.920 | 0.917 | 0.913 |
| Q4 | 0.971 | 0.975 | 0.975 | 0.978 | 0.978 | 0.912 | 0.912 | 0.913 | 0.911 | 0.907 |

Paired on the same rows, public reference to federated query: false alarms removed and introduced, with the exact p of step 3.

| quarter | removed | introduced | p |
|---|---|---|---|
| Q1 | 4 | 0 | 0.125 |
| Q2 | 4 | 0 | 0.125 |
| Q3 | 4 | 0 | 0.125 |
| Q4 | 5 | 0 | 0.0625 |

### The count query without a model

For a patient at each hospital, how many variants change call once the other two hospitals answer, of the variants in the table, one row per name.

| quarter | patient at `site_oslo` | patient at `site_karachi` | patient at `site_lagos` | likely harmless once all answer |
|---|---|---|---|---|
| Q1 | 151 | 132 | 36 | 311 |
| Q2 | 192 | 152 | 51 | 348 |
| Q3 | 204 | 151 | 54 | 357 |
| Q4 | 209 | 152 | 56 | 363 |

### Does Q4 land on the known numbers

- Q4 files against the build in `data/`: identical.
- False alarms of the pooled model, Q4 against `results_local.json`: the same. Step 3 has none 12, public 8, own 9, federated 3, ceiling 3; Q4 has none 12, public 8, own 9, federated 3, ceiling 3. Frequency weight -3.1334 in step 3, -3.1334 here.
- Calls changed by the query, Q4 against `hospital_query.overview()` on the build, missense rows: the same. overview() gives site_oslo 209, site_karachi 152, site_lagos 56.

## Limits

- A simulation from start to end. Which hospital holds which verdict and every patient count are simulated in step 2; here they are additionally cut into quarters. No number describes a real hospital or a real quarter.
- Counts among sick patients were generated from the verdict. They are demonstration only and never reach a model, in step 3 or here.
- The federated model is stood in by the pooled fit. Step 4 showed FedAvg lands on the pooled weights for this model; NVFlare was not run for the quarters.
- Cohorts and verdict piles grow uniformly, by a quarter of the final size each quarter, at all three hospitals at once. Real hospitals do not.
- The sub-cohorts are drawn from the final counts, so the quarters are nested and Q4 is the build. An independent draw per quarter would give the same distribution but would let a carrier vanish between quarters and would not reproduce Q4.
- The test set is locked and the same in every quarter, so the model is scored on a fixed yardstick while its training rows grow. The discordant benign rows are few, and step 3's caution about single-digit counts applies to every quarter.
- The count query here covers the missense table only. `hospital_query.overview()` on the build also answers for `data/other_types/` when that is built; those rows were not replayed.
