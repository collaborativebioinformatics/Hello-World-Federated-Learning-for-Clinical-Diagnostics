# Data contract: what each file is, and what you may train on

Steps 1 and 2 write everything below into `data/`. The seed is fixed and package versions are locked, so everyone who starts from the same downloaded table gets byte-identical files, on Windows, Mac or Linux.

Step 2 ends by telling you whether your build is **identical to the team's reference**, recorded in `config/reference_build.json`. It can differ for one reason only: step 1 downloads from a live API, and myvariant.info may have updated its data since the reference was made. If that happens the script says so and tells you what to do.

```
uv run python scripts/01_build_table.py
uv run python scripts/02_simulate_hospitals.py
```

```
data/
  variants.csv                  the full table from step 1. Reference only, see rule 1
  columns.json                  which columns are scores, frequencies and label
  public_reference.csv          the frequency every hospital already has
  sites.json                    settings and sizes of this run
  test/variants.csv             locked test set, never trained on
  site_oslo/verdicts.csv        Oslo's private classified variants
  site_oslo/patient_counts.csv  Oslo's private carrier counts
  site_karachi/...              same two files
  site_lagos/...                same two files
```

> **In review.** The sampling choices in step 2 are open for input from the ML side, see [step2_review.md](step2_review.md). Known issue: 39% of the `random` test rows share a gene and amino-acid position with a training row, so that part of the test set flatters the model. Until it is fixed, trust the `unseen_gene` score.

## Three rules

1. **Train only on `site_*/verdicts.csv`.** Never on `test/variants.csv`, and never on `variants.csv`, because that file still contains the test rows. The "everything pooled" baseline is the three `verdicts.csv` files stacked.
2. **Model inputs are the score columns listed under `features` in `columns.json`, plus `af_public` or `af_local`.** Nothing else. `label` is the answer.
3. **Never feed `ac_affected` or `an_affected` to a model.** Those counts were generated using the verdict, so they give the answer away. They exist to demonstrate the patient query.

## The files

### `site_*/verdicts.csv`: a hospital's private classified variants

One row per variant. Every variant belongs to exactly one hospital. A variant common in one population mostly landed at that population's hospital, so the three files differ in kind, not only in size.

| column | meaning | use |
|---|---|---|
| `name`, `gene` | readable name and gene | reading only |
| `verdict`, `label` | ClinVar's verdict in words, and as 1 or 0 | **the answer** |
| `stars` | how well reviewed the verdict is, 1 to 4 | optional row weight |
| `af_public` | frequency in the public reference. Ours covers Europeans only | input |
| `af_local` | frequency among this hospital's own unaffected patients | input |
| 13 score columns | see `table_columns.md` | inputs, listed in `columns.json` |
| `variant_id`, `clinvar_id` | exact DNA change, ClinVar id | key |

The hospital files deliberately contain no gnomAD per-population columns. A hospital only knows the public reference and its own patients.

### `site_*/patient_counts.csv`: a hospital's private carrier counts

One row for every variant in the whole table, test variants included, because patients carry what they carry whether or not anyone has classified it. This is also exactly what a hospital returns when queried about a variant in step 6.

| column | meaning |
|---|---|
| `variant_id`, `name` | which variant |
| `ac`, `an` | copies of the variant seen, out of gene copies looked at (two per patient) |
| `ac_unaffected`, `an_unaffected` | the same among patients without heart disease. **Fair evidence:** drawn from the real population frequency only, knows nothing about the verdict |
| `ac_affected`, `an_affected` | the same among heart patients. **Demo only, see rule 3** |

### `test/variants.csv`: the locked test set

All columns of `variants.csv`, plus `af_public`, plus `test_kind`:

- `random`: a random fifth of the variants, with the same mix of verdicts and of population-discordant variants as the training rows.
- `unseen_gene`: every variant of six whole genes that no hospital has. This answers "does the model work on a gene it has never seen?", which guards against a model that merely recognises genes.

This file keeps gnomAD's real `af_nfe`, `af_sas`, `af_afr`. They are the truth to score against, not inputs a hospital would have.

### `public_reference.csv`

`variant_id`, `af_public`. The European frequency, for every variant.

## Scoring a test variant under different evidence

The same trained model can be given different frequency evidence. This is the comparison that shows what federation buys.

| setting | frequency given to the model | where it comes from |
|---|---|---|
| today's world | `af_public` | `test/variants.csv` |
| own hospital only | that site's `ac_unaffected / an_unaffected` | that site's `patient_counts.csv` |
| federated | the sites' unaffected counts combined, for example the highest site frequency with a minimum count | all three `patient_counts.csv`, counts only |
| ceiling | the highest of `af_nfe`, `af_sas`, `af_afr` | `test/variants.csv`, not allowed in real life |

On `pop_discordant == 1` test rows, which are almost all benign, report the false-alarm rate rather than AUC.

## What is real and what is simulated

| real | simulated |
|---|---|
| every verdict, score and gnomAD frequency | which hospital holds which verdict |
| the gene list | every patient count |
| | the public reference covering Europeans only. That is a deliberate pretence, standing in for the many populations real references miss. gnomAD's real South Asian and African columns play the truth only the local hospital can see |

## This run

Seed 12. Settings sit at the top of `scripts/02_simulate_hospitals.py`.

| | population | patients | verdicts | pathogenic | benign | population-discordant |
|---|---|---|---|---|---|---|
| `site_oslo` | European | 20,000 | 4,057 | 2,632 | 1,425 | 81 |
| `site_karachi` | South Asian | 4,000 | 1,164 | 609 | 555 | 111 |
| `site_lagos` | African | 4,000 | 1,190 | 569 | 621 | 250 |

Test set: 2,379 rows. 1,602 random, 777 from the unseen genes CACNA1C, COL5A2, DMD, MYBPC3, TNNI3, TNNT2.

Patient settings are illustrative, not estimates: 15% of each cohort are heart patients, a fifth of those are explained by one pathogenic variant, and pathogenic variants are five times more common among the affected.
