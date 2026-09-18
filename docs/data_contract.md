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

> **Reviewed 17 September 2026.** The ML side has been through the sampling choices; the reply and what changed are in [step2_review.md](step2_review.md). The position leak is fixed, so the whole test set is now usable, not only the `unseen_gene` part.

## Three rules

1. **Train only on `site_*/verdicts.csv`.** Never on `test/variants.csv`, and never on `variants.csv`, because that file still contains the test rows. The "everything pooled" baseline is the three `verdicts.csv` files stacked.
2. **Model inputs are the score columns listed under `features` in `columns.json`, plus `af_public` or `af_local`.** Nothing else. `label` is the answer.
3. **Never feed `ac_affected` or `an_affected` to a model.** Those counts were generated using the verdict, so they give the answer away. They exist to demonstrate the patient query.

## The files

### `site_*/verdicts.csv`: a hospital's private classified variants

One row per variant, and a variant appears at most once per hospital. A variant common in one population mostly landed at that population's hospital, so the three files differ in kind, not only in size.

Every variant has one owning hospital, and with `SITE_OVERLAP = 0.5` another hospital may also have classified it, the way real labs overlap. That second draw follows population alone, so a small lab still meets what is common among its own patients. About a quarter of the training variants are held by more than one hospital. Set `SITE_OVERLAP = 0.0` for a clean partition, which is the sharper federated-versus-pooled contrast and the control run worth having.

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

- `random`: a random fifth of the variants, with the same mix of verdicts and of population-discordant variants as the training rows. Split by amino-acid **position**, not by row, so two DNA changes that both give `ACTA2 M46I` cannot sit on opposite sides of the split.
- `unseen_gene`: every variant of six whole genes that no hospital has. This answers "does the model work on a gene it has never seen?", which guards against a model that merely recognises genes.

It also carries `pop`: the site population the variant is most common in, or `none` when gnomAD never saw it in any of the three. This is for the per-population AUC in README section 5, and it is a thin instrument: 1,459 of 2,373 test rows are `none`, and the AFR slice has 12 pathogenic rows. Step 2 prints every slice's positive count. Report the evidence-setting comparison below as the headline and treat per-population AUC as secondary, always with its sample size.

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

## Every build checks itself

Step 2 stops rather than writing a split that is quietly wrong. It asserts that no test variant reached a hospital file, that no hospital lists a variant twice, that every hospital has both verdicts and at least 100 pathogenic rows, and that each hospital's `af_local` really came from its own population rather than a neighbour's cohort.

`uv run python scripts/02_simulate_hospitals.py --self-check` hands every hospital the wrong cohort on purpose and expects the run to stop. An assertion that cannot fail reads like a guarantee and is worse than none; an earlier version of this one compared `af_local` against the same constant it was derived from and passed no matter what.

## This run

Seed 12, `SITE_OVERLAP` 0.5. Settings sit at the top of `scripts/02_simulate_hospitals.py`.

| | population | patients | verdicts | pathogenic | benign | population-discordant |
|---|---|---|---|---|---|---|
| `site_oslo` | European | 20,000 | 4,319 | 2,811 | 1,508 | 95 |
| `site_karachi` | South Asian | 4,000 | 1,817 | 1,102 | 715 | 114 |
| `site_lagos` | African | 4,000 | 1,921 | 1,084 | 837 | 259 |

Test set: 2,373 rows. 1,596 random, 777 from the unseen genes CACNA1C, COL5A2, DMD, MYBPC3, TNNI3, TNNT2. 1,525 of the 6,417 training variants are held by more than one hospital.

Patient settings are illustrative, not estimates: 15% of each cohort are heart patients, a fifth of those are explained by one pathogenic variant, and pathogenic variants are five times more common among the affected.

## A second disease area

Everything above describes the heart build, which keeps its place at the top of `data/`. Any other disease area lives in a folder of its own, `data/<panel>/`, with the same file names and the same 40 table columns in the same order. The inherited cancer build is in `data/cancer/`. Build it with:

```
uv run python scripts/00_fetch_gene_panel.py --panel cancer
uv run python scripts/01_build_table.py --panel cancer
```

Step 0 writes `config/cancer_gene_panel.txt`. Step 1 writes `data/cancer/variants.csv` and `data/cancer/columns.json`. The download cache `data/raw/` is shared by every panel, because the query for a gene is the same whichever panel asks for it. With no `--panel`, both steps behave exactly as before and the heart files keep the same bytes.

Two things change from one disease area to the next. Both are recorded in files, so a script should read them from there.

- The `features` list in `columns.json` is decided per table by the 30% missing rule. The heart table recommends 13 score columns and the cancer table recommends 16. The 13 heart columns are all among the 16.
- The demo variants. `DSP N1526K` is a heart variant and is absent from the cancer table. Candidates for cancer are listed in [disease_areas.md](disease_areas.md).

**Reading it from steps 2 to 4.** Steps 2, 3 and 4 and the step 3 check take the same `--panel NAME` flag as steps 0 and 1, default `cardiac`. Each one sets its `DATA_DIR` to `data/` for `cardiac` and to `data/<panel>/` for anything else, the rule `table_folder()` follows in `scripts/01_build_table.py`, and every path it reads or writes hangs off that line. The whole cancer pipeline after step 1 is:

```
uv run python scripts/02_simulate_hospitals.py --panel cancer
uv run python scripts/03_train_local.py --panel cancer
uv run python scripts/03_check_results.py --panel cancer
uv run python scripts/04_federated_train.py --panel cancer
```

Step 2 keeps one reference build per panel: `config/reference_build.json` holds the heart fingerprints and `config/reference_build_<panel>.json` any other panel's, so the cancer ones are in `config/reference_build_cancer.json`, and `--set-reference` writes the file of the panel it was given. Step 4 keeps its workspace under `data/<panel>/fedavg_runs/` and writes `data/<panel>/results_federated.json`. The NVFlare client script reads only the run folder step 4 hands it, so it needed no change. NVFlare's simulator did not start under Windows Python on the laptop this was built on, whatever the panel, so the cancer step 4 was run in WSL Ubuntu from the same checkout; the details are in [cancer_results.md](cancer_results.md). The step 6 query scripts were not changed and still read the heart build at the top of `data/`.

With no `--panel`, steps 2, 3 and 4 write the same bytes as before: the SHA256 of every file they write was compared with the heart build made before the change, and step 2 still reports the heart build identical to the team's reference. The cancer results are in [cancer_results.md](cancer_results.md).
