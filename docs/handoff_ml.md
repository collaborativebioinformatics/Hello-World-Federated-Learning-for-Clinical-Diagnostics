# Handoff to the ML side

Written 18 September 2026 for Yan, or for an agent working on Yan's side. It covers only what bears on training: the tables, the flag, the results and the open decisions. The patient query and its screen are Mohit's side and are left out. Every number here was computed from a build on that date; the docs named below hold the full tables.

## What is on main

- **Steps 0 and 1 take `--panel`.** `cardiac` is the default and keeps writing `data/variants.csv` and `data/columns.json` with the same bytes as the team reference. Any other name reads `config/<panel>_gene_panel.txt` and writes `data/<panel>/`. Panel sets are listed in `config/panel_sets.json`. See [disease_areas.md](disease_areas.md).
- **Two more areas exist.** `cancer` is nine NHS signed-off panels, 41 genes, 4,107 rows. `all` is every NHS signed-off panel at its signed-off version, 296 panels, 4,206 genes, 123,454 rows; `config/all_panel_versions.json` records every panel and version used. Build times: heart and cancer under a minute each, `all` about six minutes for step 0 and six for step 1. Every download is cached per gene, so a rerun takes seconds.
- **Each gene's mode of inheritance is recorded** in the comment of its line in the panel file: `GENE  # panels | MONOALLELIC; BIALLELIC`. Nothing on the ML side reads it yet.
- **A table of every other small mutation exists**, `data/other_types/`, in the same layout as `data/`, 73,570 rows. It serves the patient query, which is Mohit's side. Nothing trains on it today; see point 5 below if that should change. The counts measured there: outside missense the mutation type alone matches the ClinVar label about 99% of the time. See [other_mutation_types.md](other_mutation_types.md).

## What is on the branch `panel-flag`, for you to review and merge

The branch gives `scripts/02_simulate_hospitals.py`, `03_train_local.py`, `03_check_results.py` and `04_federated_train.py` a `--panel` flag, following the same rule as step 1: `data/` for `cardiac`, `data/<panel>/` otherwise. Step 2 keeps one reference file per panel, `config/reference_build_<panel>.json`; the cancer one is on the branch. The heart outputs were checked byte for byte with no flag, and the branch merges cleanly onto main. Details and the diff notes are in the branch's [data_contract.md](data_contract.md) section "A second disease area" and in [cancer_results.md](cancer_results.md).

Design note for the review: `main()` in steps 2 and 4 sets the module-level `DATA_DIR` with `global`, and the check script and step 4 set `step3.DATA_DIR` because step 3's `load()` and `evidence_columns()` read that global. Threading a parameter through would be cleaner and a larger diff.

## Results so far

False alarms of the pooled model on population-discordant benign test rows, by the source of frequency evidence:

| source of frequency | heart, of 183 | cancer, of 100 | all panels, of 3,273 |
|---|---|---|---|
| none | 15 | 12 | 592 |
| public reference, Europeans only | 7 | 8 | 369 |
| own hospital, Oslo | 11 | 9 | 409 |
| federated count query | 2 | 3 | 117 |
| ceiling, real gnomAD | 2 | 2 | 116 |

Public reference to federated query: heart 5 removed and 0 introduced (p = 0.0625), cancer 5 and 0 (p = 0.063), all panels 252 and 0 (p = 2.8e-76). Sensitivity moves from 0.950 to 0.949 on all panels. The cancer run is official, five NVFlare runs, federated equal to pooled in every run. The all-panels numbers come from your step 2 and step 3 run unchanged on that table from a scratch script; they are not yet an official run. AUC is flat around 0.97 everywhere, and that is not the headline: on the heart test set AlphaMissense alone scores 0.950, the gene name alone scores 0.916, and within single genes the scores still separate at about 0.93 to 0.95, so the AUC mostly reflects how ClinVar labels were made.

## Things for you to decide

1. **Step 4 on `all`.** Step 3 on 123,454 rows took 45 seconds. NVFlare has not been tried on that table.
2. **The test split at scale.** `UNSEEN_GENES = 6` gives 927 of 25,509 test rows on `all`, 3.6%. On `cancer` the same six drew BRCA1 and the test set became 44.5% of the table.
3. **A shared feature list.** The 30% missing rule recommends 13 scores for heart, 16 for cancer and 16 for `all`. The 13 heart columns are in every list, so the heart list would serve all three.
4. **Two-copy genes.** Half of the `all` genes need two bad copies, and 92 of the 127 harmful population-discordant variants sit in them. The model could take a "needs two copies" input read from the panel file, or results can be reported per class; the per-class false-alarm table is in [disease_areas.md](disease_areas.md). The 0.1% line that marks `pop_discordant` in step 1 is wrong for those genes and was left alone so the heart table keeps its bytes.
5. **Other mutation types in the model.** If wanted: one starting point per mutation type, the 13 scores applied to missense only, and one frequency weight shared by all types, then a check that the shared weight matches the missense-only fit. The labels outside missense are 99% one class, so AUC would mislead there; report false alarms and wrongly cleared harmful variants instead.
6. **Heart wording hard-coded in your scripts.** "heart patients" in step 2, "12 scores" and "Fourteen numbers" in step 3's prose, the `DSP N1526K` example, "4,300 rows" and the two-thirds `none` sentence in step 4, "14 floats" in the client. None stops a run; all mislead on another panel. The branch fixed the two that were computed counts.

## Known problems

- **Step 4 does not run on Windows.** NVFlare 2.9.0 calls `os.setsid` when it launches a job. It runs under WSL Ubuntu from the same checkout with `uv run --frozen`; the five cancer runs took 320 seconds there.
- **Genes with nothing.** On `all`, 356 genes returned no records, and most are protein-coding genes whose symbol differs between PanelApp and the ClinVar index behind myvariant.info. A symbol lookup is being added for the `all` set only, so the heart and cancer tables keep their bytes. Rebuild `all` once that lands, then set its reference with `--set-reference --panel all` on the branch.
- **Variant names follow dbNSFP's first protein position**, so well-known variants can appear under unfamiliar numbers, for example MUTYH Y179C as `MUTYH Y151C`. `variant_id` and `clinvar_id` are the safe keys.

## Where the numbers live

- [disease_areas.md](disease_areas.md): how to add an area, cancer and all-panels builds, the inheritance split, what does not scale yet.
- [cancer_results.md](cancer_results.md) on the branch: the official cancer run.
- [other_mutation_types.md](other_mutation_types.md): the other-types table, the spelling problem, the MYBPC3 demo.
- [data_contract.md](data_contract.md): files and columns.
