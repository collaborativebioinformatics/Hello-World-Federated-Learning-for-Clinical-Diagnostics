# How to build and run

For developers. The [README](../README.md) is the front page for clinicians and judges. This page holds every command, what each step makes, the checks that guard a build, where the data goes and how long each step takes. [build_notes.md](build_notes.md) is the front page from while the project was being built, kept as a record with example tables and the decisions taken along the way.

Everything runs with [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and the packages on first run. Step 4 does not run on Windows; see the note under the steps. Everything else runs anywhere.

## A fresh clone

```
git clone https://github.com/collaborativebioinformatics/Mildly-Flirting-With-Federated-Learning-for-Clinical-Diagnostics.git
cd Mildly-Flirting-With-Federated-Learning-for-Clinical-Diagnostics
uv run python scripts/01_build_table.py           # once, about 4 minutes: downloads the public data
uv run python scripts/02_simulate_hospitals.py    # once, a few seconds: the three hospitals
uv run python scripts/06_query_tui.py             # the patient query, in a real terminal
```

Those three commands give the heart area with its missense variants and open the patient query. `uv run python scripts/06_query_variant.py "DSP N1526K"` prints one answer without the interactive screen.

## The other mutation types and the other areas

Two more commands add every other small mutation in the same heart genes, and two more add the inherited cancer area, which then appears in the screen's first dropdown:

```
uv run python scripts/01_build_other_types.py                   # once, about 15 minutes the first time: every other small mutation, 73,570 variants
uv run python scripts/02_simulate_other_types.py                # once, a few seconds: the three hospitals' files for them
uv run python scripts/01_build_table.py --panel cancer          # once, under a minute: the inherited cancer table
uv run python scripts/02_simulate_hospitals.py --panel cancer   # once, a few seconds: its three hospitals
```

The whole reference genome, `uv run python scripts/00_fetch_reference_genome.py`, is optional: about 1 GB to download and 3 GB unpacked. With it the spelling of insertions and deletions is read offline from the genome. Without it the other-types build fetches the letters it needs gene by gene, which is the slow part of that first run.

Add `--panel cancer` to steps 0 to 4 for the inherited cancer area, and `--panel all` for every NHS signed-off panel, which takes about six minutes for step 0 and six for step 1. With no `--panel` every step builds the heart area, named `cardiac`, into `data/`; any other area goes into `data/<area>/`, and the query lists it once `data/<area>/sites.json` exists. [disease_areas.md](disease_areas.md) says how to add an area.

## The patient query from the command line

```
uv run python scripts/06_query_tui.py                       # the screen, in a real terminal
uv run python scripts/06_query_tui.py --panel cancer        # start on the inherited cancer area
uv run python scripts/06_query_variant.py "DSP N1526K"      # one answer, printed
uv run python scripts/06_query_variant.py "MUTYH G63D" --panel cancer --json
uv run python scripts/06_check_query_rules.py               # the numbers in patient_query.md
uv run python scripts/06_query_tui_check.py                 # drives the screen headlessly, writes the pictures
```

**The spelling demo.** The 25-letter MYBPC3 deletion has 8 valid spellings, and two hospitals that write it differently count zero for each other. The query slides every insertion and deletion to one spelling against the GRCh38 genome before any hospital is asked, and `--no-spelling-fix` switches that off to show the failure:

```
uv run python scripts/06_query_variant.py "chr11:g.47332275_47332299del" --patient-at site_oslo --no-spelling-fix   # Karachi answers none of 6,800: frequency says nothing
uv run python scripts/06_query_variant.py "chr11:g.47332275_47332299del" --patient-at site_oslo                     # Karachi answers 215 of 6,800: likely harmless
```

**The model line.** The screen reads the pooled model of step 3 from `data/results_local.json`, and the NVIDIA FLARE model from `data/results_federated.json` when that file exists. The query builds its feature rows the way step 3 does, and they were checked against step 3's and match to the last digit; [patient_query.md](patient_query.md) has the proof.

## The steps

```
git clone https://github.com/collaborativebioinformatics/Mildly-Flirting-With-Federated-Learning-for-Clinical-Diagnostics.git
cd Mildly-Flirting-With-Federated-Learning-for-Clinical-Diagnostics
uv sync

uv run python scripts/00_fetch_gene_panel.py        # step 0, the gene list from PanelApp, already committed
uv run python scripts/01_build_table.py             # step 1, about 4 minutes, then cached
uv run python scripts/02_simulate_hospitals.py      # step 2, a few seconds, the hospital files
uv run python scripts/03_train_local.py             # step 3, a few seconds, the baseline models
uv run python scripts/04_federated_train.py         # step 4, about 6 minutes, NVIDIA FLARE, five runs; not on Windows, use WSL or Linux
uv run python scripts/05_quarterly_replay.py        # step 5, about 10 seconds, four quarters of growing hospitals
uv run python scripts/06_query_tui.py               # step 6, the patient query
```

What each step makes:

- **Step 0, the gene list.** `00_fetch_gene_panel.py` reads Genomics England's PanelApp at pinned panel versions and writes `config/cardiac_gene_panel.txt`: the green genes of the six NHS heart panels, each with the panels that list it and how it is inherited. The file is committed, so a fresh clone can skip this step. `--panel cancer` writes `config/cancer_gene_panel.txt`, and `--panel all` writes `config/all_gene_panel.txt` and `config/all_panel_versions.json`. Panel sets are listed in `config/panel_sets.json`.
- **Step 1, the variant table.** `01_build_table.py` downloads the ClinVar, dbNSFP and gnomAD fields through myvariant.info, one cache file per gene under `data/raw/`, and writes `data/variants.csv` and `data/columns.json`, the list of columns the later steps read. `--refresh` ignores the cache and downloads again. `01_build_other_types.py` writes `data/other_types/variants.csv` for every other small mutation in the same genes, with a `mutation_type` column and no score columns, and keeps each gene's reference sequence under `data/raw_sequence/`.
- **Step 2, the simulated hospitals.** `02_simulate_hospitals.py` locks the test set in `data/test/variants.csv`, deals the rest to `data/site_oslo/`, `data/site_karachi/` and `data/site_lagos/`, each with `verdicts.csv` and `patient_counts.csv`, and writes `data/public_reference.csv` and `data/sites.json`. It ends with the reference check below. `--self-check` breaks the split on purpose and expects the script to stop. `02_simulate_other_types.py` does the same for the other mutation types under `data/other_types/`, with a separate random generator, so the missense files are untouched.
- **Step 3, the local and pooled models.** `03_train_local.py` fits one logistic regression per hospital and one on every hospital's verdicts pooled, and writes each model's weights and cut-off to `data/results_local.json`. `--self-check` tests the maths against brute force with no data. `03_check_results.py` reprints the false-alarm table of the README, and `--panel cancer` does the same for inherited cancer.
- **Step 4, federated training.** `04_federated_train.py` runs NVIDIA FLARE in simulator mode, a server and three clients as separate processes on one machine, five runs, and writes `data/results_federated.json`. NVFlare 2.9.0 calls `os.setsid` when it launches a job, so this step does not run on Windows. Under WSL Ubuntu it runs from the same checkout with `uv run --frozen`, where the five inherited cancer runs took 320 seconds.
- **Step 5, the quarterly replay.** `05_quarterly_replay.py` grows the hospitals in four equal steps under `data/quarterly/<area>/q1` to `q4`, retrains the step 3 model after each, and writes `docs/quarterly_replay.json`, which the website's pipeline page animates, and `docs/quarterly_replay.md`.
- **Step 6, the patient query.** `06_query_tui.py` is the screen and `06_query_variant.py` the one-answer command. `scripts/hospital_query.py` holds the logic and `scripts/query_drawing.py` the look, and the two commands are thin layers over them, so they cannot disagree. `06_check_query_rules.py` computes the numbers in [patient_query.md](patient_query.md), and `06_query_tui_check.py` drives the screen headlessly and writes its pictures.

## Build status

![Recipe status: steps 0 to 4 and step 6 are built and tested, step 5 is next](recipe_status.png?v=7)

The picture predates step 5 and the second and third disease areas; the list above is current. Steps 0 to 6 run from a fresh clone with the commands shown. The heart and inherited cancer areas have run through step 4, with five NVIDIA FLARE runs each. The all-panels area has run through step 3 from a scratch script, and its NVIDIA FLARE run sits with the ML side; [handoff_ml.md](handoff_ml.md) says what that involves.

To update the picture, open [recipe_status.html](recipe_status.html), change a step's one-word status, re-render with the command at the top of that file, and raise the `?v=` number on the image link, otherwise GitHub keeps serving its cached copy of the old picture. Where headless Chrome will not run, `uv run --with weasyprint --with pypdfium2 --with pillow python scripts/render_docs_png.py` produces the same picture.

## The reference check

The seed is fixed, so everyone gets identical files, and step 2 ends by confirming that your build matches the team's reference in `config/reference_build.json`. Any other area has its own file, `config/reference_build_<area>.json`, and `--set-reference` on step 2 declares a build the reference for its area. The one thing that can break the match is myvariant.info updating its data between two people's downloads; the script detects it and says what to do.

Every step 2 run also checks itself and stops rather than write a split that is quietly wrong: no test variant in a hospital file, no variant twice in one hospital, both verdicts present at every hospital, and each hospital's local frequency really drawn from its own population.

## Where the data goes

Everything the steps make lives under `data/`, which is git-ignored. The tables are rebuilt on your machine and never committed, because the prediction scores from dbNSFP carry a non-commercial licence, so we share the recipe and never the table. Every download is cached per gene, so a rerun takes seconds.

```
data/
  test/variants.csv             2,373 rows   locked, never trained on
  public_reference.csv          8,790 rows   af_public, what every hospital already has
  sites.json                                 settings and sizes of this run
  site_oslo/verdicts.csv        4,319 rows   Oslo's private classified variants
  site_oslo/patient_counts.csv  8,790 rows   Oslo's private carrier counts
  site_karachi/...              same two files
  site_lagos/...                same two files
  results_local.json                         step 3
  results_federated.json                     step 4
  other_types/                               the other mutation types, same layout
  cancer/, all/                              the other areas, same layout
  quarterly/<area>/q1 .. q4                  step 5
  raw/, raw_sequence/                        the download caches
```

[data_contract.md](data_contract.md) lists every file and column and says what may be trained on, and [table_columns.md](table_columns.md) explains the variant table column by column.

## Timings

| What | How long |
|---|---|
| Step 1, the heart table | about 4 minutes the first time, then cached |
| Steps 2 and 3 | a few seconds each |
| Step 4, five NVIDIA FLARE runs | about 6 minutes |
| Step 5, the quarterly replay | about 10 seconds |
| The other mutation types | about 15 minutes the first time, without the reference genome |
| The inherited cancer table | under a minute |
| Every NHS signed-off panel | about six minutes for step 0 and six for step 1 |

## Read more

- [data_contract.md](data_contract.md): every file and column, and what may be trained on.
- [handoff_ml.md](handoff_ml.md): the handoff to the ML side, with what is on main and the results so far.
- [build_notes.md](build_notes.md): the old front page, every step with example tables, the experiment grid, data sources and decisions.
