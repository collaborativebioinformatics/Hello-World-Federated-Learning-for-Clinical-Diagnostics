# Hello World! Federated Learning for Clinical Diagnostics

Can hospitals help each other judge a patient's DNA variant while every patient record stays where it is?

> This is a research prototype from a hackathon. The hospitals and their patients are simulated. It must not be used for patient care.

## The problem

A DNA variant that is common among healthy people cannot be the cause of a rare, severe disease. Genetics labs rely on this every day: they look the variant up in a public frequency database, and a common variant is cleared as harmless.

Those databases are built mostly from people of European ancestry. A variant can look rare there and still be common, and harmless, in the patient's own population. Patients of African ancestry have been told they carry a cause of heart disease because of this gap, as [a 2016 study in the New England Journal of Medicine](https://www.nejm.org/doi/full/10.1056/NEJMsa1507092) documented.

The hospitals that serve under-represented populations hold the missing numbers. They cannot send patient records to each other.

## The idea

Two kinds of information can leave a hospital without exposing a patient.

1. **Counts.** "How many of your healthy patients carry this variant?" The answer is a handful of numbers.
2. **Model settings.** Every hospital trains the same small prediction model on its own data and shares only the 14 numbers that define the model. A server averages them, which is called federated learning. We run this part with NVIDIA FLARE.

We tested the idea on three simulated hospitals, each serving a different population: **Oslo** for European, **Karachi** for South Asian and **Lagos** for African ancestry.

## How it works

![Seven steps: three public sources join into one table, a test set is locked away, three hospitals train locally, a server averages the models, the result is scored, a patient's variant is sent to every hospital as a count query, and the answer becomes a verdict](docs/pipeline_flowchart.png?v=5)

- **The data are real and public.** Expert verdicts come from ClinVar, computer predictions of damage from dbNSFP, and frequencies per population from gnomAD. The table holds 8,790 variants in 97 heart disease genes used by NHS labs.
- **A test set is locked away first.** 2,373 variants are never used for training, including every variant of six whole genes.
- **Each hospital sees only its own share.** It knows the public database, which in our setup covers Europeans only, and the frequencies among its own patients.
- **The model learns from expert verdicts.** It combines 12 prediction scores with one frequency. Without being told the rule, it learned that a high frequency means harmless.

## What we found

One model, one fixed threshold, the same 183 test variants. All 183 are harmless, and each is common in one population and at least ten times rarer in another. The only thing that changes is what the model is told about frequency.

| What the model is told about frequency | Harmless variants wrongly flagged, of 183 |
|---|---|
| Nothing | 15 `███████████████` |
| The public database, Europeans only | 7 `███████` |
| Counts from all three hospitals | 2 `██` |
| The true frequency in every population, which no hospital has | 2 `██` |

- **Asking the other hospitals removed most false alarms.** The result stayed at 2 or 3 when we simulated the patients again 200 times.
- **Harmful variants were still caught.** The model flagged 92% of the 1,152 harmful test variants in every setting. Three of them lost their flag once the hospitals answered.
- **Overall accuracy did not move.** The usual summary score, AUC, stayed between 0.97 and 0.98 whatever the model was told and whichever hospital trained it. Frequency matters for a small number of variants, and those are the ones behind a wrong report.
- **A hospital that consults only its own patients does about as well as the public database.** The gain comes from combining hospitals.
- **Training across hospitals cost nothing.** The model trained with NVIDIA FLARE, where only 14 numbers per hospital travel, matched a model trained on all the data in one place. Over five runs the AUC was 0.9783 against 0.9784, and both left the same 2 false alarms.

These are small numbers from a simulation. The section on limits below says what they can and cannot show. Full tables are in [docs/step3_results.md](docs/step3_results.md) and [docs/step4_results.md](docs/step4_results.md), and `uv run python scripts/03_check_results.py` reprints the numbers in the table above.

## Try it: the patient query

Pick a variant, and every hospital answers with counts. Two fixed rules read the answers. Nothing is generated and nothing is random.

1. Common among healthy patients anywhere, at 0.1% or more: likely harmless.
2. Common, yet at least three times more frequent among the sick: keep it flagged.

![The patient query on screen: a variant picker, the reading from the patient's own hospital and from all three hospitals, and one chart with a row per source of evidence](docs/patient_query_tui.png?v=1)

```
uv run python scripts/06_query_tui.py                     # interactive, needs a real terminal
uv run python scripts/06_query_variant.py "DSP N1526K"    # run once and print
```

| Try | What it shows |
|---|---|
| `DSP N1526K` | Oslo alone cannot clear it. Among healthy patients at Lagos 15% carry it, so it is likely harmless. |
| `TTR V142I` | Common at Lagos, yet five times more common among the sick, so it stays flagged. It is a known cause of cardiac amyloidosis in people of African ancestry. |
| `MYH7 R403Q` | Seen nowhere. Frequency says nothing, and the prediction scores have to decide. |

Every bar in the chart is a frequency, and the tick marks 0.1%. A green bar past the tick clears the variant. A red bar well past the green one keeps it flagged. The last line on screen states what crossed hospital walls: one variant name out, counts back, and no patient record. For a patient at Oslo, asking the other two hospitals changes the reading for 492 of 8,682 variants.

## What is real and what is simulated

| Real | Simulated |
|---|---|
| Every expert verdict, prediction score and population frequency | Which hospital has classified which variant |
| The gene list, from NHS-approved panels | Every patient and every carrier count |
| | A public database that covers Europeans only. It stands in for the many populations that real databases cover poorly. |

## Limits

- The patients are simulated from the same public frequencies we score against, so the close match between "all three hospitals" and "the true frequency" is partly built in.
- 183 variants is a small test, and only five calls changed. A larger gene list is the remedy.
- The test set has no harmful variant that is common in one population, so the risk of wrongly clearing such a variant could not be measured. `TTR V142I` shows the case is real.
- Expert verdicts themselves lean on the same frequency databases.
- Rule 2 wrongly kept two harmless variants flagged on the strength of six carriers. It needs a proper statistical test across hospitals.
- Hiding counts under 5 is the only privacy protection. There is no formal privacy guarantee.

## Words used here

| Term | Plain meaning |
|---|---|
| Variant | A change in DNA at a known position, such as `MYH7 R403Q`. |
| Pathogenic, benign | The expert verdict: causes disease, or harmless. |
| Allele frequency | How common a variant is. 1.2% means 12 of every 1,000 gene copies carry it. |
| ClinVar | The public list of expert verdicts. Our answer key. |
| gnomAD | The public frequency database, with one frequency per population. |
| Prediction score | A computer estimate of how damaging a variant is, such as AlphaMissense or CADD. |
| Population-discordant | Common in one of our three populations and at least ten times rarer in another. 622 of our 8,790 variants. |
| False alarm | A harmless variant that the model flags as disease-causing. |
| AUC | One score for how well a model ranks variants. 0.5 is a coin flip and 1.0 is perfect. |
| Federated learning | Hospitals train locally and share only the model's numbers. |

## Build status and how to run

![Recipe status: steps 0 to 4 and step 6 are built and tested, step 5 is next](docs/recipe_status.png?v=7)

Green means the step runs from a fresh clone with the command shown. Needs [uv](https://docs.astral.sh/uv/). Python 3.12 and the packages install themselves on first run.

```
git clone https://github.com/collaborativebioinformatics/Hello-World-Federated-Learning-for-Clinical-Diagnostics.git
cd Hello-World-Federated-Learning-for-Clinical-Diagnostics
uv sync

uv run python scripts/00_fetch_gene_panel.py        # step 0, the gene list from PanelApp (already committed)
uv run python scripts/01_build_table.py             # step 1, about 4 minutes, then cached
uv run python scripts/02_simulate_hospitals.py      # step 2, a few seconds, the hospital files
uv run python scripts/03_train_local.py             # step 3, a few seconds, the baseline models
uv run python scripts/04_federated_train.py         # step 4, about 6 minutes, NVIDIA FLARE, five runs
uv run python scripts/06_query_tui.py               # step 6, the patient query
```

The seed is fixed, so everyone gets identical files, and step 2 ends by confirming that your build matches the team's reference. The tables are rebuilt on your machine and never committed, because the prediction scores carry non-commercial terms.

## Read more

| | |
|---|---|
| [docs/build_notes.md](docs/build_notes.md) | every step with example tables, the experiment grid, data sources, decisions, and how to update the pictures |
| [docs/step3_results.md](docs/step3_results.md) | the model, every result table and how to read them |
| [docs/step4_results.md](docs/step4_results.md) | federated training with NVIDIA FLARE: method, the five runs and their limits |
| [docs/step2_review.md](docs/step2_review.md) | the sampling choices behind the simulated hospitals, and their review |
| [docs/data_contract.md](docs/data_contract.md) | every file and column, and what may be trained on |
| [docs/table_columns.md](docs/table_columns.md) | the variant table, column by column |

## Team

Team 12, clinical diagnostics track.

- Yan Li
- Mohit Panwar
- Shreya Srivastava
- Oumaima Boussouis
- Fenfen Ge
- Claude Code 😉
