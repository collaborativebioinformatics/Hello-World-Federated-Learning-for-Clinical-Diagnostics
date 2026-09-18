# Hello World! Federated Learning for Clinical Diagnostics

**[Try it in your browser](https://collaborativebioinformatics.github.io/Mildly-Flirting-With-Federated-Learning-for-Clinical-Diagnostics/)**: the patient query, the pipeline in motion and the manuscript as one website. The query on its own is at [collaborativebioinformatics.github.io/Mildly-Flirting-With-Federated-Learning-for-Clinical-Diagnostics/demo/](https://collaborativebioinformatics.github.io/Mildly-Flirting-With-Federated-Learning-for-Clinical-Diagnostics/demo/), and it is the same screen as the terminal one below.

Can hospitals help each other judge a patient's DNA variant while every patient record stays where it is?

> This is a research prototype from a hackathon. The hospitals and their patients are simulated. It must not be used for patient care.

## Try it

A patient carries a DNA variant. Pick it, and every hospital answers with counts.

![The patient query on screen: the disease area, what to show, the gene and the kind of mutation at the top left; the reading from the patient's own hospital and from all three hospitals at the top right, with the gene's line and the trained model's opinion under it; and one chart with a row per source of evidence](docs/patient_query_tui.png?v=2)

```
git clone https://github.com/collaborativebioinformatics/Mildly-Flirting-With-Federated-Learning-for-Clinical-Diagnostics.git
cd Mildly-Flirting-With-Federated-Learning-for-Clinical-Diagnostics
uv run python scripts/01_build_table.py           # once, about 4 minutes: downloads the public data
uv run python scripts/02_simulate_hospitals.py    # once, a few seconds: the three hospitals
uv run python scripts/06_query_tui.py             # the patient query, in a real terminal
```

Needs [uv](https://docs.astral.sh/uv/), which installs Python and the packages on first run. `uv run python scripts/06_query_variant.py "DSP N1526K"` prints one answer without the interactive screen.

Those three commands give the heart area with its missense variants. Two more commands add every other small mutation in the same genes, and two more add the inherited cancer area, which then appears in the screen's first dropdown:

```
uv run python scripts/01_build_other_types.py                   # once, about 15 minutes the first time: every other small mutation, 73,570 variants
uv run python scripts/02_simulate_other_types.py                # once, a few seconds: the three hospitals' files for them
uv run python scripts/01_build_table.py --panel cancer          # once, under a minute: the inherited cancer table
uv run python scripts/02_simulate_hospitals.py --panel cancer   # once, a few seconds: its three hospitals
```

The whole reference genome, `uv run python scripts/00_fetch_reference_genome.py`, is optional: about 1 GB to download and 3 GB unpacked. With it the spelling of insertions and deletions is read offline from the genome. Without it the other-types build fetches the letters it needs gene by gene, which is the slow part of that first run.

| Try | What it shows |
|---|---|
| `DSP N1526K` | Oslo alone cannot clear it. Among healthy patients at Lagos 15% carry it, so it is likely harmless. |
| `TTR V142I` | Common at Lagos, yet five times more common among the sick, so it stays flagged. It is a known cause of cardiac amyloidosis in people of African ancestry. The sick-patient counts are simulated from the known verdict, so this row shows how the rule works and proves nothing about the variant. |
| `MYH7 R403Q` | Seen nowhere. Frequency says nothing, and the prediction scores have to decide. |
| `MYBPC3 c.3628-41_3628-17del` | A 25-letter deletion with 8 valid spellings. Karachi finds it in 3.2% of its healthy patients once every hospital reads the same spelling. ClinVar's labs disagree about it, so the count is all the query says. |

Two fixed rules read the answers. Nothing is generated and nothing is random.

1. Common among healthy patients anywhere, at the gene's line or above: likely harmless. The line is 0.1% where one bad copy of the gene is enough to cause disease, and 1% where both copies must be bad, because healthy carriers of one copy are then expected.
2. Common, yet at least three times more frequent among the sick: keep it flagged.

Every bar in the chart is a frequency, and the tick marks the gene's line. A green bar past the tick clears the variant. A red bar well past the green one keeps it flagged. Under the two rules the screen adds the trained model's opinion. The last line on screen states what crossed hospital walls: one variant name out, counts back, and no patient record. For a patient at Oslo, asking the other two hospitals changes the reading for 467 of 8,682 missense variants.

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

![The pipeline as built: two open APIs deliver three public sources as one table, a seeded random split locks a test set and deals the rest to three hospitals, each fits a logistic regression, an NVFlare server averages them, the locked set scores the result, and a patient query returns counts that two fixed rules read](docs/pipeline_flowchart_built.png?v=1)

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
- **Oslo consulting only its own patients did about as well as the public database.** Karachi and Lagos on their own did no better. Each hospital improved once its own counts were added to the public database, so the gain comes from combining sources.
- **Training across hospitals cost nothing.** The model trained with NVIDIA FLARE, where only 14 numbers per hospital travel, matched a model trained on all the data in one place. Over five runs the AUC was 0.9783 against 0.9784, and both left the same 2 false alarms.
- **No patient group was left behind by the shared model.** We checked each ancestry group against the model its own hospital would have built alone. European and South Asian patients came out level, and African-ancestry patients came out very slightly ahead. The shared model was also steadier: a small hospital's own model shifted three times as much between runs as the shared one did.
- **A single hospital did not fail on the ancestries it rarely sees**, which is what we had expected to find. The prediction scores work the same for everyone. What does not travel is how common a variant is, and that changes the verdict on a handful of variants while the overall ranking stays the same.
- **The result repeats in inherited cancer and holds at scale on every NHS signed-off panel.** On 41 inherited cancer genes the false alarms went 12, 8 and 3 of 100 with nothing, the public database and the counts from all three hospitals. On all 296 signed-off panels they went 566, 375 and 110 of 3,396, where the hospitals' counts removed 266 false alarms and introduced 1, p = 2.3 × 10<sup>−78</sup>. The next section has the table.
- **The summary score is mostly the prediction scores' own doing, which is why it cannot see any of this.** On the heart test set AlphaMissense alone reaches an AUC of 0.950, the gene name alone 0.916, and within single genes the scores still reach about 0.93 to 0.95, so the AUC reports how the expert verdicts were made, and the false alarms above are the measurement that needs the hospitals' counts.

These are small numbers from a simulation. The section on limits below says what they can and cannot show. Full tables are in [docs/step3_results.md](docs/step3_results.md) and [docs/step4_results.md](docs/step4_results.md) for the heart, [docs/cancer_results.md](docs/cancer_results.md) for inherited cancer and [docs/disease_areas.md](docs/disease_areas.md) for all panels. `uv run python scripts/03_check_results.py` reprints the numbers in the table above, and `--panel cancer` does the same for inherited cancer.

## Three disease areas

The pipeline was built on heart genes. A disease area here is a set of gene panels signed off by the NHS, taken from Genomics England's PanelApp, and everything after the gene list is the same code. Three areas are built. Of the genes on the panels, 7 heart genes and 128 all-panel genes returned no usable variant.

| Area | Panels | Genes | Missense variants | Harmless test variants, common in one population and rare in another |
|---|---|---|---|---|
| Heart | 6 | 104 | 8,790 | 183 |
| Inherited cancer | 9 | 41 | 4,107 | 100 |
| Every NHS signed-off panel | 296 | 4,206 | 127,618 | 3,396 |

False alarms of the pooled model on those test variants, by what the model is told about frequency. One block is 1 in 100 of the variants tested, so the bars compare across areas.

| What the model is told about frequency | Heart, of 183 | Inherited cancer, of 100 | Every panel, of 3,396 |
|---|---|---|---|
| Nothing | 15 `████████` | 12 `████████████` | 566 `█████████████████` |
| The public database, Europeans only | 7 `████` | 8 `████████` | 375 `███████████` |
| The patient's own hospital, Oslo | 11 `██████` | 9 `█████████` | 420 `████████████` |
| Counts from all three hospitals | 2 `█` | 3 `███` | 110 `███` |
| The true frequency in every population, which no hospital has | 2 `█` | 3 `███` | 110 `███` |

The heart and inherited cancer runs are official, through step 4 with five NVIDIA FLARE runs each. In inherited cancer the federated model and the pooled one again left the same 3 false alarms in every run, with an AUC of 0.9776 against 0.9779. The all-panels numbers are a first look: steps 2 and 3 run unchanged on that table from a scratch script, and the NVIDIA FLARE run on it is pending on the ML side.

Going from the public database to the counts from all three hospitals removed 5 false alarms and introduced none in the heart area, p = 0.0625, the same 5 and none in inherited cancer, p = 0.0625, and 266 with 1 introduced across every panel, p = 2.3 × 10<sup>−78</sup>. Harmful variants were still caught: the share flagged was 0.919 and 0.919 in the heart area, 0.912 and 0.911 in inherited cancer and 0.948 and 0.947 across every panel, public database first and the hospitals' counts second. Across every panel the reduction shows in each inheritance class and is largest in genes where both copies must be bad, which is also where frequency is least safe: 96 of the 136 harmful variants that are common somewhere sit in those genes.

## The patient query

![The screen on the inherited cancer area: POLD1 S173N, carried by 10.0% of healthy patients at Lagos, cleared once all hospitals answer, with the 0.1% line for a one-copy gene and the trained model's opinion under the verdict](docs/patient_query_tui_cancer.png?v=1)

The screen at the top of this page is the clinician's end of the pipeline. What it covers now:

**Every small mutation in the heart genes.** Beside the 8,790 missense variants the query answers for 73,570 other ClinVar variants in the same 104 genes: changes that cut the protein short, shift the reading frame or break a splice site, and silent, intronic and other changes. The trained model stays missense-only on purpose. Outside missense the mutation type alone matches the expert verdict for 66,099 of 66,298 variants, 99.7%, so a model would learn the type and little else. Instead each such variant starts from the presumption a lab starts from, harmful for a protein-cutting change and harmless for a silent one, and the two frequency rules can overrule it. The hospitals together clear 3 of the 84 harmless variants whose type presumes harm, and the public database wrongly clears 1 harmful frameshift of 13,056.

**One spelling per insertion or deletion.** The same deletion can be written at several positions when it sits in a repeat, and two hospitals that write it differently would count zero for each other. Of 8,890 insertions and deletions in the heart genes, 6,494 have more than one valid spelling. The query slides every one to the same spelling against the GRCh38 genome before any hospital is asked. The 25-letter MYBPC3 deletion, 8 valid spellings, carried by 3.2% of South Asians and by almost no one else, shows the failure and the fix:

```
uv run python scripts/06_query_variant.py "chr11:g.47332275_47332299del" --patient-at site_oslo --no-spelling-fix   # Karachi answers none of 6,800: frequency says nothing
uv run python scripts/06_query_variant.py "chr11:g.47332275_47332299del" --patient-at site_oslo                     # Karachi answers 215 of 6,800: likely harmless
```

**A line per gene.** Rule 1's line is 0.1% where one bad copy of the gene is enough to cause disease and 1% where both copies must be bad, read for every gene from the inheritance PanelApp records. The screen says which line applied and why. Across every panel the single 0.1% line had cleared 129 harmful variants, and the per-gene line keeps 123 of them.

**The trained model on screen.** Under the two rules the verdict panel gives the trained model's opinion, for example "the trained model puts this at 0.1% likely harmful, below its cut-off of 57%". It is the pooled model of step 3, read from its saved weights, and the query switches to the NVIDIA FLARE model once step 4 saves one. Its feature rows were checked against step 3's and match to the last digit.

**Three disease areas in one dropdown.** The first dropdown switches between the heart, inherited cancer and, once built, every NHS signed-off panel. Each area has its own examples to try; for inherited cancer they are `POLD1 S173N`, `PMS2 T511M`, `MUTYH G63D` and `RNF43 R657P`.

**F1 for help.** F1 or ? explains the keys and every part of the screen. F2 hides or shows counts under 5, and F3 moves the patient to the next hospital.

[docs/patient_query.md](docs/patient_query.md) says how every line on the screen is computed, and [docs/other_mutation_types.md](docs/other_mutation_types.md) covers the other mutation types and the spelling problem.

## What is real and what is simulated

| Real | Simulated |
|---|---|
| Every expert verdict, prediction score and population frequency | Which hospital has classified which variant |
| The gene list, from NHS-approved panels | Every patient and every carrier count |
| | A public database that covers Europeans only. It stands in for the many populations that real databases cover poorly. |

## Limits

- The patients are simulated from the same public frequencies we score against, so the close match between "all three hospitals" and "the true frequency" is partly built in.
- 183 variants is a small test, and only five calls changed. A larger gene list is the remedy, and the all-panels table is that list; its numbers are a first look without a federated run.
- The per-ancestry checks rest on 12 harmful variants for African-ancestry patients and 23 for South Asian. They show a direction only, and the numbers are too small to measure its size.
- The test set has no harmful variant that is common in one population, so the risk of wrongly clearing such a variant could not be measured. `TTR V142I` shows the case is real: scored as if unseen, the model clears it as harmless whatever it is told about frequency, and only the hand rule keeps it flagged.
- The model on the screen is the pooled model of step 3, standing in for the federated one until step 4 saves its file. Its feature rows were shown to match step 3's to the last digit, but its cut-off was set on training rows, and its probability for a variant in a gene it has seen partly reflects the gene.
- The 1% line for genes that need two bad copies is not always enough. Across every panel six common recessive alleles stay above it, `HFE C259Y` at 5.74% among them, and would be cleared; a lab would want a disease-specific line for such genes.
- Contested variants, which ClinVar's labs disagree about, have no truth here. The MYBPC3 deletion is one: the query shows a count and says nothing about whether the variant is harmful.
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
| One-copy gene, two-copy gene | Whether one bad copy of the gene is enough to cause disease, or both copies must be bad. In a two-copy gene healthy carriers of one copy are expected, so the query's line is 1% instead of 0.1%. |
| Contested | A variant that ClinVar's labs are unsure or disagree about. It has no verdict here. |
| Spelling | The same insertion or deletion inside a repeat can be written at several positions. The query slides every one to the same position before asking. |

## Build status and how to run

![Recipe status: steps 0 to 4 and step 6 are built and tested, step 5 is next](docs/recipe_status.png?v=7)

The picture predates step 5 and the second and third disease areas and will be redrawn; the list below is current. Steps 0 to 6 are built and run from a fresh clone with the commands shown: 0 the gene list, 1 the variant table, 2 the simulated hospitals, 3 the local and pooled models, 4 federated training with NVIDIA FLARE, 5 the quarterly replay in which the hospitals grow and the model is retrained each quarter, and 6 the patient query. The heart and inherited cancer areas have run through step 4. The all-panels area has run through step 3 from a scratch script, and its NVIDIA FLARE run is pending on the ML side. Needs [uv](https://docs.astral.sh/uv/). Python 3.12 and the packages install themselves on first run.

```
git clone https://github.com/collaborativebioinformatics/Mildly-Flirting-With-Federated-Learning-for-Clinical-Diagnostics.git
cd Mildly-Flirting-With-Federated-Learning-for-Clinical-Diagnostics
uv sync

uv run python scripts/00_fetch_gene_panel.py        # step 0, the gene list from PanelApp (already committed)
uv run python scripts/01_build_table.py             # step 1, about 4 minutes, then cached
uv run python scripts/02_simulate_hospitals.py      # step 2, a few seconds, the hospital files
uv run python scripts/03_train_local.py             # step 3, a few seconds, the baseline models
uv run python scripts/04_federated_train.py         # step 4, about 6 minutes, NVIDIA FLARE, five runs; not on Windows, use WSL or Linux
uv run python scripts/05_quarterly_replay.py        # step 5, about 10 seconds, four quarters of growing hospitals
uv run python scripts/06_query_tui.py               # step 6, the patient query
```

Add `--panel cancer` to steps 0 to 4 for the inherited cancer area, and `--panel all` for every NHS signed-off panel, which takes about six minutes for step 0 and six for step 1. `uv run python scripts/01_build_other_types.py` followed by `uv run python scripts/02_simulate_other_types.py` adds the other mutation types to the heart area.

The seed is fixed, so everyone gets identical files, and step 2 ends by confirming that your build matches the team's reference. The tables are rebuilt on your machine and never committed, because the prediction scores carry non-commercial terms.

## Read more

| | |
|---|---|
| [The website](https://collaborativebioinformatics.github.io/Mildly-Flirting-With-Federated-Learning-for-Clinical-Diagnostics/) | the [patient query in the browser](https://collaborativebioinformatics.github.io/Mildly-Flirting-With-Federated-Learning-for-Clinical-Diagnostics/demo/), the pipeline in motion and the manuscript as tabs of one page |
| [Manuscript.md](Manuscript.md) | the write-up with methods, results, limits and references, also as a [PDF](Manuscript.pdf) |
| [docs/team_section_shared_manuscript.md](docs/team_section_shared_manuscript.md) | the condensed story: Team 12's blocks for the shared hackathon manuscript |
| [docs/patient_query.md](docs/patient_query.md) | what the clinician's screen shows, how every line on it is computed, the per-gene line, the model line and its proof |
| [docs/disease_areas.md](docs/disease_areas.md) | how to add a disease area, the inherited cancer and all-panels builds, the split by inheritance, what does not scale yet |
| [docs/cancer_results.md](docs/cancer_results.md) | the official inherited cancer run through steps 2, 3 and 4 |
| [docs/other_mutation_types.md](docs/other_mutation_types.md) | the count query beyond missense, the spelling problem and the MYBPC3 demo |
| [docs/quarterly_replay.md](docs/quarterly_replay.md) | the hospitals grow quarter by quarter and the model is retrained each time |
| [docs/handoff_ml.md](docs/handoff_ml.md) | the handoff to the ML side: what is on main, results so far, open decisions |
| [docs/build_notes.md](docs/build_notes.md) | every step with example tables, the experiment grid, data sources, decisions, and how to update the pictures |
| [docs/step3_results.md](docs/step3_results.md) | the model, every result table and how to read them |
| [notebooks/federated_random_forest_colab.ipynb](notebooks/federated_random_forest_colab.ipynb) | Oumaima Boussouis's Colab notebook: a random forest per hospital with soft voting, on synthetic data |
| [docs/step4_results.md](docs/step4_results.md) | federated training with NVIDIA FLARE: method, the five runs and their limits |
| [docs/step2_review.md](docs/step2_review.md) | the sampling choices behind the simulated hospitals, and their review |
| [docs/data_contract.md](docs/data_contract.md) | every file and column, and what may be trained on |
| [docs/table_columns.md](docs/table_columns.md) | the variant table, column by column |

## Team

Team 12, clinical diagnostics track.

- Yan Li
- Mohit B. Panwar
- Shreya Srivastava
- Oumaima Boussouis
- Claude Code 😉
