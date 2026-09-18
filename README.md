**[Try it in your browser](https://collaborativebioinformatics.github.io/Confidently-Flirting-With-Federated-Learning-for-Clinical-Diagnostics/)**: the patient query, the pipeline in motion and the manuscript as one website. The query opens first, and it is the same screen as the terminal one below.

[![The pipeline in motion: data comes in, hospitals train together, a quarter passes, a doctor asks](docs/pipeline_live/pipeline.gif?v=1)](https://collaborativebioinformatics.github.io/Mildly-Flirting-With-Federated-Learning-for-Clinical-Diagnostics/#pipeline)

> This is a research prototype from a hackathon. The hospitals and their patients are simulated. It must not be used for patient care.

## The question

Can hospitals help each other judge a patient's DNA variant while every patient record stays where it is?

A DNA variant that is common among healthy people cannot be the cause of a rare, severe disease, and genetics labs rely on this every day: a variant that is common in the public frequency database is cleared as harmless. Those databases are built mostly from people of European ancestry, so a variant can look rare there and still be common, and harmless, in the patient's own population. Patients of African ancestry have been told they carry a cause of heart disease because of this gap, as [a 2016 study in the New England Journal of Medicine](https://www.nejm.org/doi/full/10.1056/NEJMsa1507092) documented. The hospitals that serve under-represented populations hold the missing numbers, and they cannot send patient records to each other.

## The idea

Two kinds of information can leave a hospital without exposing a patient. The first is counts: "how many of your healthy patients carry this variant?" is answered with a handful of numbers, so one variant name goes out and only counts come back. The second is model settings: every hospital trains the same small prediction model, 12 prediction scores and one frequency, on its own verdicts and shares only the 14 numbers that define it; a server averages them, and the hospitals end up with one shared model without moving a single record. That is federated learning, and we run it with NVIDIA FLARE. Without being told the rule, the model learned from the expert verdicts that a high frequency means harmless. We tested both on three simulated hospitals, each serving a different population: **Oslo** for European, **Karachi** for South Asian and **Lagos** for African ancestry.

![The pipeline as built: two open APIs deliver three public sources as one table, a seeded random split locks a test set and deals the rest to three hospitals, each fits a logistic regression, an NVFlare server averages them, the locked set scores the result, and a patient query returns counts that two fixed rules read](docs/pipeline_flowchart_built.png?v=1)

## What we found

We built three disease areas from gene panels signed off by the NHS: the heart, 6 panels, 104 genes, 97 with usable variants and 8,790 missense variants; inherited cancer, 9 panels, 41 genes and 4,107 variants; and every NHS signed-off panel, 296 panels, 4,206 genes and 127,618 variants. In each area the test variants that matter are the harmless ones that are common in one population and at least ten times rarer in another, because those are the ones a public database can get wrong: 183 in the heart area, 100 in inherited cancer and 3,396 across every panel. The table counts how many of them one model, with one fixed threshold, wrongly flags. The only thing that changes is what the model is told about frequency. One block is 1 in 100 of the variants tested, so the bars compare across areas.

| What the model is told about frequency | Heart, of 183 | Inherited cancer, of 100 | Every panel, of 3,396 |
|---|---|---|---|
| Nothing | 15 `████████` | 12 `████████████` | 566 `█████████████████` |
| The public database, Europeans only | 7 `████` | 8 `████████` | 375 `███████████` |
| The patient's own hospital, Oslo | 11 `██████` | 9 `█████████` | 420 `████████████` |
| Counts from all three hospitals | 2 `█` | 3 `███` | 110 `███` |
| The true frequency in every population, which no hospital has | 2 `█` | 3 `███` | 110 `███` |

- **Asking the other hospitals removed most false alarms.** In the heart area, going from the public database to the counts from all three hospitals removed 5 false alarms and introduced none, p = 0.0625, and the result stayed at 2 or 3 when we simulated the patients again 200 times.
- **The result repeats in inherited cancer and holds at scale.** In inherited cancer the hospitals' counts again removed 5 false alarms and introduced none, p = 0.0625. Across all 296 signed-off panels they removed 266 and introduced 1, p = 2.3 × 10<sup>−78</sup>. The reduction shows in each inheritance class and is largest in genes where both copies must be bad, which is also where frequency is least safe: 96 of the 136 harmful variants that are common somewhere sit in those genes.
- **Harmful variants were still caught.** The model flagged 92% of the 1,152 harmful heart test variants in every setting, and three of them lost their flag once the hospitals answered. With the public database first and the hospitals' counts second, the share flagged was 0.919 and 0.919 in the heart area, 0.912 and 0.911 in inherited cancer and 0.948 and 0.947 across every panel.
- **Overall accuracy did not move, and it cannot see any of this.** The usual summary score, AUC, stayed between 0.97 and 0.98 whatever the model was told and whichever hospital trained it. On the heart test set AlphaMissense alone reaches an AUC of 0.950, the gene name alone 0.916, and within single genes the scores still reach about 0.93 to 0.95, so the AUC reports how the expert verdicts were made, and the false alarms above are the measurement that needs the hospitals' counts.
- **Oslo consulting only its own patients did about as well as the public database.** Karachi and Lagos on their own did no better. Each hospital improved once its own counts were added to the public database, so the gain comes from combining sources.

The heart and inherited cancer runs are official, with five NVIDIA FLARE runs each in which the federated model, where only 14 numbers per hospital travel, matched a model trained on all the data in one place, 0.9783 against 0.9784 for the heart and 0.9776 against 0.9779 for inherited cancer, leaving the same false alarms in every run, while the all-panels numbers above are a first look from steps 2 and 3 alone. The federated arm that table was missing is below, split by clinical specialty.

### Twelve clinical specialties, and what a hospital training alone costs

Every panel gives enough genes to cut the table into twelve clinical specialties and run the whole
comparison inside each one: cardiac, cancer, neurology, neuromuscular, metabolic, renal, eye,
haematology, immunology, skeletal, skin and hearing. Three models per specialty, one trained at Oslo
alone, one federated over the three hospitals and one on all the rows in one place. The patient this is
about is the one whose ancestry the hospital rarely sees, a South Asian or African-ancestry patient who
arrives at Oslo. All three models are told the same thing about frequency, so the only difference
between them is how they were trained.

![Two panels over twelve clinical specialties. Panel A, AUC on variants of South Asian or African ancestry patients seen at Oslo, has the three lines on top of each other in every specialty. Panel B, false alarms among population-discordant benign variants, separates them: Oslo training alone is above federated and pooled in ten of the twelve specialties and level in the other two, by about a third in neurology and metabolic disease](docs/disease_areas_figure.png?v=1)

- **Federated training cost nothing in ranking, in any of the twelve.** The three lines in panel A sit
  within 0.002 AUC of each other, and which one is on top changes from specialty to specialty.
- **A hospital training alone cost false alarms.** Oslo alone was never better than federated, worse in
  ten specialties and level in two, while federated and pooled agreed everywhere: 83 against 67 of 1,489
  in neurology, 34 against 23 of 426 in metabolic disease, about a third more.
- **Not paid for by missing disease.** The share of harmful variants caught was within 0.004 of
  federated in every specialty, and higher for Oslo alone in nine of the twelve.
- **Why a hospital alone learns a weaker rule.** Only 3% of Oslo's training variants are common in a
  population other than its own, against 22% of Lagos's. Oslo sees few examples of "common and yet
  harmless", so its frequency veto comes out weaker, −3.35 against −3.99 after averaging, and its
  threshold was the stricter of the two. Federation lends it a rule its own patients cannot teach it.
- **The AUC cannot see any of it**, which is the same lesson as the heart area one metric further on.
  Frequency does not reorder variants, it vetoes a few near the threshold. Eight changed calls out of 426
  move the fourth decimal of an AUC over 4,256 variants.

The numbers behind the figure, averaged over three deals of the training variants, written by the script
into [docs/disease_areas_table.md](docs/disease_areas_table.md). Specialties share genes, so their
columns are not independent and the counts should not be added up.

| Disease area | Genes | Test variants | Travelling variants | Travelling pathogenic | AUC, Oslo alone | AUC, federated | AUC, pooled | False alarms, Oslo alone | False alarms, federated | False alarms, pooled | Discordant benign |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cardiac | 104 | 1,631 | 318 | 24 | 0.9634 (0.0005) | 0.9623 (0.0007) | 0.9625 (0.0002) | 2.0 | 2.0 | 2.0 | 122 |
| cancer | 41 | 902 | 115 | 14 | 0.9668 (0.0014) | 0.9689 (0.0000) | 0.9675 (0.0000) | 4.3 | 4.0 | 4.0 | 50 |
| neurology | 2160 | 14,278 | 3,793 | 347 | 0.9582 (0.0000) | 0.9576 (0.0000) | 0.9577 (0.0000) | 83.3 | 67.0 | 68.7 | 1,489 |
| neuromuscular | 442 | 3,541 | 1,160 | 132 | 0.9648 (0.0000) | 0.9639 (0.0002) | 0.9639 (0.0001) | 19.7 | 14.3 | 14.3 | 474 |
| metabolic | 795 | 4,256 | 1,109 | 274 | 0.9686 (0.0003) | 0.9695 (0.0001) | 0.9697 (0.0000) | 33.7 | 23.0 | 22.7 | 426 |
| renal | 260 | 1,847 | 541 | 64 | 0.9571 (0.0006) | 0.9578 (0.0002) | 0.9580 (0.0001) | 9.7 | 6.0 | 6.7 | 279 |
| eye | 595 | 4,845 | 1,406 | 150 | 0.9561 (0.0002) | 0.9561 (0.0002) | 0.9562 (0.0000) | 20.3 | 16.3 | 17.0 | 615 |
| haematology | 243 | 1,529 | 350 | 34 | 0.9685 (0.0010) | 0.9702 (0.0005) | 0.9698 (0.0001) | 7.0 | 7.0 | 7.0 | 173 |
| immunology | 378 | 2,067 | 697 | 43 | 0.9648 (0.0003) | 0.9645 (0.0002) | 0.9645 (0.0001) | 16.3 | 11.7 | 12.3 | 333 |
| skeletal | 495 | 4,295 | 1,117 | 75 | 0.9484 (0.0005) | 0.9475 (0.0007) | 0.9477 (0.0001) | 10.3 | 9.0 | 9.7 | 471 |
| skin | 281 | 2,545 | 612 | 55 | 0.9688 (0.0003) | 0.9688 (0.0002) | 0.9693 (0.0001) | 9.0 | 8.7 | 8.0 | 280 |
| hearing | 165 | 1,753 | 535 | 46 | 0.9289 (0.0006) | 0.9293 (0.0003) | 0.9294 (0.0001) | 8.7 | 5.0 | 4.3 | 195 |

This run used the all-panels table as it stood before the gene-alias pass, 123,454 variants rather than
127,618, so its counts sit beside the all-panels numbers above rather than replacing them. Differences
between specialties are not a specialty effect: each holds different variants, and only the comparison
between the three models inside one specialty means anything.

These are small numbers from a simulation, and the limits below say what they can and cannot show. Full tables: [docs/step3_results.md](docs/step3_results.md) and [docs/step4_results.md](docs/step4_results.md) for the heart, [docs/cancer_results.md](docs/cancer_results.md) for inherited cancer and [docs/disease_areas.md](docs/disease_areas.md) for every panel.

## The patient query

A patient carries a DNA variant. The clinician picks it on the screen, every hospital answers with counts, and the screen gives the call twice: what the patient's own hospital could say alone, with the public database, and what it can say once all hospitals have answered. Two fixed rules read the counts, and nothing is generated or random. Common among healthy patients anywhere, at the gene's line or above, means likely harmless. Common, yet at least three times more frequent among the sick, means keep flagged. The line is read for each gene from the NHS panel records: 0.1% where one bad copy of the gene is enough to cause disease and 1% where both copies must be bad, because healthy carriers of one copy are then expected. Across every panel this per-gene line keeps 123 of the 129 harmful variants that a single 0.1% line had cleared. Under the two rules the screen adds the trained model's opinion, for example "the trained model puts this at 0.1% likely harmful, below its cut-off of 57%". The last line states what crossed hospital walls: one variant name out, counts back, and no patient record. For a patient at Oslo, asking the other two hospitals changes the reading for 467 of 8,682 missense variants.

![The patient query on screen: the disease area, what to show, the gene and the kind of mutation at the top left; the reading from the patient's own hospital and from all three hospitals at the top right, with the gene's line and the trained model's opinion under it; and one chart with a row per source of evidence](docs/patient_query_tui.png?v=2)

Beside the 8,790 missense variants the query answers for 73,570 other small mutations in the same heart genes: changes that cut the protein short, shift the reading frame or break a splice site, and silent, intronic and other changes. The trained model scores missense changes only. Outside missense the mutation type alone matches the expert verdict for 66,099 of 66,298 variants, 99.7%, so each such variant starts from the presumption a lab starts from, harmful for a protein-cutting change and harmless for a silent one, and the two rules can overrule it. The same insertion or deletion can be written at several positions when it sits in a repeat, and two hospitals that write it differently would count zero for each other, so the query slides every one to the same spelling against the GRCh38 genome before any hospital is asked; of 8,890 insertions and deletions in the heart genes, 6,494 have more than one valid spelling. The 25-letter MYBPC3 deletion, with 8 valid spellings, is carried by 3.2% of South Asians and by almost no one else. Read as written, Karachi answers none of 6,800 healthy patients and frequency says nothing; read at one spelling, Karachi answers 215 of 6,800 and the variant is likely harmless.

![The screen on the inherited cancer area: POLD1 S173N, carried by 10.0% of healthy patients at Lagos, cleared once all hospitals answer, with the 0.1% line for a one-copy gene and the trained model's opinion under the verdict](docs/patient_query_tui_cancer.png?v=1)

The first dropdown switches between the heart, inherited cancer and every NHS signed-off panel, whichever are built, and each area has its own examples to try. F1 explains the keys and every part of the screen. For inherited cancer the examples are `POLD1 S173N`, `PMS2 T511M`, `MUTYH G63D` and `RNF43 R657P`. For the heart:

| Try | What it shows |
|---|---|
| `DSP N1526K` | Oslo alone cannot clear it. Among healthy patients at Lagos 15% carry it, so it is likely harmless. |
| `TTR V142I` | Common at Lagos, yet five times more common among the sick, so it stays flagged. It is a known cause of cardiac amyloidosis in people of African ancestry. The sick-patient counts are simulated from the known verdict, so this row shows how the rule works and proves nothing about the variant. |
| `MYH7 R403Q` | Seen nowhere. Frequency says nothing, and the prediction scores have to decide. |
| `MYBPC3 c.3628-41_3628-17del` | A 25-letter deletion with 8 valid spellings. Karachi finds it in 3.2% of its healthy patients once every hospital reads the same spelling. ClinVar's labs disagree about it, so the count is all the query says. |

## What is real and what is simulated

Expert verdicts come from ClinVar, computer predictions of damage from dbNSFP and frequencies per population from gnomAD, joined into one table of 8,790 missense variants in 97 heart disease genes used by NHS labs. A test set of 2,373 variants is locked away before anything is trained, including every variant of six whole genes, and each hospital sees only its own share of the rest.

| Real | Simulated |
|---|---|
| Every expert verdict, prediction score and population frequency | Which hospital has classified which variant |
| The gene list, from NHS-approved panels | Every patient and every carrier count |
| | A public database that covers Europeans only. It stands in for the many populations that real databases cover poorly. |

## Limits

- The patients are simulated from the same public frequencies we score against, and the expert verdicts themselves lean on those databases, so the close match between "all three hospitals" and "the true frequency" is partly built in.
- 183 variants is a small test, and only five calls changed. A larger gene list is the remedy, and the all-panels table is that list; its numbers are a first look without a federated run.
- The test set has no harmful variant that is common in one population, so the risk of wrongly clearing such a variant could not be measured. `TTR V142I` shows the case is real: scored as if unseen, the model clears it as harmless whatever it is told about frequency, and only the hand rule keeps it flagged.
- The model on the screen is the pooled model of step 3, which left the same false alarms as the federated one in every run. Its cut-off was set on training rows, and its probability for a variant in a gene it has seen partly reflects the gene.
- The 1% line for genes that need two bad copies is not always enough. Across every panel six common recessive alleles stay above it, `HFE C259Y` at 5.74% among them, and would be cleared; a lab would want a disease-specific line for such genes.
- Contested variants, which ClinVar's labs disagree about, have no truth here. The MYBPC3 deletion is one: the query shows a count and says nothing about whether the variant is harmful.
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

## Read more

- [The website](https://collaborativebioinformatics.github.io/Mildly-Flirting-With-Federated-Learning-for-Clinical-Diagnostics/): the [patient query in the browser](https://collaborativebioinformatics.github.io/Mildly-Flirting-With-Federated-Learning-for-Clinical-Diagnostics/demo/), the pipeline in motion and the manuscript as tabs of one page.
- [The manuscript as a PDF](Manuscript.pdf), written in [Manuscript.md](Manuscript.md): methods, results, limits and references.
- [docs/team_section_shared_manuscript.md](docs/team_section_shared_manuscript.md): the condensed story, Team 12's blocks for the shared hackathon manuscript.
- [docs/patient_query.md](docs/patient_query.md): what the clinician's screen shows, how every line on it is computed, the per-gene line, the model line and its proof.
- [docs/disease_areas.md](docs/disease_areas.md): how to add a disease area, the inherited cancer and all-panels builds, and the split by inheritance.
- [docs/other_mutation_types.md](docs/other_mutation_types.md): the count query beyond missense, the spelling problem and the MYBPC3 demo.
- [docs/quarterly_replay.md](docs/quarterly_replay.md): the hospitals grow quarter by quarter and the model is retrained each time.
- [docs/build.md](docs/build.md): how to build and run, for developers: every command, what each step makes, the reference check and where the data goes.
- [docs/handoff_ml.md](docs/handoff_ml.md): the handoff to the ML side, with what is on main and the results so far.

## Team

Team 12, clinical diagnostics track.

- Yan Li
- Mohit B. Panwar
- Shreya Srivastava
- Oumaima Boussouis
- Claude Code 😉
