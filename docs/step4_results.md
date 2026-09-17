# Step 4 results: federated training with NVFlare

**NVFlare 2.9.0, FedAvg, 20 aggregation rounds, simulator mode.** A server and
three clients as separate processes, each client reading only its own folder.
The client script is [`scripts/04_fedavg_client.py`](../scripts/04_fedavg_client.py);
everything inside its loop is step 3's local training, and the only federated
parts are `flare.receive()` and `flare.send()`.

![Three panels. AUC 0.9781, 0.9783 and 0.9784 for Oslo only, federated and pooled, with standard deviations of 0.0002 or less. False alarms among the 183 discordant benign variants are 2.2 for Oslo only and exactly 2.0 for federated and pooled. The paired difference between federated and each population's own hospital is -0.0011 for European, -0.0007 for South Asian and +0.0031 for African, every interval crossing zero](step4_figure.png?v=2)

Reproduce with `uv run python scripts/04_federated_train.py` (about six minutes).
Every number is in `data/results_federated.json`.

## The result

Five runs, **mean (standard deviation over runs)**, scored on the locked test
set that no arm ever trained on.

| | AUC, public frequency | AUC, federated count query | false alarms / 183, public | false alarms / 183, federated counts |
|---|---|---|---|---|
| Oslo only | 0.9758 (0.0002) | 0.9781 (0.0002) | 7.6 (0.9) | 2.2 (0.4) |
| **Federated, NVFlare FedAvg** | **0.9759 (0.0001)** | **0.9783 (0.0001)** | **7.0 (0.0)** | **2.0 (0.0)** |
| Everything pooled | 0.9760 (0.0001) | 0.9784 (0.0001) | 7.0 (0.0) | 2.0 (0.0) |

**Federated lands on pooled.** The gap is 0.0001 in AUC, which is the same size
as the run-to-run standard deviation, and identical to the last digit on false
alarms. Nothing was lost by keeping the rows at home. That is the whole claim,
and it is the right one to make: federation is not supposed to beat centralised
training, it is supposed to cost nothing against it.

**Oslo alone is behind, and it is the only arm that moves.** Its false alarms
were 3, 2, 2, 2, 2 across the five deals against a flat 2, 2, 2, 2, 2 for both
other arms, and 8, 7, 7, 7, 9 against a flat 7 under the public reference. A
single hospital's result depends on which variants it happened to be dealt.

Do not over-read this. Three against two out of 183 is **one variant**, and a
zero variance over five runs at counts this small could easily be luck rather
than stability. What these counts support is "federated is at least as good as a
single site and indistinguishable from pooled", not "federation demonstrably
reduces variance". The variance claim is better made on the per-population AUC
below, where the numbers have more resolution than a count of two.

## Per population

The test set labels each variant with the population it is most common in, by the
rule step 2 applied. AUC on each slice, federated count evidence, mean (sd) over
the five runs:

| trained on | European NFE | South Asian SAS | African AFR | none |
|---|---|---|---|---|
| test rows | 382 | 291 | 241 | 1,459 |
| of which pathogenic | 58 | **23** | **12** | 1,059 |
| Oslo only | 0.9346 (0.0009) | 0.9275 (0.0009) | 0.9539 (0.0010) | 0.9759 (0.0001) |
| Federated, FedAvg | 0.9335 (0.0004) | 0.9259 (0.0005) | 0.9518 (0.0008) | 0.9758 (0.0001) |
| Everything pooled | 0.9337 (0.0002) | 0.9260 (0.0001) | 0.9516 (0.0000) | 0.9757 (0.0000) |

`none` is the two thirds of the test set gnomAD never saw in any of the three
populations, so it belongs to none of them.

**The project's original hypothesis is refuted.** README section 5 used to
predict a single site scoring 0.90 on its own population and collapsing to 0.78
and 0.75 on the others. Oslo alone scores 0.9346, 0.9275 and 0.9539 — it does not
collapse anywhere, and it is marginally the best arm on every slice. The reason:
the twelve prediction scores carry essentially all the ranking signal and none of
them depends on ancestry. What depends on ancestry is the frequency evidence at
scoring time, and that moves the decision rather than the ranking, which is why
the false-alarm column exists.

Do not read the between-slice differences as a population effect. AFR scores
higher than NFE here because those 241 variants happen to be easier, not because
the model serves African-ancestry patients better. The slices contain different
variants; only within-slice, between-arm comparisons are meaningful.

## Does the global model let any population down?

This is the direct test of the non-IID worry, and it is paired: the same test
rows, scored by two models, so the per-run difference is the statistic.

| population | its own hospital | that hospital's model alone | federated | difference |
|---|---|---|---|---|
| European NFE | Oslo | 0.9346 (0.0009) | 0.9335 (0.0004) | −0.0011 (0.0013) |
| South Asian SAS | Karachi | 0.9266 (0.0021) | 0.9259 (0.0005) | −0.0007 (0.0024) |
| African AFR | Lagos | 0.9488 (0.0027) | 0.9518 (0.0008) | **+0.0031** (0.0023) |

**No population is served worse by the global model than by its own hospital's.**
Every interval crosses or touches zero. The only positive difference is
African-ancestry, the smallest population and the one gnomAD covers worst, which
is the population you would predict to gain — but at 12 pathogenic variants and
about 1.3 standard deviations this is a direction, not a result. The honest claim
is that federating **does not hurt anyone**.

The other column is quieter and more solid: federating **stabilises** the small
sites. Lagos alone varies by 0.0027 between deals and Karachi by 0.0021, against
0.0008 and 0.0005 federated. A small hospital's model depends on which variants
it happened to classify; the global model does not.

This is also the direct evidence for the claim the frequency coefficient only
supported indirectly. Seeing −3.57 survive the averaging says the veto is still
in the model; seeing Lagos's own slice not get worse says the averaging did not
cost the population that leans on it hardest.

## What varies across the five runs

The deal: which hospital classified which variant, re-drawn with
`deal_verdicts` under seeds 100 to 104. The test set, the patient cohorts, the
model and the feature preparation are byte-identical across runs, so nothing
else can move the numbers. Hospital sizes stayed close to the intended
0.70 / 0.15 / 0.15 split:

| seed | Oslo | Karachi | Lagos |
|---|---|---|---|
| 100 | 4,332 | 1,870 | 1,896 |
| 101 | 4,330 | 1,845 | 1,972 |
| 102 | 4,323 | 1,849 | 1,901 |
| 103 | 4,341 | 1,826 | 1,908 |
| 104 | 4,310 | 1,835 | 1,927 |

This is a deliberate choice and worth defending. The training is deterministic
full-batch gradient descent, so five runs on the *same* deal would reproduce
each other exactly and report a standard deviation of zero — a statement about
the optimiser, not about federation. The question a reader actually has is
whether "federated equals pooled" survives a different partition, and that is
what these error bars answer.

## What crosses the wire

A vector of **14 floats** and a row count, per site per round. No variant, no
score, no verdict. FedAvg weights each site's update by its row count, so the
0.70 / 0.15 / 0.15 asymmetry is respected rather than ignored.

The threshold is federated too. Each site takes the 95%-sensitivity quantile on
its own rows under the global model, and the server averages those by row count:
one more number per site, the same kind of payload as the weights. This matters
because using the pooled threshold would quietly hand the federated arm access
it does not have. It changes none of the figures above, which is worth knowing
but is not a reason to have got it wrong.

## The model FedAvg arrived at

Averaged weights land close to, but not on top of, the pooled ones — largest
coefficient difference 0.156 of 14. On the coefficient the project is about:

| | log frequency weight |
|---|---|
| Federated (FedAvg) | −3.57 |
| Pooled | −3.73 |

The veto survives averaging. This was the open question at the end of step 3:
whether weight averaging washes out the population-specific frequency evidence
that the small hospitals lean on hardest (Karachi −3.80, Lagos −4.09 against
Oslo's −3.26 in step 3). It does not, at least at this model size. No
personalisation step was needed, so the "federated, then tuned locally" row of
the README section 5 grid has nothing to fix and was not run.

## Honest limits

- **AUC was saturated before federation started.** Step 3 established that a
  14-weight model on 12 scores reaches 0.97 on 1,800 rows, so no training method
  can separate itself here. Do not present the flat AUC row as evidence that
  FedAvg works well; present it as evidence that it costs nothing. The metric
  that moves in this project is the false-alarm rate, and what moves it is the
  step 6 **count query**, not the weight averaging.
- **Three clients, one machine, no network.** Simulator mode. Stragglers, dropped
  clients, secure aggregation and real latency are all absent.
- **Single-digit false-alarm counts.** 2 against 2.2 out of 183 is one variant.
  Neither the gap nor the zero variance is strong evidence at this size; the
  per-population AUC carries the variance claim better.
- **Five runs, one source of variance.** The deal. Resampling the test set as
  well would widen every interval and is the honest next step if these numbers
  go in a paper.
- **The per-population slices are thin.** 12 pathogenic variants in the African
  slice and 23 in the South Asian one. Between-arm differences there are
  directional at best. The paired design helps, since the same rows are scored
  twice, but it cannot manufacture positives that are not in the test set.
