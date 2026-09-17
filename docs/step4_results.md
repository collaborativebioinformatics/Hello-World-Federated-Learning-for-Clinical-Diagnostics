# Step 4 results: federated training with NVFlare

**NVFlare 2.9.0, FedAvg, 20 aggregation rounds, simulator mode.** A server and
three clients as separate processes, each client reading only its own folder.
The client script is [`scripts/04_fedavg_client.py`](../scripts/04_fedavg_client.py);
everything inside its loop is step 3's local training, and the only federated
parts are `flare.receive()` and `flare.send()`.

![Two panels, three arms each. AUC 0.9781, 0.9783 and 0.9784 for Oslo only, federated and pooled, with standard deviations of 0.0002 or less. False alarms among the 183 discordant benign variants are 2.2 with standard deviation 0.4 for Oslo only, and exactly 2.0 for federated and pooled in all five runs](step4_figure.png?v=1)

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
single hospital's result depends on which variants it happened to be dealt;
federating removes that dependence. With single-digit counts the size of the gap
is not the interesting part — the **zero variance** is.

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
- **Single-digit false-alarm counts.** 2 against 2.2 out of 183 is not a
  difference anyone should quote as a rate; the zero variance is the reportable
  part.
- **Five runs, one source of variance.** The deal. Resampling the test set as
  well would widen every interval and is the honest next step if these numbers
  go in a paper.
