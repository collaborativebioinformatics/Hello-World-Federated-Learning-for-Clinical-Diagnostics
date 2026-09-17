"""The NVFlare client: one hospital, one round of local training.

NVFlare ships this script to each site and runs it there. Everything inside the
loop is the ordinary local training of step 3; the only federated parts are
`flare.receive()` and `flare.send()`. That is the whole point of the design: one
round of FedAvg asks a client to start from the server's weights and train a few
epochs on its own rows, which is exactly what `fit(X, y, weights, epochs)` does.

What crosses the wire is a vector of 14 floats and a row count. No variant, no
score, no verdict.

Run by scripts/04_federated_train.py, not by hand.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import nvflare.client as flare


def load_step3(root: Path):
    """Reuse step 3's model. Same feature prep and same fit, or the comparison is not one."""
    spec = importlib.util.spec_from_file_location("step3", root / "scripts" / "03_train_local.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["step3"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--run-dir", required=True, help="where this run's re-dealt hospital files live")
    parser.add_argument("--root", required=True)
    parser.add_argument("--local-epochs", type=int, default=150)
    args = parser.parse_args()

    root, run_dir = Path(args.root), Path(args.run_dir)
    step3 = load_step3(root)

    rows = pd.read_csv(run_dir / args.site / "verdicts.csv")
    score_columns = json.loads((run_dir / "score_columns.json").read_text())
    X = step3.features(rows, score_columns, rows.af_local.to_numpy())
    y = rows.label.to_numpy()

    flare.init()
    while flare.is_running():
        received = flare.receive()
        if received is None:
            break
        weights = np.asarray(received.params["weights"], dtype=float)

        # Record what the server handed us. The aggregate of the final round is
        # never sent back to anyone, so the driver runs one extra round and reads
        # the model received at the start of it.
        np.save(run_dir / f"global_from_{args.site}.npy", weights)

        weights = step3.fit(X, y, weights=weights, epochs=args.local_epochs)

        flare.send(flare.FLModel(
            params={"weights": weights},
            # FedAvg weights each site's update by this, so a lab with more
            # verdicts counts for more, which is the point of the 0.70/0.15/0.15 split.
            meta={"NUM_STEPS_CURRENT_ROUND": len(rows)},
        ))


if __name__ == "__main__":
    main()
