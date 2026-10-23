"""Training labels: what actually happened in the 3 months after the snapshot.

The generator already drew an outcome for every account (`true_churned_3m`, `true_acquired_3m`) from its hidden
true probabilities. In a real project this is the 'historical outcomes' table. It sits in eval/ only because
the generator wrote it there, so this module is how model code gets labels from eval/, and it copies just those
two columns to `models/labels_3m.csv`. Agents (Phase 3+) must never import this module. (`train.py` also reads
the true probabilities, but only to print reference AUCs; they are never features.)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LABEL_COLUMNS = {"true_churned_3m": "churned_3m", "true_acquired_3m": "acquired_3m"}


def extract_labels(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    truth = pd.read_csv(data_dir / "eval" / "ground_truth.csv", usecols=["account_id", *LABEL_COLUMNS])
    out = truth.rename(columns=LABEL_COLUMNS).set_index("account_id").sort_index()
    path = data_dir / "models" / "labels_3m.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path)
    return path


def load_labels(data_dir: str | Path) -> pd.DataFrame:
    path = Path(data_dir) / "models" / "labels_3m.csv"
    if not path.exists():
        extract_labels(data_dir)
    return pd.read_csv(path, index_col="account_id")
