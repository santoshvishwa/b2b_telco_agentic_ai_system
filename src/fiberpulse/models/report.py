"""Compare the real model (v1) with the stand-in (v0) on the planted eval accounts.

Reads eval/ on purpose: this is a developer report, not agent code.

    python -m fiberpulse.models.report --data data/synthetic
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def build_report(data_dir="data/synthetic") -> dict:
    data_dir = Path(data_dir)
    con = sqlite3.connect(data_dir / "lakehouse.db")
    s0 = pd.read_sql_query("SELECT account_id, signal_type, score, flagged FROM model_scores", con)
    s1 = pd.read_sql_query("SELECT account_id, signal_type, score, score_rank, flagged FROM model_scores_v1", con)
    con.close()
    ev = pd.DataFrame(json.loads((data_dir / "eval" / "eval_set.json").read_text(encoding="utf-8")))
    truth = pd.read_csv(data_dir / "eval" / "ground_truth.csv", index_col="account_id")
    out: dict = {}

    flags = {}
    for sig in ("churn", "acquisition"):
        a = set(s0[(s0.signal_type == sig) & (s0.flagged == 1)].account_id)
        b = set(s1[(s1.signal_type == sig) & (s1.flagged == 1)].account_id)
        col = "true_churned_3m" if sig == "churn" else "true_acquired_3m"
        flags[sig] = {
            "flagged_v0": len(a), "flagged_v1": len(b), "overlap": len(a & b),
            "overlap_share_of_v0": round(len(a & b) / len(a), 3),
            "outcome_rate_flagged_v0": round(float(truth.loc[sorted(a), col].mean()), 4),
            "outcome_rate_flagged_v1": round(float(truth.loc[sorted(b), col].mean()), 4),
            "outcome_rate_overall": round(float(truth[col].mean()), 4),
        }
    out["flag_overlap"] = flags

    ev = ev.rename(columns={"score": "score_v0"})           # the eval set stores the v0 score
    m = ev.merge(s1, on=["account_id", "signal_type"])
    m["still_flagged"] = m["flagged"] == 1
    n = len(s1[s1.signal_type == "churn"])
    m["rank_percentile"] = (m["score_rank"] / n * 100).round(2)
    out["planted_still_flagged"] = {
        "total": f"{int(m.still_flagged.sum())} of {len(m)}",
        "by_verdict": m.groupby("expected_verdict")["still_flagged"].sum().astype(int).to_dict(),
        "by_signal": m.groupby("signal_type")["still_flagged"].sum().astype(int).to_dict(),
        "median_rank_percentile": float(m["rank_percentile"].median()),
    }
    y = (m["expected_verdict"] == "confirmed").astype(int)
    out["can_v1_tell_paradox_from_confirmed"] = {
        "auc_v0_score": round(float(roc_auc_score(y, m["score_v0"])), 3),
        "auc_v1_score": round(float(roc_auc_score(y, m["score"])), 3),
        "auc_true_probability": round(float(roc_auc_score(y, m["true_probability_3m"])), 3),
        "note": ("0.5 = the score alone cannot tell them apart (the agents' job); 1.0 = perfectly separated. "
                 "True probability separates them fully only because it includes the fresh September events."),
    }
    out["missed"] = m.loc[~m.still_flagged, ["case_id", "account_id", "signal_type", "expected_verdict",
                                             "archetype", "rank_percentile"]].to_dict("records")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synthetic")
    print(json.dumps(build_report(p.parse_args().data), indent=2))


if __name__ == "__main__":
    main()
