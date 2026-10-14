"""Stand-in predictive layer ("simulated-v0"). Replaced by a real model in Phase 2.

Scores are propensity scores in 0-1, rank-calibrated so the flagged top slice lands around
0.70-0.99. They are NOT calibrated probabilities; ground-truth probabilities live in eval/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants as K


def score_from_z(z_sim: np.ndarray, flag_rate: float = K.FLAG_RATE):
    """Map the stand-in model's logit to (score, flagged). Flag cut-off maps to a score of 0.70."""
    zs = (z_sim - z_sim.mean()) / z_sim.std()
    cut = np.quantile(zs, 1.0 - flag_rate)
    score = 1.0 / (1.0 + np.exp(-(1.6 * (zs - cut) + 0.85)))
    return score, zs >= cut


def build_model_scores(account_ids, z_sim_c, z_sim_a) -> tuple[pd.DataFrame, dict]:
    """Return the model_scores table plus helper masks used by the planting step."""
    n = len(account_ids)
    sc_c, fl_c = score_from_z(z_sim_c)
    sc_a, fl_a = score_from_z(z_sim_a)

    def part(signal, score, flagged):
        rank = (-score).argsort(kind="stable").argsort(kind="stable") + 1
        return pd.DataFrame({
            "account_id": account_ids,
            "scored_month": K.SCORED_MONTH,
            "signal_type": signal,
            "score": np.round(score, 4),
            "model_version": K.MODEL_VERSION,
            "score_rank": rank,
            "flagged": flagged.astype(int),
        })

    table = pd.concat([part("churn", sc_c, fl_c), part("acquisition", sc_a, fl_a)], ignore_index=True)

    # "core" accounts: well inside the flagged slice, so a real Phase 2 model should still flag them
    zc = (z_sim_c - z_sim_c.mean()) / z_sim_c.std()
    za = (z_sim_a - z_sim_a.mean()) / z_sim_a.std()
    core_c = zc >= np.quantile(zc, 1.0 - K.CORE_RATE)
    core_a = za >= np.quantile(za, 1.0 - K.CORE_RATE)
    masks = {"flagged_c": fl_c, "flagged_a": fl_a, "core_c": core_c, "core_a": core_a,
             "score_c": sc_c, "score_a": sc_a}
    return table, masks
