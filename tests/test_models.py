"""Phase 2 tests: point-in-time features (no leakage), labels, training, scores table, report."""

import shutil
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from fiberpulse.models import features as F
from fiberpulse.models.labels import load_labels
from fiberpulse.models.report import build_report
from fiberpulse.models.train import SCORED_COLUMNS, available_backends, train
from fiberpulse.synth import constants as K
from fiberpulse.synth.build import build

N = 2000
_CACHE: dict = {}


def data_dir() -> Path:
    if "dir" not in _CACHE:
        out = Path(tempfile.mkdtemp(prefix="fp_models_"))
        build(N, 42, out, quiet=True)
        _CACHE["dir"] = out
    return _CACHE["dir"]


def trained() -> dict:
    if "report" not in _CACHE:
        _CACHE["report"] = train(data_dir(), backend="sklearn", quiet=True)
    return _CACHE["report"]


def features_of(db: Path) -> pd.DataFrame:
    con = sqlite3.connect(db)
    f = F.build_features(con)
    con.close()
    return f


def copy_db(tag: str) -> Path:
    dst = Path(tempfile.mkdtemp(prefix=f"fp_{tag}_")) / "lakehouse.db"
    shutil.copy(data_dir() / "lakehouse.db", dst)
    return dst


# --- leakage ---------------------------------------------------------------------------------


def test_features_do_not_change_when_everything_after_the_snapshot_is_removed():
    base = features_of(data_dir() / "lakehouse.db")
    db = copy_db("trunc")
    snap = K.MODEL_SNAPSHOT.isoformat()
    con = sqlite3.connect(db)
    con.execute(f"DELETE FROM sr_complaint WHERE opened_date > '{snap}'")
    # a ticket closed after the snapshot looks open on the snapshot date; hide the later closure
    con.execute(f"UPDATE sr_complaint SET closed_date = NULL, csat = NULL, resolution_hours = NULL "
                f"WHERE closed_date > '{snap}'")
    con.execute(f"DELETE FROM call_intent_signals WHERE call_date > '{snap}'")
    con.execute(f"DELETE FROM external_enrichment WHERE signal_date > '{snap}'")
    con.execute(f"DELETE FROM addons WHERE start_date > '{snap}'")
    con.execute(f"UPDATE addons SET end_date = NULL WHERE end_date > '{snap}'")
    con.execute(f"DELETE FROM offer_history WHERE offer_date > '{snap}'")
    con.execute(f"DELETE FROM bills WHERE bill_month > '{snap[:7]}'")
    con.commit()
    con.close()
    pd.testing.assert_frame_equal(base, features_of(db))


def test_features_ignore_the_exact_date_and_details_of_a_renewal_after_the_snapshot():
    db = copy_db("renew")
    con = sqlite3.connect(db)
    snap = K.MODEL_SNAPSHOT.isoformat()
    n = con.execute(f"SELECT COUNT(*) FROM customer_360 WHERE last_renewal_date > '{snap}'").fetchone()[0]
    assert n > 0, "the test needs some accounts that renewed after the snapshot"
    base = features_of(data_dir() / "lakehouse.db")
    con.execute(f"UPDATE customer_360 SET last_renewal_date = '2026-09-30', contract_term_months = 36, "
                f"contract_months_remaining = 36 WHERE last_renewal_date > '{snap}'")
    con.commit()
    con.close()
    pd.testing.assert_frame_equal(base, features_of(db))


def test_the_leak_check_is_not_vacuous_removing_earlier_data_does_change_features():
    base = features_of(data_dir() / "lakehouse.db")
    db = copy_db("control")
    con = sqlite3.connect(db)
    con.execute("DELETE FROM sr_complaint WHERE opened_date BETWEEN '2026-06-03' AND '2026-08-31'")
    con.commit()
    con.close()
    changed = features_of(db)
    assert not changed["tickets_90d"].equals(base["tickets_90d"])


def test_feature_table_shape_and_names():
    f = features_of(data_dir() / "lakehouse.db")
    assert len(f) == N and f.index.is_unique and f.index.name == "account_id"
    assert f.shape[1] >= 60
    assert not any(w in c for c in f.columns for w in ("true_", "fresh", "sept", "label", "score"))
    for col in ("late_last3", "tickets_90d", "plan_speed_mbps", "contract_months_remaining", "tenure_months"):
        assert f[col].notna().all(), col
    assert (f["tenure_months"] >= 0).all()
    assert set(f["plan_speed_mbps"].unique()) <= {p[1] for p in K.PLANS}


# --- labels, training, scores ------------------------------------------------------------------


def test_labels_have_realistic_base_rates():
    lab = load_labels(data_dir())
    assert len(lab) == N
    assert 0.02 < lab["churned_3m"].mean() < 0.07
    assert 0.03 < lab["acquired_3m"].mean() < 0.09


def test_training_writes_a_scores_table_in_the_model_scores_format():
    trained()
    con = sqlite3.connect(data_dir() / "lakehouse.db")
    cols = [r[1] for r in con.execute("PRAGMA table_info(model_scores_v1)")]
    assert cols == SCORED_COLUMNS == [r[1] for r in con.execute("PRAGMA table_info(model_scores)")]
    scores = pd.read_sql_query("SELECT * FROM model_scores_v1", con)
    con.close()
    assert len(scores) == 2 * N
    assert scores["score"].between(0, 1).all()
    assert (scores["model_version"] == "hgb-v1").all()
    expected = int(round(N * K.FLAG_RATE))
    for sig, g in scores.groupby("signal_type"):
        assert g["flagged"].sum() == expected, sig
        assert sorted(g["score_rank"]) == list(range(1, N + 1))
        top = g[g["score_rank"] <= expected]
        assert (top["flagged"] == 1).all() and top["score"].min() >= g[g["flagged"] == 0]["score"].max()


def test_models_beat_chance_and_stay_below_the_oracle():
    rep = trained()
    for sig in ("churn", "acquisition"):
        m = rep["signals"][sig]["metrics"]
        assert m["auc"] > 0.60, sig
        assert m["auc"] < m["auc_oracle_true_probability"] + 0.02, sig    # cannot beat the truth
        assert m["lift_at_top"] > 1.5, sig


def test_same_seed_gives_identical_scores():
    trained()
    a = pd.read_csv(data_dir() / "models" / "model_scores_v1.csv")
    out = Path(tempfile.mkdtemp(prefix="fp_models2_"))
    shutil.copytree(data_dir(), out, dirs_exist_ok=True)
    train(out, backend="sklearn", quiet=True)
    b = pd.read_csv(out / "models" / "model_scores_v1.csv")
    pd.testing.assert_frame_equal(a, b)


def test_backend_list_always_offers_sklearn_and_prefers_xgboost_when_installed():
    backends = available_backends()
    assert backends[-1] == "sklearn"
    try:
        import xgboost  # noqa: F401
        assert backends[0] == "xgboost"
    except ImportError:
        assert backends == ["sklearn"]


def test_report_compares_v1_with_the_planted_accounts():
    trained()
    r = build_report(data_dir())
    assert set(r) >= {"flag_overlap", "planted_still_flagged", "can_v1_tell_paradox_from_confirmed", "missed"}
    for sig in ("churn", "acquisition"):
        f = r["flag_overlap"][sig]
        assert f["flagged_v1"] == f["flagged_v0"]
        assert f["outcome_rate_flagged_v1"] > 2 * f["outcome_rate_overall"]     # flagged accounts really do churn/buy more
    total = int(r["planted_still_flagged"]["total"].split(" of ")[0])
    assert 0 < total <= 50
    assert len(r["missed"]) == 50 - total
