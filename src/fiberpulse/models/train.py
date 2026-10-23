"""Train churn and acquisition models, score every account out-of-fold, write `model_scores_v1`.

    python -m fiberpulse.models.train --data data/synthetic            # XGBoost if installed, else sklearn
    python -m fiberpulse.models.train --backend sklearn --mlflow

Why out-of-fold: every account is scored by a model that never saw its label, so scores are honest. The
final model (fit on everyone) is saved for later scoring of new accounts.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fiberpulse.synth import constants as K

from .features import build_features
from .labels import load_labels

SIGNALS = {"churn": "churned_3m", "acquisition": "acquired_3m"}
SCORED_COLUMNS = ["account_id", "scored_month", "signal_type", "score", "model_version", "score_rank", "flagged"]


def available_backends() -> list[str]:
    found = []
    try:
        import xgboost  # noqa: F401
        found.append("xgboost")
    except Exception:
        pass
    found.append("sklearn")
    return found


def make_model(backend: str, seed: int):
    if backend == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.04, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                             tree_method="hist", eval_metric="logloss", random_state=seed)
    if backend == "sklearn":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_depth=4,
                                              min_samples_leaf=40, l2_regularization=2.0,
                                              early_stopping=False, random_state=seed)
    raise ValueError(f"unknown backend {backend!r}")


def top_slice_metrics(y: np.ndarray, p: np.ndarray, rate: float = K.FLAG_RATE) -> dict:
    n_top = int(round(len(y) * rate))
    top = np.argsort(-p, kind="stable")[:n_top]
    base = float(y.mean())
    precision = float(y[top].mean())
    return {"top_n": n_top, "precision_at_top": precision, "lift_at_top": precision / base if base else float("nan"),
            "capture_at_top": float(y[top].sum() / max(y.sum(), 1))}


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "logloss": float(log_loss(y, p)),
            "base_rate": float(y.mean()), **top_slice_metrics(y, p)}


def out_of_fold(X: pd.DataFrame, y: np.ndarray, backend: str, seed: int, folds: int = 5) -> np.ndarray:
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        model = make_model(backend, seed)
        model.fit(X.iloc[tr], y[tr])
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]
    return oof


def score_table(account_ids, signal: str, p: np.ndarray, version: str) -> pd.DataFrame:
    n_flag = int(round(len(p) * K.FLAG_RATE))
    order = np.argsort(-p, kind="stable")
    rank = np.empty(len(p), dtype=int)
    rank[order] = np.arange(1, len(p) + 1)
    return pd.DataFrame({"account_id": account_ids, "scored_month": K.SCORED_MONTH, "signal_type": signal,
                         "score": np.round(p, 5), "model_version": version, "score_rank": rank,
                         "flagged": (rank <= n_flag).astype(int)})


def importances(model, columns) -> pd.Series:
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)
    return pd.Series(dtype=float)


def _permutation_importance(model, X, y, top=15, seed=0) -> pd.Series:
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=3, random_state=seed, n_jobs=1)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False).head(top)


def train(data_dir="data/synthetic", backend: str | None = None, seed: int = 42, use_mlflow: bool = False,
          quiet: bool = False) -> dict:
    t0 = time.time()
    data_dir = Path(data_dir)
    out_dir = data_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = backend or available_backends()[0]
    version = f"{'xgb' if backend == 'xgboost' else 'hgb'}-v1"
    log = (lambda *a: None) if quiet else print

    con = sqlite3.connect(data_dir / "lakehouse.db")
    X = build_features(con)
    labels = load_labels(data_dir).reindex(X.index)
    log(f"features: {X.shape[0]:,} accounts x {X.shape[1]} columns (snapshot {K.MODEL_SNAPSHOT}); backend={backend}")

    truth = pd.read_csv(data_dir / "eval" / "ground_truth.csv", index_col="account_id").reindex(X.index)
    v0 = pd.read_sql_query("SELECT account_id, signal_type, score FROM model_scores", con)
    tables, report = [], {"backend": backend, "model_version": version, "seed": seed, "n_accounts": len(X),
                          "n_features": X.shape[1], "snapshot": K.MODEL_SNAPSHOT.isoformat(), "signals": {}}
    run_ctx = _start_mlflow(use_mlflow, data_dir, report)
    for signal, col in SIGNALS.items():
        y = labels[col].to_numpy(int)
        oof = out_of_fold(X, y, backend, seed)
        m = evaluate(y, oof)
        v0_scores = v0[v0.signal_type == signal].set_index("account_id")["score"].reindex(X.index).to_numpy()
        true_p = truth["true_churn_prob_3m" if signal == "churn" else "true_acquisition_prob_3m"].to_numpy()
        m["auc_v0_simulated"] = float(roc_auc_score(y, v0_scores))
        m["auc_oracle_true_probability"] = float(roc_auc_score(y, true_p))
        final = make_model(backend, seed).fit(X, y)
        imp = importances(final, X.columns)
        if imp.empty:
            imp = _permutation_importance(final, X, y)
        imp.head(25).to_csv(out_dir / f"importance_{signal}.csv", header=["importance"])
        _save_model(final, out_dir / f"{signal}_{version}", backend)
        tables.append(score_table(X.index, signal, oof, version))
        report["signals"][signal] = {"metrics": m, "top_features": imp.head(10).round(4).to_dict()}
        log(f"{signal:12s} AUC {m['auc']:.3f} (v0 {m['auc_v0_simulated']:.3f}, oracle {m['auc_oracle_true_probability']:.3f}) "
            f"PR-AUC {m['pr_auc']:.3f}  lift@top {m['lift_at_top']:.1f}x  Brier {m['brier']:.4f}")
        _log_mlflow(run_ctx, signal, m, imp)

    scores = pd.concat(tables, ignore_index=True)[SCORED_COLUMNS]
    scores.to_csv(out_dir / "model_scores_v1.csv", index=False)
    con.execute("DROP TABLE IF EXISTS model_scores_v1")
    scores.to_sql("model_scores_v1", con, index=False)
    con.execute("CREATE INDEX IF NOT EXISTS idx_msv1 ON model_scores_v1(account_id, signal_type)")
    con.commit()
    con.close()
    report["seconds"] = round(time.time() - t0, 1)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _end_mlflow(run_ctx)
    log(f"wrote model_scores_v1 ({len(scores):,} rows), models and metrics.json in {report['seconds']}s")
    return report


def _save_model(model, stem: Path, backend: str) -> None:
    if backend == "xgboost":
        model.save_model(str(stem) + ".json")
    else:
        import joblib
        joblib.dump(model, str(stem) + ".joblib")


# --- optional MLflow (never allowed to break training) -----------------------------------------


def _start_mlflow(enabled: bool, data_dir: Path, report: dict):
    if not enabled:
        return None
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{(data_dir / 'mlflow.db').resolve().as_posix()}")
        mlflow.set_experiment("fiberpulse-phase2")
        run = mlflow.start_run(run_name=report["model_version"])
        mlflow.log_params({"backend": report["backend"], "seed": report["seed"],
                           "n_features": report["n_features"], "snapshot": report["snapshot"]})
        return mlflow
    except Exception as exc:                      # noqa: BLE001
        print(f"[mlflow disabled: {exc}]")
        return None


def _log_mlflow(mlflow, signal: str, metrics: dict, imp: pd.Series) -> None:
    if mlflow is None:
        return
    try:
        mlflow.log_metrics({f"{signal}_{k}": v for k, v in metrics.items() if isinstance(v, float)})
        mlflow.log_dict(imp.head(25).round(5).to_dict(), f"importance_{signal}.json")
    except Exception as exc:                      # noqa: BLE001
        print(f"[mlflow log skipped: {exc}]")


def _end_mlflow(mlflow) -> None:
    if mlflow is not None:
        try:
            mlflow.end_run()
        except Exception:                         # noqa: BLE001
            pass


def main() -> None:
    p = argparse.ArgumentParser(description="Train FiberPulse churn and acquisition models (Phase 2)")
    p.add_argument("--data", default="data/synthetic")
    p.add_argument("--backend", choices=["xgboost", "sklearn"], default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mlflow", action="store_true", help="log to a local MLflow store (needs the mlops extra)")
    a = p.parse_args()
    train(a.data, a.backend, a.seed, a.mlflow)


if __name__ == "__main__":
    main()
