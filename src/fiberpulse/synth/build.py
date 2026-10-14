"""Build the whole synthetic dataset.

    python -m fiberpulse.synth.build --n-accounts 20000 --seed 42 --out data/synthetic

Output layout (everything under --out):
    lakehouse.db            SQLite stand-in for the Databricks tables the agents may read
    raw/*.csv               the same tables as CSV
    knowledge/              policy_handbook.md and outcomes_kb.jsonl (RAG sources)
    eval/                   ground truth and eval sets - agents must NEVER read this folder
    manifest.json           seed, counts, file hashes, validation result
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import constants as K
from .archetypes import PLAN, plant
from .catalog import render_handbook
from .evalset import build_compliance_cases, build_context_truth, build_eval_set
from .events import EventBook, baseline_events
from .kb import build_outcomes_kb
from .population import build_population
from .scoring import build_model_scores
from .tables import assemble, write_sqlite
from .validate import run_checks

GENERATOR_VERSION = "1.0"
MIN_ACCOUNTS = 1000


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def build(n_accounts: int = 20000, seed: int = 42, out_dir: str | Path = "data/synthetic",
          validate: bool = True, quiet: bool = False) -> dict:
    if n_accounts < MIN_ACCOUNTS:
        raise ValueError(f"Use at least {MIN_ACCOUNTS} accounts so the 50-case eval set can be planted.")
    started = time.perf_counter()
    log = (lambda *a: None) if quiet else print
    out = Path(out_dir)
    for sub in ("raw", "knowledge", "eval"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed + 1)
    log(f"[1/6] population ({n_accounts:,} accounts)")
    pop = build_population(n_accounts, rng, pyrng)
    scores, masks = build_model_scores(pop.cust["account_id"].to_numpy(), pop.z_sim_c, pop.z_sim_a)

    log("[2/6] planted eval archetypes and background events")
    book = EventBook(pop, rng, pyrng)
    planted = plant(book, masks)
    baseline_events(book, planted)

    log("[3/6] tables, bills, ground truth")
    tables, gt = assemble(pop, book, scores, masks, planted, rng)

    log("[4/6] knowledge base, policy handbook, eval sets")
    kb = build_outcomes_kb(np.random.default_rng(seed + 1000), random.Random(seed + 1001))
    eval_rng = np.random.default_rng(seed + 2000)
    eval_set = build_eval_set(tables, gt)
    compliance = build_compliance_cases(tables, eval_rng)
    context = build_context_truth(tables, gt, eval_rng)

    log("[5/6] writing files")
    write_sqlite(out / "lakehouse.db", tables)
    for name, df in tables.items():
        df.to_csv(out / "raw" / f"{name}.csv", index=False, lineterminator="\n")
    (out / "knowledge" / "policy_handbook.md").write_text(render_handbook(), encoding="utf-8")
    (out / "knowledge" / "outcomes_kb.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in kb), encoding="utf-8")
    gt.to_csv(out / "eval" / "ground_truth.csv", index=False, lineterminator="\n")
    _write_json(out / "eval" / "eval_set.json", eval_set)
    _write_json(out / "eval" / "compliance_cases.json", compliance)
    _write_json(out / "eval" / "context_truth.json", context)

    hashed = [p for p in sorted(out.rglob("*")) if p.is_file()
              and p.name not in ("manifest.json", "lakehouse.db")]
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "n_accounts": n_accounts,
        "as_of": K.AS_OF.isoformat(),
        "model_snapshot": K.MODEL_SNAPSHOT.isoformat(),
        "model_version": K.MODEL_VERSION,
        "rows": {name: int(len(df)) for name, df in tables.items()},
        "flagged": {"churn": int(masks["flagged_c"].sum()), "acquisition": int(masks["flagged_a"].sum())},
        "planted": {p[0]: p[4] for p in PLAN},
        "knowledge": {"outcomes_kb_cases": len(kb), "compliance_cases": len(compliance),
                      "context_truth_accounts": len(context)},
        "files": {str(p.relative_to(out)).replace("\\", "/"): _sha256(p) for p in hashed},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    problems: list[str] = []
    if validate:
        log("[6/6] validating")
        _write_json(out / "manifest.json", manifest)   # validator reads eval files, not the manifest
        problems = run_checks(out)
    manifest["validation_problems"] = problems
    manifest["build_seconds"] = round(time.perf_counter() - started, 1)
    _write_json(out / "manifest.json", manifest)
    log(f"done in {manifest['build_seconds']}s -> {out}  | validation: "
        f"{'FAILED (' + str(len(problems)) + ')' if problems else 'all checks passed'}")
    for p in problems:
        log("  PROBLEM:", p)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the FiberPulse synthetic dataset")
    parser.add_argument("--n-accounts", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()
    manifest = build(args.n_accounts, args.seed, args.out, validate=not args.no_validate)
    return 1 if manifest["validation_problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
