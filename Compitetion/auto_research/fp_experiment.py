"""Standalone experiment: Morgan fingerprint encoding vs one-hot for substrate variables.

Runs GPBO (no LLM) with substrate columns encoded as Morgan fingerprints instead
of one-hot. Completely self-contained — does not modify any existing code.

Usage:
    cd Compitetion/auto_research
    uv run --project ~/project/BOagent python fp_experiment.py run --dataset buchwald_sub4 --seed 100
    uv run --project ~/project/BOagent python fp_experiment.py run --dataset suzuki --seed 100
    uv run --project ~/project/BOagent python fp_experiment.py batch --dataset buchwald_sub4 --seeds 100,200,300,400,500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CODE_ROOT = ROOT.parent.parent / "submission" / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

# ---- Fingerprint encoder --------------------------------------------------

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

_mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

# IUPAC name → SMILES mapping for all datasets
_SMILES_MAP: dict[str, str] = {
    # Buchwald Reactant2
    "1-bromo-4-(trifluoromethyl)benzene": "Brc1ccc(C(F)(F)F)cc1",
    "1-bromo-4-ethylbenzene": "CCc1ccc(Br)cc1",
    "1-bromo-4-methoxybenzene": "COc1ccc(Br)cc1",
    "1-chloro-4-(trifluoromethyl)benzene": "Clc1ccc(C(F)(F)F)cc1",
    "1-chloro-4-ethylbenzene": "CCc1ccc(Cl)cc1",
    "1-chloro-4-methoxybenzene": "COc1ccc(Cl)cc1",
    "1-ethyl-4-iodobenzene": "CCc1ccc(I)cc1",
    "1-iodo-4-(trifluoromethyl)benzene": "Ic1ccc(C(F)(F)F)cc1",
    "1-iodo-4-methoxybenzene": "COc1ccc(I)cc1",
    "2-bromopyridine": "Brc1ccccn1",
    "2-chloropyridine": "Clc1ccccn1",
    "2-iodopyridine": "Ic1ccccn1",
    "3-bromopyridine": "Brc1cccnc1",
    "3-chloropyridine": "Clc1cccnc1",
    "3-iodopyridine": "Ic1cccnc1",
    # Suzuki Electrophile
    "6-bromoquinoline": "Brc1ccc2ncccc2c1",
    "6-chloroquinoline": "Clc1ccc2ncccc2c1",
    "6-iodoquinoline": "Ic1ccc2ncccc2c1",
    "4-bromo-5-methyl-1-(oxan-2-yl)indazole": "Cc1c(Br)cc2nn(C3CCCCO3)cc2c1",
    "quinolin-6-yl trifluoromethanesulfonate": "O=S(=O)(OC(F)(F)F)c1ccc2ncccc2c1",
    # Suzuki Nucleophile
    "quinolin-6-ylboronic acid": "OB(O)c1ccc2ncccc2c1",
    "6-(4,4,5,5-tetramethyl-1,3,2-dioxaborolan-2-yl)quinoline": "CC1(C)OB(c2ccc3ncccc3c2)OC1(C)C",
    "[5-methyl-1-(oxan-2-yl)indazol-4-yl]boronic acid": "Cc1c(B(O)O)cc2nn(C3CCCCO3)cc2c1",
    "potassium trifluoro(quinolin-6-yl)boranuide": "[K+].[B-](F)(F)(F)c1ccc2ncccc2c1",
    "5-methyl-1-(oxan-2-yl)-4-(4,4,5,5-tetramethyl-1,3,2-dioxaborolan-2-yl)indazole": "Cc1c(B2OC(C)(C)C(C)(C)O2)cc2nn(C3CCCCO3)cc2c1",
    "potassium trifluoro(5-methyl-1-(tetrahydro-2H-pyran-2-yl)-1H-indazol-4-yl)borate": "[K+].[B-](F)(F)(F)c1cc2nn(C3CCCCO3)cc2cc1C",
}

# Which columns are substrates per dataset
_SUBSTRATE_COLS: dict[str, list[str]] = {
    "buchwald_sub4": ["Reactant2"],
    "suzuki": ["Electrophile", "Nucleophile"],
}

_FP_CACHE: dict[str, np.ndarray] = {}


def _fingerprint(name: str) -> np.ndarray:
    """Get or compute 1024-bit Morgan fingerprint for a molecule name."""
    if name in _FP_CACHE:
        return _FP_CACHE[name]
    smiles = _SMILES_MAP.get(name)
    if smiles is None:
        # Unknown molecule — return zero vector
        return np.zeros(1024, dtype=np.float64)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(1024, dtype=np.float64)
    fp = _mfpgen.GetFingerprint(mol)
    arr = np.array(fp, dtype=np.float64)
    _FP_CACHE[name] = arr
    return arr


class FingerprintEncoder:
    """Encoder that uses Morgan fingerprints for substrate columns, one-hot for the rest.

    Drop-in replacement for OneHotEncoder with the same interface.
    """

    def __init__(
        self,
        feature_cols: list[str],
        options: dict[str, list[str]],
        dataset: str,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.options = options
        self.dataset = dataset
        self.substrate_cols = _SUBSTRATE_COLS.get(dataset, [])
        self.non_substrate_cols = [c for c in feature_cols if c not in self.substrate_cols]

        # Pre-compute fingerprints for all known substrate options
        self._fp_map: dict[str, np.ndarray] = {}
        for col in self.substrate_cols:
            for opt in options.get(col, []):
                self._fp_map[opt] = _fingerprint(opt)

        # Build one-hot blocks for non-substrate columns (after substrate fingerprints)
        self._onehot_offsets: dict[str, tuple[int, int]] = {}
        offset = len(self.substrate_cols) * 1024
        for col in self.non_substrate_cols:
            n_opts = len(options.get(col, []))
            self._onehot_offsets[col] = (offset, offset + n_opts)
            offset += n_opts

        # Total dimension: substrate fingerprints + one-hot for rest
        self._dim = len(self.substrate_cols) * 1024 + offset

    @property
    def dim(self) -> int:
        return self._dim

    def encode_rows(
        self,
        rows: list[dict[str, str]],
        *,
        allow_unknown: bool = False,
    ) -> np.ndarray:
        N = len(rows)
        X = np.zeros((N, self._dim), dtype=np.float64)
        for i, row in enumerate(rows):
            pos = 0
            # Substrate fingerprints
            for col in self.substrate_cols:
                val = str(row.get(col, ""))
                fp = self._fp_map.get(val)
                if fp is not None:
                    X[i, pos : pos + 1024] = fp
                elif allow_unknown:
                    fp_unknown = _fingerprint(val)
                    X[i, pos : pos + 1024] = fp_unknown
                else:
                    raise ValueError(f"Unknown substrate {val!r} for column {col!r}")
                pos += 1024
            # One-hot for non-substrate columns
            for col in self.non_substrate_cols:
                val = str(row.get(col, ""))
                start, end = self._onehot_offsets[col]
                opts = self.options.get(col, [])
                if val in opts:
                    idx = opts.index(val)
                    X[i, start + idx] = 1.0
                elif allow_unknown:
                    pass  # leave zeros
                else:
                    raise ValueError(f"Unknown category {val!r} for column {col!r}")
        return X

    def encode_df(self, df, *, allow_unknown: bool = False) -> np.ndarray:
        rows = df[self.feature_cols].to_dict("records")
        return self.encode_rows(rows, allow_unknown=allow_unknown)

    def decode(self, vec: np.ndarray) -> dict[str, str]:
        """Decode is approximate for fingerprints — returns closest match."""
        result = {}
        pos = 0
        for col in self.substrate_cols:
            fp_vec = vec[pos : pos + 1024]
            best_name = None
            best_sim = -1.0
            for name, fp in self._fp_map.items():
                sim = np.dot(fp_vec, fp) / (np.linalg.norm(fp_vec) * np.linalg.norm(fp) + 1e-10)
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
            result[col] = best_name or ""
            pos += 1024
        for col in self.non_substrate_cols:
            start, end = self._onehot_offsets[col]
            opts = self.options.get(col, [])
            if end > start:
                idx = np.argmax(vec[start:end])
                result[col] = opts[idx] if vec[start + idx] > 0.5 else ""
        return result


# ---- Experiment runner ----------------------------------------------------

import components.library  # noqa: E402,F401
from bo_core.benchmark.data_loader import DATA_LOADERS, UNIFIED_DATASET_ROOT  # noqa: E402
from components.protocol import Composition  # noqa: E402
from engine import HybridEngine, compute_metrics  # noqa: E402


def run_single(
    dataset: str,
    seed: int,
    use_fingerprint: bool,
    n_iters: int = 40,
    n_initial: int = 5,
) -> dict[str, Any]:
    """Run one (dataset, seed, encoding) experiment with seeded prior."""
    comp = Composition(
        name="gpbo_ei_fp" if use_fingerprint else "gpbo_ei",
        surrogate="botorch_matern",
        acquisition="ei",
        llm_strategy="none",
        selector="argmax",
        params={},
    )

    engine = HybridEngine(comp, dataset, seed=seed, n_iters=n_iters)

    # Seeded prior: subsample n_initial rows from train
    rng = np.random.RandomState(seed)
    n = min(n_initial, len(engine.train_df))
    initial_indices = tuple(int(i) for i in rng.choice(len(engine.train_df), n, replace=False))
    engine.train_df = engine.train_df.iloc[list(initial_indices)].reset_index(drop=True)
    engine.initial_indices = initial_indices

    if use_fingerprint:
        # Replace encoder with fingerprint version
        engine.encoder = FingerprintEncoder(
            engine.feature_cols,
            engine.options,
            dataset,
        )
        # Re-encode pool
        engine.pool_X = engine.encoder.encode_df(engine.test_df)
        # Re-encode train observations
        engine.X_obs = engine.encoder.encode_df(engine.train_df, allow_unknown=True)

    t0 = time.time()
    trajectory = engine.run()
    elapsed = time.time() - t0

    global_best = float(np.max(engine.pool_yield))
    metrics = compute_metrics(trajectory, global_best)
    return {
        "dataset": dataset,
        "seed": seed,
        "encoding": "fingerprint" if use_fingerprint else "one_hot",
        "prior_protocol": "seeded_subsample",
        "n_initial": n_initial,
        "initial_indices": list(initial_indices),
        "best_found": metrics["best_found"],
        "t95": metrics["t95"],
        "elapsed_s": elapsed,
        "trajectory": trajectory,
    }


def cmd_run(args):
    result = run_single(args.dataset, args.seed, args.fingerprint, args.n_iters, args.n_initial)
    # Strip trajectory for console output
    summary = {k: v for k, v in result.items() if k != "trajectory"}
    print(json.dumps(summary, indent=2))
    # Save full result
    out_dir = ROOT / "history" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "fp" if args.fingerprint else "oh"
    out_path = out_dir / f"fp_exp_{args.dataset}_{tag}_seed{args.seed}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved: {out_path}")


def cmd_batch(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    results = []
    for seed in seeds:
        for fp in [False, True]:
            tag = "fp" if fp else "oh"
            print(f"\n--- {args.dataset} seed={seed} encoding={tag} ---")
            r = run_single(args.dataset, seed, fp, args.n_iters, args.n_initial)
            results.append(r)
            print(f"  best_found={r['best_found']:.2f}  t95={r['t95']}  elapsed={r['elapsed_s']:.1f}s")

    # Save batch results
    out_dir = ROOT / "history" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fp_batch_{args.dataset}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")

    # Summary table
    oh = [r for r in results if r["encoding"] == "one_hot"]
    fp = [r for r in results if r["encoding"] == "fingerprint"]
    print(f"\n{'':20s} {'one-hot':>15s} {'fingerprint':>15s}")
    print(f"{'best_found mean':20s} {np.mean([r['best_found'] for r in oh]):15.2f} {np.mean([r['best_found'] for r in fp]):15.2f}")
    print(f"{'t95 mean':20s} {np.mean([r['t95'] for r in oh]):15.1f} {np.mean([r['t95'] for r in fp]):15.1f}")
    print(f"{'n_seeds':20s} {len(oh):15d} {len(fp):15d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run")
    p_run.add_argument("--dataset", required=True, choices=["buchwald_sub4", "suzuki"])
    p_run.add_argument("--seed", type=int, default=100)
    p_run.add_argument("--fingerprint", action="store_true")
    p_run.add_argument("--n_iters", type=int, default=40)
    p_run.add_argument("--n_initial", type=int, default=5)

    p_batch = sub.add_parser("batch")
    p_batch.add_argument("--dataset", required=True, choices=["buchwald_sub4", "suzuki"])
    p_batch.add_argument("--seeds", default="100,200,300,400,500")
    p_batch.add_argument("--n_iters", type=int, default=40)
    p_batch.add_argument("--n_initial", type=int, default=5)

    args = parser.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "batch":
        cmd_batch(args)
    else:
        parser.print_help()
