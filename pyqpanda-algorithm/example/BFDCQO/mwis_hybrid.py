from __future__ import annotations
import argparse
from pathlib import Path
from pyqpanda_alg.BFDCQO.benchmark.mwis import run_mwis_hybrid_benchmark


def main():
    ap = argparse.ArgumentParser(description="MWIS + confidence-guided Hybrid BF-DCQO benchmark")
    ap.add_argument("--seeds", type=int, default=10, help="run seeds 0..N-1")
    ap.add_argument("--budget", type=int, default=72)
    ap.add_argument("--neighborhood-size", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--qaoa-backend", choices=["auto", "numpy", "pyqpanda"], default="auto")
    ap.add_argument("--output-dir", default="results/mwis_hybrid")
    args = ap.parse_args()
    out = run_mwis_hybrid_benchmark(
        seeds=tuple(range(args.seeds)), quantum_budget=args.budget,
        local_neighborhood_size=args.neighborhood_size, local_rounds=args.rounds,
        qaoa_backend=args.qaoa_backend, output_dir=Path(args.output_dir),
    )
    print("MWIS low-shot benchmark")
    print("config:", out["config"])
    for row in out["summary"]:
        print(
            f"{row['instance']:8s} {row['solver']:18s} "
            f"success={row['success_rate']:.3f} weight_ratio={row['mean_weight_ratio']:.4f} "
            f"feasible={row['feasible_rate']:.3f} shots={row['mean_total_shots']:.1f} "
            f"local_eval={row['mean_local_evaluations']:.1f}"
        )


if __name__ == "__main__":
    main()
