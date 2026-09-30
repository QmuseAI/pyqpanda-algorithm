from __future__ import annotations
import argparse
import json
from pyqpanda_alg.BFDCQO.benchmark.max3sat import run_max3sat_benchmark


def main():
    ap = argparse.ArgumentParser(description="Native-HUBO MAX-3-SAT benchmark")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--qaoa-backend", choices=["auto", "numpy", "pyqpanda"], default="auto")
    ap.add_argument("--output-dir", default="results/max3sat_native_hubo")
    args = ap.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    r = run_max3sat_benchmark(seeds=seeds, qaoa_backend=args.qaoa_backend, output_dir=args.output_dir)
    print("qaoa_backend:", r["qaoa_backend"], "pyqpanda3_available:", r["pyqpanda3_available"])
    print("\nReduction summary:")
    for x in r["reductions"]:
        print(json.dumps(x, sort_keys=True))
    print("\nBenchmark summary:")
    for x in r["summary"]:
        print(
            f"{x['instance']:18s} {x['solver']:24s} "
            f"n={x['solver_n']:2d} anc={x['ancillas']:2d} "
            f"p_opt={x['mean_p_opt_logical']:.4f} "
            f"E[sat]/opt={x['mean_expected_approx_ratio']:.4f} "
            f"best/opt={x['mean_best_approx_ratio']:.4f} "
            f"success={x['success_rate']:.2f} shots={x['mean_total_shots']:.0f}"
        )


if __name__ == "__main__":
    main()
