# Licensed under the Apache License, Version 2.0.
"""Matched-shot MaxCut benchmark for BF-DCQO and QAOA."""

from __future__ import annotations
from dataclasses import asdict
import csv,json,time
from pathlib import Path
import numpy as np
from .instances import default_maxcut_instances
from .metrics import maxcut_approximation
from ..solvers import ExactSolver,BFDCQO,QAOASolver
from ..backends import pyqpanda3_backend


def run_maxcut_benchmark(*,seeds=(3,7,11),qaoa_backend="auto",output_dir=None):
    """Run the standard MaxCut Exact/BF-DCQO/QAOA benchmark suite."""
    if qaoa_backend=="auto": qaoa_backend="pyqpanda" if pyqpanda3_backend.available() else "numpy"
    rows=[]
    for app in default_maxcut_instances():
        problem=app.to_ising(); exact=ExactSolver().solve(problem); opt_cut=-exact.best_energy
        for seed in seeds:
            t=time.perf_counter(); bf=BFDCQO(problem,n_trotter=3,dt=0.1,iterations=6,shots=1024,seed=seed,backend="numpy").solve(); bf_t=time.perf_counter()-t
            bf_cut=-bf.best_energy
            rows.append({
                "instance":app.name,"n":app.n,"solver":"BF-DCQO","backend":"numpy_reference","seed":seed,"opt_cut":opt_cut,"best_cut":bf_cut,"approx_ratio":maxcut_approximation(bf_cut,opt_cut),
                "p_opt":bf.history[-1].ground_state_probability,"expected_cut":-bf.history[-1].expected_energy,"expected_approx_ratio":(-bf.history[-1].expected_energy)/opt_cut,"total_shots":bf.total_shots,"circuit_evaluations":bf.circuit_evaluations,"walltime_s":bf_t,"success":bf.success,
            })
            t=time.perf_counter(); q=QAOASolver(problem,p=1,backend=qaoa_backend,shots_per_eval=256,max_evals=20,final_shots=1024,total_shot_budget=6144,seed=seed,method="COBYLA").solve(); q_t=time.perf_counter()-t
            q_cut=-q.best_energy
            rows.append({
                "instance":app.name,"n":app.n,"solver":"QAOA-p1","backend":qaoa_backend,"seed":seed,"opt_cut":opt_cut,"best_cut":q_cut,"approx_ratio":maxcut_approximation(q_cut,opt_cut),
                "p_opt":q.ground_state_probability,"expected_cut":-q.expected_energy,"expected_approx_ratio":(-q.expected_energy)/opt_cut,"total_shots":q.total_shots,"circuit_evaluations":q.circuit_evaluations,"walltime_s":q_t,"success":q.success,
            })
    summary=[]
    for inst in sorted({r["instance"] for r in rows}):
        for solver in ["BF-DCQO","QAOA-p1"]:
            rr=[r for r in rows if r["instance"]==inst and r["solver"]==solver]
            summary.append({
                "instance":inst,"solver":solver,"backend":rr[0]["backend"],"runs":len(rr),"opt_cut":rr[0]["opt_cut"],
                "mean_best_cut":float(np.mean([r["best_cut"] for r in rr])),"mean_approx_ratio":float(np.mean([r["approx_ratio"] for r in rr])),
                "mean_p_opt":float(np.mean([r["p_opt"] for r in rr])),"mean_expected_approx_ratio":float(np.mean([r["expected_approx_ratio"] for r in rr])),"std_p_opt":float(np.std([r["p_opt"] for r in rr])),
                "mean_expected_cut":float(np.mean([r["expected_cut"] for r in rr])),"mean_total_shots":float(np.mean([r["total_shots"] for r in rr])),
                "mean_circuit_evaluations":float(np.mean([r["circuit_evaluations"] for r in rr])),"success_rate":float(np.mean([r["success"] for r in rr])),
                "mean_walltime_s":float(np.mean([r["walltime_s"] for r in rr])),
            })
    result={"qaoa_backend":qaoa_backend,"pyqpanda3_available":pyqpanda3_backend.available(),"budget":{"bf":"6 iterations x 1024 shots = 6144","qaoa":"20 objective evals x 256 + 1024 final shots = 6144 (actual eval count reported)"},"rows":rows,"summary":summary}
    if output_dir:
        out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
        (out/"benchmark_raw.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
        with (out/"benchmark_raw.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        with (out/"benchmark_summary.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    return result
