"""Tools — each one single responsibility, mathematically correct, CPU-only.
MPS is reserved for Gemma 4 12B (Ollama). No tool touches MPS.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List

DEVICE = "cpu"  # all tools on CPU — MPS is Gemma's (unified memory, avoid contention)

# ============================================================
# TOOL 1: SOLVER — PuLP/CBC (LP/MIP)  CPU only
# ============================================================
def solve_optimization(problem: Dict[str, Any]) -> Dict[str, Any]:
    """Pure deterministic solver. No simulation, no LLM.
    problem = {
      "objective": {"coeffs": {"A":5,"B":4,"C":3}, "sense":"max"},
      "constraints": [
        {"coeffs":{"A":1,"B":1,"C":1}, "sense":"<=", "rhs":10000},
        {"coeffs":{"A":1}, "sense":"<=", "rhs":6000},
      ],
      "integer": []  # optional
    }
    Returns: {status, feasible, objective, allocation, solver}
    """
    import pulp
    coeffs = problem.get("objective", {}).get("coeffs", {})
    if not coeffs:
        return {"status": "Invalid", "feasible": False, "objective": None, "allocation": {}, "solver": "PULP_CBC", "error": "empty objective"}
    sense = problem.get("objective", {}).get("sense", "max")
    prob = pulp.LpProblem("model", pulp.LpMaximize if sense == "max" else pulp.LpMinimize)
    vars_ = {k: pulp.LpVariable(k, lowBound=0, cat="Integer" if k in problem.get("integer", []) else "Continuous") for k in coeffs}
    prob += pulp.lpSum(coeffs[k] * vars_[k] for k in coeffs)
    for c in problem.get("constraints", []):
        expr = pulp.lpSum(c["coeffs"][k] * vars_[k] for k in c["coeffs"] if k in vars_)
        if c["sense"] == "<=":
            prob += expr <= c["rhs"]
        elif c["sense"] == ">=":
            prob += expr >= c["rhs"]
        elif c["sense"] == "==":
            prob += expr == c["rhs"]
        else:
            return {"status": "Invalid", "feasible": False, "objective": None, "allocation": {}, "error": f"bad sense {c['sense']}"}
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[prob.status]
    return {
        "status": status,
        "feasible": status == "Optimal",
        "objective": float(pulp.value(prob.objective)) if status == "Optimal" else None,
        "allocation": {k: float(vars_[k].value()) if vars_[k].value() is not None else 0.0 for k in vars_},
        "solver": "PULP_CBC",
        "device": DEVICE,
    }

# ============================================================
# TOOL 2: VALIDATION — checks formulation, no solving
# ============================================================
def validate_formulation(problem: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    if not problem.get("objective", {}).get("coeffs"):
        errors.append("Missing objective.coeffs")
    if problem.get("objective", {}).get("sense") not in ("max", "min"):
        errors.append("objective.sense must be 'max' or 'min'")
    for i, c in enumerate(problem.get("constraints", [])):
        if not c.get("coeffs"):
            errors.append(f"Constraint {i} missing coeffs")
        if c.get("sense") not in ("<=", ">=", "=="):
            errors.append(f"Constraint {i} invalid sense {c.get('sense')}")
        if "rhs" not in c:
            errors.append(f"Constraint {i} missing rhs")
    return {"valid": len(errors) == 0, "errors": errors}

# ============================================================
# TOOL 3: MONTE CARLO — service level validator, CPU vectorized
# Does NOT solve — only evaluates a given allocation against stochastic demand.
# ============================================================
def run_monte_carlo(allocation: Dict[str, float], demand_params: Dict[str, Dict[str, float]], n: int = 10000, seed: int = 0) -> Dict[str, Any]:
    """Vectorized numpy on CPU. demand_params = {"A":{"mu":4200,"sigma":700}, ...}"""
    if not allocation:
        return {"n": n, "service_level": 0.0, "expected_stockout_cost": 0.0, "p95_cost": 0.0, "device": DEVICE, "error": "empty allocation"}
    whs = list(allocation.keys())
    alloc = np.array([allocation[w] for w in whs], dtype=float)
    mus = np.array([demand_params.get(w, {"mu": 1000, "sigma": 200})["mu"] for w in whs], dtype=float)
    sigmas = np.array([demand_params.get(w, {"mu": 1000, "sigma": 200})["sigma"] for w in whs], dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.normal(mus, sigmas, size=(n, len(whs)))
    samples = np.maximum(samples, 0)
    shorts = np.maximum(samples - alloc, 0)  # (n, whs)
    total_short = shorts.sum(axis=1)  # (n,)
    stockout_cost = total_short * 10  # penalty per unit short
    service_level = float((total_short == 0).mean())
    return {
        "n": n,
        "service_level": service_level,
        "expected_stockout_cost": float(stockout_cost.mean()),
        "p95_cost": float(np.quantile(stockout_cost, 0.95)),
        "mean_shortage": float(total_short.mean()),
        "device": DEVICE,
    }

# ============================================================
# TOOL 4: SENSITIVITY — re-evaluates given allocation under shocked demand
# (does not re-solve; re-solving is challenger→reformulate loop in graph)
# ============================================================
def sensitivity_analysis(allocation: Dict[str, float], demand_params: Dict[str, Any]) -> Dict[str, Any]:
    """Pure evaluator — no solver inside. Caller re-solves if needed."""
    results = {}
    for label, factor in [("demand_+10%", 1.1), ("demand_+20%", 1.2), ("demand_+30%", 1.3)]:
        shocked = {k: {"mu": v["mu"] * factor, "sigma": v["sigma"] * factor} for k, v in demand_params.items()}
        sim = run_monte_carlo(allocation, shocked, n=8000, seed=hash(label) % 10000)
        results[label] = {"service_level": sim["service_level"], "expected_cost": sim["expected_stockout_cost"], "mean_shortage": sim["mean_shortage"]}
    return results

# ============================================================
# TOOL 5: CVXPY — alternative convex solver (CPU, clarabel)
# ============================================================
def solve_cvxpy(c: List[float], A: List[List[float]], b: List[float]) -> Dict[str, Any]:
    import cvxpy as cp
    n = len(c)
    x = cp.Variable(n)
    objective = cp.Maximize(np.array(c) @ x)
    constraints = []
    if A and b:
        constraints.append(np.array(A) @ x <= np.array(b))
    constraints.append(x >= 0)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, verbose=False)
    return {"status": prob.status, "feasible": prob.status == "optimal", "objective": float(prob.value) if prob.value is not None else None, "allocation": x.value.tolist() if x.value is not None else [], "device": DEVICE}
