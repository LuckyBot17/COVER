#!/usr/bin/env python
"""COVER decision-layer experiment on R-Bench (doc §6).

Split-conformal per §6.2: calib images split 50/50 (seed 2026) into
standardization-fit (b_j/g_j) and conformal-calibration (q_{1-delta}). Test =
eval (image-disjoint from both calib halves).

Part A  verification axis, candidate set C_K = {r_gen}:
    Gamma_delta = {r_gen} if A(r_gen) <= q else empty
    ACCEPT iff Gamma = {r_gen}; ABSTAIN otherwise.
    Coverage guarantee (27): P(ACCEPT | claim true) >= 1-delta.
    Rejection power / risk: P(ACCEPT | claim false).
    Baseline: same conformal procedure on raw confidence (-a_orig).

Part B  two-candidate set C_K = {r_gen, r^-1}:
    candidate scores use identity-only measurements (3 ops): A(r_gen) from
    d_id:*, A(r^-1) from d_inv:* (the inverse claim's own identity evidence).
    Decision per §6.3. REPAIR output-flip measured via a_inv_orig (model's
    Yes/No log-odds on the repaired question).

Writes decision_rbench.json + decision_rbench_report.txt.
"""
import csv
import json
import math
import random

import numpy as np

BASE = "/root/autodl-tmp/cover_min_exp"
NONSPATIAL = {"wearing", "holding", "riding", "carrying", "watching", "playing",
              "eating", "drinking", "reading", "using", "interacting with",
              "looking at", "swimming", "driving"}
OPS6 = [("id", "s"), ("id", "o"), ("id", "so"), ("inv", "s"), ("inv", "o"), ("inv", "so")]
OPS3 = [("id", "s"), ("id", "o"), ("id", "so")]
GMIN = 0.01
SEED = 2026


def gnum(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError):
        return float("nan")


def load():
    rows = list(csv.DictReader(open(f"{BASE}/scored_rbench_full.csv")))
    for r in rows:
        r["label"] = int(float(r["label"]))
        r["a_orig"] = gnum(r, "a_orig")
        for v, k in OPS6:
            r[f"d_{v}:{k}"] = gnum(r, f"d_{v}:{k}")
    ai = {}
    try:
        for r in csv.DictReader(open(f"{BASE}/a_inv_orig.csv")):
            ai[(r["image_id"], r["s"], r["r"], r["o"])] = float(r["a_inv_orig"])
    except FileNotFoundError:
        pass
    for r in rows:
        r["a_inv_orig"] = ai.get((r["image_id"], r["s"], r["r"], r["o"]), float("nan"))
    return rows


def fam(r, scope):
    if scope == "all":
        return True
    return r["family"] == scope


def fit_std(rows_std):
    bj, gj = {}, {}
    for v, k in OPS6:
        key = f"d_{v}:{k}"
        df = [r[key] for r in rows_std if r["label"] == 0 and not math.isnan(r[key])]
        dt = [r[key] for r in rows_std if r["label"] == 1 and not math.isnan(r[key])]
        if not df or not dt:
            continue
        b = float(np.median(df))
        g = float(np.median(dt)) - b
        if g > GMIN:
            bj[key] = b
            gj[key] = g
    return bj, gj


def ascore(r, bj, gj, op_list, lam):
    xs = []
    for v, k in op_list:
        key = f"d_{v}:{k}"
        if key in gj and not math.isnan(r[key]):
            xs.append((r[key] - bj[key]) / (gj[key] + 1e-6))
    if not xs:
        return None
    mu = float(np.mean(xs))
    J = float(np.sum((np.array(xs) - mu) ** 2) / max(1, len(xs) - 1))
    return {"A": -mu + lam * J, "mu": mu, "J": J, "m": len(xs)}


def quantile(scores, delta):
    n = len(scores)
    if n == 0:
        return float("nan")
    s = sorted(scores)
    k = int(math.ceil((n + 1) * (1 - delta)))
    k = max(1, min(k, n))
    return s[k - 1]


def boot_ci(true_rows, false_rows, acc_fn, q, keyfn, n=2000, seed=7):
    """Image-clustered bootstrap 95% CI for coverage and false-ACCEPT risk."""
    rng = np.random.default_rng(seed)
    true_imgs = list({keyfn(r) for r in true_rows})
    false_imgs = list({keyfn(r) for r in false_rows})
    covs, risks = [], []
    for _ in range(n):
        ti = set(rng.choice(true_imgs, len(true_imgs), replace=True))
        fi = set(rng.choice(false_imgs, len(false_imgs), replace=True))
        ct = [r for r in true_rows if keyfn(r) in ti]
        cf = [r for r in false_rows if keyfn(r) in fi]
        if ct:
            covs.append(np.mean([1.0 if acc_fn(r) <= q else 0.0 for r in ct]))
        if cf:
            risks.append(np.mean([1.0 if acc_fn(r) <= q else 0.0 for r in cf]))
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if a else (float("nan"), float("nan"))
    return ci(covs), ci(risks)


def partA(rows, bj, gj, conf_true, eval_true, eval_false, lam, delta, score_name, score_fn, keyfn):
    scores = [score_fn(r) for r in conf_true]
    scores = [s for s in scores if s is not None]
    if not scores:
        return None
    q = quantile(scores, delta)
    n = len(scores)
    acc_true = [r for r in eval_true if score_fn(r) is not None and score_fn(r) <= q]
    acc_false = [r for r in eval_false if score_fn(r) is not None and score_fn(r) <= q]
    cov = len(acc_true) / len(eval_true) if eval_true else float("nan")
    risk = len(acc_false) / len(eval_false) if eval_false else float("nan")
    (cov_lo, cov_hi), (risk_lo, risk_hi) = boot_ci(
        [r for r in eval_true if score_fn(r) is not None],
        [r for r in eval_false if score_fn(r) is not None],
        lambda r: score_fn(r), q, keyfn)
    return {
        "delta": delta, "q": q, "n_conf": n, "score": score_name,
        "n_true": len(eval_true), "n_false": len(eval_false),
        "coverage(ACCEPT|true)": round(cov, 4),
        "coverage_ci": [round(cov_lo, 4), round(cov_hi, 4)],
        "risk(ACCEPT|false)": round(risk, 4),
        "risk_ci": [round(risk_lo, 4), round(risk_hi, 4)],
        "n_accept_true": len(acc_true), "n_accept_false": len(acc_false),
    }


def main():
    rows = load()
    calib = [r for r in rows if r["split"] == "calib"]
    calib_imgs = sorted({r["image_id"] for r in calib})
    rng = random.Random(SEED)
    rng.shuffle(calib_imgs)
    cut = len(calib_imgs) // 2
    std_imgs = set(calib_imgs[:cut])
    conf_imgs = set(calib_imgs[cut:])
    std = [r for r in calib if r["image_id"] in std_imgs]
    conf = [r for r in calib if r["image_id"] in conf_imgs]

    report = []
    out = {"setup": {"std_imgs": len(std_imgs), "conf_imgs": len(conf_imgs),
                     "std_claims": len(std), "conf_claims": len(conf)}}
    print(f"=== setup: std {len(std_imgs)} imgs/{len(std)} claims, "
          f"conf {len(conf_imgs)} imgs/{len(conf)} claims ===", flush=True)

    for scope in ("non-spatial", "all", "spatial"):
        std_sc = [r for r in std if fam(r, scope)]
        conf_sc = [r for r in conf if fam(r, scope)]
        ev = [r for r in rows if r["split"] == "eval" and fam(r, scope)]
        ev_true = [r for r in ev if r["label"] == 1]
        ev_false = [r for r in ev if r["label"] == 0]
        conf_true = [r for r in conf_sc if r["label"] == 1]
        conf_false = [r for r in conf_sc if r["label"] == 0]
        bj, gj = fit_std(std_sc)
        print(f"\n=== scope={scope}: std n={len(std_sc)} (true {sum(1 for r in std_sc if r['label']==1)}), "
              f"conf n={len(conf_sc)} (true {len(conf_true)}, false {len(conf_false)}), "
              f"eval n={len(ev)} (true {len(ev_true)}, false {len(ev_false)}) ===", flush=True)
        print(f"kept ops: {sorted(gj.keys())}", flush=True)
        if not gj:
            print("  no operators kept; skip", flush=True)
            continue

        scope_res = {"scope": scope,
                     "n_eval_true": len(ev_true), "n_eval_false": len(ev_false),
                     "n_conf_true": len(conf_true),
                     "operators_kept": sorted(gj.keys()),
                     "calib_b": {k: round(v, 3) for k, v in bj.items()},
                     "calib_g": {k: round(v, 3) for k, v in gj.items()}}

        keyfn = lambda r: r["image_id"]
        for lam in (0.0, 1.0):
            A = lambda r: (ascore(r, bj, gj, OPS6, lam) or {}).get("A")
            rowsA = []
            for delta in (0.05, 0.10, 0.20):
                res = partA(rows, bj, gj, conf_true, ev_true, ev_false, lam, delta,
                            f"mu6_A_lam{lam:g}", A, keyfn)
                rowsA.append(res)
            # raw-confidence baseline (score = -a_orig), lam irrelevant
            base = []
            for delta in (0.05, 0.10, 0.20):
                res = partA(rows, bj, gj, conf_true, ev_true, ev_false, lam, delta,
                            "raw_conf(-a_orig)", lambda r: -r["a_orig"], keyfn)
                base.append(res)
            scope_res[f"lam{lam:g}"] = {"COVER_A": rowsA, "baseline_raw": base}

            print(f"  --- lam={lam} ---", flush=True)
            print(f"  {'δ':>5} {'q':>7} {'n_conf':>6} | {'覆盖(ACCEPT|真)':>14} {'[CI]':>16} | "
                  f"{'风险(ACCEPT|假)':>14} {'[CI]':>16}", flush=True)
            for rA, rb in zip(rowsA, base):
                print(f"  COVER {rA['delta']:>5.2f} {rA['q']:7.3f} {rA['n_conf']:6d} | "
                      f"{rA['coverage(ACCEPT|true)']:14.3f} {rA['coverage_ci'][0]:.3f}-{rA['coverage_ci'][1]:.3f} | "
                      f"{rA['risk(ACCEPT|false)']:14.3f} {rA['risk_ci'][0]:.3f}-{rA['risk_ci'][1]:.3f}", flush=True)
            for rb in base:
                print(f"  RAW  {rb['delta']:>5.2f} {rb['q']:7.3f} {rb['n_conf']:6d} | "
                      f"{rb['coverage(ACCEPT|true)']:14.3f} {rb['coverage_ci'][0]:.3f}-{rb['coverage_ci'][1]:.3f} | "
                      f"{rb['risk(ACCEPT|false)']:14.3f} {rb['risk_ci'][0]:.3f}-{rb['risk_ci'][1]:.3f}", flush=True)

        # ---- Part B: two-candidate set {r_gen, r^-1}, identity-only scores ----
        lamB = 1.0
        A3 = lambda r: (ascore(r, bj, gj, OPS3, lamB) or {}).get("A")
        conf_true_A3 = [A3(r) for r in conf_true]
        conf_true_A3 = [a for a in conf_true_A3 if a is not None]
        partB = {}
        if len(conf_true_A3) >= 20:
            for delta in (0.05, 0.10, 0.20):
                q3 = quantile(conf_true_A3, delta)
                dec = {"ACCEPT": 0, "REPAIR": 0, "ABSTAIN": 0}
                dec_t, dec_f = dict(dec), dict(dec)
                repair_flip = {"t": 0, "f": 0, "t_n": 0, "f_n": 0}
                for r in ev:
                    ag = A3(r)
                    ai = (ascore(r, bj, gj, [("inv", "s"), ("inv", "o"), ("inv", "so")], lamB) or {}).get("A")
                    if ag is None:
                        continue
                    if ai is None:  # no inverse candidate (swimming/driving): single-candidate set
                        d = "ACCEPT" if ag <= q3 else "ABSTAIN"
                    elif ag <= q3 and ai <= q3:
                        d = "ABSTAIN"
                    elif ag <= q3:
                        d = "ACCEPT"
                    elif ai <= q3:
                        d = "REPAIR"
                    else:
                        d = "ABSTAIN"
                    dec[d] += 1
                    if r["label"] == 1:
                        dec_t[d] += 1
                    else:
                        dec_f[d] += 1
                    if d == "REPAIR":
                        key = "t" if r["label"] == 1 else "f"
                        repair_flip[f"{key}_n"] += 1
                        if not math.isnan(r["a_inv_orig"]) and r["a_inv_orig"] > 0:
                            repair_flip[key] += 1
                tot = sum(dec.values()) or 1
                tot_t = sum(dec_t.values())
                tot_f = sum(dec_f.values())
                partB[f"delta{delta:g}"] = {
                    "q": round(q3, 4), "n_conf_true": len(conf_true_A3),
                    "ACCEPT": dec["ACCEPT"], "REPAIR": dec["REPAIR"], "ABSTAIN": dec["ABSTAIN"],
                    "ACCEPT_rate": round(dec["ACCEPT"] / tot, 4),
                    "REPAIR_rate": round(dec["REPAIR"] / tot, 4),
                    "ABSTAIN_rate": round(dec["ABSTAIN"] / tot, 4),
                    "on_true": {"ACCEPT": dec_t["ACCEPT"], "REPAIR": dec_t["REPAIR"],
                                "ABSTAIN": dec_t["ABSTAIN"], "n": tot_t},
                    "on_false": {"ACCEPT": dec_f["ACCEPT"], "REPAIR": dec_f["REPAIR"],
                                 "ABSTAIN": dec_f["ABSTAIN"], "n": tot_f},
                    "repair_flip": {"true": repair_flip["t"], "true_n": repair_flip["t_n"],
                                    "false": repair_flip["f"], "false_n": repair_flip["f_n"]},
                }
                print(f"  [partB δ={delta:.2f}] q={q3:.3f} ACCEPT={dec['ACCEPT']} "
                      f"REPAIR={dec['REPAIR']} ABSTAIN={dec['ABSTAIN']} "
                      f"(true: {dec_t}, false: {dec_f}) "
                      f"repair-flip true {repair_flip['t']}/{repair_flip['t_n']} "
                      f"false {repair_flip['f']}/{repair_flip['f_n']}", flush=True)
        scope_res["partB"] = partB
        out[scope] = scope_res

    with open(f"{BASE}/decision_rbench.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n[done] decision_rbench.json written", flush=True)


if __name__ == "__main__":
    main()
