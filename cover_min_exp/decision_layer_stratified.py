#!/usr/bin/env python
"""Stratified conformal decision layer (Part C).

The global decision layer (Part A) uses one threshold over all confidence
levels and has weak rejection. The gate test showed mu separates true/false
WITHIN raw-confidence strata (especially the high stratum). Part C
operationalizes that: conformal quantiles are calibrated per stratum (stratum
boundaries fixed by conf-half a_orig terciles), applied to eval claims in the
same stratum. Does within-stratum COVER rejection beat within-stratum raw?
"""
import csv
import json
import math

import numpy as np

import decision_layer_rbench as dl

BASE = dl.BASE
rows = dl.load()
calib = [r for r in rows if r["split"] == "calib"]
calib_imgs = sorted({r["image_id"] for r in calib})
rng = np.random.default_rng(2026)
rng.shuffle(np.array(calib_imgs))
cut = len(calib_imgs) // 2
conf_imgs = set(calib_imgs[cut:])
conf = [r for r in calib if r["image_id"] in conf_imgs]
keyfn = lambda r: r["image_id"]


def terciles(a):
    return np.percentile(a, [100 / 3, 200 / 3])


def stratified(scope, lam):
    std_sc = [r for r in calib if dl.fam(r, scope) and r["image_id"] not in conf_imgs]
    conf_sc = [r for r in conf if dl.fam(r, scope)]
    ev = [r for r in rows if r["split"] == "eval" and dl.fam(r, scope)]
    bj, gj = dl.fit_std(std_sc)
    if not gj:
        return None
    A = lambda r: (dl.ascore(r, bj, gj, dl.OPS6, lam) or {}).get("A")
    # stratum boundaries from conf-half a_orig (no test leakage)
    conf_a = np.array([r["a_orig"] for r in conf_sc])
    t1, t2 = terciles(conf_a)
    bounds = [(-np.inf, t1, "低"), (t1, t2, "中"), (t2, np.inf, "高")]
    rows_out = []
    for lo, hi, name in bounds:
        conf_s = [r for r in conf_sc if lo < r["a_orig"] <= hi]
        conf_true = [r for r in conf_s if r["label"] == 1]
        conf_false = [r for r in conf_s if r["label"] == 0]
        ev_s = [r for r in ev if lo < r["a_orig"] <= hi]
        ev_true = [r for r in ev_s if r["label"] == 1]
        ev_false = [r for r in ev_s if r["label"] == 0]
        for sname, sfn in (("COVER_A", A), ("raw_conf", lambda r: -r["a_orig"])):
            scores = [sfn(r) for r in conf_true]
            scores = [s for s in scores if s is not None]
            if not scores or len(ev_true) < 5:
                rows_out.append({"stratum": name, "score": sname, "n_conf_true": len(scores),
                                 "n_true": len(ev_true), "n_false": len(ev_false),
                                 "coverage": None, "risk": None, "q": None})
                continue
            q = dl.quantile(scores, 0.10)
            acc_t = sum(1 for r in ev_true if sfn(r) is not None and sfn(r) <= q)
            acc_f = sum(1 for r in ev_false if sfn(r) is not None and sfn(r) <= q)
            rows_out.append({
                "stratum": name, "score": sname, "n_conf_true": len(scores),
                "n_true": len(ev_true), "n_false": len(ev_false),
                "coverage": round(acc_t / len(ev_true), 4),
                "risk": round(acc_f / len(ev_false), 4) if ev_false else None,
                "q": round(q, 4),
            })
    return rows_out


out = {}
print("=== Part C: stratified conformal decision layer (delta=0.10) ===", flush=True)
for scope in ("non-spatial", "all", "spatial"):
    for lam in (0.0, 1.0):
        res = stratified(scope, lam)
        out[f"{scope}_lam{lam:g}"] = res
        if not res:
            continue
        print(f"\n{scope} lam={lam}:", flush=True)
        for r in res:
            if r["coverage"] is None:
                print(f"  {r['stratum']} {r['score']:9s} n_conf={r['n_conf_true']:4d} "
                      f"n={r['n_true']}+{r['n_false']} (skip)", flush=True)
            else:
                print(f"  {r['stratum']} {r['score']:9s} n_conf={r['n_conf_true']:4d} "
                      f"n={r['n_true']}+{r['n_false']} q={r['q']:7.3f} "
                      f"coverage={r['coverage']:.3f} risk={r['risk']:.3f}", flush=True)

json.dump(out, open(f"{BASE}/decision_rbench_stratified.json", "w"), indent=2)
print("\n[done] decision_rbench_stratified.json", flush=True)
