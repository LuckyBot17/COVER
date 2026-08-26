#!/usr/bin/env python
"""Sensitivity: decision layer with FULL-calib standardization (gate-consistent).

The clean split-conformal run fit b_j/g_j on half the calib images (std-fit),
which dropped the inversion operators in the non-spatial family (g <= 0.01 on
~29 false claims). This run uses ALL calib claims to fit b_j/g_j (as the gate
analysis did) while still computing the conformal quantile on the conf half.
Caveat: the conf half then overlaps the standardization fit, so the coverage
guarantee is mildly inflated — read the rejection numbers as sensitivity only.
"""
import csv
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
std_imgs = set(calib_imgs[:cut])
conf_imgs = set(calib_imgs[cut:])
conf = [r for r in calib if r["image_id"] in conf_imgs]
keyfn = lambda r: r["image_id"]

out = {}
print("=== sensitivity: full-calib b_j/g_j, conf-half quantile ===", flush=True)
for scope in ("non-spatial", "all", "spatial"):
    full = [r for r in calib if dl.fam(r, scope)]
    conf_sc = [r for r in conf if dl.fam(r, scope)]
    ev = [r for r in rows if r["split"] == "eval" and dl.fam(r, scope)]
    ev_true = [r for r in ev if r["label"] == 1]
    ev_false = [r for r in ev if r["label"] == 0]
    conf_true = [r for r in conf_sc if r["label"] == 1]
    bj, gj = dl.fit_std(full)
    print(f"\n{scope}: full-calib n={len(full)}, kept ops={sorted(gj.keys())}", flush=True)
    out[scope] = {"kept_ops": sorted(gj.keys())}
    for lam in (0.0, 1.0):
        A = lambda r: (dl.ascore(r, bj, gj, dl.OPS6, lam) or {}).get("A")
        out[scope][f"lam{lam:g}"] = {}
        print(f"  lam={lam}: ", flush=True)
        for delta in (0.05, 0.10, 0.20):
            res = dl.partA(rows, bj, gj, conf_true, ev_true, ev_false, lam, delta,
                           "A_fullcalib", A, keyfn)
            out[scope][f"lam{lam:g}"][f"d{delta:g}"] = res
            print(f"    d={delta:.2f} coverage={res['coverage(ACCEPT|true)']:.3f} "
                  f"risk={res['risk(ACCEPT|false)']:.3f} "
                  f"(true {res['n_accept_true']}/{res['n_true']}, "
                  f"false {res['n_accept_false']}/{res['n_false']})", flush=True)

import json
json.dump(out, open(f"{BASE}/decision_rbench_fullcalib_sens.json", "w"), indent=2)
print("\n[done] decision_rbench_fullcalib_sens.json", flush=True)
