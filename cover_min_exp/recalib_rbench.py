#!/usr/bin/env python
"""Cross-image recalibration of the COVER score (fix the risk overshoot).

Symptom: the risk-controlled decision layer assumed per-claim exchangeability
(q = alpha-quantile of calibration FALSE scores), but realized risk on eval
overshot nominal alpha by ~2x for COVER A (a 6-op composite). False claims
cluster by image, and the calib/eval partitions differ, so the per-claim
guarantee under-samples the cluster variance.

Variants:
  V0  baseline: std-half b/g, conf-half q (from risk_control_rbench.py)
  V1  full-calib b/g, full-calib q: standardization and calibration on the
      same, largest distribution.
  V2  full-calib b/g + q by IMAGE-CLUSTERED BOOTSTRAP: choose the largest q
      whose bootstrap upper-95% estimate of P(ACCEPT|false) is <= alpha.
      Directly absorbs intra-image correlation and sampling noise.

Report realized risk + coverage on eval (image-disjoint from everything).
Writes recalib_rbench.json.
"""
import json
import math
import os
import random

import numpy as np

import decision_layer_rbench as dl

dl.BASE = os.path.dirname(os.path.abspath(__file__))
BASE = dl.BASE

ALPHAS = (0.05, 0.10, 0.20)
NB = 3000


def quantile_risk(scores, alpha):
    n = len(scores)
    if n == 0:
        return float("nan")
    s = sorted(scores)
    k = max(1, min(n, int(math.floor(alpha * (n + 1)))))
    return s[k - 1]


def make_A(bj, gj, lam=0.0):
    return lambda r: (dl.ascore(r, bj, gj, dl.OPS6, lam) or {}).get("A")


def eval_risk_cov(ev_true, ev_false, sfn, q):
    if not math.isfinite(q):
        return 0.0, 0.0, 0, 0
    at = [r for r in ev_true if sfn(r) is not None and sfn(r) <= q]
    af = [r for r in ev_false if sfn(r) is not None and sfn(r) <= q]
    cov = len(at) / len(ev_true) if ev_true else float("nan")
    risk = len(af) / len(ev_false) if ev_false else float("nan")
    return cov, risk, len(at), len(af)


def bootstrap_q(false_rows, sfn, keyfn, alpha, seed=2026):
    """Largest q with image-clustered bootstrap upper-95% risk <= alpha."""
    imgs = list({keyfn(r) for r in false_rows if sfn(r) is not None})
    by_img = {im: sorted([sfn(r) for r in false_rows
                          if sfn(r) is not None and keyfn(r) == im]) for im in imgs}
    grid = sorted({s for v in by_img.values() for s in v})
    rng = np.random.default_rng(seed)
    chosen = -np.inf
    # evaluate risk at each candidate q on the full calib set
    for q in grid:
        # cluster bootstrap: risk estimate = claim fraction <= q over resampled images
        b = np.empty(NB)
        for i in range(NB):
            sel = rng.choice(imgs, len(imgs), replace=True)
            cnt, tot = 0, 0
            for im in sel:
                vs = by_img[im]
                cnt += sum(1 for s in vs if s <= q)
                tot += len(vs)
            b[i] = cnt / tot if tot else 0.0
        ub = float(np.percentile(b, 95))
        if ub <= alpha:
            chosen = q
        else:
            break  # monotone: once rejected, larger q also rejected
    return chosen


def run(rows, calib, std, conf, scope, keyfn):
    std_sc = [r for r in std if dl.fam(r, scope)]
    cal_sc = [r for r in calib if dl.fam(r, scope)]
    conf_sc = [r for r in conf if dl.fam(r, scope)]
    ev = [r for r in rows if r["split"] == "eval" and dl.fam(r, scope)]
    ev_true = [r for r in ev if r["label"] == 1]
    ev_false = [r for r in ev if r["label"] == 0]
    out = {"scope": scope, "n_eval_true": len(ev_true), "n_eval_false": len(ev_false)}

    for sname in ("COVER_A(lam0)", "raw_conf"):
        # score function differs by standardization
        if sname == "COVER_A(lam0)":
            bj0, gj0 = dl.fit_std(std_sc)
            bj1, gj1 = dl.fit_std(cal_sc)
            s0, s1 = make_A(bj0, gj0), make_A(bj1, gj1)
        else:
            s0 = s1 = lambda r: -r["a_orig"]
        conf_false0 = [r for r in conf_sc if r["label"] == 0]
        cal_false = [r for r in cal_sc if r["label"] == 0]
        f0 = [s0(r) for r in conf_false0 if s0(r) is not None]
        f1 = [s1(r) for r in cal_false if s1(r) is not None]
        rec = {"n_conf_false_v0": len(f0), "n_cal_false_v1": len(f1)}
        for alpha in ALPHAS:
            # V0
            q0 = quantile_risk(f0, alpha)
            c0, r0, at0, af0 = eval_risk_cov(ev_true, ev_false, s0, q0)
            # V1
            q1 = quantile_risk(f1, alpha)
            c1, r1, at1, af1 = eval_risk_cov(ev_true, ev_false, s1, q1)
            # V2
            q2 = bootstrap_q(cal_false, s1, keyfn, alpha)
            c2, r2, at2, af2 = eval_risk_cov(ev_true, ev_false, s1, q2)
            rec[f"a{alpha:g}"] = {
                "V0": {"q": round(q0, 3), "coverage": round(c0, 4), "risk": round(r0, 4)},
                "V1": {"q": round(q1, 3), "coverage": round(c1, 4), "risk": round(r1, 4)},
                "V2": {"q": round(q2, 3) if math.isfinite(q2) else None,
                       "coverage": round(c2, 4), "risk": round(r2, 4)},
            }
        out[sname] = rec
        print(f"\n  {scope}/{sname}", flush=True)
        for alpha in ALPHAS:
            d = rec[f"a{alpha:g}"]
            print(f"    α={alpha:.2f}  V0 q={d['V0']['q']:7.3f} cov={d['V0']['coverage']:.3f} "
                  f"risk={d['V0']['risk']:.3f} | V1 q={d['V1']['q']:7.3f} "
                  f"cov={d['V1']['coverage']:.3f} risk={d['V1']['risk']:.3f} | "
                  f"V2 q={d['V2']['q']} cov={d['V2']['coverage']:.3f} risk={d['V2']['risk']:.3f}",
                  flush=True)
    return out


def main():
    rows = dl.load()
    calib = [r for r in rows if r["split"] == "calib"]
    calib_imgs = sorted({r["image_id"] for r in calib})
    rng = random.Random(dl.SEED)
    rng.shuffle(calib_imgs)
    cut = len(calib_imgs) // 2
    std_imgs, conf_imgs = set(calib_imgs[:cut]), set(calib_imgs[cut:])
    std = [r for r in calib if r["image_id"] in std_imgs]
    conf = [r for r in calib if r["image_id"] in conf_imgs]
    keyfn = lambda r: r["image_id"]
    out = {"note": "V0 baseline | V1 full-calib bg+q | V2 cluster-bootstrap q",
           "NB": NB}
    print("=== cross-image recalibration (non-spatial, all) ===", flush=True)
    for scope in ("non-spatial", "all"):
        out[scope] = run(rows, calib, std, conf, scope, keyfn)
    with open(f"{BASE}/recalib_rbench.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n[done] recalib_rbench.json", flush=True)


if __name__ == "__main__":
    main()
