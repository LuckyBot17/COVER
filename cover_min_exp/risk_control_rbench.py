#!/usr/bin/env python
"""Risk-controlled decision layer on R-Bench (fix of the operating point).

PROBLEM FIXED: the old decision layer calibrated the acceptance threshold on
TRUE-claim scores (split-conformal coverage guarantee, q = (1-delta)-quantile
of true scores). False claims rank below the true distribution on average, so
that q admitted essentially all of them: P(ACCEPT|false) ~ 1 even though the
gate AUROC was healthy (non-spatial high-stratum risk hit 1.000). The rule
answered "cover the true claims", not "reject the false ones".

FIX (Neyman-Pearson / conformal risk control): calibrate q on FALSE-claim
scores. ACCEPT iff score <= q, q = alpha-th empirical quantile of the
calibration-half FALSE scores, finite-sample conformal k = max(1,
floor(alpha*(n+1))), so E[P(ACCEPT|false)] <= alpha by exchangeability. The
experiment asks the question that decides the fix: at each controlled risk
alpha, how much true coverage survives, COVER A(lambda=0) vs raw confidence,
globally and per confidence stratum.

No new model inference: all scores already exist (scored_rbench_full.csv diffs
+ a_orig; a_inv_orig.csv for the REPAIR flip).

Parts:
  A'  global risk-coverage curves, alpha in {.01,.05,.10,.20,.30,.50}
  C'  per-confidence-stratum risk control (alpha=.10), non-spatial + all
  B'  two-candidate ACCEPT/REPAIR/ABSTAIN under alpha-controlled thresholds

Writes risk_control_rbench.json + prints a text report.
"""
import csv
import json
import math
import os
import random

import numpy as np

import decision_layer_rbench as dl

# location-independent: BASE = directory containing the CSVs
dl.BASE = os.path.dirname(os.path.abspath(__file__))
BASE = dl.BASE

ALPHAS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.50)
INV_OPS = [("inv", "s"), ("inv", "o"), ("inv", "so")]


def quantile_risk(scores, alpha):
    """q s.t. P(new false score <= q) <= alpha (finite-sample conformal).
    k = max(1, floor(alpha*(n+1)))."""
    n = len(scores)
    if n == 0:
        return float("nan")
    s = sorted(scores)
    k = max(1, min(n, int(math.floor(alpha * (n + 1)))))
    return s[k - 1]


def make_scores(std_sc):
    bj, gj = dl.fit_std(std_sc)
    A0 = lambda r: (dl.ascore(r, bj, gj, dl.OPS6, 0.0) or {}).get("A")
    RAW = lambda r: -r["a_orig"]
    return {"COVER_A(lam0)": A0, "raw_conf": RAW}, gj


def partA(rows, std, conf, scope):
    std_sc = [r for r in std if dl.fam(r, scope)]
    conf_sc = [r for r in conf if dl.fam(r, scope)]
    ev = [r for r in rows if r["split"] == "eval" and dl.fam(r, scope)]
    ev_true = [r for r in ev if r["label"] == 1]
    ev_false = [r for r in ev if r["label"] == 0]
    conf_false = [r for r in conf_sc if r["label"] == 0]
    scores, gj = make_scores(std_sc)
    if not gj:
        return None
    res = {"scope": scope, "n_conf_false": len(conf_false),
           "n_eval_true": len(ev_true), "n_eval_false": len(ev_false),
           "operators_kept": sorted(gj.keys())}
    print(f"\n{scope}: conf-half FALSE n={len(conf_false)}, "
          f"eval T{len(ev_true)}/F{len(ev_false)}", flush=True)
    for sname, sfn in scores.items():
        fsc = [sfn(r) for r in conf_false if sfn(r) is not None]
        curves = []
        print(f"  {sname:16s} " + "  ".join(f"{a:>4} {'cov':>6} {'risk':>6} {'covCI':>13} {'riskCI':>13}" for a in ALPHAS))
        for alpha in ALPHAS:
            q = quantile_risk(fsc, alpha)
            act = [r for r in ev_true if sfn(r) is not None and sfn(r) <= q]
            acf = [r for r in ev_false if sfn(r) is not None and sfn(r) <= q]
            cov = len(act) / len(ev_true) if ev_true else float("nan")
            risk = len(acf) / len(ev_false) if ev_false else float("nan")
            (clo, chi), (rlo, rhi) = dl.boot_ci(
                [r for r in ev_true if sfn(r) is not None],
                [r for r in ev_false if sfn(r) is not None],
                lambda r: sfn(r), q, lambda r: r["image_id"])
            curves.append({"alpha": alpha, "q": round(q, 4),
                           "coverage": round(cov, 4),
                           "coverage_ci": [round(clo, 4), round(chi, 4)],
                           "risk": round(risk, 4),
                           "risk_ci": [round(rlo, 4), round(rhi, 4)],
                           "n_acc_true": len(act), "n_acc_false": len(acf)})
        for c in curves:
            print(f"    α={c['alpha']:.2f} q={c['q']:7.3f} cov={c['coverage']:.3f} "
                  f"risk={c['risk']:.3f} covCI={c['coverage_ci']} riskCI={c['risk_ci']} "
                  f"(T{c['n_acc_true']}/{c['n_acc_false']}F)", flush=True)
        res[sname] = curves
    return res


def partC(rows, std, conf, scope):
    std_sc = [r for r in std if dl.fam(r, scope)]
    conf_sc = [r for r in conf if dl.fam(r, scope)]
    ev = [r for r in rows if r["split"] == "eval" and dl.fam(r, scope)]
    scores, gj = make_scores(std_sc)
    if not gj:
        return None
    # stratum boundaries from conf-half a_orig (per scope, no test leakage)
    t1, t2 = np.percentile([r["a_orig"] for r in conf_sc], [100 / 3, 200 / 3])
    rows_out = []
    print(f"\n{scope} (terciles {t1:.2f}/{t2:.2f})", flush=True)
    for lo, hi, name in ((-np.inf, t1, "低"), (t1, t2, "中"), (t2, np.inf, "高")):
        cf = [r for r in conf_sc if lo < r["a_orig"] <= hi and r["label"] == 0]
        et = [r for r in ev if lo < r["a_orig"] <= hi and r["label"] == 1]
        ef = [r for r in ev if lo < r["a_orig"] <= hi and r["label"] == 0]
        rec = {"stratum": name, "n_conf_false": len(cf),
               "n_true": len(et), "n_false": len(ef)}
        for sname, sfn in scores.items():
            fsc = [sfn(r) for r in cf if sfn(r) is not None]
            if not fsc or len(et) < 3:
                rec[sname] = None
                continue
            q = quantile_risk(fsc, 0.10)
            acc_t = [r for r in et if sfn(r) is not None and sfn(r) <= q]
            acc_f = [r for r in ef if sfn(r) is not None and sfn(r) <= q]
            cov = len(acc_t) / len(et) if et else float("nan")
            risk = (len(acc_f) / len(ef) if ef else float("nan"))
            (clo, chi), (rlo, rhi) = dl.boot_ci(
                [r for r in et if sfn(r) is not None],
                [r for r in ef if sfn(r) is not None],
                lambda r: sfn(r), q, lambda r: r["image_id"])
            rec[sname] = {"q": round(q, 4), "coverage": round(cov, 4),
                          "coverage_ci": [round(clo, 4), round(chi, 4)],
                          "risk": round(risk, 4),
                          "risk_ci": [round(rlo, 4), round(rhi, 4)]}
            print(f"  {name} {sname:16s} n_conf_f={len(fsc):3d} q={q:7.3f} "
                  f"cov={cov:.3f} [{clo:.3f},{chi:.3f}] risk={risk:.3f} "
                  f"(eval T{len(et)}/F{len(ef)})", flush=True)
        rows_out.append(rec)
    return rows_out


def partB(rows, std, conf, scope):
    std_sc = [r for r in std if dl.fam(r, scope)]
    conf_sc = [r for r in conf if dl.fam(r, scope)]
    ev = [r for r in rows if r["split"] == "eval" and dl.fam(r, scope)]
    bj, gj = dl.fit_std(std_sc)
    if not gj:
        return None
    s3 = lambda r: (dl.ascore(r, bj, gj, dl.OPS3, 0.0) or {}).get("A")
    sinv = lambda r: (dl.ascore(r, bj, gj, INV_OPS, 0.0) or {}).get("A")
    cf_false = [r for r in conf_sc if r["label"] == 0]
    fsc = [s3(r) for r in cf_false if s3(r) is not None]
    res = {}
    print(f"\n{scope}: conf-half FALSE n={len(fsc)} (identity-only A3)", flush=True)
    for alpha in (0.05, 0.10, 0.20):
        q = quantile_risk(fsc, alpha)
        dec = {"ACCEPT": 0, "REPAIR": 0, "ABSTAIN": 0}
        dt, df = dict(dec), dict(dec)
        flip = {"t": 0, "t_n": 0, "f": 0, "f_n": 0}
        for r in ev:
            ag, ai = s3(r), sinv(r)
            if ag is None:
                continue
            if ai is None:  # no inverse candidate (swimming/driving): single-candidate set
                d = "ACCEPT" if ag <= q else "ABSTAIN"
            elif ag <= q and ai <= q:
                d = "ABSTAIN"
            elif ag <= q:
                d = "ACCEPT"
            elif ai <= q:
                d = "REPAIR"
            else:
                d = "ABSTAIN"
            dec[d] += 1
            (dt if r["label"] == 1 else df)[d] += 1
            if d == "REPAIR":
                k = "t" if r["label"] == 1 else "f"
                flip[f"{k}_n"] += 1
                if not math.isnan(r["a_inv_orig"]) and r["a_inv_orig"] > 0:
                    flip[k] += 1
        res[f"a{alpha:g}"] = {"q": round(q, 4), "n_conf_false": len(fsc),
                              "ACCEPT": dec["ACCEPT"], "REPAIR": dec["REPAIR"],
                              "ABSTAIN": dec["ABSTAIN"], "on_true": dt,
                              "on_false": df, "flip": flip}
        print(f"  α={alpha:.2f} q={q:.3f} ACCEPT {dec['ACCEPT']} REPAIR {dec['REPAIR']} "
              f"ABSTAIN {dec['ABSTAIN']} (true {dt} false {df}) flip {flip}", flush=True)
    return res


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
    print(f"setup: std {len(std_imgs)}imgs/{len(std)}cl  conf {len(conf_imgs)}imgs/{len(conf)}cl",
          flush=True)

    out = {"setup": {"std_imgs": len(std_imgs), "conf_imgs": len(conf_imgs),
                     "std_claims": len(std), "conf_claims": len(conf)},
           "alphas": list(ALPHAS), "note": "q calibrated on conf-half FALSE scores"}

    print("=== Part A': global risk-coverage (q from FALSE calibration) ===", flush=True)
    for scope in ("non-spatial", "all", "spatial"):
        r = partA(rows, std, conf, scope)
        if r:
            out[scope] = r

    print("\n=== Part C': per-stratum risk control, alpha=0.10 ===", flush=True)
    strat = {}
    for s in ("non-spatial", "all"):
        r = partC(rows, std, conf, s)
        if r:
            strat[s] = r
    out["stratified"] = strat

    print("\n=== Part B': two-candidate decisions, alpha-controlled ===", flush=True)
    pb = {}
    for s in ("non-spatial", "all"):
        r = partB(rows, std, conf, s)
        if r:
            pb[s] = r
    out["partB"] = pb

    # Sensitivity: q from ALL calib false claims (larger calibration; eval is
    # image-disjoint from calib so the risk guarantee still holds). b/g remain
    # std-half. Checks whether the conf-half q results (esp. high stratum)
    # survive a larger calibration sample.
    print("\n=== Sensitivity: q from ALL calib false claims ===", flush=True)
    sens = {}
    for s in ("non-spatial", "all"):
        sens[s] = partA(rows, std, calib, s)
    for s in ("non-spatial",):
        sens[f"stratified_{s}"] = partC(rows, std, calib, s)
    out["sensitivity_fullcalib_q"] = sens

    with open(f"{BASE}/risk_control_rbench.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n[done] risk_control_rbench.json written", flush=True)


if __name__ == "__main__":
    main()
