#!/usr/bin/env python
"""Per-stratum V2 check: does the high-stratum COVER-vs-raw gap survive the
cluster-bootstrap (conservative) q? Computes V0 and V2 per confidence tercile
for non-spatial scope, at alpha=0.10, so the recalibration report can state
the headline honestly. Pure analysis, no new inference."""
import json
import math
import os

import numpy as np

import decision_layer_rbench as dl

dl.BASE = os.path.dirname(os.path.abspath(__file__))
BASE = dl.BASE

ALPHA = 0.10
NB = 3000


def quantile_risk(scores, alpha):
    n = len(scores)
    if n == 0:
        return float("nan")
    s = sorted(scores)
    k = max(1, min(n, int(math.floor(alpha * (n + 1)))))
    return s[k - 1]


def make_A(bj, gj):
    return lambda r: (dl.ascore(r, bj, gj, dl.OPS6, 0.0) or {}).get("A")


def bootstrap_q(false_rows, sfn, keyfn, alpha):
    imgs = list({keyfn(r) for r in false_rows if sfn(r) is not None})
    by_img = {im: sorted([sfn(r) for r in false_rows
                          if sfn(r) is not None and keyfn(r) == im]) for im in imgs}
    grid = sorted({s for v in by_img.values() for s in v})
    rng = np.random.default_rng(2026)
    chosen = -np.inf
    for q in grid:
        b = np.empty(NB)
        for i in range(NB):
            sel = rng.choice(imgs, len(imgs), replace=True)
            cnt = tot = 0
            for im in sel:
                vs = by_img[im]
                cnt += sum(1 for s in vs if s <= q)
                tot += len(vs)
            b[i] = cnt / tot if tot else 0.0
        if float(np.percentile(b, 95)) <= alpha:
            chosen = q
        else:
            break
    return chosen


def main():
    rows = dl.load()
    calib = [r for r in rows if r["split"] == "calib"]
    calib_imgs = sorted({r["image_id"] for r in calib})
    import random
    rng = random.Random(dl.SEED)
    rng.shuffle(calib_imgs)
    cut = len(calib_imgs) // 2
    std_imgs, conf_imgs = set(calib_imgs[:cut]), set(calib_imgs[cut:])
    std = [r for r in calib if r["image_id"] in std_imgs]
    conf = [r for r in calib if r["image_id"] in conf_imgs]
    scope = "non-spatial"
    std_sc = [r for r in std if dl.fam(r, scope)]
    conf_sc = [r for r in conf if dl.fam(r, scope)]
    cal_sc = [r for r in calib if dl.fam(r, scope)]
    ev = [r for r in rows if r["split"] == "eval" and dl.fam(r, scope)]
    bj, gj = dl.fit_std(std_sc)
    sA = make_A(bj, gj)
    sR = lambda r: -r["a_orig"]
    t1, t2 = np.percentile([r["a_orig"] for r in conf_sc], [100 / 3, 200 / 3])
    out = []
    print(f"non-spatial per-stratum V0 (conf-half q) vs V2 (cluster-bootstrap q), alpha={ALPHA}")
    for lo, hi, name in ((-np.inf, t1, "低"), (t1, t2, "中"), (t2, np.inf, "高")):
        cf = [r for r in conf_sc if lo < r["a_orig"] <= hi and r["label"] == 0]
        et = [r for r in ev if lo < r["a_orig"] <= hi and r["label"] == 1]
        ef = [r for r in ev if lo < r["a_orig"] <= hi and r["label"] == 0]
        rec = {"stratum": name, "n_conf_false": len(cf), "n_true": len(et), "n_false": len(ef)}
        for sname, sfn in (("COVER_A(lam0)", sA), ("raw_conf", sR)):
            fsc = [sfn(r) for r in cf if sfn(r) is not None]
            calf = [r for r in cal_sc if lo < r["a_orig"] <= hi and r["label"] == 0]
            calf_s = [sfn(r) for r in calf if sfn(r) is not None]
            q0 = quantile_risk(fsc, ALPHA)
            q2 = bootstrap_q(calf, sfn, lambda r: r["image_id"], ALPHA)
            for tag, q in (("V0", q0), ("V2", q2)):
                at = sum(1 for r in et if sfn(r) is not None and sfn(r) <= q)
                af = sum(1 for r in ef if sfn(r) is not None and sfn(r) <= q)
                cov = at / len(et) if et else float("nan")
                risk = af / len(ef) if ef else float("nan")
                rec[f"{sname}.{tag}"] = {"q": round(q, 3), "coverage": round(cov, 4),
                                         "risk": round(risk, 4)}
                print(f"  {name} {sname:16s} {tag} q={q:7.3f} cov={cov:.3f} risk={risk:.3f} "
                      f"(eval T{len(et)}/F{len(ef)}, calibF {len(calf_s)})", flush=True)
        out.append(rec)
    with open(f"{BASE}/recalib_strat.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("[done] recalib_strat.json", flush=True)


if __name__ == "__main__":
    main()
