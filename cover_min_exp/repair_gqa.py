#!/usr/bin/env python
"""GQA REPAIR ground-truth experiment (Test A, zero new inference).

QUESTION: is the REPAIR axis (two-candidate {r_gen, r^-1}) correct, judged
against scene-graph ground truth? R-Bench could not test this (only 6
same-pair sibling claims); GQA gives a clean test: present_rels is the true
relation set, and we VERIFIED inverse truth == label for all 280 claims
(inverse of a false claim is always false in the scene graph), so:

  - REPAIR output (o, r^-1, s) is true  <=> label == 1.
  - REPAIR on a FALSE claim can never fix it (semantically impossible).
  - REPAIR on a TRUE claim outputs a true statement (rescues coverage the
    identity threshold missed) but is a spurious flip.

So the measurable quantities are: (1) REPAIR precision P(label=1 | REPAIR)
and fire rates; (2) rescue (REPAIR on true) vs harm (REPAIR on false);
(3) net decision correctness with- vs without-REPAIR; (4) the inverse score
as an independent verifier (AUROC, complementary coverage beyond A_id).

Protocol mirrors risk_control_rbench.py partB: b/g fit on std-half calib,
q = alpha-quantile of conf-half FALSE identity scores, decisions on eval.
alpha in {.05,.10,.20}. Full-calib q sensitivity included.

Writes repair_gqa.json + text report.
"""
import csv
import json
import math
import os
import random

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
GMIN = 0.01
SEED = 2026
OPS = ("s", "o", "so")
ALPHAS = (0.05, 0.10, 0.20)


def load():
    claims = list(csv.DictReader(open(f"{BASE}/claims.csv")))
    scored = {c["image_id"] + "|" + c["s"] + "|" + c["r"] + "|" + c["o"]: c
              for c in csv.DictReader(open(f"{BASE}/scored_claims_blur.csv"))}
    rows = []
    for c in claims:
        k = c["image_id"] + "|" + c["s"] + "|" + c["r"] + "|" + c["o"]
        s = scored.get(k)
        if not s:
            print("  [skip] no score for", k)
            continue
        row = dict(c)
        row["label"] = int(c["label"])
        row["present"] = [x.strip() for x in c["present_rels"].split("|")]
        row["inv_true"] = row["r"] in row["present"]  # verified == label
        for pfx in ("d_id", "d_inv"):
            for op in OPS:
                v = s.get(f"{pfx}:{op}", "")
                row[f"{pfx}:{op}"] = float(v) if v not in ("", "nan") else math.nan
        row["a_orig"] = float(s["a_orig"])
        rows.append(row)
    return rows


def fit_bg(rows):
    """Per-op (b=median false d, g=median true d - b) for identity diffs."""
    bg = {}
    for op in OPS:
        tr = [r[f"d_id:{op}"] for r in rows if r["label"] == 1 and not math.isnan(r[f"d_id:{op}"])]
        fa = [r[f"d_id:{op}"] for r in rows if r["label"] == 0 and not math.isnan(r[f"d_id:{op}"])]
        if not tr or not fa:
            continue
        b = float(np.median(fa))
        g = float(np.median(tr)) - b
        if g > GMIN:
            bg[op] = (b, g)
    return bg


def fit_bg_inv(rows):
    bg = {}
    for op in OPS:
        tr = [r[f"d_inv:{op}"] for r in rows if r["label"] == 1 and not math.isnan(r[f"d_inv:{op}"])]
        fa = [r[f"d_inv:{op}"] for r in rows if r["label"] == 0 and not math.isnan(r[f"d_inv:{op}"])]
        if not tr or not fa:
            continue
        b = float(np.median(fa))
        g = float(np.median(tr)) - b
        if g > GMIN:
            bg[op] = (b, g)
    return bg


def ascore(r, bg, pfx):
    xs = []
    for op, (b, g) in bg.items():
        v = r[f"{pfx}:{op}"]
        if math.isnan(v):
            continue
        xs.append((v - b) / (g + 1e-6))
    if not xs:
        return None
    xs = np.asarray(xs)
    mu = float(xs.mean())
    J = float(xs.var(ddof=1) / (len(xs) - 1)) if len(xs) > 2 else 0.0
    return -mu + 0.0 * J


def quantile_risk(scores, alpha):
    n = len(scores)
    if n == 0:
        return float("nan")
    s = sorted(scores)
    k = max(1, min(n, int(math.floor(alpha * (n + 1)))))
    return s[k - 1]


def auroc(scores, labels):
    sc = sorted(zip(scores, labels), key=lambda t: t[0])
    ranks = []
    i = 0
    n = len(sc)
    while i < n:
        j = i
        while j + 1 < n and sc[j + 1][0] == sc[i][0]:
            j += 1
        avg = (i + j + 2) / 2.0
        for t in range(i, j + 1):
            ranks.append((avg, sc[t][1]))
        i = j + 1
        n_pos = sum(l for _, l in sc)
        n_neg = len(sc) - n_pos
    u = sum(r for r, l in ranks if l == 1) - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg) if n_pos and n_neg else float("nan")


def decide(ev, sA, sI, q):
    out = []
    for r in ev:
        a, i = sA(r), sI(r)
        if a is None:
            out.append({"claim": r, "dec": "NA", "a": None, "i": None})
            continue
        if a <= q:
            d = "ACCEPT"
        elif i is not None and i <= q:
            d = "REPAIR"
        else:
            d = "ABSTAIN"
        out.append({"claim": r, "dec": d, "a": a, "i": i})
    return out


def report_decision(ev, sA, sI, q, q_inv, tag, res):
    ds = decide(ev, sA, sI, q)
    acc = {"ACCEPT": 0, "REPAIR": 0, "ABSTAIN": 0}
    by_lab = {l: dict(acc) for l in (1, 0)}
    for d in ds:
        if d["dec"] == "NA":
            continue
        by_lab[d["claim"]["label"]][d["dec"]] += 1
        acc[d["dec"]] += 1
    n_fire_true = by_lab[1]["REPAIR"]
    n_fire_false = by_lab[0]["REPAIR"]
    n_rescue = n_fire_true
    n_harm = n_fire_false
    # identity-only baseline: REPAIR -> ABSTAIN
    acc_base = acc["ACCEPT"]
    # correct among decided outputs (ACCEPT or REPAIR): correct iff label==1
    n_decided = acc["ACCEPT"] + acc["REPAIR"]
    prec_with = (sum(1 for d in ds if d["dec"] in ("ACCEPT", "REPAIR")
                     and d["claim"]["label"] == 1) / n_decided) if n_decided else float("nan")
    prec_base = (sum(1 for d in ds if d["dec"] == "ACCEPT" and d["claim"]["label"] == 1) /
                 acc_base) if acc_base else float("nan")
    cov_with = sum(1 for d in ds if d["dec"] in ("ACCEPT", "REPAIR") and d["claim"]["label"] == 1)
    cov_base = sum(1 for d in ds if d["dec"] == "ACCEPT" and d["claim"]["label"] == 1)
    n_true = sum(1 for d in ds if d["claim"]["label"] == 1)
    n_false = sum(1 for d in ds if d["claim"]["label"] == 0)
    # cross-tab A_id<=q vs A_inv<=q (REPAIR candidate zone = (A_id>q, A_inv<=q))
    cross = {"id_le": {"inv_le": 0, "inv_gt": 0}, "id_gt": {"inv_le": 0, "inv_gt": 0}}
    cross_lab = {"true": dict(cross), "false": dict(cross)}
    for d in ds:
        if d["dec"] == "NA" or d["i"] is None:
            continue
        ix = "id_le" if d["a"] <= q else "id_gt"
        iy = "inv_le" if d["i"] <= q else "inv_gt"
        cross[ix][iy] += 1
        cross_lab["true" if d["claim"]["label"] == 1 else "false"][ix][iy] += 1
    # REPAIR-B: identity rejected, inverse passes its OWN risk quantile q_inv
    rep_b = sum(1 for d in ds if d["dec"] != "NA" and d["i"] is not None
                and d["a"] > q and d["i"] <= q_inv)
    rep_b_true = sum(1 for d in ds if d["dec"] != "NA" and d["i"] is not None
                     and d["a"] > q and d["i"] <= q_inv and d["claim"]["label"] == 1)
    rec = {
        "q": round(q, 3), "q_inv": round(q_inv, 3),
        "n_eval_true": n_true, "n_eval_false": n_false,
        "decisions": acc,
        "by_label": {"true": by_lab[1], "false": by_lab[0]},
        "cross_tab": cross, "cross_tab_by_label": cross_lab,
        "repair_b_fires": rep_b, "repair_b_true": rep_b_true,
        "repair_precision": round(n_fire_true / (n_fire_true + n_fire_false), 4)
        if (n_fire_true + n_fire_false) else None,
        "repair_fire_rate_true": round(n_fire_true / n_true, 4) if n_true else None,
        "repair_fire_rate_false": round(n_fire_false / n_false, 4) if n_false else None,
        "rescue_true_via_repair": n_fire_true,
        "harm_false_via_repair": n_fire_false,
        "identity_only": {"precision": round(prec_base, 4),
                          "true_outputs": cov_base, "n_decided": acc_base},
        "with_repair": {"precision": round(prec_with, 4),
                        "true_outputs": cov_with, "n_decided": n_decided},
    }
    print(f"  {tag} q_id={q:.3f} q_inv={q_inv:.3f} ACCEPT {acc['ACCEPT']} "
          f"REPAIR {acc['REPAIR']} ABSTAIN {acc['ABSTAIN']} | true {by_lab[1]} "
          f"false {by_lab[0]} | REPAIR precision {rec['repair_precision']} | "
          f"rescue {n_fire_true} harm {n_fire_false} | precision-with {prec_with:.3f} "
          f"vs identity-only {prec_base:.3f} | REPAIR-B {rep_b} ({rep_b_true} true)", flush=True)
    res[tag] = rec


def main():
    rows = load()
    calib = [r for r in rows if r["split"] == "calib"]
    ev = [r for r in rows if r["split"] == "eval"]
    calib_imgs = sorted({r["image_id"] for r in calib})
    rng = random.Random(SEED)
    rng.shuffle(calib_imgs)
    cut = len(calib_imgs) // 2
    conf_imgs = set(calib_imgs[cut:])
    conf = [r for r in calib if r["image_id"] in conf_imgs]
    print(f"setup: conf {len(conf)} / eval {len(ev)} "
          f"(true {sum(1 for r in ev if r['label']==1)}, "
          f"false {sum(1 for r in ev if r['label']==0)})", flush=True)

    # GQA identity negative-control diffs are weak (g ~ 0.03-0.16 full calib,
    # <0 on any image-split half), so b/g are fit on FULL calib (V1-style;
    # q stays image-disjoint from eval -> risk guarantee intact). Reported
    # honestly: all GQA verifiers have AUROC 0.56-0.59 (vs R-Bench 0.67-0.71).
    bg_id = fit_bg(calib)
    bg_inv = fit_bg_inv(calib)
    print(f"identity ops kept: {list(bg_id)}  inverse ops kept: {list(bg_inv)}", flush=True)
    if not bg_id:
        print("  [fatal] no identity ops survive g_min; aborting")
        return
    sA = lambda r: ascore(r, bg_id, "d_id")
    sI = lambda r: ascore(r, bg_inv, "d_inv") if bg_inv else None

    out = {"note": "full-calib b/g (GQA identity diffs too weak for half-fit); "
                   "q from conf-half FALSE A_id (main) or full-calib (sensitivity)",
           "ops_kept": {"id": list(bg_id), "inv": list(bg_inv)},
           "n": {"calib": len(calib), "conf": len(conf), "eval": len(ev)}}

    # AUROC of each verifier on eval, decision-layer convention: lower score
    # = more acceptable, so AUROC = P(score_true < score_false) = 1 - auroc().
    for name, sfn in (("A_id", sA), ("A_inv", sI), ("a_orig", lambda r: -r["a_orig"])):
        sc, lab = [], []
        for r in ev:
            v = sfn(r)
            if v is not None:
                sc.append(v)
                lab.append(r["label"])
        a = 1.0 - auroc(sc, lab)
        out.setdefault("verifier_auroc", {})[name] = round(a, 4)
        print(f"  accept-AUROC eval  {name:6s} = {a:.3f}", flush=True)

    # main protocol: q from conf-half false A_id (identity) and false A_inv (inverse)
    print("\n=== main protocol (q from conf-half FALSE A_id / A_inv) ===", flush=True)
    conf_false_id = [sA(r) for r in conf if r["label"] == 0 and sA(r) is not None]
    conf_false_inv = [sI(r) for r in conf if r["label"] == 0 and sI(r) is not None]
    for alpha in ALPHAS:
        q = quantile_risk(conf_false_id, alpha)
        q_inv = quantile_risk(conf_false_inv, alpha)
        report_decision(ev, sA, sI, q, q_inv, f"a{alpha:g}", out.setdefault("main", {}))

    # sensitivity: q from ALL calib false scores
    print("\n=== sensitivity (q from ALL calib FALSE A_id / A_inv) ===", flush=True)
    cal_false_id = [sA(r) for r in calib if r["label"] == 0 and sA(r) is not None]
    cal_false_inv = [sI(r) for r in calib if r["label"] == 0 and sI(r) is not None]
    for alpha in ALPHAS:
        q = quantile_risk(cal_false_id, alpha)
        q_inv = quantile_risk(cal_false_inv, alpha)
        report_decision(ev, sA, sI, q, q_inv, f"a{alpha:g}", out.setdefault("sens_fullq", {}))

    # correlation between identity and inverse scores (explains REPAIR rarity)
    p = []
    for r in ev:
        a, i = sA(r), sI(r)
        if a is not None and i is not None:
            p.append((a, i))
    if p:
        a_vec = np.array([x for x, _ in p])
        i_vec = np.array([y for _, y in p])
        corr = float(np.corrcoef(a_vec, i_vec)[0, 1])
        out["score_corr_pearson"] = round(corr, 4)
        print(f"  Pearson corr(A_id, A_inv) on eval = {corr:.3f} (n={len(p)})", flush=True)

    # structural check: inverse-of-false is always false in scene graph
    inv_false_true = sum(1 for r in rows if r["label"] == 0 and r["inv_true"])
    out["structural_inverse_of_false_is_true"] = inv_false_true
    out["verified_inv_true_eq_label"] = sum(1 for r in rows if r["inv_true"] == (r["label"] == 1))
    print(f"\nstructural: inverse-of-false claims that are true in scene graph: "
          f"{inv_false_true}/0 expected; inv_true==label holds for "
          f"{out['verified_inv_true_eq_label']}/{len(rows)}", flush=True)

    with open(f"{BASE}/repair_gqa.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n[done] repair_gqa.json", flush=True)


if __name__ == "__main__":
    main()
