#!/usr/bin/env python
"""Diagnose GQA identity vs inverse op signal and eval AUROC."""
import csv
import math
import random

import numpy as np

BASE = "/root/autodl-tmp/cover_min_exp"


def main():
    rows = list(csv.DictReader(open(f"{BASE}/claims.csv")))
    scored = {c["image_id"] + "|" + c["s"] + "|" + c["r"] + "|" + c["o"]: c
              for c in csv.DictReader(open(f"{BASE}/scored_claims_blur.csv"))}
    full = []
    for c in rows:
        s = scored.get(c["image_id"] + "|" + c["s"] + "|" + c["r"] + "|" + c["o"])
        if not s:
            continue
        r = dict(c)
        r["label"] = int(c["label"])
        for pfx in ("d_id", "d_inv"):
            for op in ("s", "o", "so"):
                v = s.get(f"{pfx}:{op}", "")
                r[f"{pfx}:{op}"] = float(v) if v not in ("", "nan") else math.nan
        r["a_orig"] = float(s["a_orig"])
        full.append(r)
    calib = [r for r in full if r["split"] == "calib"]
    imgs = sorted({r["image_id"] for r in calib})
    rng = random.Random(2026)
    rng.shuffle(imgs)
    cut = len(imgs) // 2
    std_half = [r for r in calib if r["image_id"] in set(imgs[:cut])]
    print("std_half n =", len(std_half))
    for pfx in ("d_id", "d_inv"):
        for op in ("s", "o", "so"):
            tr = [r[f"{pfx}:{op}"] for r in std_half
                  if r["label"] == 1 and not math.isnan(r[f"{pfx}:{op}"])]
            fa = [r[f"{pfx}:{op}"] for r in std_half
                  if r["label"] == 0 and not math.isnan(r[f"{pfx}:{op}"])]
            if tr and fa:
                b = float(np.median(fa))
                g = float(np.median(tr)) - b
                print(f"  {pfx}:{op} b={b:+.3f} g={g:+.3f} keep={'Y' if g > 0.01 else 'n'}")

    def auroc(scs, labs):
        sc = sorted(zip(scs, labs), key=lambda t: t[0])
        nn = len(sc)
        ranks = []
        i = 0
        while i < nn:
            j = i
            while j + 1 < nn and sc[j + 1][0] == sc[i][0]:
                j += 1
            avg = (i + j + 2) / 2.0
            for t in range(i, j + 1):
                ranks.append((avg, sc[t][1]))
            i = j + 1
        npos = sum(labs)
        nneg = len(labs) - npos
        u = sum(rk for rk, lab in ranks if lab == 1) - npos * (npos + 1) / 2
        return u / (npos * nneg) if npos and nneg else float("nan")

    def fitbg(sub, pfx):
        bg = {}
        for op in ("s", "o", "so"):
            tr = [r[f"{pfx}:{op}"] for r in sub
                  if r["label"] == 1 and not math.isnan(r[f"{pfx}:{op}"])]
            fa = [r[f"{pfx}:{op}"] for r in sub
                  if r["label"] == 0 and not math.isnan(r[f"{pfx}:{op}"])]
            if tr and fa:
                b = float(np.median(fa))
                g = float(np.median(tr)) - b
                if g > 0.01:
                    bg[op] = (b, g)
        return bg

    def A(r, bg, pfx):
        xs = [(r[f"{pfx}:{op}"] - b) / (g + 1e-6)
              for op, (b, g) in bg.items() if not math.isnan(r[f"{pfx}:{op}"])]
        return -float(np.mean(xs)) if xs else None

    ev = [r for r in full if r["split"] == "eval"]
    for tag, sub in (("std_half", std_half), ("full_calib", calib)):
        bgi = fitbg(sub, "d_id")
        bgi2 = fitbg(sub, "d_inv")
        for name, sfn in (("A_id", lambda r: A(r, bgi, "d_id")),
                          ("A_inv", lambda r: A(r, bgi2, "d_inv")),
                          ("a_orig", lambda r: -r["a_orig"])):
            scs2, labs2 = [], []
            for r in ev:
                v = sfn(r)
                if v is not None:
                    scs2.append(v)
                    labs2.append(r["label"])
            print(f"eval AUROC {tag:11s} {name:6s} ops(id={list(bgi)} inv={list(bgi2)}) "
                  f"= {auroc(scs2, labs2):.3f}")


if __name__ == "__main__":
    main()
