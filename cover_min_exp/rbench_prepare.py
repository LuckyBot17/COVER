#!/usr/bin/env python
"""R-Bench instance-level -> (s, r, o) claims for the COVER gate test.

Parses instance-level_filterd.json (has subject/object + normalized boxes) into
clean relation claims: canonical relation whitelist, pixel boxes in
"x|y|w|h" (same format as GQA claims.csv), image-level 40/60 calib/eval split,
and a stratified ~500-claim sample prioritizing the non-spatial family.
"""
import argparse
import json
import os
import random
import re
from collections import defaultdict

from PIL import Image

NONSPATIAL = {"wearing", "holding", "riding", "carrying", "watching", "playing",
              "eating", "drinking", "reading", "using", "interacting with",
              "looking at", "swimming", "driving"}
SPATIAL = {"sitting on", "standing on", "next to", "near", "on", "on top of",
           "behind", "in front of", "above", "below",
           "to the left of", "to the right of", "under"}

RULES = [  # longest phrase first
    ("in front of", "in front of"),
    ("to the right of", "to the right of"),
    ("to the left of", "to the left of"),
    ("interacting with", "interacting with"),
    ("interact with", "interacting with"),
    ("standing next to", "next to"),
    ("sitting next to", "next to"),
    ("parked next to", "next to"),
    ("sitting on", "sitting on"),
    ("standing on", "standing on"),
    ("looking at", "looking at"),
    ("look at", "looking at"),
    ("riding on", "riding"),
    ("parked on", "on"),
    ("on top of", "on top of"),
    ("next to", "next to"),
    ("wearing", "wearing"),
    ("wear", "wearing"),
    ("holding", "holding"),
    ("hold", "holding"),
    ("riding", "riding"),
    ("ride", "riding"),
    ("carrying", "carrying"),
    ("carry", "carrying"),
    ("watching", "watching"),
    ("watch", "watching"),
    ("playing", "playing"),
    ("play", "playing"),
    ("eating", "eating"),
    ("eat", "eating"),
    ("drinking", "drinking"),
    ("drink", "drinking"),
    ("reading", "reading"),
    ("read", "reading"),
    ("using", "using"),
    ("use", "using"),
    ("swimming", "swimming"),
    ("swim", "swimming"),
    ("driving", "driving"),
    ("drive", "driving"),
    ("behind", "behind"),
    ("above", "above"),
    ("below", "below"),
    ("under", "under"),
    ("near", "near"),
    (" on ", "on"),
]

STOP_WORDS = (" all ", " each ", " every ", " both ", " same ")

# 逆关系：方法文档的 argument inversion 要求 (s,r,o)->(o,r^-1,s)
INV = {
    "wearing": "worn by", "holding": "held by", "riding": "ridden by",
    "carrying": "carried by", "watching": "watched by", "playing": "played by",
    "eating": "eaten by", "drinking": "drunk by", "reading": "read by",
    "using": "used by", "looking at": "looked at by",
    "interacting with": "interacted with by",
    "sitting on": "sat on by", "standing on": "stood on by",
    "next to": "next to", "near": "near",
    "on": "under", "under": "on top of",
    "behind": "in front of", "in front of": "behind",
    "above": "below", "below": "above",
    "to the left of": "to the right of", "to the right of": "to the left of",
    "on top of": "under",
    "swimming": None, "driving": None,  # 无干净逆关系，跳过 inversion 视角
}


def norm_truth(qt, lb):
    """Return 1/0 (true/false relation claim) or None to skip."""
    qt = (qt or "").strip().lower().split("(")[0].strip().rstrip(".")
    if not qt.startswith(("positive", "opposite", "random")):
        return None
    lb = (lb or "").strip().lower().split("(")[0].strip().rstrip(".")
    if lb in ("yes", "yes."):
        return 1
    if lb in ("no", "no."):
        return 0
    return None


def bare_noun(x):
    x = x.lower().strip()
    x = re.sub(r"^(a|an|the)\s+", "", x)
    return x.strip()


def extract_rel(text, s, o):
    t = text.lower().strip()
    s = bare_noun(s)
    o = bare_noun(o)
    if not s or not o:
        return None
    sm = re.search(r"(?<![a-z])%s(?![a-z])" % re.escape(s), t)
    if not sm:
        return None
    om = re.search(r"(?<![a-z])%s(?![a-z])" % re.escape(o), t)
    if not om or om.start() <= sm.end():
        return None
    phrase = t[sm.end():om.start()]
    phrase = re.sub(r"\bin the image\b", " ", phrase)
    phrase = re.sub(r"\bin the\b", " ", phrase)
    phrase = re.sub(r"\b(a|an|the)\b", " ", phrase)
    phrase = re.sub(r"[.,?!]", " ", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    if not phrase:
        return None
    probe = " " + phrase + " "
    for pat, canon in RULES:
        if " " + pat + " " in probe:
            return canon
    return None


def box_ok(norm):
    try:
        x1, y1, x2, y2 = (float(v) for v in norm)
    except (TypeError, ValueError):
        return False
    return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", default="/root/autodl-tmp/data/R-bench/data_filterd/instance-level_filterd.json")
    ap.add_argument("--imgdir", default="/root/autodl-tmp/data/R-bench/validation")
    ap.add_argument("--out", default="/root/autodl-tmp/cover_min_exp/rbench_claims.csv")
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--nonspatial-frac", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    inst = json.load(open(args.inst))
    stats = {"total": len(inst), "no_s_or_o": 0, "bad_box": 0, "not_relation_q": 0,
             "no_parse": 0, "not_whitelisted": 0, "quantified": 0}
    claims = []
    for q in inst:
        text = (q.get("text") or "").strip()
        s = (q.get("subject") or "").strip()
        o = (q.get("object") or "").strip()
        if not s or not o:
            stats["no_s_or_o"] += 1
            continue
        if not (box_ok(q.get("sub_box")) and box_ok(q.get("obj_box"))):
            stats["bad_box"] += 1
            continue
        truth = norm_truth(q.get("qtype"), q.get("label"))
        if truth is None:
            stats["not_relation_q"] += 1
            continue
        if any(w in text.lower() for w in STOP_WORDS):
            stats["quantified"] += 1
            continue
        rel = extract_rel(text, s, o)
        if rel is None:
            stats["no_parse"] += 1
            continue
        if rel in NONSPATIAL:
            family = "non-spatial"
        elif rel in SPATIAL:
            family = "spatial"
        else:
            stats["not_whitelisted"] += 1
            continue
        claims.append({
            "image_id": q["image"], "s": bare_noun(s), "o": bare_noun(o),
            "r": rel, "label": truth,
            "family": family, "qtype": q.get("qtype", ""), "text": text,
            "sub_box": q.get("sub_box"), "obj_box": q.get("obj_box"),
        })
    stats["kept"] = len(claims)
    print("=== 清洗统计 ===", flush=True)
    for k, v in stats.items():
        print(f"  {k}: {v}", flush=True)

    # image-level 40/60 split
    images = sorted(set(c["image_id"] for c in claims))
    rnd = random.Random(args.seed)
    rnd.shuffle(images)
    cut = int(round(0.4 * len(images)))
    calib_img = set(images[:cut])
    for c in claims:
        c["split"] = "calib" if c["image_id"] in calib_img else "eval"

    # interleave labels within each relation for balance
    by_rel = defaultdict(list)
    for c in claims:
        by_rel[c["r"]].append(c)
    for r in by_rel:
        lst = by_rel[r]
        t = [c for c in lst if c["label"] == 1]
        f = [c for c in lst if c["label"] == 0]
        rnd.shuffle(t)
        rnd.shuffle(f)
        inter = [v for pair in zip(t, f) for v in pair]
        inter += t[len(f):] + f[len(t):]
        by_rel[r] = inter

    # round-robin across relations per family until target reached (0 = all)
    selected = []
    for fam, frac in (("non-spatial", args.nonspatial_frac), ("spatial", 1 - args.nonspatial_frac)):
        fam_claims = [c for c in claims if c["family"] == fam]
        fam_rel = sorted({c["r"] for c in fam_claims}, key=lambda r: -sum(1 for c in fam_claims if c["r"] == r))
        if args.target > 0:
            target = int(round(args.target * frac))
            pools = [by_rel[r] for r in fam_rel]
            idx = [0] * len(fam_rel)
            got = 0
            while got < target:
                progress = False
                for i in range(len(fam_rel)):
                    if idx[i] < len(pools[i]):
                        selected.append(pools[i][idx[i]])
                        idx[i] += 1
                        got += 1
                        progress = True
                        if got >= target:
                            break
                if not progress:
                    break
            print(f"  [family] {fam}: 可用 {len(fam_claims)} / 目标 {target} / 取到 {got}", flush=True)
        else:
            for r in fam_rel:
                selected.extend(by_rel[r])
            print(f"  [family] {fam}: 全部 {len(fam_claims)} 条", flush=True)

    # pixel boxes via actual image size
    size_cache = {}
    imgdir = args.imgdir
    sel = []
    for c in selected:
        img_id = c["image_id"]
        if img_id not in size_cache:
            with Image.open(os.path.join(imgdir, img_id)) as im:
                size_cache[img_id] = im.size
        W, H = size_cache[img_id]
        b = []
        for norm in (c["sub_box"], c["obj_box"]):
            x1, y1, x2, y2 = (float(v) for v in norm)
            bx = round(x1 * W)
            by = round(y1 * H)
            bw = round(x2 * W) - bx
            bh = round(y2 * H) - by
            if bw < 2 or bh < 2:
                b = None
                break
            b.append((bx, by, bw, bh))
        if b is None:
            continue
        sel.append({
            "image_id": img_id,
            "image_path": os.path.join(imgdir, img_id),
            "s": c["s"], "o": c["o"], "r": c["r"], "r_inv": INV.get(c["r"], c["r"]),
            "label": c["label"], "split": c["split"], "family": c["family"],
            "qtype": c["qtype"], "orig_text": c["text"],
            "sub_box": "|".join(map(str, b[0])), "obj_box": "|".join(map(str, b[1])),
        })

    # dedup by image-level split presence
    n_cal = sum(1 for c in sel if c["split"] == "calib")
    n_ev = sum(1 for c in sel if c["split"] == "eval")
    print(f"=== 抽样结果：共 {len(sel)} 条，calib {n_cal} / eval {n_ev} ===", flush=True)
    from collections import Counter
    print("  家族×标签:", flush=True)
    for fam in ("non-spatial", "spatial"):
        for lb in (1, 0):
            n = sum(1 for c in sel if c["family"] == fam and c["label"] == lb)
            print(f"    {fam:12s} 标签{lb}: {n}", flush=True)
    print("  关系分布:", flush=True)
    for r, n in Counter(c["r"] for c in sel).most_common():
        print(f"    {r:16s} {n}", flush=True)
    print("  qtype 分布:", dict(Counter(c["qtype"] for c in sel)), flush=True)

    # write
    fieldnames = ["image_id", "image_path", "s", "o", "r", "r_inv", "label",
                  "split", "family", "qtype", "orig_text", "sub_box", "obj_box"]
    with open(args.out, "w", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for c in sel:
            w.writerow(c)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
