#!/usr/bin/env python
"""COVER minimal experiment -- data preparation.

Build true/false (subject, relation, object) claims from GQA val scene graphs.
True claims come from scene-graph relations; false claims corrupt the relation
to one that is absent for the same entity pair. Every claim carries the
subject/object bounding boxes needed for the negative-control masks, and is
assigned to a calibration or evaluation split grouped by image id.
"""
import argparse
import csv
import json
import random
from pathlib import Path

RELATIONS = {
    "wearing": {"inv": "worn by"},
    "holding": {"inv": "held by"},
    "riding": {"inv": "ridden by"},
    "carrying": {"inv": "carried by"},
    "watching": {"inv": "watched by"},
    "next to": {"inv": "next to", "sym": True},
    "near": {"inv": "near", "sym": True},
    "on top of": {"inv": "under"},
    "under": {"inv": "on top of"},
    "to the left of": {"inv": "to the right of"},
    "to the right of": {"inv": "to the left of"},
    "above": {"inv": "below"},
    "below": {"inv": "above"},
    "behind": {"inv": "in front of"},
    "in front of": {"inv": "behind"},
}

STOP_OBJECTS = {
    "air", "sky", "water", "sea", "ground", "floor", "wall", "street", "road",
    "tent", "grass", "tree", "building", "background", "ceiling", "cloud",
    "dirt", "grounds", "room", "hill", "field", "sidewalk", "door", "window",
}


def box_ok(box, W, H):
    x, y, w, h = box
    if x < 0 or y < 0 or x + w > W + 2 or y + h > H + 2:
        return False
    area = w * h / (W * H)
    return 0.005 < area < 0.7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sg", default="/root/autodl-tmp/data/GQA/val_sceneGraphs.json")
    ap.add_argument("--img-root", default="/root/autodl-tmp/data/GQA/images")
    ap.add_argument("--out", default="/root/autodl-tmp/cover_min_exp/claims.csv")
    ap.add_argument("--n-true", type=int, default=140)
    ap.add_argument("--n-false", type=int, default=140)
    ap.add_argument("--per-rel-true", type=int, default=40)
    ap.add_argument("--calib-frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sg = json.load(open(args.sg))
    img_root = Path(args.img_root)

    # Pass 1: collect candidate true claims with valid entity boxes.
    candidates = []  # (image_id, s, r, o, sbox, obox, W, H)
    pair_rels = {}   # (image_id, s, o) -> set of relations in scene graph
    for iid, g in sg.items():
        if not (img_root / f"{iid}.jpg").exists():
            continue
        W, H = g.get("width"), g.get("height")
        objs = g["objects"]
        by_id = {}
        for oid, o in objs.items():
            name = o.get("name", "").strip().lower()
            b = (o.get("x"), o.get("y"), o.get("w"), o.get("h"))
            if name in STOP_OBJECTS or len(name) > 24:
                continue
            if None in b or b[2] <= 0 or b[3] <= 0:
                continue
            by_id[oid] = (name, b)
        for oid, (name, b) in by_id.items():
            for rel in o.get("relations", []):
                rname = rel.get("name", "").strip()
                if rname not in RELATIONS:
                    continue
                ooid = rel.get("object")
                if ooid not in by_id:
                    continue
                oname, ob = by_id[ooid]
                if oname == name:
                    continue
                if not box_ok(b, W, H) or not box_ok(ob, W, H):
                    continue
                candidates.append((iid, name, rname, oname, b, ob, W, H))
                pair_rels.setdefault((iid, name, oname), set()).add(rname)

    rng.shuffle(candidates)

    # Per-relation quota sampling so spatial relations do not dominate.
    by_rel = {}
    for c in candidates:
        by_rel.setdefault(c[2], []).append(c)
    per_rel = {}
    true_claims = []
    rel_order = sorted(by_rel.keys())
    # proportional quotas: fill each relation round-robin so coverage is balanced
    quotas = {r: min(args.per_rel_true, len(by_rel[r])) for r in rel_order}
    idx = {r: 0 for r in rel_order}
    progressed = True
    while progressed and len(true_claims) < args.n_true:
        progressed = False
        for r in rel_order:
            if idx[r] >= quotas[r]:
                continue
            true_claims.append(by_rel[r][idx[r]])
            per_rel[r] = per_rel.get(r, 0) + 1
            idx[r] += 1
            progressed = True
            if len(true_claims) >= args.n_true:
                break

    # Build false claims: same entities, relation replaced by one absent for the pair.
    rel_keys = list(RELATIONS.keys())
    false_claims = []
    for (iid, s, r, o, sb, ob, W, H) in true_claims:
        present = pair_rels.get((iid, s, o), set())
        rng.shuffle(rel_keys)
        rp = None
        for cand_r in rel_keys:
            if cand_r != r and cand_r not in present:
                rp = cand_r
                break
        if rp is None:
            continue
        false_claims.append((iid, s, rp, o, sb, ob, W, H))
        if len(false_claims) >= args.n_false:
            break

    claims = [(c, 1) for c in true_claims] + [(c, 0) for c in false_claims]
    rng.shuffle(claims)

    # Split by image id so no image straddles calibration / evaluation.
    images = list({c[0] for c, _ in claims})
    rng.shuffle(images)
    calib_imgs = set(images[: max(1, int(len(images) * args.calib_frac))])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "image_id", "image_path", "s", "o", "r", "r_inv",
                    "label", "s_box", "o_box", "width", "height", "present_rels"])
        for (iid, s, r, o, sb, ob, W, H), label in claims:
            split = "calib" if iid in calib_imgs else "eval"
            w.writerow([
                split, iid, f"{img_root}/{iid}.jpg", s, o, r, RELATIONS[r]["inv"],
                label, "|".join(map(str, sb)), "|".join(map(str, ob)), W, H,
                "|".join(sorted(pair_rels.get((iid, s, o), set()))),
            ])

    n_true = sum(1 for _, lab in claims if lab == 1)
    n_false = sum(1 for _, lab in claims if lab == 0)
    print(f"images used: {len(images)} | true: {n_true} | false: {n_false} | total: {len(claims)}")
    print("true per relation:", dict(sorted(per_rel.items())))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
