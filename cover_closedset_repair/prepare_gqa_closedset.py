#!/usr/bin/env python3
"""Prepare object-ID-safe GQA pairs for COVER closed-set relation repair.

Scientific question
-------------------
Can COVER choose the true relation for a known subject/object pair from the
fixed 15-relation GQA vocabulary?  This script constructs the exhaustive,
multi-label ground truth needed to answer that question.  Splits are grouped
by image so no image crosses standardization, risk calibration, pilot, or the
final held-out evaluation set.

Outputs: all_pairs.csv, pilot_pairs.csv, smoke_pairs.csv, and census.json.
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parent
SEED = 42
RELATIONS = (
    "wearing", "holding", "riding", "carrying", "watching",
    "next to", "near", "on top of", "under",
    "to the left of", "to the right of", "above", "below",
    "behind", "in front of",
)
RELATION_SET = set(RELATIONS)
STOP_OBJECTS = {
    "air", "sky", "water", "sea", "ground", "floor", "wall", "street",
    "road", "tent", "grass", "tree", "building", "background", "ceiling",
    "cloud", "dirt", "grounds", "room", "hill", "field", "sidewalk",
    "door", "window",
}
SPLIT_FRACTIONS = (
    ("standardization", 0.15),
    ("risk_calibration", 0.15),
    ("pilot_evaluation", 0.10),
    ("final_evaluation", 0.60),
)
PAIR_FIELDS = (
    "pair_id", "split", "image_id", "image_path",
    "subject_id", "subject_name", "subject_box",
    "object_id", "object_name", "object_box",
    "width", "height", "present_rels",
)


def box_ok(box, width, height):
    """Match the original COVER box filter while rejecting malformed images."""
    if not width or not height or width <= 0 or height <= 0:
        return False
    x, y, w, h = box
    if any(v is None for v in box) or w <= 0 or h <= 0:
        return False
    if x < 0 or y < 0 or x + w > width + 2 or y + h > height + 2:
        return False
    area = w * h / (width * height)
    return 0.005 < area < 0.7


def pair_key(row):
    return (str(row["image_id"]), str(row["subject_id"]), str(row["object_id"]))


def make_pair_id(image_id, subject_id, object_id):
    return f"{image_id}|{subject_id}|{object_id}"


def box_text(box):
    return "|".join(str(int(v)) for v in box)


def parse_relations(value):
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [v for v in str(value).split("|") if v]


def extract_pairs(scene_graph, img_root, require_images=True):
    """Extract eligible directed entity pairs and audit every emitted edge.

    Relations are deliberately read from ``objects[subject_id]`` in the same
    loop iteration.  This prevents the stale-loop-variable bug in the earlier
    GQA preparation script.  Object IDs, not names, define pair identity.
    """
    root = Path(img_root) if img_root is not None else None
    pair_map = {}
    audit = Counter()

    for image_id, graph in scene_graph.items():
        image_id = str(image_id)
        image_path = root / f"{image_id}.jpg" if root is not None else Path(f"{image_id}.jpg")
        if require_images and not image_path.exists():
            audit["missing_images"] += 1
            continue
        width, height = graph.get("width"), graph.get("height")
        objects = graph.get("objects", {})
        valid = {}
        for object_id, obj in objects.items():
            object_id = str(object_id)
            name = str(obj.get("name", "")).strip().lower()
            box = (obj.get("x"), obj.get("y"), obj.get("w"), obj.get("h"))
            if not name or name in STOP_OBJECTS or len(name) > 24:
                audit["filtered_object_name"] += 1
                continue
            if not box_ok(box, width, height):
                audit["filtered_object_box"] += 1
                continue
            valid[object_id] = {"name": name, "box": box, "raw": obj}

        for subject_id, subject in valid.items():
            # Important: use this subject's raw object, never a prior loop value.
            raw_relations = subject["raw"].get("relations", [])
            for relation in raw_relations:
                relation_name = str(relation.get("name", "")).strip().lower()
                if relation_name not in RELATION_SET:
                    continue
                audit["vocabulary_edges_seen"] += 1
                object_id = str(relation.get("object"))
                if object_id not in valid:
                    audit["filtered_target"] += 1
                    continue
                target = valid[object_id]
                key = (image_id, subject_id, object_id)
                if key not in pair_map:
                    pair_map[key] = {
                        "pair_id": make_pair_id(*key),
                        "image_id": image_id,
                        "image_path": str(image_path),
                        "subject_id": subject_id,
                        "subject_name": subject["name"],
                        "subject_box": tuple(subject["box"]),
                        "object_id": object_id,
                        "object_name": target["name"],
                        "object_box": tuple(target["box"]),
                        "width": int(width),
                        "height": int(height),
                        "present_rels": set(),
                    }
                pair_map[key]["present_rels"].add(relation_name)

                # Immediate source-of-truth check: the exact edge must occur in
                # the current subject record and point to the current object ID.
                found = any(
                    str(r.get("name", "")).strip().lower() == relation_name
                    and str(r.get("object")) == object_id
                    for r in objects[subject_id].get("relations", [])
                )
                if not found:
                    audit["edge_lookup_failures"] += 1
                else:
                    audit["emitted_edges"] += 1

    pairs = []
    for key in sorted(pair_map):
        row = pair_map[key]
        truth = sorted(row["present_rels"], key=RELATIONS.index)
        if not truth or len(truth) == len(RELATIONS):
            audit["pairs_without_both_classes"] += 1
            continue
        row["present_rels"] = truth
        pairs.append(row)
    # Materialize zero-valued critical counters so audits never confuse
    # "no failures" with "field missing".
    audit["edge_lookup_failures"] += 0
    audit["eligible_pairs"] = len(pairs)
    audit["eligible_images"] = len({p["image_id"] for p in pairs})
    return pairs, dict(audit)


def _split_counts(n_images):
    if n_images < 4:
        raise ValueError("at least four eligible images are required for four image-disjoint splits")
    counts = [max(1, int(n_images * frac)) for _, frac in SPLIT_FRACTIONS[:-1]]
    while sum(counts) >= n_images:
        largest = max(range(len(counts)), key=counts.__getitem__)
        if counts[largest] == 1:
            raise ValueError("not enough images to create non-empty splits")
        counts[largest] -= 1
    counts.append(n_images - sum(counts))
    return counts


def assign_image_splits(pairs, seed=SEED):
    """Return copied pair records with deterministic, image-disjoint splits."""
    images = sorted({str(p["image_id"]) for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(images)
    counts = _split_counts(len(images))
    image_split = {}
    start = 0
    for (name, _), count in zip(SPLIT_FRACTIONS, counts):
        for image_id in images[start:start + count]:
            image_split[image_id] = name
        start += count
    out = []
    for pair in pairs:
        row = dict(pair)
        row["split"] = image_split[str(pair["image_id"])]
        out.append(row)
    return out


def iter_candidate_rows(pairs):
    """Yield the exhaustive candidate grid without materializing full GQA."""
    for pair in pairs:
        truth = set(parse_relations(pair["present_rels"]))
        for relation in RELATIONS:
            row = dict(pair)
            row["candidate_r"] = relation
            row["candidate_true"] = int(relation in truth)
            yield row


def candidate_rows(pairs):
    """Materialize candidates for tests and small diagnostic subsets."""
    return list(iter_candidate_rows(pairs))


def _balanced_subset(rows, target, seed):
    """Deterministically favor underrepresented true relations."""
    if len(rows) < target:
        raise ValueError(f"requested {target} pairs but split contains only {len(rows)}")
    rng = random.Random(seed)
    pool = [dict(r) for r in rows]
    rng.shuffle(pool)
    relation_counts = Counter()
    selected = []
    while pool and len(selected) < target:
        best_i = min(
            range(len(pool)),
            key=lambda i: (
                min(relation_counts[r] for r in parse_relations(pool[i]["present_rels"])),
                sum(relation_counts[r] for r in parse_relations(pool[i]["present_rels"])),
                pool[i]["pair_id"],
            ),
        )
        chosen = pool.pop(best_i)
        selected.append(chosen)
        for relation in parse_relations(chosen["present_rels"]):
            relation_counts[relation] += 1
    return selected


def select_pilot(pairs, per_split=50, seed=SEED):
    selected = []
    wanted = ("standardization", "risk_calibration", "pilot_evaluation")
    for offset, split in enumerate(wanted):
        rows = [p for p in pairs if p["split"] == split]
        selected.extend(_balanced_subset(rows, per_split, seed + offset))
    return selected


def select_smoke(pilot_pairs, per_split=2):
    selected = []
    for split in ("standardization", "risk_calibration", "pilot_evaluation"):
        rows = sorted(
            (p for p in pilot_pairs if p["split"] == split),
            key=lambda p: p["pair_id"],
        )
        if len(rows) < per_split:
            raise ValueError(f"pilot split {split} has fewer than {per_split} pairs")
        selected.extend(rows[:per_split])
    return selected


def serialize_pair(row):
    rec = dict(row)
    rec["subject_box"] = box_text(row["subject_box"]) if not isinstance(row["subject_box"], str) else row["subject_box"]
    rec["object_box"] = box_text(row["object_box"]) if not isinstance(row["object_box"], str) else row["object_box"]
    rec["present_rels"] = "|".join(parse_relations(row["present_rels"]))
    return rec


def write_pairs(path, pairs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for pair in pairs:
            writer.writerow(serialize_pair(pair))


def relation_counts(pairs):
    counts = Counter()
    for pair in pairs:
        counts.update(parse_relations(pair["present_rels"]))
    return {r: counts[r] for r in RELATIONS}


def build_census(pairs, pilot, smoke, audit):
    split_pairs = Counter(p["split"] for p in pairs)
    split_images = defaultdict(set)
    for pair in pairs:
        split_images[pair["split"]].add(pair["image_id"])
    return {
        "seed": SEED,
        "relation_vocabulary": list(RELATIONS),
        "n_relations": len(RELATIONS),
        "eligible_pairs": len(pairs),
        "eligible_images": len({p["image_id"] for p in pairs}),
        "candidate_evaluations": len(pairs) * len(RELATIONS),
        "multilabel_pairs": sum(len(parse_relations(p["present_rels"])) > 1 for p in pairs),
        "truth_relations_per_pair": dict(Counter(len(parse_relations(p["present_rels"])) for p in pairs)),
        "relation_counts": relation_counts(pairs),
        "split_pair_counts": dict(split_pairs),
        "split_image_counts": {k: len(v) for k, v in split_images.items()},
        "pilot_pairs": len(pilot),
        "pilot_split_counts": dict(Counter(p["split"] for p in pilot)),
        "pilot_relation_counts": relation_counts(pilot),
        "smoke_pairs": len(smoke),
        "source_audit": audit,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sg", default="/root/autodl-tmp/data/GQA/val_sceneGraphs.json")
    parser.add_argument("--img-root", default="/root/autodl-tmp/data/GQA/images")
    parser.add_argument("--outdir", default=str(BASE))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pilot-per-split", type=int, default=50)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    print(f"[info] loading scene graph: {args.sg}", flush=True)
    with open(args.sg, encoding="utf-8") as f:
        scene_graph = json.load(f)
    pairs, audit = extract_pairs(scene_graph, args.img_root, require_images=True)
    if audit.get("edge_lookup_failures", 0):
        raise RuntimeError(f"scene-graph edge audit failed: {audit['edge_lookup_failures']}")
    pairs = assign_image_splits(pairs, seed=args.seed)
    pilot = select_pilot(pairs, per_split=args.pilot_per_split, seed=args.seed)
    smoke = select_smoke(pilot)

    write_pairs(outdir / "all_pairs.csv", pairs)
    write_pairs(outdir / "pilot_pairs.csv", pilot)
    write_pairs(outdir / "smoke_pairs.csv", smoke)
    census = build_census(pairs, pilot, smoke, audit)
    census["seed"] = args.seed
    with (outdir / "census.json").open("w", encoding="utf-8") as f:
        json.dump(census, f, indent=2, ensure_ascii=False)
    print(
        f"[info] eligible pairs={len(pairs)} images={census['eligible_images']} "
        f"candidate evaluations={census['candidate_evaluations']}", flush=True
    )
    print(f"[info] split pairs={census['split_pair_counts']}", flush=True)
    print(f"[done] wrote preparation outputs to {outdir}", flush=True)


if __name__ == "__main__":
    main()
