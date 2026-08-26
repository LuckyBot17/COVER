#!/usr/bin/env python3
"""Analyze COVER closed-set candidate ranking and selective relation repair.

The main decision threshold is calibrated on the minimum false-candidate risk
inside each entity-pair candidate set.  Image-cluster bootstrap calibration
therefore targets P(any false relation passes | entity pair), not the weaker
per-candidate error rate.  A relation is emitted only when the passing set is
a singleton; otherwise the method abstains.
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from prepare_gqa_closedset import BASE, RELATIONS, parse_relations
from score_gqa_closedset import STAGE_FILES, candidate_key


SEED = 42
G_MIN = 0.01
ALPHAS = (0.05, 0.10, 0.20)
PRIMARY_ALPHA = 0.10
N_BOOTSTRAP = 3000
OPS = ("s", "o", "so")
SPATIAL_RELATIONS = {
    "next to", "near", "on top of", "under", "to the left of",
    "to the right of", "above", "below", "behind", "in front of",
}
NON_SPATIAL_RELATIONS = set(RELATIONS) - SPATIAL_RELATIONS


def as_int(value):
    return int(float(value))


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_pairs(path):
    with Path(path).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["present_rels"] = parse_relations(row["present_rels"])
    return rows


def load_scores(path):
    rows, seen = [], set()
    duplicates = []
    with Path(path).open(encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row = dict(raw)
            row["candidate_true"] = as_int(row["candidate_true"])
            for field in ("a_orig", "d_s", "d_o", "d_so"):
                row[field] = as_float(row[field])
            key = candidate_key(row)
            if key in seen:
                duplicates.append(key)
            seen.add(key)
            rows.append(row)
    return rows, duplicates


def fit_bg(rows, g_min=G_MIN):
    """Fit false baseline b and true-vs-false gap g per control operator."""
    bg = {}
    for op in OPS:
        key = f"d_{op}"
        false_values = [r[key] for r in rows if as_int(r["candidate_true"]) == 0 and math.isfinite(r[key])]
        true_values = [r[key] for r in rows if as_int(r["candidate_true"]) == 1 and math.isfinite(r[key])]
        if not false_values or not true_values:
            continue
        baseline = float(np.median(false_values))
        gap = float(np.median(true_values)) - baseline
        if gap > g_min:
            bg[op] = {"b": baseline, "g": gap}
    return bg


def add_cover_scores(rows, bg):
    out = []
    for source in rows:
        row = dict(source)
        standardized = []
        for op, values in bg.items():
            value = as_float(row[f"d_{op}"])
            if math.isfinite(value):
                standardized.append((value - values["b"]) / (values["g"] + 1e-6))
        if standardized:
            array = np.asarray(standardized, dtype=float)
            row["mu"] = float(array.mean())
            row["J"] = float(array.var(ddof=1)) if len(array) > 1 else 0.0
            row["A"] = -row["mu"]
        else:
            row["mu"] = row["J"] = row["A"] = float("nan")
        out.append(row)
    return out


def auroc(labels, scores):
    pairs = [(float(s), int(y)) for y, s in zip(labels, scores) if math.isfinite(float(s))]
    n_pos = sum(y for _, y in pairs)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    pairs.sort(key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average_rank = (index + end + 2) / 2.0
        rank_sum += average_rank * sum(pairs[i][1] for i in range(index, end + 1))
        index = end + 1
    u = rank_sum - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def group_by_pair(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["pair_id"]].append(row)
    return groups


def pair_false_minima(rows):
    groups = group_by_pair(rows)
    out = []
    for pair_id in sorted(groups):
        group = groups[pair_id]
        false_scores = [as_float(r["A"]) for r in group if as_int(r["candidate_true"]) == 0 and math.isfinite(as_float(r["A"]))]
        if false_scores:
            out.append({
                "pair_id": pair_id,
                "image_id": group[0]["image_id"],
                "false_min_A": min(false_scores),
            })
    return out


def cluster_bootstrap_threshold(minima, alpha, n_boot=N_BOOTSTRAP, seed=SEED):
    """Largest q whose image-cluster bootstrap upper-95% false-pass risk <= alpha."""
    by_image = defaultdict(list)
    for row in minima:
        value = as_float(row["false_min_A"])
        if math.isfinite(value):
            by_image[str(row["image_id"])].append(value)
    images = sorted(by_image)
    if not images:
        return float("-inf"), {
            "n_images": 0, "n_pairs": 0, "bootstrap_upper_risk": 0.0,
            "empirical_risk": 0.0,
        }
    grid = sorted({v for image in images for v in by_image[image]})
    totals = np.asarray([len(by_image[image]) for image in images], dtype=float)
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(images), 1.0 / len(images))
    weights = rng.multinomial(len(images), probabilities, size=n_boot).astype(np.float64)
    bootstrap_totals = weights @ totals

    cache = {}

    def risk_at(q):
        if q == float("-inf"):
            return 0.0, 0.0
        hits = np.asarray([
            sum(value <= q for value in by_image[image]) for image in images
        ], dtype=float)
        empirical = float(hits.sum() / totals.sum())
        risks = (weights @ hits) / np.maximum(bootstrap_totals, 1.0)
        upper = float(np.percentile(risks, 95))
        return empirical, upper

    lo, hi, chosen = 0, len(grid) - 1, -1
    while lo <= hi:
        mid = (lo + hi) // 2
        empirical, upper = risk_at(grid[mid])
        cache[mid] = (empirical, upper)
        if upper <= alpha:
            chosen = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if chosen < 0:
        q, empirical, upper = float("-inf"), 0.0, 0.0
    else:
        q = grid[chosen]
        empirical, upper = cache.get(chosen, risk_at(q))
    return q, {
        "n_images": len(images),
        "n_pairs": int(totals.sum()),
        "bootstrap_upper_risk": upper,
        "empirical_risk": empirical,
        "alpha": alpha,
        "n_bootstrap": n_boot,
    }


def decide_singleton(original_relation, candidate_scores, threshold):
    passing = sorted(r for r, value in candidate_scores.items() if math.isfinite(value) and value <= threshold)
    if len(passing) != 1:
        return "ABSTAIN", None
    selected = passing[0]
    if selected == original_relation:
        return "ACCEPT", selected
    return "REPAIR", selected


def empty_decision_counts():
    return {
        label: {decision: 0 for decision in ("ACCEPT", "REPAIR", "ABSTAIN")}
        for label in ("true", "false")
    }


def structural_audit(pairs, scores, duplicates):
    pair_ids = {p["pair_id"] for p in pairs}
    selected = [r for r in scores if r["pair_id"] in pair_ids]
    groups = group_by_pair(selected)
    missing_pairs = sorted(pair_ids - set(groups))
    bad_candidate_counts = {}
    truth_mismatches = []
    nonfinite = []
    for pair in pairs:
        group = groups.get(pair["pair_id"], [])
        relations = [r["candidate_r"] for r in group]
        if len(group) != len(RELATIONS) or set(relations) != set(RELATIONS):
            bad_candidate_counts[pair["pair_id"]] = len(group)
        truth = set(parse_relations(pair["present_rels"]))
        for row in group:
            expected = int(row["candidate_r"] in truth)
            if as_int(row["candidate_true"]) != expected:
                truth_mismatches.append((pair["pair_id"], row["candidate_r"]))
            if not all(math.isfinite(as_float(row[field])) for field in ("a_orig", "d_s", "d_o", "d_so")):
                nonfinite.append((pair["pair_id"], row["candidate_r"]))
    selected_pair_keys = {
        (str(p["image_id"]), str(p["subject_id"]), str(p["object_id"]))
        for p in pairs
    }
    duplicate_subset = [key for key in duplicates if key[:3] in selected_pair_keys]
    checks = {
        "pair_count": len(pairs),
        "score_rows": len(selected),
        "expected_score_rows": len(pairs) * len(RELATIONS),
        "missing_pairs": len(missing_pairs),
        "bad_candidate_sets": len(bad_candidate_counts),
        "truth_mismatches": len(truth_mismatches),
        "nonfinite_scores": len(nonfinite),
        "duplicate_score_keys": len(duplicate_subset),
        "image_split_overlap": image_split_overlap(pairs),
    }
    checks["pass"] = all([
        checks["score_rows"] == checks["expected_score_rows"],
        checks["missing_pairs"] == 0,
        checks["bad_candidate_sets"] == 0,
        checks["truth_mismatches"] == 0,
        checks["nonfinite_scores"] == 0,
        checks["duplicate_score_keys"] == 0,
        checks["image_split_overlap"] == 0,
    ])
    return checks, selected


def image_split_overlap(pairs):
    by_image = defaultdict(set)
    for pair in pairs:
        by_image[pair["image_id"]].add(pair["split"])
    return sum(len(splits) > 1 for splits in by_image.values())


def ranking_metrics(rows):
    groups = group_by_pair(rows)
    report = {}
    for name, risk_key in (("COVER", "A"), ("raw", "raw_risk")):
        top1, reciprocal = [], []
        for group in groups.values():
            ranked = sorted(group, key=lambda r: (r[risk_key], r["candidate_r"]))
            top1.append(as_int(ranked[0]["candidate_true"]))
            first_true = next((i for i, row in enumerate(ranked, 1) if as_int(row["candidate_true"]) == 1), None)
            reciprocal.append(1.0 / first_true if first_true else 0.0)
        report[name] = {
            "top1_accuracy": float(np.mean(top1)) if top1 else float("nan"),
            "mrr": float(np.mean(reciprocal)) if reciprocal else float("nan"),
            "n_pairs": len(top1),
        }
    labels = [as_int(r["candidate_true"]) for r in rows]
    report["candidate_auroc_mu"] = auroc(labels, [r["mu"] for r in rows])
    report["candidate_auroc_raw"] = auroc(labels, [r["a_orig"] for r in rows])
    return report


def closedset_baselines(pairs, rankings):
    """Baselines under exhaustive, uniformly weighted closed-set starts.

    The no-repair number is a controlled benchmark quantity, not an estimate
    of the relation distribution produced by an LVLM in the wild.
    """
    n_true_starts = sum(len(parse_relations(pair["present_rels"])) for pair in pairs)
    n_all_starts = len(pairs) * len(RELATIONS)
    return {
        "no_repair_uniform_start_accuracy": safe_ratio(n_true_starts, n_all_starts),
        "raw_always_rerank_accuracy": rankings["raw"]["top1_accuracy"],
        "cover_always_rerank_accuracy": rankings["COVER"]["top1_accuracy"],
        "oracle_top1_accuracy": 1.0 if pairs else None,
        "oracle_candidate_recall": 1.0 if pairs else None,
        "note": "No-repair weights all 15 possible starting relations equally; it is not an LVLM output prior.",
    }


def build_pair_records(pairs, scored_rows, threshold):
    score_groups = group_by_pair(scored_rows)
    records = []
    for pair in pairs:
        group = score_groups[pair["pair_id"]]
        truth = set(parse_relations(pair["present_rels"]))
        cover_scores = {r["candidate_r"]: r["A"] for r in group}
        raw_scores = {r["candidate_r"]: r["raw_risk"] for r in group}
        passing = sorted(r for r, value in cover_scores.items() if math.isfinite(value) and value <= threshold)
        selected = passing[0] if len(passing) == 1 else None
        selected_true = int(selected in truth) if selected else 0
        n_true, n_false = len(truth), len(RELATIONS) - len(truth)
        singleton = int(selected is not None)
        repair_total = (len(RELATIONS) - 1) if selected else 0
        record = {
            "pair_id": pair["pair_id"],
            "image_id": pair["image_id"],
            "present_rels": "|".join(sorted(truth, key=RELATIONS.index)),
            "passing_rels": "|".join(passing),
            "selected_relation": selected or "",
            "selected_true": selected_true,
            "singleton": singleton,
            "candidate_set_size": len(passing),
            "false_candidate_accepted": int(any(r not in truth for r in passing)),
            "cover_top1_true": int(min(cover_scores, key=lambda r: (cover_scores[r], r)) in truth),
            "raw_top1_true": int(min(raw_scores, key=lambda r: (raw_scores[r], r)) in truth),
            "true_starts": n_true,
            "false_starts": n_false,
            "repair_total": repair_total,
            "repair_correct": repair_total * selected_true,
            "successful_repairs": n_false * selected_true if selected else 0,
            "corruptions": n_true * (1 - selected_true) if selected else 0,
            "exact_preserved": selected_true,
            "semantic_preserved": n_true * selected_true if selected else 0,
            "emitted": len(RELATIONS) * singleton,
            "emitted_correct": len(RELATIONS) * selected_true if selected else 0,
            "abstentions": len(RELATIONS) * (1 - singleton),
            "all_starts": len(RELATIONS),
        }
        records.append(record)
    return records


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def aggregate_pair_records(records):
    sums = Counter()
    for row in records:
        for key in (
            "singleton", "false_candidate_accepted", "cover_top1_true", "raw_top1_true",
            "true_starts", "false_starts", "repair_total", "repair_correct",
            "successful_repairs", "corruptions", "exact_preserved",
            "semantic_preserved", "emitted", "emitted_correct", "abstentions", "all_starts",
        ):
            sums[key] += row[key]
    n_pairs = len(records)
    return {
        "n_pairs": n_pairs,
        "singleton_pairs": sums["singleton"],
        "singleton_coverage": safe_ratio(sums["singleton"], n_pairs),
        "pair_false_acceptance_risk": safe_ratio(sums["false_candidate_accepted"], n_pairs),
        "cover_top1_accuracy": safe_ratio(sums["cover_top1_true"], n_pairs),
        "raw_top1_accuracy": safe_ratio(sums["raw_top1_true"], n_pairs),
        "repair_precision": safe_ratio(sums["repair_correct"], sums["repair_total"]),
        "repair_recall": safe_ratio(sums["successful_repairs"], sums["false_starts"]),
        "successful_repairs": sums["successful_repairs"],
        "introduced_errors": sums["corruptions"],
        "net_errors_reduced": sums["successful_repairs"] - sums["corruptions"],
        "semantic_corruption_rate": safe_ratio(sums["corruptions"], sums["true_starts"]),
        "exact_preservation": safe_ratio(sums["exact_preserved"], sums["true_starts"]),
        "semantic_preservation": safe_ratio(sums["semantic_preserved"], sums["true_starts"]),
        "coverage": safe_ratio(sums["emitted"], sums["all_starts"]),
        "abstention_rate": safe_ratio(sums["abstentions"], sums["all_starts"]),
        "selective_accuracy": safe_ratio(sums["emitted_correct"], sums["emitted"]),
    }


def bootstrap_confidence_intervals(records, n_boot=N_BOOTSTRAP, seed=SEED):
    by_image = defaultdict(list)
    for row in records:
        by_image[row["image_id"]].append(row)
    images = sorted(by_image)
    if not images:
        return {}
    keys = (
        "singleton_coverage", "pair_false_acceptance_risk", "cover_top1_accuracy",
        "raw_top1_accuracy", "repair_precision", "repair_recall",
        "semantic_corruption_rate", "semantic_preservation", "coverage",
        "selective_accuracy",
    )
    values = {key: [] for key in keys}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        sample = rng.choice(images, len(images), replace=True)
        sampled_records = [row for image in sample for row in by_image[image]]
        metrics = aggregate_pair_records(sampled_records)
        for key in keys:
            value = metrics[key]
            if value is not None and math.isfinite(value):
                values[key].append(value)
    return {
        key: {
            "low": float(np.percentile(sample, 2.5)),
            "high": float(np.percentile(sample, 97.5)),
        }
        for key, sample in values.items() if sample
    }


def relation_diagnostics(eval_pairs, scored_rows):
    groups = group_by_pair(scored_rows)
    per_relation = {}
    for relation in RELATIONS:
        subset = [p for p in eval_pairs if relation in set(parse_relations(p["present_rels"]))]
        ranks, top1 = [], []
        for pair in subset:
            ranked = sorted(groups[pair["pair_id"]], key=lambda r: (r["A"], r["candidate_r"]))
            rank = next(i for i, row in enumerate(ranked, 1) if row["candidate_r"] == relation)
            ranks.append(rank)
            top1.append(int(rank == 1))
        per_relation[relation] = {
            "n_pairs": len(subset),
            "mean_rank": float(np.mean(ranks)) if ranks else None,
            "relation_is_top1": float(np.mean(top1)) if top1 else None,
        }

    families = {}
    for name, family in (("non_spatial", NON_SPATIAL_RELATIONS), ("spatial", SPATIAL_RELATIONS)):
        subset_ids = {
            p["pair_id"] for p in eval_pairs
            if set(parse_relations(p["present_rels"])) & family
        }
        rows = [r for r in scored_rows if r["pair_id"] in subset_ids]
        families[name] = ranking_metrics(rows) if rows else {"n_pairs": 0}
    return {"per_relation": per_relation, "families": families}


def write_pair_decisions(path, records):
    fields = (
        "pair_id", "image_id", "present_rels", "passing_rels",
        "candidate_set_size", "selected_relation", "selected_true",
        "singleton", "false_candidate_accepted", "cover_top1_true", "raw_top1_true",
    )
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def gate_checks(pairs, structural, rankings, primary_metrics):
    split_counts = Counter(p["split"] for p in pairs)
    pilot_eval = [p for p in pairs if p["split"] == "pilot_evaluation"]
    relation_coverage = set(r for p in pilot_eval for r in parse_relations(p["present_rels"]))
    checks = {
        "structural_audit": structural["pass"],
        "exactly_150_pairs": len(pairs) == 150,
        "exactly_50_per_split": all(split_counts[name] == 50 for name in (
            "standardization", "risk_calibration", "pilot_evaluation"
        )),
        "pilot_eval_covers_15_relations": relation_coverage == set(RELATIONS),
        "candidate_auroc_mu_gt_0_5": rankings["candidate_auroc_mu"] > 0.5,
        "cover_top1_ge_raw": rankings["COVER"]["top1_accuracy"] >= rankings["raw"]["top1_accuracy"],
        "at_least_10_singleton_pairs": primary_metrics["singleton_pairs"] >= 10,
        "repair_precision_ge_0_80": (
            primary_metrics["repair_precision"] is not None
            and primary_metrics["repair_precision"] >= 0.80
        ),
        "semantic_corruption_le_0_10": (
            primary_metrics["semantic_corruption_rate"] is not None
            and primary_metrics["semantic_corruption_rate"] <= 0.10
        ),
        "net_error_reduction_positive": primary_metrics["net_errors_reduced"] > 0,
    }
    return checks


def analyze(stage, outdir, n_boot=N_BOOTSTRAP, seed=SEED):
    outdir = Path(outdir)
    pair_path = outdir / STAGE_FILES[stage]
    score_path = outdir / "scored_candidates.csv"
    if stage == "full":
        gate_path = outdir / "pilot_gate.json"
        if not gate_path.exists() or json.load(gate_path.open(encoding="utf-8")).get("status") != "PASS":
            raise RuntimeError("full analysis is blocked until pilot_gate.json has status PASS")
    pairs = load_pairs(pair_path)
    scores, duplicates = load_scores(score_path)
    structural, selected_scores = structural_audit(pairs, scores, duplicates)
    if stage == "smoke":
        report = {"stage": stage, "status": "PASS" if structural["pass"] else "FAIL", "structural": structural}
        with (outdir / "smoke_report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return report
    if not structural["pass"]:
        report = {"stage": stage, "status": "FAIL", "structural": structural}
        target = outdir / ("pilot_gate.json" if stage == "pilot" else "analysis_full.json")
        with target.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return report

    standardization = [r for r in selected_scores if r["split"] == "standardization"]
    bg = fit_bg(standardization)
    if not bg:
        report = {
            "stage": stage, "status": "FAIL", "structural": structural,
            "failure": "no negative-control operator passed g_min",
        }
        target = outdir / ("pilot_gate.json" if stage == "pilot" else "analysis_full.json")
        with target.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return report

    scored = add_cover_scores(selected_scores, bg)
    for row in scored:
        row["raw_risk"] = -row["a_orig"]
    risk_rows = [r for r in scored if r["split"] == "risk_calibration"]
    eval_split = "pilot_evaluation" if stage == "pilot" else "final_evaluation"
    eval_pairs = [p for p in pairs if p["split"] == eval_split]
    eval_ids = {p["pair_id"] for p in eval_pairs}
    eval_rows = [r for r in scored if r["pair_id"] in eval_ids]
    rankings = ranking_metrics(eval_rows)
    thresholds, decisions = {}, {}
    minima = pair_false_minima(risk_rows)
    for alpha in ALPHAS:
        q, calibration = cluster_bootstrap_threshold(minima, alpha, n_boot=n_boot, seed=seed)
        records = build_pair_records(eval_pairs, eval_rows, q)
        metrics = aggregate_pair_records(records)
        thresholds[f"a{alpha:g}"] = {
            "q": q if math.isfinite(q) else None,
            "calibration": calibration,
        }
        decisions[f"a{alpha:g}"] = metrics
        if alpha == PRIMARY_ALPHA:
            primary_records = records
            primary_metrics = metrics

    report = {
        "stage": stage,
        "seed": seed,
        "g_min": G_MIN,
        "alphas": list(ALPHAS),
        "primary_alpha": PRIMARY_ALPHA,
        "score_definition": "A=-mu; J diagnostic only",
        "structural": structural,
        "operators": bg,
        "ranking": rankings,
        "baselines": closedset_baselines(eval_pairs, rankings),
        "thresholds": thresholds,
        "decisions": decisions,
        "diagnostics": relation_diagnostics(eval_pairs, eval_rows),
        "bootstrap_95_ci": bootstrap_confidence_intervals(primary_records, n_boot=n_boot, seed=seed),
        "runtime": {
            "scored_candidates": len(selected_scores),
            "total_candidate_seconds": float(np.sum([as_float(r.get("elapsed_sec")) for r in selected_scores
                                                       if math.isfinite(as_float(r.get("elapsed_sec")))])),
            "mean_candidate_seconds": float(np.mean([as_float(r.get("elapsed_sec")) for r in selected_scores
                                                       if math.isfinite(as_float(r.get("elapsed_sec")))]))
            if any(math.isfinite(as_float(r.get("elapsed_sec"))) for r in selected_scores) else None,
        },
        "claim_boundary": (
            "Ground-truth boxes and a closed 15-relation GQA vocabulary; this does not by itself "
            "establish mitigation of free-form LVLM generations."
        ),
    }
    write_pair_decisions(outdir / f"repair_decisions_{stage}.csv", primary_records)
    if stage == "pilot":
        checks = gate_checks(pairs, structural, rankings, primary_metrics)
        report["checks"] = checks
        report["status"] = "PASS" if all(checks.values()) else "FAIL"
        target = outdir / "pilot_gate.json"
    else:
        report["status"] = "COMPLETE"
        target = outdir / "analysis_full.json"
    with target.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("smoke", "pilot", "full"))
    parser.add_argument("--outdir", default=str(BASE))
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    report = analyze(args.stage, args.outdir, n_boot=args.n_bootstrap, seed=args.seed)
    print(f"[info] stage={args.stage} status={report['status']}", flush=True)
    if args.stage == "pilot" and report["status"] != "PASS":
        print("[done] pilot failed pre-registered gate; full experiment remains blocked", flush=True)
        raise SystemExit(2)
    if args.stage == "smoke" and report["status"] != "PASS":
        print("[done] smoke structural audit failed", flush=True)
        raise SystemExit(2)
    print(f"[done] analysis written under {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
