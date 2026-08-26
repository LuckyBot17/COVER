#!/usr/bin/env python3
"""Regression tests for the COVER closed-set repair experiment."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import analyze_gqa_closedset as analyze
import prepare_gqa_closedset as prepare
import score_gqa_closedset as score


def fixture_scene_graph():
    """Small graph where every subject has a different outgoing relation."""
    return {
        "img-a": {
            "width": 100,
            "height": 100,
            "objects": {
                "1": {
                    "name": "person",
                    "x": 5, "y": 5, "w": 20, "h": 20,
                    "relations": [
                        {"name": "holding", "object": "3"},
                        {"name": "watching", "object": "3"},
                    ],
                },
                "2": {
                    "name": "person",
                    "x": 35, "y": 5, "w": 20, "h": 20,
                    "relations": [{"name": "near", "object": "3"}],
                },
                "3": {
                    "name": "ball",
                    "x": 60, "y": 40, "w": 20, "h": 20,
                    "relations": [],
                },
            },
        }
    }


class PreparationTests(unittest.TestCase):
    def test_subject_relations_are_read_from_current_object(self):
        pairs, audit = prepare.extract_pairs(
            fixture_scene_graph(), img_root=None, require_images=False
        )
        keyed = {(p["subject_id"], p["object_id"]): set(p["present_rels"]) for p in pairs}
        self.assertEqual(keyed[("1", "3")], {"holding", "watching"})
        self.assertEqual(keyed[("2", "3")], {"near"})
        self.assertEqual(audit["edge_lookup_failures"], 0)

    def test_same_named_objects_remain_distinct_by_object_id(self):
        pairs, _ = prepare.extract_pairs(
            fixture_scene_graph(), img_root=None, require_images=False
        )
        ids = {(p["subject_id"], p["object_id"]) for p in pairs}
        self.assertIn(("1", "3"), ids)
        self.assertIn(("2", "3"), ids)
        self.assertEqual(len(ids), 2)

    def test_candidate_truth_is_multilabel_and_exhaustive(self):
        pairs, _ = prepare.extract_pairs(
            fixture_scene_graph(), img_root=None, require_images=False
        )
        pair = next(p for p in pairs if p["subject_id"] == "1")
        rows = prepare.candidate_rows([pair])
        self.assertEqual(len(rows), 15)
        truth = {r["candidate_r"] for r in rows if r["candidate_true"] == 1}
        self.assertEqual(truth, {"holding", "watching"})
        self.assertTrue(any(r["candidate_true"] == 0 for r in rows))

    def test_image_splits_are_deterministic_and_disjoint(self):
        pairs = []
        for i in range(40):
            pairs.append({
                "pair_id": f"img-{i}|s|o",
                "image_id": f"img-{i}",
                "present_rels": ["near"],
            })
            pairs.append({
                "pair_id": f"img-{i}|x|y",
                "image_id": f"img-{i}",
                "present_rels": ["holding"],
            })
        a = prepare.assign_image_splits(pairs, seed=42)
        b = prepare.assign_image_splits(pairs, seed=42)
        self.assertEqual(
            [(p["pair_id"], p["split"]) for p in a],
            [(p["pair_id"], p["split"]) for p in b],
        )
        by_image = {}
        for p in a:
            by_image.setdefault(p["image_id"], set()).add(p["split"])
        self.assertTrue(all(len(v) == 1 for v in by_image.values()))
        self.assertEqual(
            set().union(*(set(v) for v in by_image.values())),
            {"standardization", "risk_calibration", "pilot_evaluation", "final_evaluation"},
        )


class ScoringCheckpointTests(unittest.TestCase):
    def test_existing_checkpoint_rows_are_not_pending(self):
        candidates = [
            {"image_id": "i", "subject_id": "s", "object_id": "o", "candidate_r": "near"},
            {"image_id": "i", "subject_id": "s", "object_id": "o", "candidate_r": "holding"},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "scores.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(candidates[0]))
                writer.writeheader()
                writer.writerow(candidates[0])
            done = score.load_done_keys(path)
            pending = score.pending_candidates(candidates, done)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["candidate_r"], "holding")

    def test_checkpoint_rejects_mixing_mask_modes(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "scores.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["mask"])
                writer.writeheader()
                writer.writerow({"mask": "blur"})
            with self.assertRaisesRegex(ValueError, "mask mode"):
                score.validate_checkpoint_mask(path, "gray")


class DecisionTests(unittest.TestCase):
    def test_pair_false_minimum_uses_only_false_candidates(self):
        rows = [
            {"pair_id": "p", "image_id": "i", "candidate_true": 1, "A": -9.0},
            {"pair_id": "p", "image_id": "i", "candidate_true": 0, "A": 0.4},
            {"pair_id": "p", "image_id": "i", "candidate_true": 0, "A": 0.2},
        ]
        minima = analyze.pair_false_minima(rows)
        self.assertEqual(minima, [{"pair_id": "p", "image_id": "i", "false_min_A": 0.2}])

    def test_singleton_policy_accepts_repairs_or_abstains(self):
        scores = {"near": 0.1, "holding": 0.8, "watching": 0.9}
        self.assertEqual(analyze.decide_singleton("near", scores, 0.2), ("ACCEPT", "near"))
        self.assertEqual(analyze.decide_singleton("holding", scores, 0.2), ("REPAIR", "near"))
        self.assertEqual(analyze.decide_singleton("holding", scores, 0.05), ("ABSTAIN", None))
        self.assertEqual(analyze.decide_singleton("holding", scores, 0.85), ("ABSTAIN", None))

    def test_decision_count_tables_do_not_share_nested_state(self):
        counts = analyze.empty_decision_counts()
        counts["true"]["ACCEPT"] += 1
        self.assertEqual(counts["false"]["ACCEPT"], 0)

    def test_pair_level_threshold_respects_cluster_upper_risk(self):
        minima = [
            {"image_id": f"i{i}", "false_min_A": float(i)} for i in range(20)
        ]
        q, detail = analyze.cluster_bootstrap_threshold(
            minima, alpha=0.20, n_boot=300, seed=42
        )
        self.assertLessEqual(detail["bootstrap_upper_risk"], 0.20)
        self.assertIn(q, [m["false_min_A"] for m in minima] + [float("-inf")])

    def test_stage_audit_ignores_duplicates_from_unselected_pair_in_same_image(self):
        pair = {
            "pair_id": "i|s|o", "image_id": "i", "subject_id": "s", "object_id": "o",
            "split": "pilot_evaluation", "present_rels": ["near"],
        }
        scores = []
        for relation in prepare.RELATIONS:
            scores.append({
                "pair_id": pair["pair_id"], "image_id": "i", "subject_id": "s",
                "object_id": "o", "candidate_r": relation,
                "candidate_true": int(relation == "near"),
                "a_orig": 1.0, "d_s": 1.0, "d_o": 1.0, "d_so": 1.0,
            })
        duplicates = [("i", "different-subject", "different-object", "near")]
        audit, _ = analyze.structural_audit([pair], scores, duplicates)
        self.assertEqual(audit["duplicate_score_keys"], 0)
        self.assertTrue(audit["pass"])


class SyntheticAnalysisTests(unittest.TestCase):
    def test_fit_standardization_and_candidate_scores(self):
        rows = []
        for label, base in ((0, 0.0), (1, 2.0)):
            for i in range(4):
                rows.append({
                    "candidate_true": label,
                    "d_s": base + i * 0.01,
                    "d_o": base + i * 0.01,
                    "d_so": base + i * 0.01,
                })
        bg = analyze.fit_bg(rows, g_min=0.01)
        self.assertEqual(set(bg), {"s", "o", "so"})
        scored = analyze.add_cover_scores(rows, bg)
        true_mu = [r["mu"] for r in scored if r["candidate_true"] == 1]
        false_mu = [r["mu"] for r in scored if r["candidate_true"] == 0]
        self.assertGreater(min(true_mu), max(false_mu))
        self.assertTrue(all(r["A"] == -r["mu"] for r in scored))

    def test_closedset_baselines_include_no_repair_raw_cover_and_oracle(self):
        pairs = [
            {"present_rels": ["holding"]},
            {"present_rels": ["near", "next to"]},
        ]
        rankings = {
            "raw": {"top1_accuracy": 0.5},
            "COVER": {"top1_accuracy": 1.0},
        }
        report = analyze.closedset_baselines(pairs, rankings)
        self.assertEqual(report["no_repair_uniform_start_accuracy"], 3 / 30)
        self.assertEqual(report["raw_always_rerank_accuracy"], 0.5)
        self.assertEqual(report["cover_always_rerank_accuracy"], 1.0)
        self.assertEqual(report["oracle_top1_accuracy"], 1.0)
        self.assertEqual(report["oracle_candidate_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
