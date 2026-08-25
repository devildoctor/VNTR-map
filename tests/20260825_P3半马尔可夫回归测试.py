#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "程序" / "20260825_P3半马尔可夫概率分解与置信度.py"
INPUT = ROOT / "测试数据" / "IRF2BPL.HPRC.fa.3034e58.11fba48.904d69e.smooth.final.gfa"
P1 = (
    ROOT
    / "outputs"
    / "IRF2BPL"
    / "20260825_P1唯一侧翼锚点与共识边界"
    / "20260825_IRF2BPL_P1唯一侧翼锚点与共识边界_路径边界共识.tsv"
)
P2_DIR = ROOT / "outputs" / "IRF2BPL" / "20260825_P2从头发现主体motif与MDL分解"
P2_PREFIX = "20260825_IRF2BPL_P2从头发现主体motif与MDL分解"


def load_program():
    spec = importlib.util.spec_from_file_location("vntr_p3_test", PROGRAM)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {PROGRAM}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_tsv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


P3 = load_program()


def simple_model(canonical: str, duration_mean: float = 12.0):
    transitions = {
        ("START", canonical): 0.90,
        ("START", "C"): 0.10,
        (canonical, "C"): 0.20,
        (canonical, "END"): 0.80,
        ("C", canonical): 0.80,
        ("C", "END"): 0.20,
    }
    return P3.ProbabilityModel(
        exact_probability={canonical: 0.95},
        variant_probability={canonical: 0.05},
        duration_mean={canonical: duration_mean},
        duration_sd={canonical: 2.0},
        insertion_run_mean={canonical: 4.0},
        complex_run_mean=30.0,
        transition_probability=transitions,
        training_paths=("synthetic_train",),
    )


def motif_observation(path: str, index: int, start: int, motif: str):
    return P3.Observation(path, index, start, start + len(motif), "exact_motif", motif, motif, motif, 0)


class P3SemiMarkovUnitTest(unittest.TestCase):
    def test_path_suffix_weights_are_parsed_and_validated(self):
        self.assertEqual(
            P3.path_weights(["seq1_158.0", "seq2_37.5"], "suffix"),
            {"seq1_158.0": 158.0, "seq2_37.5": 37.5},
        )
        self.assertEqual(P3.path_weights(["seq1", "seq2"], "equal"), {"seq1": 1.0, "seq2": 1.0})
        with self.assertRaisesRegex(ValueError, "numeric suffix weight"):
            P3.path_weights(["seq_without_weight"], "suffix")
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            P3.path_weights(["seq_0"], "suffix")

    def test_weighted_split_keeps_alleles_intact_and_balances_effective_weight(self):
        names = ["seq1_158.0", "seq2_54.5", "seq3_37.5", "seq4_29.0"] + [
            f"seq{index}_1.0" for index in range(5, 26)
        ]
        weights = P3.path_weights(names, "suffix")
        roles, hashes = P3.split_paths(names, weights)
        totals = Counter()
        for name, role in roles.items():
            totals[role] += weights[name]
        self.assertEqual(set(roles), set(names))
        self.assertEqual(set(hashes), set(names))
        self.assertEqual(set(totals), {"train", "tune", "test"})
        self.assertEqual(Counter(roles.values()), Counter({"train": 15, "tune": 5, "test": 5}))
        self.assertEqual(roles["seq1_158.0"], "train")
        self.assertGreater(totals["train"], totals["tune"])
        self.assertGreater(totals["train"], totals["test"])

    def test_short_insertion_is_absorbed_without_losing_sequence(self):
        family = P3.Family("M1", "CAG", 3, "CAG")
        observations = []
        position = 0
        for index in range(1, 7):
            observations.append(motif_observation("insertion", index, position, "CAG"))
            position += 3
        observations.append(P3.Observation("insertion", 7, position, position + 4, "background", "TTTT", "", "", 0))
        position += 4
        for index in range(8, 14):
            observations.append(motif_observation("insertion", index, position, "CAG"))
            position += 3

        decoded = P3.decode_path(
            "insertion",
            "test",
            observations,
            [family],
            simple_model("CAG"),
            duration_weight=1.0,
            min_copies=4,
            min_density=0.60,
            max_insert_bp=12,
            max_total_insert_bp=18,
        )
        self.assertEqual(len(decoded.blocks), 1)
        block = decoded.blocks[0].segment
        self.assertEqual(block.motif_copies, 12)
        self.assertEqual(block.inserted_bp, 4)
        self.assertEqual(block.insertion_parts, ((18, "TTTT"),))
        self.assertEqual(block.start_bp, 0)
        self.assertEqual(block.end_bp, position)
        self.assertTrue(all(abs(sum(row.values()) - 1.0) < 1e-8 for row in decoded.token_posterior))
        self.assertNotEqual(decoded.second_log_score, P3.NEG_INF)

    def test_long_complex_gap_keeps_two_position_blocks(self):
        family = P3.Family("M1", "CAG", 3, "CAG")
        observations = []
        position = 0
        for index in range(1, 7):
            observations.append(motif_observation("two_blocks", index, position, "CAG"))
            position += 3
        observations.append(P3.Observation("two_blocks", 7, position, position + 50, "background", "A" * 50, "", "", 0))
        position += 50
        for index in range(8, 14):
            observations.append(motif_observation("two_blocks", index, position, "CAG"))
            position += 3

        decoded = P3.decode_path(
            "two_blocks",
            "test",
            observations,
            [family],
            simple_model("CAG", duration_mean=6.0),
            duration_weight=1.0,
            min_copies=4,
            min_density=0.60,
            max_insert_bp=12,
            max_total_insert_bp=18,
        )
        self.assertEqual(len(decoded.blocks), 2)
        self.assertEqual([block.segment.motif_copies for block in decoded.blocks], [6, 6])
        self.assertLessEqual(decoded.blocks[0].segment.end_bp, decoded.blocks[1].segment.start_bp)

    def test_short_foreign_motif_run_is_an_explicit_interruption(self):
        ccg = P3.Family("M2", "CCG", 3, "GCG")
        agc = P3.Family("M1", "AGC", 3, "GCA")
        observations = []
        position = 0
        for index in range(1, 7):
            observations.append(motif_observation("foreign", index, position, "CCG"))
            position += 3
        for index in range(7, 9):
            observations.append(motif_observation("foreign", index, position, "AGC"))
            position += 3
        for index in range(9, 17):
            observations.append(motif_observation("foreign", index, position, "CCG"))
            position += 3

        model = simple_model("CCG", duration_mean=14.0)
        model.exact_probability["AGC"] = 0.95
        model.variant_probability["AGC"] = 0.05
        model.duration_mean["AGC"] = 8.0
        model.duration_sd["AGC"] = 2.0
        model.insertion_run_mean["AGC"] = 3.0
        model.transition_probability.update(
            {
                ("START", "CCG"): 0.85,
                ("START", "AGC"): 0.05,
                ("CCG", "C"): 0.15,
                ("CCG", "AGC"): 0.05,
                ("AGC", "CCG"): 0.80,
                ("AGC", "C"): 0.10,
                ("AGC", "END"): 0.10,
                ("C", "CCG"): 0.75,
                ("C", "AGC"): 0.05,
            }
        )
        decoded = P3.decode_path(
            "foreign",
            "test",
            observations,
            [agc, ccg],
            model,
            duration_weight=1.0,
            min_copies=4,
            min_density=0.60,
            max_insert_bp=12,
            max_total_insert_bp=18,
        )
        self.assertEqual(len(decoded.blocks), 1)
        block = decoded.blocks[0].segment
        self.assertEqual(block.state, "CCG")
        self.assertEqual(block.motif_copies, 14)
        self.assertEqual(block.inserted_bp, 6)
        self.assertEqual(block.insertion_parts, ((18, "AGC"), (21, "AGC")))

    def test_isolated_motif_does_not_form_repeat_block(self):
        family = P3.Family("M1", "CAG", 3, "CAG")
        observations = [
            P3.Observation("isolated", 1, 0, 10, "background", "A" * 10, "", "", 0),
            motif_observation("isolated", 2, 10, "CAG"),
            P3.Observation("isolated", 3, 13, 23, "background", "T" * 10, "", "", 0),
        ]
        decoded = P3.decode_path(
            "isolated",
            "test",
            observations,
            [family],
            simple_model("CAG"),
            duration_weight=1.0,
            min_copies=4,
            min_density=0.60,
            max_insert_bp=12,
            max_total_insert_bp=18,
        )
        self.assertEqual(decoded.blocks, [])
        self.assertEqual(P3.family_sequence(decoded.best_segments), ())


class P3IRF2BPLEndToEndTest(unittest.TestCase):
    def test_probability_outputs_holdout_and_gfa_reconstruction(self):
        with tempfile.TemporaryDirectory(prefix="vntr_p3_") as temporary:
            outdir = Path(temporary)
            prefix = "p3_test"
            subprocess.run(
                [
                    sys.executable,
                    str(PROGRAM),
                    "--input",
                    str(INPUT),
                    "--p1-boundaries",
                    str(P1),
                    "--p2-dictionary",
                    str(P2_DIR / f"{P2_PREFIX}_共享motif字典.tsv"),
                    "--p2-tokens",
                    str(P2_DIR / f"{P2_PREFIX}_逐路径motif_token.tsv"),
                    "--p2-blocks",
                    str(P2_DIR / f"{P2_PREFIX}_逐路径repeat区块.tsv"),
                    "--outdir",
                    str(outdir),
                    "--prefix",
                    prefix,
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
            )

            qa = {row["metric"]: row["value"] for row in read_tsv(outdir / f"{prefix}_P3验证汇总.tsv")}
            self.assertEqual(qa["path_count"], "47")
            self.assertEqual(qa["training_path_count"], "28")
            self.assertEqual(qa["tuning_path_count"], "9")
            self.assertEqual(qa["heldout_path_count"], "10")
            self.assertEqual(qa["selected_duration_weight"], "1.0000")
            self.assertEqual(qa["repeat_block_count"], "141")
            self.assertEqual(qa["modal_repeat_block_count_per_path"], "3")
            self.assertEqual(qa["paths_with_modal_repeat_block_count"], "47")
            self.assertEqual(qa["posterior_available_paths"], "47")
            self.assertEqual(qa["second_best_available_paths"], "47")
            self.assertEqual(qa["explicit_loop_positions"], "R1,R2,R3")
            self.assertEqual(qa["exact_gfa_reconstruction_paths"], "47")

            split = read_tsv(outdir / f"{prefix}_训练调参与留出划分.tsv")
            self.assertEqual(Counter(row["split"] for row in split), Counter({"train": 28, "test": 10, "tune": 9}))

            tokens = read_tsv(outdir / f"{prefix}_逐路径token后验.tsv")
            self.assertTrue(tokens)
            posterior_columns = [column for column in tokens[0] if column.startswith("posterior_") and column != "posterior_entropy_bits"]
            for row in tokens:
                self.assertAlmostEqual(sum(float(row[column]) for column in posterior_columns), 1.0, places=6)

            gfa = (outdir / f"{prefix}_P3半马尔可夫多环图.gfa").read_text(encoding="utf-8").splitlines()
            self.assertIn("TS:Z:p3_hidden_semi_markov_assembly_only", gfa[0])
            self.assertEqual(sum(line.startswith("P\t") for line in gfa), 47)
            self.assertTrue(all("\tVS:Z:assembly_only_probabilistic" in line for line in gfa if line.startswith("P\t")))


if __name__ == "__main__":
    unittest.main()
