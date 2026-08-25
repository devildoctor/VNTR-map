#!/usr/bin/env python3

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
P4_PROGRAM = ROOT / "程序" / "20260825_P4变异read证据分级.py"
P5_PROGRAM = ROOT / "程序" / "20260825_P5位置特异SCC循环图.py"
HPRC_P3 = ROOT / "outputs" / "IRF2BPL" / "20260825_P3半马尔可夫概率分解与置信度"
HPRC_P3_PREFIX = "20260825_IRF2BPL_P3半马尔可夫概率分解与置信度"


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def qa_values(path: Path) -> dict[str, str]:
    return {row["metric"]: row["value"] for row in read_tsv(path)}


TOKEN_FIELDS = [
    "path", "split", "token_index", "start_bp", "end_bp", "observed", "p2_token_type",
    "p2_family", "p3_block_state", "p3_emission_state", "selected_state_posterior",
    "posterior_entropy_bits", "coordinate_system", "posterior_complex", "posterior_AGC",
]

BLOCK_FIELDS = [
    "path", "split", "repeat_block", "start_bp", "end_bp", "start_ci_low_bp",
    "start_ci_high_bp", "end_ci_low_bp", "end_ci_high_bp", "family_id", "canonical_family",
    "oriented_motif", "copies", "exact_copies", "variant_copies", "inserted_bp", "insertions",
    "repeat_density", "segment_posterior", "mean_token_posterior", "min_token_posterior",
    "confidence", "interpretation_status", "coordinate_system",
]


class P4ReadEvidenceTest(unittest.TestCase):
    def test_read_evidence_separates_high_confidence_and_error_like(self):
        with tempfile.TemporaryDirectory(prefix="vntr_p4_synthetic_") as temporary:
            root = Path(temporary)
            gfa = root / "input.gfa"
            tokens = root / "tokens.tsv"
            blocks = root / "blocks.tsv"
            first_out = root / "assembly_only"
            second_out = root / "read_backed"
            gfa.write_text(
                "\n".join(
                    [
                        "H\tVN:Z:1.0\tTS:Z:synthetic",
                        "S\tB0_a\tA\tBT:Z:background_or_flank",
                        "S\tR1_exact_CAG\tCAG\tLC:Z:R1\tCM:Z:AGC\tOM:Z:CAG\tBT:Z:exact_motif",
                        "S\tR1_variant_CAA\tCAA\tLC:Z:R1\tCM:Z:AGC\tOM:Z:CAG\tBT:Z:variant_motif",
                        "S\tR1_INS_T\tT\tLC:Z:R1\tBT:Z:explicit_insertion_or_complex",
                        "S\tB1_c\tC\tBT:Z:background_or_flank",
                        "P\tp1\tB0_a+,R1_exact_CAG+,R1_variant_CAA+,R1_exact_CAG+,R1_INS_T+,B1_c+\t*",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            token_rows = [
                (1, 1, 4, "CAG", "exact_motif", "AGC"),
                (2, 4, 7, "CAA", "variant_motif", "AGC"),
                (3, 7, 10, "CAG", "exact_motif", "AGC"),
                (4, 10, 11, "T", "local_insertion", "-"),
            ]
            write_tsv(
                tokens,
                [
                    {
                        "path": "p1", "split": "train", "token_index": index, "start_bp": start,
                        "end_bp": end, "observed": observed, "p2_token_type": "background" if emission == "local_insertion" else emission,
                        "p2_family": family, "p3_block_state": "AGC", "p3_emission_state": emission,
                        "selected_state_posterior": "0.99", "posterior_entropy_bits": "0.01",
                        "coordinate_system": "0-based_half-open", "posterior_complex": "0.01", "posterior_AGC": "0.99",
                    }
                    for index, start, end, observed, emission, family in token_rows
                ],
                TOKEN_FIELDS,
            )
            block = {
                "path": "p1", "split": "train", "repeat_block": "R1", "start_bp": 1, "end_bp": 11,
                "start_ci_low_bp": 1, "start_ci_high_bp": 1, "end_ci_low_bp": 11, "end_ci_high_bp": 11,
                "family_id": "M1", "canonical_family": "AGC", "oriented_motif": "CAG", "copies": 3,
                "exact_copies": 2, "variant_copies": 1, "inserted_bp": 1, "insertions": "10:T",
                "repeat_density": "0.9", "segment_posterior": "0.99", "mean_token_posterior": "0.99",
                "min_token_posterior": "0.99", "confidence": "high",
                "interpretation_status": "assembly_only_probabilistic", "coordinate_system": "0-based_half-open",
            }
            write_tsv(blocks, [block], BLOCK_FIELDS)

            subprocess.run(
                [sys.executable, str(P4_PROGRAM), "--p3-gfa", str(gfa), "--p3-tokens", str(tokens),
                 "--p3-blocks", str(blocks), "--outdir", str(first_out), "--prefix", "p4"],
                cwd=ROOT, check=True, capture_output=True,
            )
            assembly_events = read_tsv(first_out / "p4_P4组装事件与read判定.tsv")
            self.assertEqual(len(assembly_events), 2)
            self.assertEqual({row["validation_status"] for row in assembly_events}, {"assembly_only_unvalidated"})

            evidence_rows = []
            for event in assembly_events:
                for index in range(1, 7):
                    is_variant = event["event_type"] == "motif_substitution"
                    evidence_rows.append(
                        {
                            "event_id": event["event_id"], "path": event["path"], "repeat_block": event["repeat_block"],
                            "start_bp": event["start_bp"], "end_bp": event["end_bp"], "event_type": event["event_type"],
                            "read_id": f"{event['event_id']}_r{index}", "platform": "HiFi",
                            "strand": "+" if index % 2 else "-", "mapq": 60, "base_quality": 35,
                            "spanning": 1, "supports_event": int(is_variant and index <= 4), "caller": "GraphAligner",
                        }
                    )
            evidence = root / "evidence.tsv"
            write_tsv(evidence, evidence_rows, list(evidence_rows[0]))
            subprocess.run(
                [sys.executable, str(P4_PROGRAM), "--p3-gfa", str(gfa), "--p3-tokens", str(tokens),
                 "--p3-blocks", str(blocks), "--read-evidence", str(evidence),
                 "--outdir", str(second_out), "--prefix", "p4"],
                cwd=ROOT, check=True, capture_output=True,
            )
            validated = read_tsv(second_out / "p4_P4组装事件与read判定.tsv")
            status_by_type = {row["event_type"]: row["validation_status"] for row in validated}
            self.assertEqual(status_by_type["motif_substitution"], "read_backed_high_confidence")
            self.assertEqual(status_by_type["local_insertion"], "error_like")
            qa = qa_values(second_out / "p4_P4验证汇总.tsv")
            self.assertEqual(qa["exact_gfa_reconstruction_paths"], "1")


class P5GraphTheoryTest(unittest.TestCase):
    def test_abab_is_one_multinode_scc_and_positions_stay_separate(self):
        with tempfile.TemporaryDirectory(prefix="vntr_p5_synthetic_") as temporary:
            root = Path(temporary)
            gfa = root / "input.gfa"
            blocks = root / "blocks.tsv"
            outdir = root / "out"
            gfa.write_text(
                "\n".join(
                    [
                        "H\tVN:Z:1.0\tTS:Z:synthetic",
                        "S\tB0_a\tA\tBT:Z:background_or_flank",
                        "S\tR1_A\tAAA\tLC:Z:R1\tCM:Z:AAA\tBT:Z:exact_motif",
                        "S\tR1_B\tCCC\tLC:Z:R1\tCM:Z:CCC\tBT:Z:exact_motif",
                        "S\tR2_A\tAAA\tLC:Z:R2\tCM:Z:AAA\tBT:Z:exact_motif",
                        "S\tB2_c\tC\tBT:Z:background_or_flank",
                        "P\tp1\tB0_a+,R1_A+,R1_B+,R1_A+,R1_B+,R2_A+,R2_A+,B2_c+\t*",
                        "P\tp2\tB0_a+,R1_A+,R1_B+,R1_A+,R1_B+,R2_A+,R2_A+,B2_c+\t*",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows = []
            for path in ("p1", "p2"):
                for repeat_block, family, copies in (("R1", "AAA", 4), ("R2", "AAA", 2)):
                    rows.append(
                        {
                            "path": path, "split": "train", "repeat_block": repeat_block,
                            "start_bp": 1, "end_bp": 13, "start_ci_low_bp": 1, "start_ci_high_bp": 1,
                            "end_ci_low_bp": 13, "end_ci_high_bp": 13, "family_id": "M1",
                            "canonical_family": family, "oriented_motif": family, "copies": copies,
                            "exact_copies": copies, "variant_copies": 0, "inserted_bp": 0, "insertions": "-",
                            "repeat_density": 1, "segment_posterior": 1, "mean_token_posterior": 1,
                            "min_token_posterior": 1, "confidence": "high",
                            "interpretation_status": "assembly_only_probabilistic", "coordinate_system": "0-based_half-open",
                        }
                    )
            write_tsv(blocks, rows, BLOCK_FIELDS)
            subprocess.run(
                [sys.executable, str(P5_PROGRAM), "--input-gfa", str(gfa), "--p3-blocks", str(blocks),
                 "--outdir", str(outdir), "--prefix", "p5"],
                cwd=ROOT, check=True, capture_output=True,
            )
            qa = qa_values(outdir / "p5_P5验证汇总.tsv")
            self.assertEqual(qa["cyclic_scc_count"], "2")
            self.assertEqual(qa["repeat_locations_with_cyclic_scc"], "R1,R2")
            self.assertEqual(qa["cross_position_cycle_count"], "0")
            self.assertEqual(qa["eligible_high_order_cycle_count"], "1")
            self.assertEqual(qa["exact_gfa_reconstruction_paths"], "2")
            candidates = read_tsv(outdir / "p5_P5_高阶循环候选.tsv")
            self.assertEqual(candidates[0]["pattern"], "AAA>CCC")
            scc_rows = read_tsv(outdir / "p5_P5_SCC目录.tsv")
            r1 = [row for row in scc_rows if row["primary_for_location"] == "R1"]
            r2 = [row for row in scc_rows if row["primary_for_location"] == "R2"]
            self.assertEqual(len(r1), 1)
            self.assertEqual(r1[0]["segment_count"], "2")
            self.assertNotEqual(r1[0]["scc_id"], r2[0]["scc_id"])


class P4P5HPRCEndToEndTest(unittest.TestCase):
    def test_hprc_assembly_only_and_scc_invariants(self):
        with tempfile.TemporaryDirectory(prefix="vntr_p4p5_hprc_") as temporary:
            root = Path(temporary)
            p4_out = root / "P4"
            p5_out = root / "P5"
            p3_gfa = HPRC_P3 / f"{HPRC_P3_PREFIX}_P3半马尔可夫多环图.gfa"
            p3_tokens = HPRC_P3 / f"{HPRC_P3_PREFIX}_逐路径token后验.tsv"
            p3_blocks = HPRC_P3 / f"{HPRC_P3_PREFIX}_逐路径P3_repeat区块.tsv"
            subprocess.run(
                [sys.executable, str(P4_PROGRAM), "--p3-gfa", str(p3_gfa), "--p3-tokens", str(p3_tokens),
                 "--p3-blocks", str(p3_blocks), "--outdir", str(p4_out), "--prefix", "p4"],
                cwd=ROOT, check=True, capture_output=True,
            )
            p4_qa = qa_values(p4_out / "p4_P4验证汇总.tsv")
            self.assertEqual(p4_qa["path_count"], "47")
            self.assertEqual(p4_qa["exact_gfa_reconstruction_paths"], "47")
            self.assertEqual(p4_qa["assembly_event_count"], p4_qa["assembly_only_unvalidated_event_count"])
            subprocess.run(
                [sys.executable, str(P5_PROGRAM), "--input-gfa", str(p4_out / "p4_P4变异证据标注图.gfa"),
                 "--p3-blocks", str(p3_blocks), "--outdir", str(p5_out), "--prefix", "p5"],
                cwd=ROOT, check=True, capture_output=True,
            )
            p5_qa = qa_values(p5_out / "p5_P5验证汇总.tsv")
            self.assertEqual(p5_qa["path_count"], "47")
            self.assertEqual(p5_qa["path_supported_edge_count"], "47")
            self.assertEqual(p5_qa["cyclic_scc_count"], "3")
            self.assertEqual(p5_qa["repeat_locations_with_cyclic_scc"], "R1,R2,R3")
            self.assertEqual(p5_qa["repeat_locations_without_cyclic_scc"], "-")
            self.assertEqual(p5_qa["cross_position_cycle_count"], "0")
            self.assertEqual(p5_qa["paths_with_backward_repeat_order"], "0")
            self.assertEqual(p5_qa["exact_gfa_reconstruction_paths"], "47")


if __name__ == "__main__":
    unittest.main()
