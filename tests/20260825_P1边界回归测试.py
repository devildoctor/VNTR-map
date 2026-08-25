#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "程序" / "20260825_P1唯一侧翼锚点与共识边界.py"
INPUT = ROOT / "测试数据" / "IRF2BPL.HPRC.fa.3034e58.11fba48.904d69e.smooth.final.gfa"


def read_tsv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class P1BoundaryRegressionTest(unittest.TestCase):
    def test_irf2bpl_boundaries_and_reconstruction(self):
        with tempfile.TemporaryDirectory(prefix="vntr_p1_") as temporary:
            outdir = Path(temporary)
            prefix = "p1_test"
            subprocess.run(
                [
                    sys.executable,
                    str(PROGRAM),
                    "--input",
                    str(INPUT),
                    "--outdir",
                    str(outdir),
                    "--prefix",
                    prefix,
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            qa = {row["metric"]: row["value"] for row in read_tsv(outdir / f"{prefix}_P1验证汇总.tsv")}
            self.assertEqual(qa["path_count"], "47")
            self.assertEqual(qa["unique_left_anchor_paths"], "47")
            self.assertEqual(qa["unique_right_anchor_paths"], "47")
            self.assertEqual(qa["ordered_anchor_paths"], "47")
            self.assertEqual(qa["internally_supported_boundary_paths"], "47")
            self.assertEqual(qa["within_one_motif_boundary_paths"], "47")
            self.assertEqual(qa["exact_locus_reconstruction_paths"], "47")

            boundaries = read_tsv(outdir / f"{prefix}_路径边界共识.tsv")
            self.assertEqual(len(boundaries), 47)
            self.assertTrue(all(row["boundary_status"] == "provisional_internal_consensus" for row in boundaries))
            self.assertTrue(all(row["structural_term"] == "complex/mosaic TR" for row in boundaries))
            self.assertTrue(all(int(row["left_anchor_end_bp"]) < int(row["consensus_start_bp"]) for row in boundaries))
            self.assertTrue(all(int(row["consensus_end_bp"]) < int(row["right_anchor_start_bp"]) for row in boundaries))

            gfa_lines = (outdir / f"{prefix}_P1边界标注多环图.gfa").read_text(encoding="utf-8").splitlines()
            self.assertIn("P1:Z:unique_flank_consensus_boundary", gfa_lines[0])
            path_lines = [line for line in gfa_lines if line.startswith("P\t")]
            self.assertEqual(len(path_lines), 47)
            self.assertTrue(all("\tRS:i:" in line and "\tRE:i:" in line and "\tTT:Z:" in line for line in path_lines))

            fasta_headers = sum(
                line.startswith(">")
                for line in (outdir / f"{prefix}_路径序列.fa").read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual(fasta_headers, 47)


if __name__ == "__main__":
    unittest.main()
