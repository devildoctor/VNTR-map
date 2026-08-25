#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "程序" / "20260825_P2从头发现主体motif与MDL分解.py"
INPUT = ROOT / "测试数据" / "IRF2BPL.HPRC.fa.3034e58.11fba48.904d69e.smooth.final.gfa"
BOUNDARIES = (
    ROOT
    / "outputs"
    / "IRF2BPL"
    / "20260825_P1唯一侧翼锚点与共识边界"
    / "20260825_IRF2BPL_P1唯一侧翼锚点与共识边界_路径边界共识.tsv"
)


def load_program():
    spec = importlib.util.spec_from_file_location("vntr_p2_test", PROGRAM)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {PROGRAM}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_tsv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


P2 = load_program()


class P2PrimitiveMotifUnitTest(unittest.TestCase):
    def test_rotation_reverse_complement_and_primitive_root(self):
        cag_family = P2.canonical_motif("CAG")
        self.assertEqual(cag_family, P2.canonical_motif("AGC"))
        self.assertEqual(cag_family, P2.canonical_motif("GCA"))
        self.assertEqual(cag_family, P2.canonical_motif("CTG"))
        self.assertEqual("CAG", P2.primitive_root("CAGCAG"))

        gcc_family = P2.canonical_motif("GCC")
        self.assertEqual(gcc_family, P2.canonical_motif("GCG"))
        self.assertNotEqual(cag_family, gcc_family)

    def test_local_alignment_keeps_clear_insertion_explicit(self):
        sequence = "GCC" * 4 + "TTTTTTT" + "GCC" * 4
        region = P2.Region("synthetic_insertion", sequence, 0, len(sequence))
        _score, actions = P2.align_family_interval(region, 0, len(sequence), P2.canonical_motif("GCC"))
        inserted = "".join(action.observed for action in actions if action.kind == "background")
        motifs = [action for action in actions if action.kind != "background"]
        self.assertEqual(inserted, "TTTTTTT")
        self.assertEqual(len(motifs), 8)
        self.assertTrue(all(action.kind == "exact_motif" for action in motifs))

    def test_abab_is_reported_as_higher_order_not_long_primary(self):
        left = P2.canonical_motif("CAG")
        right = P2.canonical_motif("GCC")
        actions = []
        position = 0
        for _cycle in range(5):
            for observed, canonical in (("CAG", left), ("GCC", right)):
                actions.append(
                    P2.Action(
                        "synthetic_ab",
                        position,
                        position + 3,
                        "exact_motif",
                        observed,
                        canonical,
                        observed,
                        0,
                    )
                )
                position += 3
        ids = {left: "M1", right: "M2"}
        rows = P2.higher_order_rows({"synthetic_ab": actions}, ids)
        self.assertTrue(any(int(row["pattern_token_length"]) == 2 and int(row["cycle_count"]) == 5 for row in rows))
        self.assertTrue(all(row["status"] == "higher_order_candidate_not_primary_override" for row in rows))


class P2IRF2BPLEndToEndTest(unittest.TestCase):
    def test_de_novo_dictionary_blocks_and_gfa_reconstruction(self):
        with tempfile.TemporaryDirectory(prefix="vntr_p2_") as temporary:
            outdir = Path(temporary)
            prefix = "p2_test"
            subprocess.run(
                [
                    sys.executable,
                    str(PROGRAM),
                    "--input",
                    str(INPUT),
                    "--p1-boundaries",
                    str(BOUNDARIES),
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

            qa = {row["metric"]: row["value"] for row in read_tsv(outdir / f"{prefix}_P2验证汇总.tsv")}
            self.assertEqual(qa["path_count"], "47")
            self.assertEqual(qa["selected_primitive_family_count"], "2")
            self.assertEqual(qa["selected_periods_bp"], "3,3")
            self.assertEqual(qa["selected_composite_family_count"], "0")
            self.assertEqual(qa["path_order_stable_dictionary"], "1")
            self.assertEqual(qa["repeat_block_count"], "141")
            self.assertEqual(qa["modal_repeat_block_count_per_path"], "3")
            self.assertEqual(qa["paths_with_modal_repeat_block_count"], "47")
            self.assertEqual(qa["exact_gfa_reconstruction_paths"], "47")
            self.assertEqual(qa["explicit_loop_positions"], "R1,R2,R3")

            dictionary = read_tsv(outdir / f"{prefix}_共享motif字典.tsv")
            expected = {P2.canonical_motif("CAG"), P2.canonical_motif("GCC")}
            self.assertEqual({row["canonical_family"] for row in dictionary}, expected)
            self.assertTrue(all(row["path_support"] == "47" for row in dictionary))

            blocks = read_tsv(outdir / f"{prefix}_逐路径repeat区块.tsv")
            self.assertEqual(len(blocks), 141)
            by_path = {}
            for row in blocks:
                by_path.setdefault(row["path"], []).append(row)
            self.assertEqual(len(by_path), 47)
            self.assertTrue(all([row["repeat_block"] for row in rows] == ["R1", "R2", "R3"] for rows in by_path.values()))
            self.assertTrue(all(rows[0]["family_id"] == rows[2]["family_id"] != rows[1]["family_id"] for rows in by_path.values()))

            gfa = (outdir / f"{prefix}_P2_de_novo_motif图.gfa").read_text(encoding="utf-8").splitlines()
            self.assertIn("TS:Z:p2_de_novo_primitive_motif_mdl", gfa[0])
            self.assertEqual(sum(line.startswith("P\t") for line in gfa), 47)
            self.assertTrue(any("\tRL:i:1" in line for line in gfa if line.startswith("L\t")))


if __name__ == "__main__":
    unittest.main()
