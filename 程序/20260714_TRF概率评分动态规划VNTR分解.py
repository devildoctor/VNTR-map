#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path


BASE_SCRIPT = Path(__file__).with_name("20260714_三状态动态规划VNTR分解与图建模.py")
DEFAULT_INPUT = Path("测试数据") / "IRF2BPL.HPRC.fa.3034e58.11fba48.904d69e.smooth.final.gfa"
DEFAULT_OUTDIR = Path("outputs") / "IRF2BPL" / "20260714_TRF概率评分动态规划"
DEFAULT_PREFIX = "20260714_IRF2BPL_TRF概率评分"

TRF_MATCH_PROBABILITY = 0.80
TRF_INDEL_PROBABILITY = 0.10
ENTER_REPEAT_PRIOR = 0.02
SWITCH_REPEAT_PRIOR = 0.01
BACKGROUND_BASE_PROBABILITY = 0.25
MIN_RING_COPIES = 4
MIN_CORE_COPIES = 5
MIN_RING_DENSITY = 0.75
MAX_LOCAL_INSERT_BP = 12


def load_base_module(name: str):
    spec = importlib.util.spec_from_file_location(name, BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base program: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.MIN_REPEAT_UNITS = MIN_RING_COPIES
    module.MIN_CORE_UNITS = MIN_CORE_COPIES
    module.MIN_REPEAT_DENSITY = MIN_RING_DENSITY
    module.MAX_LOCAL_INSERT_BP = MAX_LOCAL_INSERT_BP
    return module


def emission_log_odds(observed: str, canonical: str) -> float:
    mismatch_probability = (1.0 - TRF_MATCH_PROBABILITY) / 3.0
    score = 0.0
    for observed_base, canonical_base in zip(observed, canonical):
        probability = TRF_MATCH_PROBABILITY if observed_base == canonical_base else mismatch_probability
        score += math.log(probability / BACKGROUND_BASE_PROBABILITY)
    return score


def transition_log_odds(base, previous: str, current: str) -> float:
    if previous == current:
        return 0.0
    if previous == base.STATE_INSERT and current != base.STATE_INSERT:
        return math.log(ENTER_REPEAT_PRIOR / (1.0 - ENTER_REPEAT_PRIOR))
    if previous != base.STATE_INSERT and current == base.STATE_INSERT:
        return math.log(TRF_INDEL_PROBABILITY / (1.0 - TRF_INDEL_PROBABILITY))
    return math.log(SWITCH_REPEAT_PRIOR / (1.0 - SWITCH_REPEAT_PRIOR))


def probabilistic_viterbi(base, region: str, offset: int):
    length = len(region)
    scores: list[dict[str, float]] = [dict() for _ in range(length + 1)]
    back: dict[tuple[int, str], tuple[int, str, object]] = {}
    scores[0][base.STATE_INSERT] = 0.0

    def update(end: int, state: str, score: float, previous_pos: int, previous_state: str, atom) -> None:
        if score > scores[end].get(state, float("-inf")):
            scores[end][state] = score
            back[(end, state)] = (previous_pos, previous_state, atom)

    for pos in range(length):
        if not scores[pos]:
            continue
        for previous_state, previous_score in list(scores[pos].items()):
            insertion = base.Atom(
                base.STATE_INSERT,
                offset + pos,
                offset + pos + 1,
                region[pos : pos + 1],
                "insertion",
            )
            update(
                pos + 1,
                base.STATE_INSERT,
                previous_score + transition_log_odds(base, previous_state, base.STATE_INSERT),
                pos,
                previous_state,
                insertion,
            )

            if pos + 3 > length:
                continue
            observed = region[pos : pos + 3]
            for group, state in base.GROUP_TO_STATE.items():
                call = base.call_for_group(observed, group)
                if call is None:
                    continue
                atom = base.Atom(state, offset + pos, offset + pos + 3, observed, call.kind, call.canonical)
                score = (
                    previous_score
                    + transition_log_odds(base, previous_state, state)
                    + emission_log_odds(observed, call.canonical)
                )
                update(pos + 3, state, score, pos, previous_state, atom)

    state = max(scores[length], key=scores[length].get)
    pos = length
    atoms = []
    while pos > 0:
        previous_pos, previous_state, atom = back[(pos, state)]
        atoms.append(atom)
        pos = previous_pos
        state = previous_state
    atoms.reverse()
    return atoms


def configure_probabilistic_dp(base) -> None:
    base.viterbi_three_state = lambda region, offset: probabilistic_viterbi(base, region, offset)


def block_score(base, block, atoms) -> tuple[float, float]:
    selected = [atom for atom in atoms if block.start <= atom.start and atom.end <= block.end]
    repeat_atoms = [atom for atom in selected if atom.state != base.STATE_INSERT]
    aligned_bases = 3 * len(repeat_atoms)
    matched_bases = sum(3 if atom.kind == "exact" else 2 for atom in repeat_atoms)
    identity = matched_bases / aligned_bases if aligned_bases else 0.0
    log_odds = 0.0
    previous = base.STATE_INSERT
    for atom in selected:
        log_odds += transition_log_odds(base, previous, atom.state)
        if atom.state != base.STATE_INSERT:
            log_odds += emission_log_odds(atom.observed, atom.canonical)
        previous = atom.state
    return identity, log_odds


def augmented_detail_rows(base, detail_rows, blocks_by_path, atoms_by_path):
    block_lookup = {
        (path_name, block.index): block
        for path_name, blocks in blocks_by_path.items()
        for block in blocks
    }
    rows = []
    for row in detail_rows:
        block = block_lookup[(str(row["path"]), int(row["block_index"]))]
        identity, log_odds = block_score(base, block, atoms_by_path[str(row["path"])])
        enriched = dict(row)
        enriched["TRF_pM"] = f"{TRF_MATCH_PROBABILITY:.2f}"
        enriched["TRF_pI"] = f"{TRF_INDEL_PROBABILITY:.2f}"
        enriched["motif_identity"] = f"{identity:.4f}"
        enriched["TRF_log_odds"] = f"{log_odds:.4f}"
        rows.append(enriched)
    return rows


def range_text(rows, column: str) -> str:
    values = [int(row[column]) for row in rows]
    return f"{min(values)}-{max(values)}"


def block_range(blocks_by_path, block_index: int, attribute: str) -> str:
    values = [int(getattr(blocks[block_index - 1], attribute)) for blocks in blocks_by_path.values()]
    return f"{min(values)}-{max(values)}"


def block_length_range(blocks_by_path, block_index: int) -> str:
    values = [
        int(blocks[block_index - 1].end - blocks[block_index - 1].start)
        for blocks in blocks_by_path.values()
    ]
    return f"{min(values)}-{max(values)}"


def comparison_rows(fixed_summary, fixed_blocks, probabilistic_summary, probabilistic_blocks):
    fixed_by_path = {str(row["path"]): row for row in fixed_summary}
    changed_paths = sum(
        fixed_by_path[str(row["path"])]["block_signature"] != row["block_signature"]
        for row in probabilistic_summary
    )
    return [
        {"metric": "path_count", "fixed_score": len(fixed_summary), "TRF_probability_score": len(probabilistic_summary)},
        {
            "metric": "block_count",
            "fixed_score": ",".join(str(value) for value in sorted({int(row['block_count']) for row in fixed_summary})),
            "TRF_probability_score": ",".join(
                str(value) for value in sorted({int(row['block_count']) for row in probabilistic_summary})
            ),
        },
        {
            "metric": "CAG_like_units",
            "fixed_score": range_text(fixed_summary, "CAG_like_units"),
            "TRF_probability_score": range_text(probabilistic_summary, "CAG_like_units"),
        },
        {
            "metric": "GCC_like_units",
            "fixed_score": range_text(fixed_summary, "GCC_like_units"),
            "TRF_probability_score": range_text(probabilistic_summary, "GCC_like_units"),
        },
        {
            "metric": "variant_units",
            "fixed_score": range_text(fixed_summary, "variant_units"),
            "TRF_probability_score": range_text(probabilistic_summary, "variant_units"),
        },
        {
            "metric": "central_complex_bp",
            "fixed_score": block_length_range(fixed_blocks, 4),
            "TRF_probability_score": block_length_range(probabilistic_blocks, 4),
        },
        {
            "metric": "L3_repeat_units",
            "fixed_score": block_range(fixed_blocks, 5, "repeat_units"),
            "TRF_probability_score": block_range(probabilistic_blocks, 5, "repeat_units"),
        },
        {
            "metric": "changed_block_signature_paths",
            "fixed_score": 0,
            "TRF_probability_score": changed_paths,
        },
    ]


def per_path_comparison(fixed_summary, probability_summary):
    fixed = {str(row["path"]): row for row in fixed_summary}
    rows = []
    for row in probability_summary:
        path = str(row["path"])
        rows.append(
            {
                "path": path,
                "fixed_CAG_units": fixed[path]["CAG_like_units"],
                "probability_CAG_units": row["CAG_like_units"],
                "fixed_GCC_units": fixed[path]["GCC_like_units"],
                "probability_GCC_units": row["GCC_like_units"],
                "fixed_variant_units": fixed[path]["variant_units"],
                "probability_variant_units": row["variant_units"],
                "fixed_signature": fixed[path]["block_signature"],
                "probability_signature": row["block_signature"],
            }
        )
    return rows


def run_analysis(input_path: Path):
    fixed = load_base_module("vntr_fixed_score_comparison")
    _segments, fixed_paths = fixed.read_gfa(input_path)
    fixed_summary, _fixed_detail, _fixed_evidence, fixed_blocks, _fixed_atoms = fixed.analyze(fixed_paths)

    probability = load_base_module("vntr_trf_probability")
    configure_probabilistic_dp(probability)
    _segments, paths = probability.read_gfa(input_path)
    summary, details, evidence, blocks, atoms = probability.analyze(paths)
    return (
        probability,
        paths,
        summary,
        details,
        evidence,
        blocks,
        atoms,
        fixed_summary,
        fixed_blocks,
    )


def validate(base, paths, summary_rows, blocks_by_path, atoms_by_path) -> None:
    summary = {str(row["path"]): row for row in summary_rows}
    for record in paths:
        start = int(summary[record.name]["region_start_bp"])
        end = int(summary[record.name]["region_end_bp"])
        expected = record.seq[start:end]
        if "".join(atom.observed for atom in atoms_by_path[record.name]) != expected:
            raise AssertionError(f"Atom reconstruction failed: {record.name}")
        if "".join(block.seq for block in blocks_by_path[record.name]) != expected:
            raise AssertionError(f"Block reconstruction failed: {record.name}")
        for block in blocks_by_path[record.name]:
            if block.group != "nonrepeat" and (
                block.repeat_units < MIN_RING_COPIES or block.repeat_density < MIN_RING_DENSITY
            ):
                raise AssertionError(f"Loop threshold failed: {record.name} block {block.index}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TRF-probability-scored three-state VNTR decomposition.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    (
        base,
        paths,
        summary,
        details,
        evidence,
        blocks,
        atoms,
        fixed_summary,
        fixed_blocks,
    ) = run_analysis(args.input)
    validate(base, paths, summary, blocks, atoms)
    enriched_details = augmented_detail_rows(base, details, blocks, atoms)

    summary_path = args.outdir / f"{args.prefix}_路径汇总.tsv"
    detail_path = args.outdir / f"{args.prefix}_block明细.tsv"
    evidence_path = args.outdir / f"{args.prefix}_主体motif证据.tsv"
    comparison_path = args.outdir / f"{args.prefix}_评分模型汇总对比.tsv"
    path_comparison_path = args.outdir / f"{args.prefix}_逐路径评分对比.tsv"
    gfa_path = args.outdir / f"{args.prefix}_状态路径图.gfa"
    png_path = args.outdir / f"{args.prefix}_block图.png"

    base.write_tsv(
        summary_path,
        summary,
        [
            "path", "path_suffix_label", "region_start_bp", "region_end_bp", "region_bp", "primary_motifs",
            "block_count", "CAG_like_units", "GCC_like_units", "exact_units", "variant_units", "inserted_bp",
            "switches", "primary_blocks", "variant_blocks", "interrupted_blocks", "complex_blocks", "block_signature",
        ],
    )
    base.write_tsv(
        detail_path,
        enriched_details,
        [
            "path", "block_index", "role", "group", "start_bp", "end_bp", "bp", "nodes", "anchor_supported",
            "repeat_units", "exact_units", "variant_units", "inserted_bp", "repeat_density", "motif_identity",
            "TRF_log_odds", "TRF_pM", "TRF_pI", "motif_counts", "variant_counts", "insertions", "sequence",
        ],
    )
    base.write_tsv(
        evidence_path,
        evidence,
        ["group", "canonical_motif", "observed_motif", "type", "anchor_path_count", "anchor_units"],
    )
    base.write_tsv(
        comparison_path,
        comparison_rows(fixed_summary, fixed_blocks, summary, blocks),
        ["metric", "fixed_score", "TRF_probability_score"],
    )
    base.write_tsv(
        path_comparison_path,
        per_path_comparison(fixed_summary, summary),
        [
            "path", "fixed_CAG_units", "probability_CAG_units", "fixed_GCC_units", "probability_GCC_units",
            "fixed_variant_units", "probability_variant_units", "fixed_signature", "probability_signature",
        ],
    )
    base.write_state_gfa(gfa_path, atoms, summary)
    base.plot_blocks(png_path, summary, enriched_details)

    print(f"Paths: {len(paths)}")
    print(f"CAG-like units: {range_text(summary, 'CAG_like_units')}")
    print(f"GCC-like units: {range_text(summary, 'GCC_like_units')}")
    print(f"Variant units: {range_text(summary, 'variant_units')}")
    for output in (
        summary_path,
        detail_path,
        evidence_path,
        comparison_path,
        path_comparison_path,
        gfa_path,
        png_path,
    ):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
