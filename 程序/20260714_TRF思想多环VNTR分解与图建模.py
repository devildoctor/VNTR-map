#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle


BASE_SCRIPT = Path(__file__).with_name("20260714_三状态动态规划VNTR分解与图建模.py")
DEFAULT_INPUT = Path("测试数据") / "IRF2BPL.HPRC.fa.3034e58.11fba48.904d69e.smooth.final.gfa"
DEFAULT_OUTDIR = Path("outputs") / "IRF2BPL" / "20260714_TRF思想多环VNTR试验"
DEFAULT_PREFIX = "20260714_IRF2BPL_TRF思想多环"

MIN_RING_COPIES = 4
MIN_CORE_COPIES = 5
MIN_RING_DENSITY = 0.75
MAX_LOCAL_INSERT_BP = 12
MAX_PERIOD = 18
TRF_MATCH = 2
TRF_MISMATCH = -7
TRF_INDEL = -7


@dataclass
class PeriodCandidate:
    path: str
    loop_id: str
    block_index: int
    group: str
    start: int
    end: int
    period: int
    copies: float
    consensus: str
    identity: float
    indel_rate: float
    tuple_support: float
    alignment_score: int
    selected: bool = False


def load_base_module():
    spec = importlib.util.spec_from_file_location("vntr_dp_base", BASE_SCRIPT)
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


def consensus_for_period(seq: str, period: int) -> str:
    columns: list[str] = []
    for phase in range(period):
        counts = Counter(seq[phase::period])
        columns.append(counts.most_common(1)[0][0] if counts else "N")
    return "".join(columns)


def tuple_support(seq: str, period: int, tuple_size: int = 2) -> float:
    comparable = 0
    matches = 0
    for pos in range(period, len(seq) - tuple_size + 1):
        comparable += 1
        if seq[pos : pos + tuple_size] == seq[pos - period : pos - period + tuple_size]:
            matches += 1
    return matches / comparable if comparable else 0.0


def score_period(
    path_name: str,
    loop_id: str,
    block,
    repeat_seq: str,
    period: int,
) -> PeriodCandidate:
    consensus = consensus_for_period(repeat_seq, period)
    matches = sum(base == consensus[index % period] for index, base in enumerate(repeat_seq))
    mismatches = len(repeat_seq) - matches
    inserted = block.inserted_bp
    score = TRF_MATCH * matches + TRF_MISMATCH * mismatches + TRF_INDEL * inserted
    return PeriodCandidate(
        path=path_name,
        loop_id=loop_id,
        block_index=block.index,
        group=block.group,
        start=block.start,
        end=block.end,
        period=period,
        copies=len(repeat_seq) / period,
        consensus=consensus,
        identity=matches / max(1, len(repeat_seq)),
        indel_rate=inserted / max(1, block.end - block.start),
        tuple_support=tuple_support(repeat_seq, period),
        alignment_score=score,
    )


def choose_primitive_candidate(candidates: list[PeriodCandidate]) -> PeriodCandidate:
    max_identity = max(candidate.identity for candidate in candidates)
    near_best = [candidate for candidate in candidates if candidate.identity >= max_identity - 0.02]
    return min(near_best, key=lambda candidate: (candidate.period, -candidate.tuple_support, -candidate.alignment_score))


def discover_period_candidates(base, blocks_by_path, atoms_by_path):
    all_candidates: list[PeriodCandidate] = []
    selected_by_path_loop: dict[tuple[str, str], PeriodCandidate] = {}
    loop_blocks_by_path: dict[str, list[tuple[str, object]]] = {}

    for path_name, blocks in blocks_by_path.items():
        loop_number = 0
        loop_blocks: list[tuple[str, object]] = []
        for block in blocks:
            if block.group == "nonrepeat":
                continue
            loop_number += 1
            loop_id = f"L{loop_number}"
            loop_blocks.append((loop_id, block))
            repeat_atoms = [
                atom
                for atom in atoms_by_path[path_name]
                if atom.state != base.STATE_INSERT
                and block.start <= atom.start
                and atom.end <= block.end
                and atom.group == block.group
            ]
            repeat_seq = "".join(atom.observed for atom in repeat_atoms)
            max_period = min(MAX_PERIOD, len(repeat_seq) // MIN_RING_COPIES)
            candidates = []
            for period in range(1, max_period + 1):
                candidate = score_period(path_name, loop_id, block, repeat_seq, period)
                if candidate.copies >= MIN_RING_COPIES and candidate.identity >= MIN_RING_DENSITY:
                    candidates.append(candidate)
            if not candidates:
                continue
            selected = choose_primitive_candidate(candidates)
            selected.selected = True
            selected_by_path_loop[(path_name, loop_id)] = selected
            all_candidates.extend(candidates)
        loop_blocks_by_path[path_name] = loop_blocks
    return all_candidates, selected_by_path_loop, loop_blocks_by_path


def build_loop_catalog(selected_by_path_loop, loop_blocks_by_path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    loop_ids = sorted({loop_id for _path, loop_id in selected_by_path_loop})
    for loop_id in loop_ids:
        selected = [candidate for (path, lid), candidate in selected_by_path_loop.items() if lid == loop_id]
        blocks = [
            block
            for path_name, loop_blocks in loop_blocks_by_path.items()
            for lid, block in loop_blocks
            if lid == loop_id and (path_name, lid) in selected_by_path_loop
        ]
        period_counts = Counter(candidate.period for candidate in selected)
        period = period_counts.most_common(1)[0][0]
        motif_counts = Counter(candidate.consensus for candidate in selected if candidate.period == period)
        consensus = motif_counts.most_common(1)[0][0]
        copies = [block.repeat_units for block in blocks]
        variants = [block.variant_units for block in blocks]
        insertions = [block.inserted_bp for block in blocks]
        rows.append(
            {
                "loop_id": loop_id,
                "group": selected[0].group,
                "path_support": len(selected),
                "selected_period": period,
                "consensus_motif": consensus,
                "copy_min": min(copies),
                "copy_median": f"{statistics.median(copies):.1f}",
                "copy_max": max(copies),
                "variant_min": min(variants),
                "variant_max": max(variants),
                "inserted_bp_min": min(insertions),
                "inserted_bp_max": max(insertions),
                "mean_identity": f"{statistics.mean(candidate.identity for candidate in selected):.4f}",
                "mean_indel_rate": f"{statistics.mean(candidate.indel_rate for candidate in selected):.4f}",
                "mean_tuple_support": f"{statistics.mean(candidate.tuple_support for candidate in selected):.4f}",
            }
        )
    return rows


def candidate_rows(candidates: list[PeriodCandidate]) -> list[dict[str, object]]:
    return [
        {
            "path": candidate.path,
            "loop_id": candidate.loop_id,
            "block_index": candidate.block_index,
            "group": candidate.group,
            "start_bp": candidate.start,
            "end_bp": candidate.end,
            "period": candidate.period,
            "copies": f"{candidate.copies:.3f}",
            "consensus": candidate.consensus,
            "identity": f"{candidate.identity:.4f}",
            "indel_rate": f"{candidate.indel_rate:.4f}",
            "tuple_support": f"{candidate.tuple_support:.4f}",
            "alignment_score": candidate.alignment_score,
            "selected": int(candidate.selected),
        }
        for candidate in candidates
    ]


def block_loop_map(loop_blocks: list[tuple[str, object]]) -> dict[int, str]:
    return {block.index: loop_id for loop_id, block in loop_blocks}


def atom_block(atom, blocks):
    for block in blocks:
        if block.start <= atom.start and atom.end <= block.end:
            return block
    raise ValueError(f"Atom {atom.start}-{atom.end} does not belong to a block")


def graph_node_for_atom(base, atom, block, loop_id: str | None) -> tuple[str, str, list[str]]:
    if block.group == "nonrepeat":
        digest = hashlib.sha1(atom.observed.encode("ascii")).hexdigest()[:12]
        node_id = f"B{block.index}_INS_{digest}"
        tags = [f"LC:Z:B{block.index}", "BT:Z:complex_or_flank", "GR:Z:nonrepeat"]
        return node_id, atom.observed, tags

    if loop_id is None:
        raise ValueError("Repeat block is missing a loop id")
    if atom.state == base.STATE_INSERT:
        digest = hashlib.sha1(atom.observed.encode("ascii")).hexdigest()[:12]
        node_id = f"{loop_id}_INS_{digest}"
        tags = [f"LC:Z:{loop_id}", "BT:Z:loop_insertion", f"GR:Z:{block.group}"]
        return node_id, atom.observed, tags
    if atom.kind == "exact":
        node_id = f"{loop_id}_M_{atom.observed}"
        tags = [f"LC:Z:{loop_id}", "BT:Z:exact_motif", f"GR:Z:{block.group}", f"CA:Z:{atom.canonical}"]
    else:
        node_id = f"{loop_id}_V_{atom.canonical}_to_{atom.observed}"
        tags = [
            f"LC:Z:{loop_id}",
            "BT:Z:variant_motif",
            f"GR:Z:{block.group}",
            f"CA:Z:{atom.canonical}",
            "ED:i:1",
        ]
    return node_id, atom.observed, tags


def write_multiloop_gfa(base, path, blocks_by_path, atoms_by_path, loop_blocks_by_path, summary_rows):
    nodes: dict[str, tuple[str, list[str]]] = {}
    edges: Counter[tuple[str, str]] = Counter()
    path_steps: dict[str, list[str]] = {}
    summary_by_path = {str(row["path"]): row for row in summary_rows}

    for path_name, atoms in atoms_by_path.items():
        blocks = blocks_by_path[path_name]
        loop_map = block_loop_map(loop_blocks_by_path[path_name])
        steps: list[str] = []
        for atom in atoms:
            block = atom_block(atom, blocks)
            node_id, seq, tags = graph_node_for_atom(base, atom, block, loop_map.get(block.index))
            nodes[node_id] = (seq, tags)
            steps.append(node_id)
        path_steps[path_name] = steps
        edges.update(zip(steps, steps[1:]))

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("H\tVN:Z:1.0\tTS:Z:trf_inspired_position_specific_multiloop\n")
        for node_id in sorted(nodes):
            seq, tags = nodes[node_id]
            handle.write(f"S\t{node_id}\t{seq}\t" + "\t".join(tags) + "\n")
        for (left, right), support in sorted(edges.items()):
            handle.write(f"L\t{left}\t+\t{right}\t+\t0M\tPS:i:{support}\n")
        for path_name, steps in path_steps.items():
            overlaps = ",".join("0M" for _ in steps[1:]) if len(steps) > 1 else "*"
            summary = summary_by_path[path_name]
            tags = (
                f"CU:i:{summary['CAG_like_units']}\tGU:i:{summary['GCC_like_units']}"
                f"\tVU:i:{summary['variant_units']}\tIB:i:{summary['inserted_bp']}"
                f"\tBS:Z:{summary['block_signature']}"
            )
            handle.write(f"P\t{path_name}\t{','.join(step + '+' for step in steps)}\t{overlaps}\t{tags}\n")


def draw_loop(ax, x: float, y: float, radius: float, color: str, label: str, detail: str) -> None:
    circle = Circle((x, y), radius, facecolor=color, edgecolor="#27303f", linewidth=1.3, alpha=0.95)
    ax.add_patch(circle)
    arc = Arc((x, y), radius * 1.45, radius * 1.45, theta1=35, theta2=320, color="white", linewidth=2.0)
    ax.add_patch(arc)
    angle = math.radians(35)
    arrow_x = x + radius * 0.725 * math.cos(angle)
    arrow_y = y + radius * 0.725 * math.sin(angle)
    ax.add_patch(
        FancyArrowPatch(
            (arrow_x - 0.14, arrow_y - 0.03),
            (arrow_x, arrow_y),
            arrowstyle="-|>",
            mutation_scale=12,
            color="white",
            linewidth=1.4,
        )
    )
    ax.text(x, y + 0.12, label, ha="center", va="center", fontsize=11, color="white", fontweight="bold")
    ax.text(x, y - 0.30, detail, ha="center", va="center", fontsize=8, color="white")


def arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, color="#4b5563", linewidth=1.5)
    )


def plot_multiloop_overview(path: Path, loop_catalog: list[dict[str, object]]) -> None:
    catalog = {str(row["loop_id"]): row for row in loop_catalog}
    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.set_xlim(-0.5, 14.8)
    ax.set_ylim(-2.2, 2.3)
    ax.axis("off")

    ax.add_patch(Rectangle((0.0, -0.42), 1.25, 0.84, facecolor="#9aa1ad", edgecolor="#27303f"))
    ax.text(0.625, 0, "left flank\n27 bp", ha="center", va="center", fontsize=9, color="white")
    draw_loop(
        ax,
        3.0,
        0,
        1.05,
        "#4d9b68",
        "L1 GCC-like",
        f"{catalog['L1']['copy_min']}-{catalog['L1']['copy_max']} copies",
    )
    draw_loop(
        ax,
        6.3,
        0,
        1.05,
        "#3478b8",
        "L2 CAG-like",
        f"{catalog['L2']['copy_min']}-{catalog['L2']['copy_max']} copies",
    )
    ax.add_patch(Rectangle((8.1, -0.52), 1.55, 1.04, facecolor="#9aa1ad", edgecolor="#27303f"))
    ax.text(8.875, 0, "complex\n75 bp", ha="center", va="center", fontsize=9, color="white")
    draw_loop(
        ax,
        11.3,
        0,
        1.12,
        "#4d9b68",
        "L3 GCC-like",
        f"{catalog['L3']['copy_min']}-{catalog['L3']['copy_max']} copies",
    )
    ax.add_patch(Rectangle((13.35, -0.42), 1.0, 0.84, facecolor="#9aa1ad", edgecolor="#27303f"))
    ax.text(13.85, 0, "right\n8 bp", ha="center", va="center", fontsize=9, color="white")

    arrow(ax, (1.25, 0), (1.85, 0))
    arrow(ax, (4.05, 0), (5.15, 0))
    arrow(ax, (7.35, 0), (8.1, 0))
    arrow(ax, (9.65, 0), (10.18, 0))
    arrow(ax, (12.42, 0), (13.35, 0))

    ax.add_patch(Rectangle((10.45, -1.82), 1.7, 0.52, facecolor="#d1d5db", edgecolor="#4b5563"))
    ax.text(11.3, -1.56, "7 bp insertion", ha="center", va="center", fontsize=8)
    arrow(ax, (10.82, -1.28), (10.82, -0.96))
    arrow(ax, (11.78, -0.96), (11.78, -1.28))

    ax.text(
        7.15,
        1.92,
        "IRF2BPL TRF-inspired position-specific multi-loop graph",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )
    ax.text(
        7.15,
        1.53,
        "Main anchors >=5 copies; independent loops >=4 copies; density >=75%; variants branch within each loop instance.",
        ha="center",
        va="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def validate_reconstruction(base, paths, summary_rows, blocks_by_path, atoms_by_path) -> None:
    summary = {str(row["path"]): row for row in summary_rows}
    for record in paths:
        start = int(summary[record.name]["region_start_bp"])
        end = int(summary[record.name]["region_end_bp"])
        expected = record.seq[start:end]
        atom_seq = "".join(atom.observed for atom in atoms_by_path[record.name])
        block_seq = "".join(block.seq for block in blocks_by_path[record.name])
        if atom_seq != expected or block_seq != expected:
            raise AssertionError(f"Sequence reconstruction failed for {record.name}")
        for block in blocks_by_path[record.name]:
            if block.group != "nonrepeat" and (
                block.repeat_units < MIN_RING_COPIES or block.repeat_density < MIN_RING_DENSITY
            ):
                raise AssertionError(f"Loop threshold failed for {record.name} block {block.index}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TRF-inspired position-specific multi-loop VNTR graph trial.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    base = load_base_module()
    _segments, paths = base.read_gfa(args.input)
    summary, details, evidence, blocks_by_path, atoms_by_path = base.analyze(paths)
    validate_reconstruction(base, paths, summary, blocks_by_path, atoms_by_path)

    candidates, selected, loop_blocks = discover_period_candidates(base, blocks_by_path, atoms_by_path)
    loop_catalog = build_loop_catalog(selected, loop_blocks)

    summary_path = args.outdir / f"{args.prefix}_路径汇总.tsv"
    detail_path = args.outdir / f"{args.prefix}_block明细.tsv"
    candidate_path = args.outdir / f"{args.prefix}_TRF周期候选.tsv"
    catalog_path = args.outdir / f"{args.prefix}_多环目录.tsv"
    evidence_path = args.outdir / f"{args.prefix}_主体motif证据.tsv"
    gfa_path = args.outdir / f"{args.prefix}_位置特异多环图.gfa"
    block_png = args.outdir / f"{args.prefix}_路径block图.png"
    overview_png = args.outdir / f"{args.prefix}_多环结构总览.png"

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
        details,
        [
            "path", "block_index", "role", "group", "start_bp", "end_bp", "bp", "nodes", "anchor_supported",
            "repeat_units", "exact_units", "variant_units", "inserted_bp", "repeat_density", "motif_counts",
            "variant_counts", "insertions", "sequence",
        ],
    )
    base.write_tsv(
        evidence_path,
        evidence,
        ["group", "canonical_motif", "observed_motif", "type", "anchor_path_count", "anchor_units"],
    )
    base.write_tsv(
        candidate_path,
        candidate_rows(candidates),
        [
            "path", "loop_id", "block_index", "group", "start_bp", "end_bp", "period", "copies", "consensus",
            "identity", "indel_rate", "tuple_support", "alignment_score", "selected",
        ],
    )
    base.write_tsv(
        catalog_path,
        loop_catalog,
        [
            "loop_id", "group", "path_support", "selected_period", "consensus_motif", "copy_min", "copy_median",
            "copy_max", "variant_min", "variant_max", "inserted_bp_min", "inserted_bp_max", "mean_identity",
            "mean_indel_rate", "mean_tuple_support",
        ],
    )
    write_multiloop_gfa(base, gfa_path, blocks_by_path, atoms_by_path, loop_blocks, summary)
    base.plot_blocks(block_png, summary, details)
    plot_multiloop_overview(overview_png, loop_catalog)

    print(f"Paths: {len(paths)}")
    print(f"Loops: {len(loop_catalog)}")
    for row in loop_catalog:
        print(
            f"{row['loop_id']} {row['group']} period={row['selected_period']} motif={row['consensus_motif']} "
            f"copies={row['copy_min']}-{row['copy_max']}"
        )
    for output in (
        summary_path,
        detail_path,
        candidate_path,
        catalog_path,
        evidence_path,
        gfa_path,
        block_png,
        overview_png,
    ):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
