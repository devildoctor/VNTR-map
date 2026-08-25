#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


FORMAL_SCRIPT = Path(__file__).with_name("20260714_TRF概率评分位置特异多环建图.py")
DEFAULT_INPUT = Path("测试数据") / "IRF2BPL.HPRC.fa.3034e58.11fba48.904d69e.smooth.final.gfa"
DEFAULT_OUTDIR = Path("outputs") / "IRF2BPL" / "20260825_P1唯一侧翼锚点与共识边界"
DEFAULT_PREFIX = "20260825_IRF2BPL_P1唯一侧翼锚点与共识边界"

DEFAULT_ANCHOR_K = 21
DEFAULT_FLANK_WINDOW_BP = 180
DEFAULT_MIN_ANCHOR_SUPPORT = 0.90
COORDINATE_SYSTEM = "0-based_half-open"


@dataclass(frozen=True)
class CohortAnchor:
    side: str
    sequence: str
    reference_path: str
    reference_start: int
    length: int
    present_paths: int
    unique_paths: int
    path_count: int
    median_boundary_distance: float

    @property
    def support_fraction(self) -> float:
        return self.unique_paths / max(1, self.path_count)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load program: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def occurrences(sequence: str, query: str) -> list[int]:
    positions: list[int] = []
    start = 0
    while True:
        position = sequence.find(query, start)
        if position < 0:
            return positions
        positions.append(position)
        start = position + 1


def shannon_entropy(sequence: str) -> float:
    counts = Counter(sequence)
    return -sum((count / len(sequence)) * math.log2(count / len(sequence)) for count in counts.values())


def repeat_bounds(blocks) -> tuple[int, int]:
    repeat_blocks = [block for block in blocks if block.group != "nonrepeat"]
    if not repeat_blocks:
        raise ValueError("No supported repeat block")
    return repeat_blocks[0].start, repeat_blocks[-1].end


def discover_cohort_anchor(
    paths,
    provisional_bounds: dict[str, tuple[int, int]],
    side: str,
    k: int,
    flank_window_bp: int,
    min_gap_bp: int,
) -> CohortAnchor:
    reference = paths[0]
    reference_boundary = provisional_bounds[reference.name][0 if side == "left" else 1]
    if side == "left":
        first = max(0, reference_boundary - flank_window_bp)
        last = reference_boundary - k
    else:
        first = reference_boundary
        last = min(len(reference.seq) - k, reference_boundary + flank_window_bp)
    if last < first:
        raise ValueError(f"No {side} flank search space for {reference.name}")

    candidates = []
    for start in range(first, last + 1):
        sequence = reference.seq[start : start + k]
        if len(sequence) != k or "N" in sequence or shannon_entropy(sequence) < 1.25:
            continue
        counts = {path.name: len(occurrences(path.seq, sequence)) for path in paths}
        present_paths = sum(count > 0 for count in counts.values())
        distances = []
        usable_paths = 0
        for path in paths:
            positions = occurrences(path.seq, sequence)
            if len(positions) != 1:
                continue
            boundary = provisional_bounds[path.name][0 if side == "left" else 1]
            distance = boundary - (positions[0] + k) if side == "left" else positions[0] - boundary
            if distance >= min_gap_bp:
                distances.append(distance)
                usable_paths += 1
        if not distances:
            continue
        reference_distance = reference_boundary - (start + k) if side == "left" else start - reference_boundary
        score = (
            usable_paths,
            present_paths,
            -statistics.median(distances),
            -reference_distance,
            shannon_entropy(sequence),
        )
        candidates.append((score, start, sequence, statistics.median(distances)))

    if not candidates:
        raise ValueError(f"No usable {side} cohort anchor")
    _score, start, sequence, median_distance = max(candidates, key=lambda item: item[0])
    counts = [len(occurrences(path.seq, sequence)) for path in paths]
    usable_paths = 0
    for path in paths:
        positions = occurrences(path.seq, sequence)
        if len(positions) != 1:
            continue
        boundary = provisional_bounds[path.name][0 if side == "left" else 1]
        distance = boundary - (positions[0] + k) if side == "left" else positions[0] - boundary
        usable_paths += distance >= min_gap_bp
    return CohortAnchor(
        side=side,
        sequence=sequence,
        reference_path=reference.name,
        reference_start=start,
        length=k,
        present_paths=sum(count > 0 for count in counts),
        unique_paths=usable_paths,
        path_count=len(paths),
        median_boundary_distance=float(median_distance),
    )


def anchor_rows(
    paths,
    left: CohortAnchor,
    right: CohortAnchor,
    provisional_bounds: dict[str, tuple[int, int]],
    min_gap_bp: int,
) -> tuple[list[dict[str, object]], dict[str, dict[str, int]]]:
    rows: list[dict[str, object]] = []
    positions_by_path: dict[str, dict[str, int]] = {}
    for path in paths:
        positions_by_path[path.name] = {}
        for anchor in (left, right):
            positions = occurrences(path.seq, anchor.sequence)
            unique = len(positions) == 1
            start = positions[0] if unique else -1
            end = start + anchor.length if unique else -1
            boundary = provisional_bounds[path.name][0 if anchor.side == "left" else 1]
            gap = boundary - end if anchor.side == "left" else start - boundary
            side_consistent = unique and gap >= min_gap_bp
            positions_by_path[path.name][f"{anchor.side}_start"] = start if side_consistent else -1
            positions_by_path[path.name][f"{anchor.side}_end"] = end if side_consistent else -1
            rows.append(
                {
                    "path": path.name,
                    "side": anchor.side,
                    "anchor_sequence": anchor.sequence,
                    "anchor_k": anchor.length,
                    "path_start_bp": start,
                    "path_end_bp": end,
                    "occurrences_in_path": len(positions),
                    "unique_in_path": int(unique),
                    "flank_gap_bp": gap if unique else -1,
                    "side_consistent": int(side_consistent),
                    "usable_unique_anchor": int(side_consistent),
                    "cohort_present_paths": anchor.present_paths,
                    "cohort_unique_paths": anchor.unique_paths,
                    "cohort_path_count": anchor.path_count,
                    "cohort_unique_fraction": f"{anchor.support_fraction:.4f}",
                    "coordinate_system": COORDINATE_SYSTEM,
                }
            )
    return rows, positions_by_path


def parse_external_evidence(paths_by_name, files: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            header = handle.readline().rstrip("\r\n").split("\t")
            required = {"path", "method", "start_bp", "end_bp"}
            if not required.issubset(header):
                raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")
            for line_number, raw in enumerate(handle, start=2):
                if not raw.strip():
                    continue
                values = raw.rstrip("\r\n").split("\t")
                record = dict(zip(header, values))
                path_name = record["path"]
                if path_name not in paths_by_name:
                    raise ValueError(f"Unknown path {path_name!r} in {path}:{line_number}")
                start = int(record["start_bp"])
                end = int(record["end_bp"])
                if not 0 <= start < end <= len(paths_by_name[path_name].seq):
                    raise ValueError(f"Invalid boundary in {path}:{line_number}")
                rows.append(
                    {
                        "path": path_name,
                        "method": record["method"],
                        "method_family": f"external:{record['method']}",
                        "start_bp": start,
                        "end_bp": end,
                        "motif": record.get("motif", "-"),
                        "source": record.get("source", str(path)),
                        "external": 1,
                    }
                )
    return rows


def internal_evidence_rows(base, paths, blocks_by_path, loop_blocks_by_path, selected) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        _window_start, _window_end, motif_anchors = base.locate_primary_region(path.seq)
        if motif_anchors:
            rows.append(
                {
                    "path": path.name,
                    "method": "motif_run",
                    "method_family": "motif_seed",
                    "start_bp": min(anchor.start for anchor in motif_anchors),
                    "end_bp": max(anchor.end for anchor in motif_anchors),
                    "motif": ";".join(sorted({anchor.group for anchor in motif_anchors})),
                    "source": "anchor_candidates",
                    "external": 0,
                }
            )

        start, end = repeat_bounds(blocks_by_path[path.name])
        rows.append(
            {
                "path": path.name,
                "method": "probabilistic_dp",
                "method_family": "probabilistic_dp",
                "start_bp": start,
                "end_bp": end,
                "motif": ";".join(
                    sorted({block.group for block in blocks_by_path[path.name] if block.group != "nonrepeat"})
                ),
                "source": "TRF_probability_three_state_DP",
                "external": 0,
            }
        )

        loop_blocks = loop_blocks_by_path[path.name]
        loop_start = min(block.start for _loop_id, block in loop_blocks)
        loop_end = max(block.end for _loop_id, block in loop_blocks)
        motif = ";".join(
            f"{loop_id}:{selected[(path.name, loop_id)].consensus}"
            for loop_id, _block in loop_blocks
        )
        rows.append(
            {
                "path": path.name,
                "method": "periodicity_loops",
                "method_family": "periodicity",
                "start_bp": loop_start,
                "end_bp": loop_end,
                "motif": motif,
                "source": "TRF_period_candidates",
                "external": 0,
            }
        )
    return rows


def structural_term(blocks, motif_period: int, supported: bool) -> tuple[str, int, int, int]:
    repeat_blocks = [block for block in blocks if block.group != "nonrepeat"]
    if not repeat_blocks or not supported:
        return "TR", 0, 0, 0
    first, last = repeat_blocks[0], repeat_blocks[-1]
    internal_complex_bp = sum(
        block.end - block.start
        for block in blocks
        if block.group == "nonrepeat" and first.end <= block.start and block.end <= last.start
    )
    inserted_bp = sum(block.inserted_bp for block in repeat_blocks)
    group_count = len({block.group for block in repeat_blocks})
    if group_count > 1 or internal_complex_bp >= motif_period or inserted_bp > 0:
        return "complex/mosaic TR", group_count, internal_complex_bp, inserted_bp
    if motif_period <= 6:
        return "STR-like", group_count, internal_complex_bp, inserted_bp
    return "VNTR-like", group_count, internal_complex_bp, inserted_bp


def supported_boundary_cluster(rows: list[dict[str, object]], column: str, tolerance: int):
    best = None
    for center_row in rows:
        center = int(center_row[column])
        cluster = [row for row in rows if abs(int(row[column]) - center) <= tolerance]
        values = [int(row[column]) for row in cluster]
        families = {str(row["method_family"]) for row in cluster}
        score = (len(families), len(cluster), -(max(values) - min(values)), -center)
        if best is None or score > best[0]:
            best = (score, cluster)
    assert best is not None
    cluster = best[1]
    values = [int(row[column]) for row in cluster]
    return (
        int(statistics.median(values)),
        min(values),
        max(values),
        sorted({str(row["method_family"]) for row in cluster}),
        sorted({str(row["method"]) for row in cluster}),
    )


def consensus_rows(
    paths,
    blocks_by_path,
    positions_by_path,
    evidence_rows,
    motif_period: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    evidence_by_path: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in evidence_rows:
        evidence_by_path[str(row["path"])].append(row)

    boundaries: list[dict[str, object]] = []
    terminology: list[dict[str, object]] = []
    for path in paths:
        positions = positions_by_path[path.name]
        left_start = positions["left_start"]
        left_end = positions["left_end"]
        right_start = positions["right_start"]
        right_end = positions["right_end"]
        left_anchor_unique = left_start >= 0
        right_anchor_unique = right_start >= 0
        anchors_unique = left_anchor_unique and right_anchor_unique
        anchors_ordered = anchors_unique and left_end <= right_start

        accepted = []
        for row in evidence_by_path[path.name]:
            within_anchors = anchors_ordered and left_end <= int(row["start_bp"]) < int(row["end_bp"]) <= right_start
            row["within_unique_flanks"] = int(within_anchors)
            row["accepted_for_consensus"] = int(within_anchors)
            if within_anchors:
                accepted.append(row)
        if not accepted:
            raise AssertionError(f"No accepted boundary method for {path.name}")

        method_families = sorted({str(row["method_family"]) for row in accepted})
        internal_families = sorted(family for family in method_families if not family.startswith("external:"))
        methods = sorted({str(row["method"]) for row in accepted})
        external_methods = sorted({str(row["method"]) for row in accepted if int(row["external"])})
        consensus_start, start_low, start_high, start_families, start_methods = supported_boundary_cluster(
            accepted, "start_bp", motif_period
        )
        consensus_end, end_low, end_high, end_families, end_methods = supported_boundary_cluster(
            accepted, "end_bp", motif_period
        )
        boundary_support_methods = sorted(set(start_methods) & set(end_methods))
        boundary_stable = (
            len(start_families) >= 2
            and len(end_families) >= 2
            and max(start_high - start_low, end_high - end_low) <= motif_period
        )
        internally_supported = len(internal_families) >= 2
        multi_tool_confirmed = len(external_methods) >= 1 and len(method_families) >= 2
        if not anchors_unique or not anchors_ordered:
            status = "anchor_failed"
        elif multi_tool_confirmed and boundary_stable:
            status = "confirmed_multi_tool"
        elif internally_supported and boundary_stable:
            status = "provisional_internal_consensus"
        elif internally_supported:
            status = "provisional_broad_boundary"
        else:
            status = "single_method_only"

        term, group_count, internal_complex_bp, inserted_bp = structural_term(
            blocks_by_path[path.name], motif_period, internally_supported
        )
        row = {
            "path": path.name,
            "coordinate_system": COORDINATE_SYSTEM,
            "left_anchor_start_bp": left_start,
            "left_anchor_end_bp": left_end,
            "right_anchor_start_bp": right_start,
            "right_anchor_end_bp": right_end,
            "left_anchor_unique": int(left_anchor_unique),
            "right_anchor_unique": int(right_anchor_unique),
            "anchors_unique": int(anchors_unique),
            "anchors_ordered": int(anchors_ordered),
            "homology_window_start_bp": left_end,
            "homology_window_end_bp": right_start,
            "consensus_start_bp": consensus_start,
            "start_ci_low_bp": start_low,
            "start_ci_high_bp": start_high,
            "consensus_end_bp": consensus_end,
            "end_ci_low_bp": end_low,
            "end_ci_high_bp": end_high,
            "start_spread_bp": start_high - start_low,
            "end_spread_bp": end_high - end_low,
            "motif_uncertainty_bp": motif_period,
            "boundary_within_one_motif": int(boundary_stable),
            "method_count": len(methods),
            "method_family_count": len(method_families),
            "internal_method_family_count": len(internal_families),
            "methods": ";".join(methods),
            "start_support_family_count": len(start_families),
            "start_support_methods": ";".join(start_methods),
            "end_support_family_count": len(end_families),
            "end_support_methods": ";".join(end_methods),
            "boundary_support_methods": ";".join(boundary_support_methods),
            "disagreement_method_count": len(set(methods) - set(boundary_support_methods)),
            "external_method_count": len(external_methods),
            "external_methods": ";".join(external_methods) or "-",
            "boundary_status": status,
            "umbrella_term": "TR",
            "structural_term": term,
        }
        boundaries.append(row)
        terminology.append(
            {
                "path": path.name,
                "umbrella_term": "TR",
                "structural_term": term,
                "dominant_period_bp": motif_period,
                "repeat_group_count": group_count,
                "internal_complex_bp": internal_complex_bp,
                "inserted_bp": inserted_bp,
                "classification_basis": (
                    "multiple_motif_groups_or_interruptions"
                    if term == "complex/mosaic TR"
                    else "period_and_consensus_support"
                ),
                "boundary_status": status,
            }
        )
    return boundaries, terminology


def write_fasta(path: Path, paths) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in paths:
            handle.write(f">{record.name}\n")
            for start in range(0, len(record.seq), 80):
                handle.write(record.seq[start : start + 80] + "\n")


def annotate_gfa(path: Path, boundary_rows: list[dict[str, object]]) -> None:
    by_path = {str(row["path"]): row for row in boundary_rows}
    output: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            fields = line.split("\t")
            if fields[0] == "H":
                line += f"\tP1:Z:unique_flank_consensus_boundary\tCS:Z:{COORDINATE_SYSTEM}"
            elif fields[0] == "P" and fields[1] in by_path:
                row = by_path[fields[1]]
                tags = [
                    f"RS:i:{row['consensus_start_bp']}",
                    f"RE:i:{row['consensus_end_bp']}",
                    f"SL:i:{row['start_ci_low_bp']}",
                    f"SH:i:{row['start_ci_high_bp']}",
                    f"EL:i:{row['end_ci_low_bp']}",
                    f"EH:i:{row['end_ci_high_bp']}",
                    f"LA:i:{row['left_anchor_start_bp']}",
                    f"RA:i:{row['right_anchor_start_bp']}",
                    f"BM:i:{row['method_family_count']}",
                    f"EM:i:{row['external_method_count']}",
                    f"BT:Z:{row['boundary_status']}",
                    f"TT:Z:{str(row['structural_term']).replace('/', '_').replace(' ', '_')}",
                ]
                line += "\t" + "\t".join(tags)
            output.append(line)
    path.write_text("\n".join(output) + "\n", encoding="utf-8")


def plot_boundaries(path: Path, boundary_rows: list[dict[str, object]]) -> None:
    ordered = list(boundary_rows)
    figure_height = max(8.0, 0.23 * len(ordered) + 2.7)
    fig, ax = plt.subplots(figsize=(14, figure_height))
    for index, row in enumerate(ordered):
        y = len(ordered) - index - 1
        left_start = int(row["left_anchor_start_bp"])
        left_end = int(row["left_anchor_end_bp"])
        right_start = int(row["right_anchor_start_bp"])
        right_end = int(row["right_anchor_end_bp"])
        start = int(row["consensus_start_bp"])
        end = int(row["consensus_end_bp"])
        ax.hlines(y, left_end, right_start, color="#cbd5e1", linewidth=5, zorder=1)
        ax.hlines(y, start, end, color="#2e74b5", linewidth=7, zorder=2)
        ax.hlines(
            y,
            int(row["start_ci_low_bp"]),
            int(row["start_ci_high_bp"]),
            color="#d49b24",
            linewidth=11,
            zorder=3,
        )
        ax.hlines(
            y,
            int(row["end_ci_low_bp"]),
            int(row["end_ci_high_bp"]),
            color="#d49b24",
            linewidth=11,
            zorder=3,
        )
        ax.hlines(y, left_start, left_end, color="#172b3a", linewidth=9, zorder=4)
        ax.hlines(y, right_start, right_end, color="#172b3a", linewidth=9, zorder=4)

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([str(row["path"]) for row in reversed(ordered)], fontsize=6.5)
    ax.set_xlabel("Path coordinate (bp; 0-based half-open)")
    ax.set_title("P1 unique-flank anchored consensus boundaries", fontsize=15, fontweight="bold")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.7)
    ax.legend(
        handles=[
            Patch(facecolor="#172b3a", label="unique flank anchors"),
            Patch(facecolor="#cbd5e1", label="homology search window"),
            Patch(facecolor="#2e74b5", label="consensus repeat span"),
            Patch(facecolor="#d49b24", label="boundary uncertainty"),
        ],
        loc="upper right",
        frameon=False,
        ncol=2,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def validation_metrics(paths, boundary_rows, terminology_rows, graph_qa) -> dict[str, object]:
    return {
        "path_count": len(paths),
        "unique_left_anchor_paths": sum(int(row["left_anchor_unique"]) for row in boundary_rows),
        "unique_right_anchor_paths": sum(int(row["right_anchor_unique"]) for row in boundary_rows),
        "ordered_anchor_paths": sum(int(row["anchors_ordered"]) for row in boundary_rows),
        "internally_supported_boundary_paths": sum(
            int(row["internal_method_family_count"]) >= 2 for row in boundary_rows
        ),
        "external_confirmed_boundary_paths": sum(int(row["external_method_count"]) >= 1 for row in boundary_rows),
        "within_one_motif_boundary_paths": sum(int(row["boundary_within_one_motif"]) for row in boundary_rows),
        "complex_mosaic_TR_paths": sum(row["structural_term"] == "complex/mosaic TR" for row in terminology_rows),
        "exact_locus_reconstruction_paths": graph_qa["exact_gfa_reconstruction_paths"],
        "gfa_node_count": graph_qa["node_count"],
        "gfa_edge_count": graph_qa["edge_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 unique-flank anchored consensus boundaries and TR terminology.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--anchor-k", type=int, default=DEFAULT_ANCHOR_K)
    parser.add_argument("--flank-window-bp", type=int, default=DEFAULT_FLANK_WINDOW_BP)
    parser.add_argument("--min-anchor-support", type=float, default=DEFAULT_MIN_ANCHOR_SUPPORT)
    parser.add_argument("--external-evidence", type=Path, action="append", default=[])
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    formal = load_module(FORMAL_SCRIPT, "vntr_p1_formal")
    probability = formal.load_module(formal.PROBABILITY_SCRIPT, "vntr_p1_probability")
    multiloop = formal.load_module(formal.MULTILOOP_SCRIPT, "vntr_p1_multiloop")
    (
        base,
        paths,
        summary,
        _details,
        _evidence,
        blocks,
        atoms,
        _fixed_summary,
        _fixed_blocks,
    ) = probability.run_analysis(args.input)
    probability.validate(base, paths, summary, blocks, atoms)
    candidates, selected, loop_blocks = multiloop.discover_period_candidates(base, blocks, atoms)
    selected, global_periods = formal.normalize_to_global_period(candidates, selected)
    motif_period = statistics.mode(global_periods.values())

    internal = internal_evidence_rows(base, paths, blocks, loop_blocks, selected)
    internal_by_path: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in internal:
        internal_by_path[str(row["path"])].append(row)
    provisional_bounds = {
        path.name: (
            min(int(row["start_bp"]) for row in internal_by_path[path.name]),
            max(int(row["end_bp"]) for row in internal_by_path[path.name]),
        )
        for path in paths
    }
    left = discover_cohort_anchor(
        paths, provisional_bounds, "left", args.anchor_k, args.flank_window_bp, motif_period
    )
    right = discover_cohort_anchor(
        paths, provisional_bounds, "right", args.anchor_k, args.flank_window_bp, motif_period
    )
    if left.support_fraction < args.min_anchor_support or right.support_fraction < args.min_anchor_support:
        raise AssertionError(
            f"Unique flank support below threshold: left={left.support_fraction:.3f}, right={right.support_fraction:.3f}"
        )

    flanks, positions_by_path = anchor_rows(paths, left, right, provisional_bounds, motif_period)
    paths_by_name = {path.name: path for path in paths}
    external = parse_external_evidence(paths_by_name, args.external_evidence)
    evidence_rows = internal + external
    boundaries, terminology = consensus_rows(paths, blocks, positions_by_path, evidence_rows, motif_period)

    flank_path = args.outdir / f"{args.prefix}_唯一侧翼锚点.tsv"
    evidence_path = args.outdir / f"{args.prefix}_边界方法证据.tsv"
    boundary_path = args.outdir / f"{args.prefix}_路径边界共识.tsv"
    terminology_path = args.outdir / f"{args.prefix}_术语分类.tsv"
    fasta_path = args.outdir / f"{args.prefix}_路径序列.fa"
    template_path = args.outdir / f"{args.prefix}_外部边界证据模板.tsv"
    gfa_path = args.outdir / f"{args.prefix}_P1边界标注多环图.gfa"
    png_path = args.outdir / f"{args.prefix}_边界共识总览.png"
    qa_path = args.outdir / f"{args.prefix}_P1验证汇总.tsv"

    base.write_tsv(flank_path, flanks, list(flanks[0]))
    base.write_tsv(evidence_path, evidence_rows, list(evidence_rows[0]))
    base.write_tsv(boundary_path, boundaries, list(boundaries[0]))
    base.write_tsv(terminology_path, terminology, list(terminology[0]))
    base.write_tsv(template_path, [], ["path", "method", "start_bp", "end_bp", "motif", "source"])
    write_fasta(fasta_path, paths)

    nodes, edges, path_steps = formal.write_formal_gfa(
        probability, multiloop, base, gfa_path, blocks, atoms, loop_blocks, summary
    )
    graph_qa = formal.validate_formal_graph(
        probability,
        base,
        paths,
        summary,
        blocks,
        atoms,
        loop_blocks,
        selected,
        nodes,
        edges,
        path_steps,
    )
    annotate_gfa(gfa_path, boundaries)
    plot_boundaries(png_path, boundaries)

    qa = validation_metrics(paths, boundaries, terminology, graph_qa)
    base.write_tsv(qa_path, [{"metric": key, "value": value} for key, value in qa.items()], ["metric", "value"])

    status_counts = Counter(str(row["boundary_status"]) for row in boundaries)
    term_counts = Counter(str(row["structural_term"]) for row in terminology)
    print(f"Paths: {len(paths)}")
    print(
        f"Left anchor: {left.sequence} unique={left.unique_paths}/{left.path_count}; "
        f"right anchor: {right.sequence} unique={right.unique_paths}/{right.path_count}"
    )
    print("Boundary status: " + ", ".join(f"{key}={value}" for key, value in sorted(status_counts.items())))
    print("Terminology: " + ", ".join(f"{key}={value}" for key, value in sorted(term_counts.items())))
    print(f"Exact locus reconstruction: {qa['exact_locus_reconstruction_paths']}/{len(paths)}")
    if not external:
        print("P1 phase: internal consensus complete; external TRF/MotifScope/uTR/vamos confirmation pending")
    else:
        print(f"P1 phase: loaded {len(external)} external boundary records")
    for output in (
        flank_path,
        evidence_path,
        boundary_path,
        terminology_path,
        fasta_path,
        template_path,
        gfa_path,
        png_path,
        qa_path,
    ):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
