#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path


PROBABILITY_SCRIPT = Path(__file__).with_name("20260714_TRF概率评分动态规划VNTR分解.py")
MULTILOOP_SCRIPT = Path(__file__).with_name("20260714_TRF思想多环VNTR分解与图建模.py")
DEFAULT_INPUT = Path("测试数据") / "IRF2BPL.HPRC.fa.3034e58.11fba48.904d69e.smooth.final.gfa"
DEFAULT_OUTDIR = Path("outputs") / "IRF2BPL" / "20260714_TRF概率评分位置特异多环正式版"
DEFAULT_PREFIX = "20260714_IRF2BPL_TRF概率评分位置特异多环"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load program: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def path_loop_rows(probability, base, loop_blocks_by_path, selected, atoms_by_path):
    rows = []
    for path_name, loop_blocks in loop_blocks_by_path.items():
        for loop_id, block in loop_blocks:
            candidate = selected[(path_name, loop_id)]
            identity, log_odds = probability.block_score(base, block, atoms_by_path[path_name])
            rows.append(
                {
                    "path": path_name,
                    "loop_id": loop_id,
                    "block_index": block.index,
                    "group": block.group,
                    "start_bp": block.start,
                    "end_bp": block.end,
                    "repeat_units": block.repeat_units,
                    "exact_units": block.exact_units,
                    "variant_units": block.variant_units,
                    "inserted_bp": block.inserted_bp,
                    "repeat_density": f"{block.repeat_density:.4f}",
                    "motif_identity": f"{identity:.4f}",
                    "TRF_log_odds": f"{log_odds:.4f}",
                    "selected_period": candidate.period,
                    "selected_consensus": candidate.consensus,
                    "period_identity": f"{candidate.identity:.4f}",
                    "tuple_support": f"{candidate.tuple_support:.4f}",
                    "motif_counts": ";".join(f"{key}:{value}" for key, value in sorted(block.exact_counts.items())),
                    "variant_counts": ";".join(
                        f"{key}:{value}" for key, value in sorted(block.variant_counts.items())
                    ) or "-",
                    "insertions": ";".join(f"{pos}:{seq}" for pos, seq in block.insertion_parts) or "-",
                }
            )
    return rows


def normalize_to_global_period(candidates, selected):
    period_by_loop = {}
    for loop_id in sorted({loop_id for _path_name, loop_id in selected}):
        period_counts = Counter(
            candidate.period
            for (_path_name, candidate_loop_id), candidate in selected.items()
            if candidate_loop_id == loop_id
        )
        period_by_loop[loop_id] = period_counts.most_common(1)[0][0]

    candidate_lookup = {
        (candidate.path, candidate.loop_id, candidate.period): candidate
        for candidate in candidates
    }
    for candidate in candidates:
        candidate.selected = False
    normalized = {}
    for path_loop in selected:
        path_name, loop_id = path_loop
        key = (path_name, loop_id, period_by_loop[loop_id])
        if key not in candidate_lookup:
            raise AssertionError(f"Global period is not supported for {path_name} {loop_id}")
        normalized[path_loop] = candidate_lookup[key]
        normalized[path_loop].selected = True
    return normalized, period_by_loop


def path_loop_matrix(loop_rows):
    by_path = {}
    for row in loop_rows:
        by_path.setdefault(str(row["path"]), {})[str(row["loop_id"])] = row
    rows = []
    for path_name, loops in by_path.items():
        output = {"path": path_name}
        for loop_id in ("L1", "L2", "L3"):
            row = loops[loop_id]
            output[f"{loop_id}_group"] = row["group"]
            output[f"{loop_id}_repeat_units"] = row["repeat_units"]
            output[f"{loop_id}_variant_units"] = row["variant_units"]
            output[f"{loop_id}_inserted_bp"] = row["inserted_bp"]
            output[f"{loop_id}_consensus"] = row["selected_consensus"]
        rows.append(output)
    return rows


def integer_range(values) -> str:
    values = list(values)
    return str(min(values)) if min(values) == max(values) else f"{min(values)}-{max(values)}"


def plot_formal_overview(multiloop, path: Path, loop_catalog, blocks_by_path) -> None:
    catalog = {str(row["loop_id"]): row for row in loop_catalog}
    left_bp = integer_range(blocks[0].end - blocks[0].start for blocks in blocks_by_path.values())
    complex_bp = integer_range(blocks[3].end - blocks[3].start for blocks in blocks_by_path.values())
    right_bp = integer_range(blocks[5].end - blocks[5].start for blocks in blocks_by_path.values())
    insertion_bp = integer_range(blocks[4].inserted_bp for blocks in blocks_by_path.values())

    fig, ax = multiloop.plt.subplots(figsize=(15, 5.2))
    ax.set_xlim(-0.5, 14.8)
    ax.set_ylim(-2.2, 2.3)
    ax.axis("off")
    rectangle = multiloop.Rectangle

    ax.add_patch(rectangle((0.0, -0.42), 1.25, 0.84, facecolor="#9aa1ad", edgecolor="#27303f"))
    ax.text(0.625, 0, f"left flank\n{left_bp} bp", ha="center", va="center", fontsize=9, color="white")
    multiloop.draw_loop(
        ax, 3.0, 0, 1.05, "#4d9b68",
        f"L1 GCC-like ({catalog['L1']['consensus_motif']})",
        f"{catalog['L1']['copy_min']}-{catalog['L1']['copy_max']} copies",
    )
    multiloop.draw_loop(
        ax, 6.3, 0, 1.05, "#3478b8",
        f"L2 CAG-like ({catalog['L2']['consensus_motif']})",
        f"{catalog['L2']['copy_min']}-{catalog['L2']['copy_max']} copies",
    )
    ax.add_patch(rectangle((8.1, -0.52), 1.55, 1.04, facecolor="#9aa1ad", edgecolor="#27303f"))
    ax.text(8.875, 0, f"complex\n{complex_bp} bp", ha="center", va="center", fontsize=9, color="white")
    multiloop.draw_loop(
        ax, 11.3, 0, 1.12, "#4d9b68",
        f"L3 GCC-like ({catalog['L3']['consensus_motif']})",
        f"{catalog['L3']['copy_min']}-{catalog['L3']['copy_max']} copies",
    )
    ax.add_patch(rectangle((13.35, -0.42), 1.0, 0.84, facecolor="#9aa1ad", edgecolor="#27303f"))
    ax.text(13.85, 0, f"right\n{right_bp} bp", ha="center", va="center", fontsize=9, color="white")

    multiloop.arrow(ax, (1.25, 0), (1.85, 0))
    multiloop.arrow(ax, (4.05, 0), (5.15, 0))
    multiloop.arrow(ax, (7.35, 0), (8.1, 0))
    multiloop.arrow(ax, (9.65, 0), (10.18, 0))
    multiloop.arrow(ax, (12.42, 0), (13.35, 0))

    ax.add_patch(rectangle((10.45, -1.82), 1.7, 0.52, facecolor="#d1d5db", edgecolor="#4b5563"))
    ax.text(11.3, -1.56, f"{insertion_bp} bp insertion", ha="center", va="center", fontsize=8)
    multiloop.arrow(ax, (10.82, -1.28), (10.82, -0.96))
    multiloop.arrow(ax, (11.78, -0.96), (11.78, -1.28))

    ax.text(
        7.15, 1.92, "IRF2BPL TRF-probability position-specific multi-loop graph",
        ha="center", va="center", fontsize=15, fontweight="bold",
    )
    ax.text(
        7.15, 1.53,
        "Global period = 3 bp; main anchors >=5 copies; independent loops >=4 copies; density >=75%.",
        ha="center", va="center", fontsize=9, color="#4b5563",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    multiloop.plt.close(fig)


def write_formal_gfa(
    probability,
    multiloop,
    base,
    path: Path,
    blocks_by_path,
    atoms_by_path,
    loop_blocks_by_path,
    summary_rows,
):
    nodes: dict[str, tuple[str, list[str]]] = {}
    edges: Counter[tuple[str, str]] = Counter()
    path_steps: dict[str, list[str]] = {}
    summary_by_path = {str(row["path"]): row for row in summary_rows}
    loop_stats = {
        path_name: {
            loop_id: (block.repeat_units, block.variant_units, block.inserted_bp)
            for loop_id, block in loop_blocks
        }
        for path_name, loop_blocks in loop_blocks_by_path.items()
    }

    for path_name, atoms in atoms_by_path.items():
        blocks = blocks_by_path[path_name]
        loop_map = multiloop.block_loop_map(loop_blocks_by_path[path_name])
        steps = []
        for atom in atoms:
            block = multiloop.atom_block(atom, blocks)
            node_id, seq, tags = multiloop.graph_node_for_atom(base, atom, block, loop_map.get(block.index))
            nodes[node_id] = (seq, tags)
            steps.append(node_id)
        path_steps[path_name] = steps
        edges.update(zip(steps, steps[1:]))

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "H\tVN:Z:1.0\tTS:Z:trf_probability_position_specific_multiloop"
            f"\tPM:f:{probability.TRF_MATCH_PROBABILITY:.2f}"
            f"\tPI:f:{probability.TRF_INDEL_PROBABILITY:.2f}"
            f"\tMC:i:{probability.MIN_RING_COPIES}"
            f"\tMD:f:{probability.MIN_RING_DENSITY:.2f}\n"
        )
        for node_id in sorted(nodes):
            seq, tags = nodes[node_id]
            handle.write(f"S\t{node_id}\t{seq}\t" + "\t".join(tags) + "\n")
        for (left, right), support in sorted(edges.items()):
            loop_tag = "\tRL:i:1" if left == right and left.startswith(("L1_", "L2_", "L3_")) else ""
            handle.write(f"L\t{left}\t+\t{right}\t+\t0M\tPS:i:{support}{loop_tag}\n")
        for path_name, steps in path_steps.items():
            overlaps = ",".join("0M" for _ in steps[1:]) if len(steps) > 1 else "*"
            summary = summary_by_path[path_name]
            tags = [
                f"CU:i:{summary['CAG_like_units']}",
                f"GU:i:{summary['GCC_like_units']}",
                f"VU:i:{summary['variant_units']}",
                f"IB:i:{summary['inserted_bp']}",
                f"BS:Z:{summary['block_signature']}",
            ]
            for loop_id in ("L1", "L2", "L3"):
                copies, variants, inserted = loop_stats[path_name][loop_id]
                tags.append(f"{loop_id}:Z:copies={copies};variants={variants};inserted_bp={inserted}")
            handle.write(
                f"P\t{path_name}\t{','.join(step + '+' for step in steps)}\t{overlaps}\t"
                + "\t".join(tags)
                + "\n"
            )
    return nodes, edges, path_steps


def validate_formal_graph(
    probability,
    base,
    paths,
    summary_rows,
    blocks_by_path,
    atoms_by_path,
    loop_blocks_by_path,
    selected,
    nodes,
    edges,
    path_steps,
):
    probability.validate(base, paths, summary_rows, blocks_by_path, atoms_by_path)
    expected_loop_ids = ["L1", "L2", "L3"]
    expected_groups = ["GCC_like", "CAG_like", "GCC_like"]
    for path_name, loop_blocks in loop_blocks_by_path.items():
        if [loop_id for loop_id, _block in loop_blocks] != expected_loop_ids:
            raise AssertionError(f"Expected L1/L2/L3 for {path_name}")
        if [block.group for _loop_id, block in loop_blocks] != expected_groups:
            raise AssertionError(f"Unexpected loop groups for {path_name}")
        if any((path_name, loop_id) not in selected for loop_id in expected_loop_ids):
            raise AssertionError(f"Missing selected period for {path_name}")
        expected = "".join(block.seq for block in blocks_by_path[path_name])
        observed = "".join(nodes[node_id][0] for node_id in path_steps[path_name])
        if observed != expected:
            raise AssertionError(f"GFA reconstruction failed for {path_name}")
        if any(not any(node.startswith(f"{loop_id}_") for node in path_steps[path_name]) for loop_id in expected_loop_ids):
            raise AssertionError(f"GFA path is missing a loop namespace for {path_name}")

    self_loop_ids = {
        left.split("_", 1)[0]
        for left, right in edges
        if left == right and left.startswith(("L1_", "L2_", "L3_"))
    }
    if self_loop_ids != set(expected_loop_ids):
        raise AssertionError(f"Missing explicit self-loop instances: {self_loop_ids}")
    return {
        "path_count": len(paths),
        "path_loop_pairs": sum(len(value) for value in loop_blocks_by_path.values()),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "self_loop_edge_count": sum(left == right for left, right in edges),
        "self_loop_instances": ",".join(sorted(self_loop_ids)),
        "exact_gfa_reconstruction_paths": len(path_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="TRF-probability-scored position-specific multi-loop VNTR graph.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    probability = load_module(PROBABILITY_SCRIPT, "vntr_probability_formal")
    multiloop = load_module(MULTILOOP_SCRIPT, "vntr_multiloop_helpers")
    (
        base,
        paths,
        summary,
        details,
        evidence,
        blocks,
        atoms,
        _fixed_summary,
        _fixed_blocks,
    ) = probability.run_analysis(args.input)
    probability.validate(base, paths, summary, blocks, atoms)

    candidates, selected, loop_blocks = multiloop.discover_period_candidates(base, blocks, atoms)
    selected, global_periods = normalize_to_global_period(candidates, selected)
    loop_catalog = multiloop.build_loop_catalog(selected, loop_blocks)
    enriched_details = probability.augmented_detail_rows(base, details, blocks, atoms)
    loop_rows = path_loop_rows(probability, base, loop_blocks, selected, atoms)
    matrix_rows = path_loop_matrix(loop_rows)

    summary_path = args.outdir / f"{args.prefix}_路径汇总.tsv"
    detail_path = args.outdir / f"{args.prefix}_block明细.tsv"
    evidence_path = args.outdir / f"{args.prefix}_主体motif证据.tsv"
    candidate_path = args.outdir / f"{args.prefix}_TRF周期候选.tsv"
    catalog_path = args.outdir / f"{args.prefix}_多环目录.tsv"
    loop_path = args.outdir / f"{args.prefix}_逐路径环统计.tsv"
    matrix_path = args.outdir / f"{args.prefix}_逐路径环次数矩阵.tsv"
    gfa_path = args.outdir / f"{args.prefix}_正式位置特异多环图.gfa"
    block_png = args.outdir / f"{args.prefix}_路径block图.png"
    overview_png = args.outdir / f"{args.prefix}_多环结构总览.png"
    qa_path = args.outdir / f"{args.prefix}_验证汇总.tsv"

    base.write_tsv(summary_path, summary, list(summary[0]))
    base.write_tsv(detail_path, enriched_details, list(enriched_details[0]))
    base.write_tsv(evidence_path, evidence, list(evidence[0]))
    base.write_tsv(candidate_path, multiloop.candidate_rows(candidates), list(multiloop.candidate_rows(candidates)[0]))
    base.write_tsv(catalog_path, loop_catalog, list(loop_catalog[0]))
    base.write_tsv(loop_path, loop_rows, list(loop_rows[0]))
    base.write_tsv(matrix_path, matrix_rows, list(matrix_rows[0]))
    nodes, edges, path_steps = write_formal_gfa(
        probability, multiloop, base, gfa_path, blocks, atoms, loop_blocks, summary
    )
    qa = validate_formal_graph(
        probability, base, paths, summary, blocks, atoms, loop_blocks, selected, nodes, edges, path_steps
    )
    base.write_tsv(qa_path, [{"metric": key, "value": value} for key, value in qa.items()], ["metric", "value"])
    base.plot_blocks(block_png, summary, enriched_details)
    plot_formal_overview(multiloop, overview_png, loop_catalog, blocks)

    print(f"Paths: {len(paths)}")
    print(f"Loops: {len(loop_catalog)}")
    print("Global periods: " + ", ".join(f"{key}={value}" for key, value in global_periods.items()))
    for row in loop_catalog:
        print(
            f"{row['loop_id']} {row['group']} period={row['selected_period']} "
            f"consensus={row['consensus_motif']} copies={row['copy_min']}-{row['copy_max']}"
        )
    print(f"GFA nodes={qa['node_count']} edges={qa['edge_count']} self_loops={qa['self_loop_edge_count']}")
    for output in (
        summary_path, detail_path, evidence_path, candidate_path, catalog_path, loop_path, matrix_path,
        gfa_path, block_png, overview_png, qa_path,
    ):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
