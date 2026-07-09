#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


DEFAULT_INPUT = Path("测试数据") / "IRF2BPL.HPRC.fa.3034e58.11fba48.904d69e.smooth.final.gfa"
DEFAULT_OUTDIR = Path("outputs") / "IRF2BPL" / "20260709_PGGB图后处理_v4真实VNTR区块图"
DEFAULT_PREFIX = "20260709_IRF2BPL"

CAG_LIKE = ("CAG", "CAA")
GCC_LIKE = ("GCG", "GCC", "GCA", "GCT", "GGC")
REPEAT_MOTIFS = tuple(dict.fromkeys(CAG_LIKE + GCC_LIKE))


@dataclass
class Segment:
    name: str
    seq: str


@dataclass
class PathRecord:
    name: str
    steps: list[str]


@dataclass
class BlockDef:
    block_id: str
    role: str
    start_node: int
    end_node: int
    group: str
    nodes: list[int]


def read_gfa(path: Path) -> tuple[dict[str, Segment], list[PathRecord], list[tuple[str, str]]]:
    segments: dict[str, Segment] = {}
    paths: list[PathRecord] = []
    links: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            fields = raw.rstrip("\n\r").split("\t")
            if not fields or not fields[0]:
                continue
            if fields[0] == "S":
                segments[fields[1]] = Segment(fields[1], fields[2].upper())
            elif fields[0] == "L":
                links.append((fields[1], fields[3]))
            elif fields[0] == "P":
                steps = [step[:-1] for step in fields[2].split(",") if step]
                paths.append(PathRecord(fields[1], steps))
    return segments, paths, links


def numeric_node(name: str) -> int:
    return int(name)


def path_seq(path: PathRecord, segments: dict[str, Segment], start: int, end: int) -> str:
    return "".join(segments[node].seq for node in path.steps if start <= numeric_node(node) <= end)


def greedy_motif_counts(seq: str, motifs: tuple[str, ...]) -> Counter[str]:
    counts: Counter[str] = Counter()
    i = 0
    motif_set = set(motifs)
    while i < len(seq):
        piece = seq[i : i + 3]
        if len(piece) == 3 and piece in motif_set:
            counts[piece] += 1
            i += 3
        else:
            i += 1
    return counts


def repeat_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    counts = greedy_motif_counts(seq, REPEAT_MOTIFS)
    return min(1.0, 3 * sum(counts.values()) / len(seq))


def path_suffix_value(name: str) -> str:
    match = re.search(r"_([0-9]+(?:\.[0-9]+)?)$", name)
    return match.group(1) if match else ""


def node_usage(paths: list[PathRecord]) -> Counter[str]:
    usage: Counter[str] = Counter()
    for path in paths:
        usage.update(path.steps)
    return usage


def bridge_ok(previous_seed: int, next_seed: int, segments: dict[str, Segment]) -> bool:
    if next_seed <= previous_seed + 1:
        return True
    between = range(previous_seed + 1, next_seed)
    total_len = sum(len(segments[str(node)].seq) for node in between if str(node) in segments)
    has_long_low_repeat = any(
        len(segments[str(node)].seq) >= 60 and repeat_fraction(segments[str(node)].seq) < 0.25
        for node in between
        if str(node) in segments
    )
    return total_len <= 60 and not has_long_low_repeat


def infer_core_interval(segments: dict[str, Segment]) -> tuple[int, int, list[int]]:
    seeds = [
        numeric_node(name)
        for name, segment in segments.items()
        if len(segment.seq) >= 3 and repeat_fraction(segment.seq) >= 0.55
    ]
    seeds.sort()
    if not seeds:
        raise ValueError("No repeat-enriched seed nodes found.")

    clusters: list[list[int]] = [[seeds[0]]]
    for seed in seeds[1:]:
        if bridge_ok(clusters[-1][-1], seed, segments):
            clusters[-1].append(seed)
        else:
            clusters.append([seed])

    def score(cluster: list[int]) -> tuple[int, int, int]:
        repeat_bp = sum(len(segments[str(node)].seq) for node in cluster)
        span = cluster[-1] - cluster[0] + 1
        return (repeat_bp, len(cluster), span)

    best = max(clusters, key=score)
    return best[0], best[-1], best


def infer_blocks(
    segments: dict[str, Segment], paths: list[PathRecord], core_start: int, core_end: int
) -> list[BlockDef]:
    usage = node_usage(paths)
    total_paths = len(paths)
    common_bridges = [
        node
        for node in range(core_start, core_end + 1)
        if usage[str(node)] == total_paths and len(segments[str(node)].seq) >= 60
    ]

    blocks: list[BlockDef] = []
    span_start = core_start
    variable_index = 1
    for bridge in common_bridges + [core_end + 1]:
        span_end = bridge - 1
        if span_start <= span_end:
            span_blocks = infer_span_blocks(
                segments,
                paths,
                usage,
                total_paths,
                span_start,
                span_end,
                bridge,
                bridge <= core_end,
                variable_index,
            )
            blocks.extend(span_blocks)
            variable_index += sum(1 for block in span_blocks if block.role == "variable_loop")

        if bridge <= core_end:
            blocks.append(
                BlockDef(
                    block_id=f"BRIDGE_{bridge}",
                    role="common_bridge",
                    start_node=bridge,
                    end_node=bridge,
                    group="bridge",
                    nodes=[bridge],
                )
            )
        span_start = bridge + 1
    return blocks


def infer_span_blocks(
    segments: dict[str, Segment],
    paths: list[PathRecord],
    usage: Counter[str],
    total_paths: int,
    span_start: int,
    span_end: int,
    next_bridge: int,
    next_bridge_is_real: bool,
    variable_index: int,
) -> list[BlockDef]:
    seed_groups: list[tuple[int, str]] = []
    for node in range(span_start, span_end + 1):
        seq = segments[str(node)].seq
        if len(seq) >= 3 and repeat_fraction(seq) >= 0.55:
            seed_groups.append((node, dominant_node_group(seq)))
    if not seed_groups:
        return []

    clusters: list[list[tuple[int, str]]] = [[seed_groups[0]]]
    for node, group in seed_groups[1:]:
        previous_node, previous_group = clusters[-1][-1]
        if group == previous_group and bridge_ok(previous_node, node, segments):
            clusters[-1].append((node, group))
        else:
            clusters.append([(node, group)])

    out: list[BlockDef] = []
    previous_end = span_start - 1
    for cluster in clusters:
        seed_start = cluster[0][0]
        seed_end = cluster[-1][0]
        group = cluster[0][1]
        block_start = max(span_start, previous_end + 1 if previous_end >= span_start else seed_start)
        block_end = seed_end
        if cluster is clusters[-1]:
            block_end = span_end

        has_variable_node = any(usage[str(node)] < total_paths for node in range(block_start, block_end + 1))
        leading_before_first_bridge = next_bridge_is_real and block_end < next_bridge and not out
        role = "variable_loop" if has_variable_node else "conserved_repeat"
        if leading_before_first_bridge and group != "CAG_like":
            role = "repeat_context"

        block_id = (
            f"VAR{variable_index}_{group}"
            if role == "variable_loop"
            else f"{role.upper()}_{block_start}_{block_end}"
        )
        if role == "variable_loop":
            variable_index += 1

        out.append(
            BlockDef(
                block_id=block_id,
                role=role,
                start_node=block_start,
                end_node=block_end,
                group=group,
                nodes=list(range(block_start, block_end + 1)),
            )
        )
        previous_end = block_end
    return out


def dominant_node_group(seq: str) -> str:
    cag = sum(greedy_motif_counts(seq, CAG_LIKE).values())
    gcc = sum(greedy_motif_counts(seq, GCC_LIKE).values())
    return "CAG_like" if cag >= gcc else "GCC_like"


def dominant_group(paths: list[PathRecord], segments: dict[str, Segment], start: int, end: int) -> str:
    cag = 0
    gcc = 0
    for path in paths:
        seq = path_seq(path, segments, start, end)
        cag += sum(greedy_motif_counts(seq, CAG_LIKE).values())
        gcc += sum(greedy_motif_counts(seq, GCC_LIKE).values())
    return "CAG_like" if cag >= gcc else "GCC_like"


def write_tsv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(col, "")) for col in columns) + "\n")


def write_block_gfa(
    out_path: Path,
    segments: dict[str, Segment],
    paths: list[PathRecord],
    blocks: list[BlockDef],
    summary_rows: list[dict[str, object]],
) -> None:
    summary_by_path = {str(row["path"]): row for row in summary_rows}
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("H\tVN:Z:1.0\tTS:Z:true_vntr_block_model\n")

        for block in blocks:
            if block.role == "variable_loop":
                seq = "CAG" if block.group == "CAG_like" else "GCC"
                handle.write(
                    f"S\t{block.block_id}\t{seq}\tRO:Z:{block.role}\tGR:Z:{block.group}"
                    f"\tNS:Z:{block.start_node}-{block.end_node}\n"
                )
            else:
                seq = "".join(segments[str(node)].seq for node in block.nodes)
                handle.write(
                    f"S\t{block.block_id}\t{seq}\tRO:Z:{block.role}\tGR:Z:{block.group}"
                    f"\tNS:Z:{block.start_node}-{block.end_node}\n"
                )

        for left, right in zip(blocks, blocks[1:]):
            handle.write(f"L\t{left.block_id}\t+\t{right.block_id}\t+\t0M\n")
        for block in blocks:
            if block.role == "variable_loop":
                handle.write(f"L\t{block.block_id}\t+\t{block.block_id}\t+\t0M\n")

        path_steps = ",".join(f"{block.block_id}+" for block in blocks)
        overlaps = ",".join("0M" for _ in blocks[1:]) if len(blocks) > 1 else "*"
        for path in paths:
            row = summary_by_path[path.name]
            tags = [
                f"PS:Z:{row['path_suffix_label']}",
                f"CL:i:{row['CAG_like_variable_units']}",
                f"GL:i:{row['GCC_like_variable_units']}",
                f"CB:i:{row['common_bridge_CAG_like_units']}",
                f"GB:i:{row['common_bridge_GCC_like_units']}",
                f"SG:Z:{row['block_signature']}",
            ]
            handle.write(f"P\t{path.name}\t{path_steps}\t{overlaps}\t" + "\t".join(tags) + "\n")


def plot_png(
    out_path: Path,
    rows: list[dict[str, object]],
    core_start: int,
    core_end: int,
) -> None:
    sorted_rows = sorted(rows, key=lambda r: (int(r["CAG_like_variable_units"]), int(r["GCC_like_variable_units"]), str(r["path"])))
    fig_h = max(8, 0.24 * len(sorted_rows) + 2.5)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    colors = {
        "left": "#8a8f98",
        "cag": "#3b82c4",
        "bridge": "#d6a947",
        "gcc": "#4f9d69",
    }

    for y, row in enumerate(sorted_rows):
        x = 0.0
        ax.add_patch(Rectangle((x, y - 0.34), 2.0, 0.68, color=colors["left"], alpha=0.65))
        x += 2.0
        cag = int(row["CAG_like_variable_units"])
        gcc = int(row["GCC_like_variable_units"])
        ax.add_patch(Rectangle((x, y - 0.34), max(cag, 0.25), 0.68, color=colors["cag"], alpha=0.9))
        ax.text(x + max(cag, 0.25) / 2, y, str(cag), va="center", ha="center", fontsize=6, color="white")
        x += max(cag, 0.25)
        ax.add_patch(Rectangle((x, y - 0.34), 2.0, 0.68, color=colors["bridge"], alpha=0.8))
        x += 2.0
        ax.add_patch(Rectangle((x, y - 0.34), max(gcc, 0.25), 0.68, color=colors["gcc"], alpha=0.9))
        ax.text(x + max(gcc, 0.25) / 2, y, str(gcc), va="center", ha="center", fontsize=6, color="white")

    ax.set_yticks(range(len(sorted_rows)))
    ax.set_yticklabels([str(row["path"]) for row in sorted_rows], fontsize=6)
    ax.set_xlabel("Variable loop unit count")
    ax.set_title(f"IRF2BPL VNTR block model, inferred core nodes {core_start}-{core_end}")
    ax.text(0, -1.2, "gray=conserved repeat, blue=CAG-like variable loop, gold=common bridge, green=GCC-like variable loop", fontsize=8)
    ax.text(0, -0.75, "Path suffix is retained as a label/weight; it is not used as the loop count.", fontsize=8)
    ax.set_ylim(len(sorted_rows) - 0.4, -1.6)
    ax.set_xlim(0, max(int(row["CAG_like_variable_units"]) + int(row["GCC_like_variable_units"]) + 5 for row in sorted_rows))
    ax.grid(axis="x", color="#d0d5dd", linewidth=0.5, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact VNTR block graph from a PGGB GFA.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    segments, paths, _links = read_gfa(args.input)
    core_start, core_end, seed_nodes = infer_core_interval(segments)
    blocks = infer_blocks(segments, paths, core_start, core_end)

    summary_rows: list[dict[str, object]] = []
    for path in paths:
        core = path_seq(path, segments, core_start, core_end)
        cag_units = 0
        gcc_units = 0
        bridge_cag = 0
        bridge_gcc = 0
        parts: list[str] = []
        for block in blocks:
            seq = path_seq(path, segments, block.start_node, block.end_node)
            cag = sum(greedy_motif_counts(seq, CAG_LIKE).values())
            gcc = sum(greedy_motif_counts(seq, GCC_LIKE).values())
            if block.role == "variable_loop":
                if block.group == "CAG_like":
                    cag_units += cag
                    parts.append(f"CAG_like:{cag}")
                else:
                    gcc_units += gcc
                    parts.append(f"GCC_like:{gcc}")
            elif block.role == "common_bridge":
                bridge_cag += cag
                bridge_gcc += gcc
                parts.append(f"bridge:{len(seq)}bp")
            else:
                parts.append(f"{block.role}:{len(seq)}bp")

        summary_rows.append(
            {
                "path": path.name,
                "path_suffix_label": path_suffix_value(path.name),
                "core_node_start": core_start,
                "core_node_end": core_end,
                "core_bp": len(core),
                "CAG_like_variable_units": cag_units,
                "GCC_like_variable_units": gcc_units,
                "common_bridge_CAG_like_units": bridge_cag,
                "common_bridge_GCC_like_units": bridge_gcc,
                "block_signature": "|".join(parts),
            }
        )

    block_rows = [
        {
            "block_id": block.block_id,
            "role": block.role,
            "group": block.group,
            "node_start": block.start_node,
            "node_end": block.end_node,
            "nodes": ",".join(str(node) for node in block.nodes),
        }
        for block in blocks
    ]
    block_rows.insert(
        0,
        {
            "block_id": "INFERRED_CORE",
            "role": "core_interval",
            "group": "repeat_enriched_graph_cluster",
            "node_start": core_start,
            "node_end": core_end,
            "nodes": ",".join(str(node) for node in seed_nodes),
        },
    )

    summary_tsv = args.outdir / f"{args.prefix}_真实VNTR区块统计.tsv"
    block_tsv = args.outdir / f"{args.prefix}_真实VNTR区块定义.tsv"
    gfa_path = args.outdir / f"{args.prefix}_真实VNTR区块图.gfa"
    png_path = args.outdir / f"{args.prefix}_真实VNTR区块图.png"

    write_tsv(
        summary_tsv,
        summary_rows,
        [
            "path",
            "path_suffix_label",
            "core_node_start",
            "core_node_end",
            "core_bp",
            "CAG_like_variable_units",
            "GCC_like_variable_units",
            "common_bridge_CAG_like_units",
            "common_bridge_GCC_like_units",
            "block_signature",
        ],
    )
    write_tsv(block_tsv, block_rows, ["block_id", "role", "group", "node_start", "node_end", "nodes"])
    write_block_gfa(gfa_path, segments, paths, blocks, summary_rows)
    plot_png(png_path, summary_rows, core_start, core_end)

    print(f"Core VNTR interval: nodes {core_start}-{core_end}")
    print("Blocks:")
    for block in blocks:
        print(f"  {block.block_id}: {block.role}, {block.group}, nodes {block.start_node}-{block.end_node}")
    print(f"Wrote {summary_tsv}")
    print(f"Wrote {block_tsv}")
    print(f"Wrote {gfa_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
