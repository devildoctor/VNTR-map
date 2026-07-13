#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


DEFAULT_INPUT = Path("测试数据") / "IRF2BPL.HPRC.fa.3034e58.11fba48.904d69e.smooth.final.gfa"
DEFAULT_OUTDIR = Path("outputs") / "IRF2BPL" / "20260713_不规则VNTR区域自动分解"
DEFAULT_PREFIX = "20260713_IRF2BPL"

CAG_LIKE = ("CAG", "CAA")
GCC_LIKE = ("GCG", "GCC", "GCA", "GCT", "GGC")
MOTIFS = tuple(dict.fromkeys(CAG_LIKE + GCC_LIKE))
MOTIF_GROUP = {motif: "CAG_like" for motif in CAG_LIKE} | {motif: "GCC_like" for motif in GCC_LIKE}


@dataclass
class Segment:
    name: str
    seq: str


@dataclass
class NodeSpan:
    node: str
    path_start: int
    path_end: int
    seq: str


@dataclass
class PathRecord:
    name: str
    steps: list[str]
    seq: str = ""
    spans: list[NodeSpan] = field(default_factory=list)


@dataclass
class Piece:
    node: str
    seq: str
    start: int
    end: int
    role: str
    group: str
    exact_counts: Counter[str]
    variant_counts: Counter[str]
    repeat_bp: int


@dataclass
class Block:
    path: str
    index: int
    start: int
    end: int
    nodes: list[str]
    role: str
    group: str
    seq: str
    exact_counts: Counter[str]
    variant_counts: Counter[str]
    insertion_bp: int
    switches: int

    @property
    def exact_units(self) -> int:
        return sum(self.exact_counts.values())

    @property
    def variant_units(self) -> int:
        return sum(self.variant_counts.values())


def read_gfa(path: Path) -> tuple[dict[str, Segment], list[PathRecord]]:
    segments: dict[str, Segment] = {}
    paths: list[PathRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            fields = raw.rstrip("\n\r").split("\t")
            if not fields or not fields[0]:
                continue
            if fields[0] == "S":
                segments[fields[1]] = Segment(fields[1], fields[2].upper())
            elif fields[0] == "P":
                steps = [step[:-1] for step in fields[2].split(",") if step]
                paths.append(PathRecord(fields[1], steps))
    for path_record in paths:
        seq_parts: list[str] = []
        offset = 0
        for node in path_record.steps:
            seq = segments[node].seq
            seq_parts.append(seq)
            path_record.spans.append(NodeSpan(node, offset, offset + len(seq), seq))
            offset += len(seq)
        path_record.seq = "".join(seq_parts)
    return segments, paths


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(ca != cb for ca, cb in zip(a, b))


def exact_motif_at(seq: str, pos: int) -> str | None:
    piece = seq[pos : pos + 3]
    return piece if len(piece) == 3 and piece in MOTIF_GROUP else None


def nearby_exact(seq: str, pos: int, radius: int = 9) -> bool:
    start = max(0, pos - radius)
    end = min(len(seq) - 2, pos + radius)
    return any(exact_motif_at(seq, i) is not None for i in range(start, end + 1))


def best_variant_at(seq: str, pos: int) -> tuple[str, str] | None:
    piece = seq[pos : pos + 3]
    if len(piece) != 3 or not nearby_exact(seq, pos):
        return None
    candidates = [(motif, hamming(piece, motif)) for motif in MOTIFS]
    motif, distance = min(candidates, key=lambda item: item[1])
    if distance == 1:
        return motif, piece
    return None


def greedy_counts(seq: str, allow_variant: bool = False) -> tuple[Counter[str], Counter[str], int, int]:
    exact: Counter[str] = Counter()
    variant: Counter[str] = Counter()
    switches = 0
    last_group: str | None = None
    i = 0
    while i < len(seq):
        motif = exact_motif_at(seq, i)
        observed = motif
        is_variant = False
        if motif is None and allow_variant:
            hit = best_variant_at(seq, i)
            if hit is not None:
                motif, observed = hit
                is_variant = True
        if motif is None:
            i += 1
            continue
        group = MOTIF_GROUP[motif]
        if last_group is not None and group != last_group:
            switches += 1
        last_group = group
        if is_variant:
            variant[f"{motif}>{observed}"] += 1
        else:
            exact[motif] += 1
        i += 3
    repeat_bp = 3 * (sum(exact.values()) + sum(variant.values()))
    return exact, variant, repeat_bp, switches


def repeat_density(seq: str) -> float:
    _exact, _variant, repeat_bp, _switches = greedy_counts(seq, allow_variant=False)
    return repeat_bp / max(1, len(seq))


def locate_repeat_region(
    seq: str,
    window_bp: int = 60,
    step_bp: int = 6,
    density_threshold: float = 0.70,
    merge_gap_bp: int = 150,
    max_region_bp: int = 460,
) -> tuple[int, int]:
    windows: list[tuple[int, int, float]] = []
    if len(seq) < window_bp:
        return 0, len(seq)
    for start in range(0, len(seq) - window_bp + 1, step_bp):
        end = start + window_bp
        density = repeat_density(seq[start:end])
        if density >= density_threshold:
            windows.append((start, end, density))
    if not windows:
        return 0, len(seq)

    clusters: list[list[tuple[int, int, float]]] = []
    for window in windows:
        if not clusters or window[0] - clusters[-1][-1][1] > 30:
            clusters.append([window])
        else:
            clusters[-1].append(window)

    def score(cluster: list[tuple[int, int, float]]) -> tuple[float, int]:
        span = cluster[-1][1] - cluster[0][0]
        mean_density = sum(item[2] for item in cluster) / len(cluster)
        return span * mean_density, span

    best_i = max(range(len(clusters)), key=lambda i: score(clusters[i]))
    start = clusters[best_i][0][0]
    end = clusters[best_i][-1][1]

    i = best_i - 1
    while i >= 0:
        candidate_start = clusters[i][0][0]
        if start - clusters[i][-1][1] <= merge_gap_bp and end - candidate_start <= max_region_bp:
            start = candidate_start
            i -= 1
        else:
            break
    i = best_i + 1
    while i < len(clusters):
        candidate_end = clusters[i][-1][1]
        if clusters[i][0][0] - end <= merge_gap_bp and candidate_end - start <= max_region_bp:
            end = candidate_end
            i += 1
        else:
            break

    start, end = trim_to_repeat_edges(seq, start, end)
    return start, end


def trim_to_repeat_edges(seq: str, start: int, end: int) -> tuple[int, int]:
    region = seq[start:end]
    repeat_positions = [i for i in range(len(region) - 2) if exact_motif_at(region, i) is not None]
    if not repeat_positions:
        return start, end
    left = max(0, repeat_positions[0] - 12)
    right = min(len(region), repeat_positions[-1] + 15)
    return start + left, start + right


def clipped_pieces(path: PathRecord, start: int, end: int) -> list[tuple[str, str, int, int]]:
    pieces: list[tuple[str, str, int, int]] = []
    for span in path.spans:
        left = max(start, span.path_start)
        right = min(end, span.path_end)
        if left >= right:
            continue
        local_left = left - span.path_start
        local_right = right - span.path_start
        pieces.append((span.node, span.seq[local_left:local_right], left, right))
    return pieces


def classify_piece(node: str, seq: str, start: int, end: int) -> Piece:
    exact, variant, repeat_bp, switches = greedy_counts(seq, allow_variant=False)
    total_units = sum(exact.values()) + sum(variant.values())
    group_counts: Counter[str] = Counter()
    for motif, count in exact.items():
        group_counts[MOTIF_GROUP[motif]] += count
    for key, count in variant.items():
        motif = key.split(">", 1)[0]
        group_counts[MOTIF_GROUP[motif]] += count

    density = repeat_bp / max(1, len(seq))
    insertion_bp = max(0, len(seq) - repeat_bp)
    if total_units == 0 or density < 0.25:
        role = "inserted_sequence"
        group = "nonrepeat"
    elif len(group_counts) >= 2 and min(group_counts.values()) >= 2 and switches >= 1:
        role = "compound_mosaic"
        group = "mixed"
    elif density >= 0.80 and insertion_bp <= 6 and sum(variant.values()) == 0:
        role = "pure_repeat"
        group = group_counts.most_common(1)[0][0]
    else:
        role = "interrupted_or_variant_repeat"
        group = group_counts.most_common(1)[0][0]
    return Piece(node, seq, start, end, role, group, exact, variant, repeat_bp)


def merge_pieces(path_name: str, pieces: list[Piece], max_insert_bp: int = 18) -> list[Block]:
    blocks: list[Block] = []
    pending_insert: Piece | None = None

    def new_block(piece: Piece) -> Block:
        return Block(
            path=path_name,
            index=len(blocks) + 1,
            start=piece.start,
            end=piece.end,
            nodes=[piece.node],
            role=piece.role,
            group=piece.group,
            seq=piece.seq,
            exact_counts=Counter(piece.exact_counts),
            variant_counts=Counter(piece.variant_counts),
            insertion_bp=max(0, len(piece.seq) - piece.repeat_bp),
            switches=greedy_counts(piece.seq, allow_variant=False)[3],
        )

    for piece in pieces:
        if piece.role == "inserted_sequence" and len(piece.seq) <= max_insert_bp and blocks:
            pending_insert = piece
            continue

        if pending_insert is not None:
            if piece.group == blocks[-1].group and blocks[-1].group != "mixed":
                target = blocks[-1]
                target.seq += pending_insert.seq + piece.seq
                target.end = piece.end
                target.nodes.extend([pending_insert.node, piece.node])
                target.exact_counts.update(piece.exact_counts)
                target.variant_counts.update(piece.variant_counts)
                target.insertion_bp += len(pending_insert.seq) + max(0, len(piece.seq) - piece.repeat_bp)
                target.role = "split_or_interrupted_repeat"
                target.switches = greedy_counts(target.seq, allow_variant=False)[3]
                pending_insert = None
                continue
            blocks[-1].seq += pending_insert.seq
            blocks[-1].end = pending_insert.end
            blocks[-1].nodes.append(pending_insert.node)
            blocks[-1].insertion_bp += len(pending_insert.seq)
            if blocks[-1].role == "pure_repeat":
                blocks[-1].role = "interrupted_or_variant_repeat"
            pending_insert = None

        if blocks and can_merge(blocks[-1], piece):
            target = blocks[-1]
            target.seq += piece.seq
            target.end = piece.end
            target.nodes.append(piece.node)
            target.exact_counts.update(piece.exact_counts)
            target.variant_counts.update(piece.variant_counts)
            target.insertion_bp += max(0, len(piece.seq) - piece.repeat_bp)
            if piece.role != target.role:
                target.role = merged_role(target.role, piece.role)
            target.switches = greedy_counts(target.seq, allow_variant=False)[3]
        else:
            blocks.append(new_block(piece))

    if pending_insert is not None and blocks:
        blocks[-1].seq += pending_insert.seq
        blocks[-1].end = pending_insert.end
        blocks[-1].nodes.append(pending_insert.node)
        blocks[-1].insertion_bp += len(pending_insert.seq)
        if blocks[-1].role == "pure_repeat":
            blocks[-1].role = "interrupted_or_variant_repeat"

    for index, block in enumerate(blocks, start=1):
        block.index = index
    return blocks


def can_merge(block: Block, piece: Piece) -> bool:
    if piece.role == "compound_mosaic" or block.role == "compound_mosaic":
        return False
    if piece.group == block.group and piece.group != "nonrepeat":
        return True
    return False


def merged_role(left: str, right: str) -> str:
    if "compound" in {left, right}:
        return "compound_mosaic"
    if "split" in {left, right}:
        return "split_or_interrupted_repeat"
    if "interrupted" in {left, right}:
        return "interrupted_or_variant_repeat"
    return left


def motif_text(counter: Counter[str]) -> str:
    return ";".join(f"{key}:{counter[key]}" for key in sorted(counter)) or "-"


def path_suffix_value(name: str) -> str:
    match = re.search(r"_([0-9]+(?:\.[0-9]+)?)$", name)
    return match.group(1) if match else ""


def write_tsv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(col, "")) for col in columns) + "\n")


def write_gfa(path: Path, blocks_by_path: dict[str, list[Block]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("H\tVN:Z:1.0\tTS:Z:path_specific_irregular_vntr\n")
        for path_name, blocks in blocks_by_path.items():
            for block in blocks:
                node_id = f"{sanitize(path_name)}_B{block.index}_{short_role(block.role)}"
                tags = [
                    f"PN:Z:{path_name}",
                    f"BI:i:{block.index}",
                    f"BT:Z:{block.role}",
                    f"GR:Z:{block.group}",
                    f"UC:i:{block.exact_units + block.variant_units}",
                    f"IV:i:{block.insertion_bp}",
                    f"SW:i:{block.switches}",
                    f"MT:Z:{motif_text(block.exact_counts)}",
                    f"VA:Z:{motif_text(block.variant_counts)}",
                    f"ND:Z:{','.join(block.nodes)}",
                ]
                handle.write(f"S\t{node_id}\t{block.seq}\t" + "\t".join(tags) + "\n")
            for left, right in zip(blocks, blocks[1:]):
                left_id = f"{sanitize(path_name)}_B{left.index}_{short_role(left.role)}"
                right_id = f"{sanitize(path_name)}_B{right.index}_{short_role(right.role)}"
                handle.write(f"L\t{left_id}\t+\t{right_id}\t+\t0M\n")
        for path_name, blocks in blocks_by_path.items():
            steps = ",".join(f"{sanitize(path_name)}_B{block.index}_{short_role(block.role)}+" for block in blocks)
            overlaps = ",".join("0M" for _ in blocks[1:]) if len(blocks) > 1 else "*"
            signature = "|".join(f"{block.group}:{block.exact_units + block.variant_units}:{block.role}" for block in blocks)
            handle.write(f"P\t{path_name}\t{steps}\t{overlaps}\tSG:Z:{signature}\n")


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def short_role(role: str) -> str:
    return {
        "pure_repeat": "pure",
        "interrupted_or_variant_repeat": "intr",
        "split_or_interrupted_repeat": "split",
        "compound_mosaic": "mosaic",
        "inserted_sequence": "ins",
    }.get(role, role[:8])


def plot_png(path: Path, summary_rows: list[dict[str, object]], block_rows: list[dict[str, object]]) -> None:
    blocks_by_path: dict[str, list[dict[str, object]]] = {}
    for row in block_rows:
        blocks_by_path.setdefault(str(row["path"]), []).append(row)

    sorted_paths = [str(row["path"]) for row in sorted(summary_rows, key=lambda r: (int(r["region_bp"]), str(r["path"])))]
    fig_h = max(9, 0.26 * len(sorted_paths) + 2.6)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    colors = {
        "CAG_like": "#3b82c4",
        "GCC_like": "#4f9d69",
        "mixed": "#9b5de5",
        "nonrepeat": "#9aa1ad",
    }
    hatch_by_role = {
        "pure_repeat": "",
        "interrupted_or_variant_repeat": "//",
        "split_or_interrupted_repeat": "xx",
        "compound_mosaic": "..",
        "inserted_sequence": "",
    }

    max_x = 1
    for y, path_name in enumerate(sorted_paths):
        x = 0.0
        for block in blocks_by_path[path_name]:
            units = int(block["exact_units"]) + int(block["variant_units"])
            width = max(0.35, units if units else int(block["bp"]) / 9)
            color = colors.get(str(block["group"]), "#9aa1ad")
            rect = Rectangle((x, y - 0.34), width, 0.68, facecolor=color, edgecolor="white", linewidth=0.7)
            rect.set_hatch(hatch_by_role.get(str(block["role"]), ""))
            ax.add_patch(rect)
            if width >= 1.2:
                label = str(units) if units else f"{block['bp']}bp"
                ax.text(x + width / 2, y, label, va="center", ha="center", fontsize=6, color="white")
            x += width
        max_x = max(max_x, x)

    ax.set_yticks(range(len(sorted_paths)))
    ax.set_yticklabels(sorted_paths, fontsize=6)
    ax.set_xlabel("Block width: repeat-unit count; nonrepeat width scaled by bp")
    ax.set_title("IRF2BPL path-specific irregular VNTR decomposition")
    ax.text(
        0,
        -1.25,
        "blue=CAG-like, green=GCC-like, purple=compound/mosaic; hatch marks interrupted/split blocks.",
        fontsize=8,
    )
    ax.set_xlim(0, max_x + 1)
    ax.set_ylim(len(sorted_paths) - 0.4, -1.8)
    ax.grid(axis="x", color="#d0d5dd", linewidth=0.5, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(paths: list[PathRecord]) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, list[Block]]]:
    summary_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    blocks_by_path: dict[str, list[Block]] = {}
    for path_record in paths:
        start, end = locate_repeat_region(path_record.seq)
        raw_pieces = clipped_pieces(path_record, start, end)
        pieces = [classify_piece(node, seq, piece_start, piece_end) for node, seq, piece_start, piece_end in raw_pieces]
        blocks = merge_pieces(path_record.name, pieces)
        blocks_by_path[path_record.name] = blocks

        total_exact = sum(block.exact_units for block in blocks)
        total_variant = sum(block.variant_units for block in blocks)
        total_insert = sum(block.insertion_bp for block in blocks)
        total_switches = sum(block.switches for block in blocks)
        role_counts = Counter(block.role for block in blocks)
        group_units: Counter[str] = Counter()
        for block in blocks:
            group_units[block.group] += block.exact_units + block.variant_units

        summary_rows.append(
            {
                "path": path_record.name,
                "path_suffix_label": path_suffix_value(path_record.name),
                "region_start_bp": start,
                "region_end_bp": end,
                "region_bp": end - start,
                "block_count": len(blocks),
                "CAG_like_units": group_units["CAG_like"],
                "GCC_like_units": group_units["GCC_like"],
                "mixed_units": group_units["mixed"],
                "exact_units": total_exact,
                "variant_units": total_variant,
                "inserted_bp": total_insert,
                "switches": total_switches,
                "pure_blocks": role_counts["pure_repeat"],
                "interrupted_blocks": role_counts["interrupted_or_variant_repeat"] + role_counts["split_or_interrupted_repeat"],
                "compound_blocks": role_counts["compound_mosaic"],
                "block_signature": "|".join(
                    f"{block.group}:{block.exact_units + block.variant_units}:{block.role}" for block in blocks
                ),
            }
        )

        for block in blocks:
            block_rows.append(
                {
                    "path": block.path,
                    "block_index": block.index,
                    "role": block.role,
                    "group": block.group,
                    "start_bp": block.start,
                    "end_bp": block.end,
                    "bp": block.end - block.start,
                    "nodes": ",".join(block.nodes),
                    "exact_units": block.exact_units,
                    "variant_units": block.variant_units,
                    "inserted_bp": block.insertion_bp,
                    "switches": block.switches,
                    "motif_counts": motif_text(block.exact_counts),
                    "variant_counts": motif_text(block.variant_counts),
                    "sequence": block.seq,
                }
            )
    return summary_rows, block_rows, blocks_by_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Path-specific irregular VNTR decomposition from a PGGB GFA.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    _segments, paths = read_gfa(args.input)
    summary_rows, block_rows, blocks_by_path = analyze(paths)

    summary_tsv = args.outdir / f"{args.prefix}_路径不规则VNTR区域汇总.tsv"
    detail_tsv = args.outdir / f"{args.prefix}_路径不规则VNTR_block明细.tsv"
    gfa_path = args.outdir / f"{args.prefix}_路径不规则VNTR_block图.gfa"
    png_path = args.outdir / f"{args.prefix}_路径不规则VNTR_block图.png"

    write_tsv(
        summary_tsv,
        summary_rows,
        [
            "path",
            "path_suffix_label",
            "region_start_bp",
            "region_end_bp",
            "region_bp",
            "block_count",
            "CAG_like_units",
            "GCC_like_units",
            "mixed_units",
            "exact_units",
            "variant_units",
            "inserted_bp",
            "switches",
            "pure_blocks",
            "interrupted_blocks",
            "compound_blocks",
            "block_signature",
        ],
    )
    write_tsv(
        detail_tsv,
        block_rows,
        [
            "path",
            "block_index",
            "role",
            "group",
            "start_bp",
            "end_bp",
            "bp",
            "nodes",
            "exact_units",
            "variant_units",
            "inserted_bp",
            "switches",
            "motif_counts",
            "variant_counts",
            "sequence",
        ],
    )
    write_gfa(gfa_path, blocks_by_path)
    plot_png(png_path, summary_rows, block_rows)

    print(f"Wrote {summary_tsv}")
    print(f"Wrote {detail_tsv}")
    print(f"Wrote {gfa_path}")
    print(f"Wrote {png_path}")
    cag_values = [int(row["CAG_like_units"]) for row in summary_rows]
    gcc_values = [int(row["GCC_like_units"]) for row in summary_rows]
    mixed_values = [int(row["mixed_units"]) for row in summary_rows]
    print(f"CAG-like units: {min(cag_values)}-{max(cag_values)}")
    print(f"GCC-like units: {min(gcc_values)}-{max(gcc_values)}")
    print(f"Mixed/mosaic units: {min(mixed_values)}-{max(mixed_values)}")


if __name__ == "__main__":
    main()
