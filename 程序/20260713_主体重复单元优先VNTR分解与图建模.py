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
DEFAULT_OUTDIR = Path("outputs") / "IRF2BPL" / "20260713_主体重复单元优先分解"
DEFAULT_PREFIX = "20260713_IRF2BPL_主体优先"

CAG_LIKE = ("CAG", "CAA")
GCC_LIKE = ("GCG", "GCC", "GCA", "GCT", "GGC")
MOTIFS = tuple(dict.fromkeys(CAG_LIKE + GCC_LIKE))
MOTIF_GROUP = {motif: "CAG_like" for motif in CAG_LIKE} | {motif: "GCC_like" for motif in GCC_LIKE}
PRIMARY_CANONICAL = {
    "CAG_like": frozenset({"CAG"}),
    "GCC_like": frozenset({"GCC", "GCG"}),
}
MIN_PRIMARY_UNITS = 3
MIN_CORE_UNITS = 5
MAX_LOCAL_COMPLEX_BP = 12
LEFT_CONTEXT_BP = 27
RIGHT_CONTEXT_BP = 15


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


@dataclass
class PrimaryAnchor:
    start: int
    end: int
    group: str
    motifs: list[str]
    score: int

    @property
    def units(self) -> int:
        return len(self.motifs)


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
        handle.write("H\tVN:Z:1.0\tTS:Z:primary_repeat_first_vntr\n")
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
        "primary_repeat": "primary",
        "local_variant_repeat": "variant",
        "locally_interrupted_repeat": "interrupted",
        "local_insertion": "localins",
        "complex_sequence": "complex",
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
        "nonrepeat": "#9aa1ad",
    }
    hatch_by_role = {
        "primary_repeat": "",
        "local_variant_repeat": "//",
        "locally_interrupted_repeat": "xx",
        "local_insertion": "//",
        "complex_sequence": "..",
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
    ax.set_xlabel("Block width: primary repeat-unit count; nonrepeat width scaled by bp")
    ax.set_title("IRF2BPL primary-repeat-first VNTR decomposition")
    ax.text(
        0,
        -1.25,
        "blue=CAG-like primary blocks, green=GCC-like primary blocks, gray=local insertion or complex sequence.",
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


def primary_anchor_candidates(seq: str, start: int = 0, end: int | None = None) -> list[PrimaryAnchor]:
    """Find uninterrupted three-base repeat runs before considering local complexity."""
    end = len(seq) if end is None else end
    candidates: list[PrimaryAnchor] = []
    for frame in range(3):
        run_start: int | None = None
        run_group: str | None = None
        run_motifs: list[str] = []

        def flush() -> None:
            nonlocal run_start, run_group, run_motifs
            if run_start is not None and run_group is not None and len(run_motifs) >= MIN_PRIMARY_UNITS:
                canonical = sum(motif in PRIMARY_CANONICAL[run_group] for motif in run_motifs)
                score = 5 * canonical + 2 * (len(run_motifs) - canonical)
                candidates.append(
                    PrimaryAnchor(run_start, run_start + 3 * len(run_motifs), run_group, list(run_motifs), score)
                )
            run_start = None
            run_group = None
            run_motifs = []

        first = start + ((frame - start) % 3)
        for pos in range(first, end - 2, 3):
            motif = seq[pos : pos + 3]
            group = MOTIF_GROUP.get(motif)
            if group is None:
                flush()
                continue
            if run_group is None:
                run_start = pos
                run_group = group
                run_motifs = [motif]
            elif group == run_group:
                run_motifs.append(motif)
            else:
                flush()
                run_start = pos
                run_group = group
                run_motifs = [motif]
        flush()
    return candidates


def select_nonoverlapping_anchors(candidates: list[PrimaryAnchor]) -> list[PrimaryAnchor]:
    """Choose the strongest compatible anchors, favoring canonical motif frames."""
    ordered = sorted(candidates, key=lambda anchor: (anchor.end, anchor.start, -anchor.score))
    if not ordered:
        return []
    best_scores = [0] * (len(ordered) + 1)
    best_sets: list[list[PrimaryAnchor]] = [[] for _ in range(len(ordered) + 1)]
    for i, anchor in enumerate(ordered, start=1):
        previous = 0
        for j in range(i - 1, 0, -1):
            if ordered[j - 1].end <= anchor.start:
                previous = j
                break
        include_score = best_scores[previous] + anchor.score
        exclude_score = best_scores[i - 1]
        if include_score > exclude_score:
            best_scores[i] = include_score
            best_sets[i] = best_sets[previous] + [anchor]
        else:
            best_scores[i] = exclude_score
            best_sets[i] = list(best_sets[i - 1])
    return sorted(best_sets[-1], key=lambda anchor: anchor.start)


def locate_primary_region(seq: str, merge_gap_bp: int = 150, max_region_bp: int = 460) -> tuple[int, int, list[PrimaryAnchor]]:
    anchors = select_nonoverlapping_anchors(primary_anchor_candidates(seq))
    strong_anchors = [anchor for anchor in anchors if anchor.units >= MIN_CORE_UNITS]
    if not strong_anchors:
        start, end = locate_repeat_region(seq)
        return start, end, []

    clusters: list[list[PrimaryAnchor]] = []
    for anchor in strong_anchors:
        if (
            not clusters
            or anchor.start - clusters[-1][-1].end > merge_gap_bp
            or anchor.end - clusters[-1][0].start > max_region_bp
        ):
            clusters.append([anchor])
        else:
            clusters[-1].append(anchor)
    core = max(clusters, key=lambda items: (sum(item.score for item in items), sum(item.units for item in items)))

    selected = list(core)
    changed = True
    while changed:
        changed = False
        for anchor in anchors:
            if anchor in selected:
                continue
            if any(
                anchor.group == chosen.group
                and (0 <= anchor.start - chosen.end <= MAX_LOCAL_COMPLEX_BP
                     or 0 <= chosen.start - anchor.end <= MAX_LOCAL_COMPLEX_BP)
                for chosen in selected
            ):
                selected.append(anchor)
                changed = True
    selected.sort(key=lambda anchor: anchor.start)
    start = max(0, selected[0].start - LEFT_CONTEXT_BP)
    end = min(len(seq), selected[-1].end + RIGHT_CONTEXT_BP)
    return start, end, selected


def nodes_for_interval(path: PathRecord, start: int, end: int) -> list[str]:
    return [span.node for span in path.spans if span.path_start < end and span.path_end > start]


def variant_gap_counts(seq: str, group: str) -> Counter[str] | None:
    """Interpret only a small, in-frame gap as variant units; otherwise retain it as an insertion."""
    if not seq or len(seq) % 3 != 0:
        return None
    variants: Counter[str] = Counter()
    allowed = [motif for motif in MOTIFS if MOTIF_GROUP[motif] == group]
    for pos in range(0, len(seq), 3):
        observed = seq[pos : pos + 3]
        if observed in allowed:
            return None
        motif, distance = min(((motif, hamming(observed, motif)) for motif in allowed), key=lambda item: item[1])
        if distance != 1:
            return None
        variants[f"{motif}>{observed}"] += 1
    return variants


def anchor_block(path: PathRecord, anchor: PrimaryAnchor) -> Block:
    return Block(
        path=path.name,
        index=0,
        start=anchor.start,
        end=anchor.end,
        nodes=nodes_for_interval(path, anchor.start, anchor.end),
        role="primary_repeat",
        group=anchor.group,
        seq=path.seq[anchor.start : anchor.end],
        exact_counts=Counter(anchor.motifs),
        variant_counts=Counter(),
        insertion_bp=0,
        switches=0,
    )


def insertion_block(path: PathRecord, start: int, end: int) -> Block:
    length = end - start
    role = "local_insertion" if length <= MAX_LOCAL_COMPLEX_BP else "complex_sequence"
    return Block(
        path=path.name,
        index=0,
        start=start,
        end=end,
        nodes=nodes_for_interval(path, start, end),
        role=role,
        group="nonrepeat",
        seq=path.seq[start:end],
        exact_counts=Counter(),
        variant_counts=Counter(),
        insertion_bp=length,
        switches=0,
    )


def decompose_primary_region(
    path: PathRecord, start: int, end: int, anchors: list[PrimaryAnchor]
) -> list[Block]:
    anchors = [anchor for anchor in anchors if anchor.start >= start and anchor.end <= end]
    if not anchors:
        return [insertion_block(path, start, end)]

    blocks: list[Block] = []
    if start < anchors[0].start:
        blocks.append(insertion_block(path, start, anchors[0].start))

    current = anchor_block(path, anchors[0])
    for anchor in anchors[1:]:
        gap_start = current.end
        gap_end = anchor.start
        gap_bp = gap_end - gap_start
        if anchor.group == current.group and 0 <= gap_bp <= MAX_LOCAL_COMPLEX_BP:
            gap_seq = path.seq[gap_start:gap_end]
            variants = variant_gap_counts(gap_seq, current.group)
            current.end = anchor.end
            current.seq = path.seq[current.start : current.end]
            current.nodes = nodes_for_interval(path, current.start, current.end)
            current.exact_counts.update(anchor.motifs)
            if variants is not None:
                current.variant_counts.update(variants)
                current.role = "local_variant_repeat"
            elif gap_bp:
                current.insertion_bp += gap_bp
                current.role = "locally_interrupted_repeat"
            continue

        blocks.append(current)
        if gap_bp > 0:
            blocks.append(insertion_block(path, gap_start, gap_end))
        current = anchor_block(path, anchor)
    blocks.append(current)

    if blocks[-1].end < end:
        blocks.append(insertion_block(path, blocks[-1].end, end))
    for index, block in enumerate(blocks, start=1):
        block.index = index
    return blocks


def analyze(paths: list[PathRecord]) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, list[Block]]]:
    summary_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    blocks_by_path: dict[str, list[Block]] = {}
    located: list[tuple[PathRecord, int, int, list[PrimaryAnchor]]] = []
    primary_evidence: Counter[str] = Counter()
    for path_record in paths:
        start, end, anchors = locate_primary_region(path_record.seq)
        located.append((path_record, start, end, anchors))
        for anchor in anchors:
            primary_evidence.update(anchor.motifs)
    primary_labels: list[str] = []
    for group, canonical in PRIMARY_CANONICAL.items():
        supported = sorted((motif for motif in canonical if primary_evidence[motif]), key=lambda motif: -primary_evidence[motif])
        if supported:
            primary_labels.append(f"{group}={','.join(supported)}")
    primary_motifs = ";".join(primary_labels)

    for path_record, start, end, anchors in located:
        blocks = decompose_primary_region(path_record, start, end, anchors)
        blocks_by_path[path_record.name] = blocks

        total_exact = sum(block.exact_units for block in blocks)
        total_variant = sum(block.variant_units for block in blocks)
        total_insert = sum(block.insertion_bp for block in blocks)
        repeat_groups = [block.group for block in blocks if block.group != "nonrepeat"]
        total_switches = sum(left != right for left, right in zip(repeat_groups, repeat_groups[1:]))
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
                "primary_motifs": primary_motifs,
                "block_count": len(blocks),
                "CAG_like_units": group_units["CAG_like"],
                "GCC_like_units": group_units["GCC_like"],
                "exact_units": total_exact,
                "variant_units": total_variant,
                "inserted_bp": total_insert,
                "switches": total_switches,
                "primary_blocks": role_counts["primary_repeat"],
                "interrupted_blocks": role_counts["locally_interrupted_repeat"] + role_counts["local_variant_repeat"],
                "complex_blocks": role_counts["complex_sequence"],
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
    parser = argparse.ArgumentParser(description="Primary-repeat-first VNTR decomposition from a PGGB GFA.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    _segments, paths = read_gfa(args.input)
    summary_rows, block_rows, blocks_by_path = analyze(paths)

    summary_tsv = args.outdir / f"{args.prefix}_主体重复优先_VNTR区域汇总.tsv"
    detail_tsv = args.outdir / f"{args.prefix}_主体重复优先_block明细.tsv"
    gfa_path = args.outdir / f"{args.prefix}_主体重复优先_block图.gfa"
    png_path = args.outdir / f"{args.prefix}_主体重复优先_block图.png"

    write_tsv(
        summary_tsv,
        summary_rows,
        [
            "path",
            "path_suffix_label",
            "region_start_bp",
            "region_end_bp",
            "region_bp",
            "primary_motifs",
            "block_count",
            "CAG_like_units",
            "GCC_like_units",
            "exact_units",
            "variant_units",
            "inserted_bp",
            "switches",
            "primary_blocks",
            "interrupted_blocks",
            "complex_blocks",
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
    print(f"CAG-like units: {min(cag_values)}-{max(cag_values)}")
    print(f"GCC-like units: {min(gcc_values)}-{max(gcc_values)}")
    print(f"Primary motifs: {summary_rows[0]['primary_motifs']}")


if __name__ == "__main__":
    main()
