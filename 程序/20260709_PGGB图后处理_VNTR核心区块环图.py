#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle


DEFAULT_INPUT = Path("测试数据") / "IRF2BPL.HPRC.fa.3034e58.11fba48.904d69e.smooth.final.gfa"
DEFAULT_OUTDIR = Path("outputs") / "IRF2BPL" / "20260709_PGGB图后处理_v3核心区块环图"
DEFAULT_PREFIX = "20260709_IRF2BPL"
DEFAULT_MOTIFS = ("CAG", "GCC", "GCG", "GGC", "CAA")


@dataclass
class Segment:
    name: str
    seq: str


@dataclass
class PathRecord:
    name: str
    steps: list[tuple[str, str]]


@dataclass
class Token:
    kind: str
    motif: str
    seq: str
    edits: int
    start: int
    end: int


@dataclass
class CoreBlock:
    path: str
    tokens: list[Token]
    start: int
    end: int
    repeat_count: int
    exact_counts: Counter[str]
    variant_counts: Counter[str]
    insert_bp: int
    transitions: Counter[tuple[str, str]]


def revcomp(seq: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1].upper()


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
                steps = [(step[:-1], step[-1]) for step in fields[2].split(",") if step]
                paths.append(PathRecord(fields[1], steps))
    return segments, paths


def oriented_sequence(segment: Segment, orient: str) -> str:
    return segment.seq if orient == "+" else revcomp(segment.seq)


def path_sequence(path: PathRecord, segments: dict[str, Segment]) -> str:
    return "".join(oriented_sequence(segments[name], orient) for name, orient in path.steps)


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(ca != cb for ca, cb in zip(a, b))


def best_motif(piece: str, motifs: tuple[str, ...], max_mismatches: int) -> tuple[str, int] | None:
    best: tuple[str, int] | None = None
    for motif in motifs:
        if len(piece) != len(motif):
            continue
        edits = hamming(piece, motif)
        if edits <= max_mismatches and (best is None or edits < best[1]):
            best = (motif, edits)
    return best


def decompose(seq: str, motifs: tuple[str, ...], max_mismatches: int) -> list[Token]:
    motif_len = len(motifs[0])
    tokens: list[Token] = []
    i = 0
    nonrepeat_start: int | None = None

    def flush_nonrepeat(end: int) -> None:
        nonlocal nonrepeat_start
        if nonrepeat_start is not None and end > nonrepeat_start:
            tokens.append(Token("nonrepeat", "-", seq[nonrepeat_start:end], 0, nonrepeat_start, end))
            nonrepeat_start = None

    while i < len(seq):
        piece = seq[i : i + motif_len]
        hit = best_motif(piece, motifs, max_mismatches) if len(piece) == motif_len else None
        if hit is None:
            if nonrepeat_start is None:
                nonrepeat_start = i
            i += 1
            continue
        flush_nonrepeat(i)
        motif, edits = hit
        kind = "exact" if edits == 0 else "variant"
        tokens.append(Token(kind, motif, piece, edits, i, i + motif_len))
        i += motif_len
    flush_nonrepeat(len(seq))
    return tokens


def token_repeat_count(tokens: list[Token]) -> int:
    return sum(1 for token in tokens if token.kind in {"exact", "variant"})


def token_insert_bp(tokens: list[Token]) -> int:
    return sum(len(token.seq) for token in tokens if token.kind == "nonrepeat")


def locate_core_block(tokens: list[Token], bridge_bp: int, min_repeats: int) -> list[Token]:
    blocks: list[list[Token]] = []
    current: list[Token] = []
    pending_nonrepeat: list[Token] = []

    for token in tokens:
        if token.kind in {"exact", "variant"}:
            if not current:
                current = pending_nonrepeat[-1:] if pending_nonrepeat and token_insert_bp(pending_nonrepeat) <= bridge_bp else []
            else:
                current.extend(pending_nonrepeat)
            pending_nonrepeat = []
            current.append(token)
        else:
            if current:
                pending_nonrepeat.append(token)
                if token_insert_bp(pending_nonrepeat) > bridge_bp:
                    blocks.append(current)
                    current = []
                    pending_nonrepeat = [token]
            else:
                pending_nonrepeat = [token]

    if current:
        current.extend(t for t in pending_nonrepeat if token_insert_bp([t]) <= bridge_bp)
        blocks.append(current)

    good_blocks = [block for block in blocks if token_repeat_count(block) >= min_repeats]
    if not good_blocks:
        return max(blocks, key=token_repeat_count) if blocks else []
    return max(good_blocks, key=lambda block: (token_repeat_count(block), -token_insert_bp(block), len(block)))


def summarize_core(path_name: str, block_tokens: list[Token]) -> CoreBlock:
    exact: Counter[str] = Counter()
    variant: Counter[str] = Counter()
    transitions: Counter[tuple[str, str]] = Counter()
    last_motif: str | None = None
    for token in block_tokens:
        if token.kind == "exact":
            exact[token.motif] += 1
        elif token.kind == "variant":
            variant[f"{token.motif}:{token.seq}"] += 1
        if token.kind in {"exact", "variant"}:
            if last_motif is not None:
                transitions[(last_motif, token.motif)] += 1
            last_motif = token.motif
    start = block_tokens[0].start if block_tokens else 0
    end = block_tokens[-1].end if block_tokens else 0
    return CoreBlock(
        path=path_name,
        tokens=block_tokens,
        start=start,
        end=end,
        repeat_count=sum(exact.values()) + sum(variant.values()),
        exact_counts=exact,
        variant_counts=variant,
        insert_bp=token_insert_bp(block_tokens),
        transitions=transitions,
    )


def variant_counts_by_motif(variant_counts: Counter[str]) -> Counter[str]:
    by_motif: Counter[str] = Counter()
    for key, value in variant_counts.items():
        by_motif[key.split(":", 1)[0]] += value
    return by_motif


def write_tsv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(col, "")) for col in columns) + "\n")


def write_gfa(path: Path, blocks: list[CoreBlock], motifs: tuple[str, ...]) -> None:
    transition_counts: Counter[tuple[str, str]] = Counter()
    motif_path_counts: Counter[str] = Counter()
    motif_repeat_counts: Counter[str] = Counter()
    for block in blocks:
        transition_counts.update(block.transitions)
        counts = block.exact_counts + variant_counts_by_motif(block.variant_counts)
        for motif, count in counts.items():
            motif_path_counts[motif] += 1
            motif_repeat_counts[motif] += count

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("H\tVN:Z:1.0\n")
        handle.write("H\tCL:Z:20260709_VNTR_core_block_loop_graph\n")
        handle.write("S\tLEFT_FLANK\t*\tTP:Z:flank\n")
        for motif in motifs:
            handle.write(
                "\t".join(
                    [
                        "S",
                        f"R_{motif}",
                        motif,
                        "TP:Z:repeat_loop",
                        f"PC:i:{motif_path_counts[motif]}",
                        f"RC:i:{motif_repeat_counts[motif]}",
                    ]
                )
                + "\n"
            )
        handle.write("S\tINSERTION\t*\tTP:Z:short_interspersed_nonrepeat\n")
        handle.write("S\tRIGHT_FLANK\t*\tTP:Z:flank\n")
        for motif in motifs:
            handle.write("\t".join(["L", f"R_{motif}", "+", f"R_{motif}", "+", "0M", f"LC:i:{motif_repeat_counts[motif]}"]) + "\n")
        for (left, right), count in sorted(transition_counts.items()):
            if left == right:
                continue
            handle.write("\t".join(["L", f"R_{left}", "+", f"R_{right}", "+", "0M", f"RC:i:{count}"]) + "\n")
        for block in blocks:
            counts = block.exact_counts + variant_counts_by_motif(block.variant_counts)
            used = [motif for motif in motifs if counts[motif] > 0]
            steps = ["LEFT_FLANK+"] + [f"R_{motif}+" for motif in used]
            loop_counts = ["1"] + [str(counts[motif]) for motif in used]
            if block.insert_bp:
                steps.append("INSERTION+")
                loop_counts.append(str(block.insert_bp))
            steps.append("RIGHT_FLANK+")
            loop_counts.append("1")
            handle.write(
                "\t".join(
                    [
                        "P",
                        block.path,
                        ",".join(steps),
                        "*",
                        f"LC:Z:{','.join(loop_counts)}",
                        f"ST:i:{block.start}",
                        f"EN:i:{block.end}",
                    ]
                )
                + "\n"
            )


def color_for(motif: str) -> str:
    return {
        "CAA": "#b56576",
        "CAG": "#f4a261",
        "GCC": "#2a9d8f",
        "GCG": "#457b9d",
        "GGC": "#e9c46a",
    }.get(motif, "#adb5bd")


def make_png(path: Path, blocks: list[CoreBlock], motifs: tuple[str, ...]) -> None:
    fig_h = max(9, 3.6 + len(blocks) * 0.24)
    fig = plt.figure(figsize=(16, fig_h), dpi=230)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, max(5, len(blocks) * 0.24)], hspace=0.22)
    ax_top = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])

    ax_top.set_title("Core VNTR block graph: one loop node per motif family", fontsize=13)
    ax_top.set_xlim(-0.8, len(motifs) - 0.2)
    ax_top.set_ylim(-1.0, 1.2)
    ax_top.axis("off")

    transition_counts: Counter[tuple[str, str]] = Counter()
    repeat_counts: Counter[str] = Counter()
    path_counts: Counter[str] = Counter()
    for block in blocks:
        transition_counts.update(block.transitions)
        counts = block.exact_counts + variant_counts_by_motif(block.variant_counts)
        for motif, count in counts.items():
            repeat_counts[motif] += count
            path_counts[motif] += 1

    positions = {motif: (idx, 0.0) for idx, motif in enumerate(motifs)}
    for (left, right), count in transition_counts.most_common(40):
        if left not in positions or right not in positions or left == right:
            continue
        x1, y1 = positions[left]
        x2, y2 = positions[right]
        patch = FancyArrowPatch(
            (x1, y1 - 0.18),
            (x2, y2 - 0.18),
            connectionstyle=f"arc3,rad={0.18 if x2 >= x1 else -0.18}",
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=min(3.5, 0.6 + math.log2(count + 1) * 0.35),
            color="#667085",
            alpha=0.38,
        )
        ax_top.add_patch(patch)

    for motif in motifs:
        x, y = positions[motif]
        ax_top.add_patch(Circle((x, y), 0.26, color=color_for(motif), ec="#1f2937", lw=1.1, zorder=3))
        ax_top.add_patch(FancyArrowPatch((x - 0.26, y + 0.37), (x + 0.26, y + 0.37), connectionstyle="arc3,rad=-0.7", arrowstyle="-|>", mutation_scale=9, lw=1.1, color="#111827"))
        ax_top.text(x, y, motif, ha="center", va="center", fontsize=9)
        ax_top.text(x, y - 0.48, f"paths:{path_counts[motif]} repeats:{repeat_counts[motif]}", ha="center", va="center", fontsize=7, color="#4b5563")

    path_names = [block.path for block in blocks]
    ax.set_title("Per-path core VNTR block summary: cell number is repeat count in the selected core block", fontsize=13)
    ax.set_xlim(-0.8, len(motifs) + 1.5)
    ax.set_ylim(-1, len(blocks))
    ax.set_xticks(list(range(len(motifs))) + [len(motifs), len(motifs) + 1])
    ax.set_xticklabels(list(motifs) + ["insert bp", "core bp"], fontsize=9)
    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels(path_names, fontsize=6)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#e5e7eb", lw=0.45)

    for row, block in enumerate(blocks):
        variant_by_motif = variant_counts_by_motif(block.variant_counts)
        counts = block.exact_counts + variant_by_motif
        for col, motif in enumerate(motifs):
            count = counts[motif]
            if count == 0:
                continue
            alpha = min(1.0, 0.35 + count / 90)
            ax.add_patch(Rectangle((col - 0.42, row - 0.38), 0.84, 0.76, facecolor=color_for(motif), edgecolor="#344054", lw=0.3, alpha=alpha))
            label = str(count) if variant_by_motif[motif] == 0 else f"{count}\nv{variant_by_motif[motif]}"
            ax.text(col, row, label, ha="center", va="center", fontsize=6.2, color="#111827")
        ax.text(len(motifs), row, str(block.insert_bp), ha="center", va="center", fontsize=6.5)
        ax.text(len(motifs) + 1, row, str(block.end - block.start), ha="center", va="center", fontsize=6.5)

    legend_handles = [Rectangle((0, 0), 1, 1, facecolor=color_for(motif), edgecolor="#344054", label=motif) for motif in motifs]
    ax.legend(handles=legend_handles, ncol=len(motifs), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Locate core VNTR blocks and draw block-level loop graph.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--motifs", default=",".join(DEFAULT_MOTIFS))
    parser.add_argument("--max-mismatches", type=int, default=1)
    parser.add_argument("--bridge-bp", type=int, default=18)
    parser.add_argument("--min-repeats", type=int, default=20)
    args = parser.parse_args()

    motifs = tuple(m.strip().upper() for m in args.motifs.split(",") if m.strip())
    args.outdir.mkdir(parents=True, exist_ok=True)

    segments, paths = read_gfa(args.input)
    blocks: list[CoreBlock] = []
    run_rows: list[dict[str, object]] = []
    for path_record in sorted(paths, key=lambda p: p.name):
        seq = path_sequence(path_record, segments)
        tokens = decompose(seq, motifs, args.max_mismatches)
        core_tokens = locate_core_block(tokens, args.bridge_bp, args.min_repeats)
        block = summarize_core(path_record.name, core_tokens)
        blocks.append(block)
        for token in core_tokens:
            if token.kind == "nonrepeat":
                continue
            run_rows.append(
                {
                    "path": path_record.name,
                    "start": token.start,
                    "end": token.end,
                    "kind": token.kind,
                    "motif": token.motif,
                    "observed_seq": token.seq,
                    "edit_distance": token.edits,
                }
            )

    summary_rows: list[dict[str, object]] = []
    for block in blocks:
        total_counts = block.exact_counts + variant_counts_by_motif(block.variant_counts)
        summary_rows.append(
            {
                "path": block.path,
                "core_start": block.start,
                "core_end": block.end,
                "core_length": block.end - block.start,
                "repeat_count": block.repeat_count,
                "insert_bp": block.insert_bp,
                "motif_counts": ";".join(f"{motif}={total_counts[motif]}" for motif in motifs),
                "exact_counts": ";".join(f"{k}={v}" for k, v in sorted(block.exact_counts.items())),
                "variant_counts": ";".join(f"{k}={v}" for k, v in sorted(block.variant_counts.items())),
            }
        )

    gfa_path = args.outdir / f"{args.prefix}_VNTR核心区块环图.gfa"
    png_path = args.outdir / f"{args.prefix}_VNTR核心区块环图可视化.png"
    summary_tsv = args.outdir / f"{args.prefix}_路径核心VNTR区块统计.tsv"
    detail_tsv = args.outdir / f"{args.prefix}_核心区块重复单元明细.tsv"
    summary_json = args.outdir / f"{args.prefix}_运行摘要.json"

    write_gfa(gfa_path, blocks, motifs)
    make_png(png_path, blocks, motifs)
    write_tsv(
        summary_tsv,
        summary_rows,
        ["path", "core_start", "core_end", "core_length", "repeat_count", "insert_bp", "motif_counts", "exact_counts", "variant_counts"],
    )
    write_tsv(detail_tsv, run_rows, ["path", "start", "end", "kind", "motif", "observed_seq", "edit_distance"])

    manifest = {
        "input": str(args.input),
        "motifs": motifs,
        "max_mismatches": args.max_mismatches,
        "bridge_bp": args.bridge_bp,
        "min_repeats": args.min_repeats,
        "paths": len(blocks),
        "outputs": {
            "gfa": str(gfa_path),
            "png": str(png_path),
            "summary_tsv": str(summary_tsv),
            "detail_tsv": str(detail_tsv),
        },
    }
    summary_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
