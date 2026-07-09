#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle


DEFAULT_INPUT = Path("测试数据") / "IRF2BPL.HPRC.fa.3034e58.11fba48.904d69e.smooth.final.gfa"
DEFAULT_OUTDIR = Path("outputs") / "IRF2BPL" / "20260709_PGGB图后处理_v2近似环图"
DEFAULT_PREFIX = "20260709_IRF2BPL"
DEFAULT_MOTIFS = ("CAG", "GCC", "GCG", "GGC", "CAA")


@dataclass
class Segment:
    name: str
    seq: str
    tags: list[str]


@dataclass
class PathRecord:
    name: str
    steps: list[tuple[str, str]]
    overlaps: str
    tags: list[str]


@dataclass
class Token:
    node_id: str
    kind: str
    motif: str
    seq: str
    edits: int


@dataclass
class LoopRun:
    node_id: str
    kind: str
    motif: str
    seq: str
    tokens: list[Token]

    @property
    def loop_count(self) -> int:
        if self.kind == "repeat_family":
            return len(self.tokens)
        return 1

    @property
    def nonrepeat_bp(self) -> int:
        if self.kind == "nonrepeat":
            return sum(len(token.seq) for token in self.tokens)
        return 0


def revcomp(seq: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1].upper()


def parse_step(step: str) -> tuple[str, str]:
    return step[:-1], step[-1]


def read_gfa(path: Path) -> tuple[list[str], dict[str, Segment], list[tuple[str, str, str, str, str]], list[PathRecord]]:
    headers: list[str] = []
    segments: dict[str, Segment] = {}
    links: list[tuple[str, str, str, str, str]] = []
    paths: list[PathRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n\r")
            if not line:
                continue
            fields = line.split("\t")
            if fields[0] == "H":
                headers.append(line)
            elif fields[0] == "S":
                segments[fields[1]] = Segment(fields[1], fields[2].upper(), fields[3:])
            elif fields[0] == "L":
                links.append((fields[1], fields[2], fields[3], fields[4], fields[5]))
            elif fields[0] == "P":
                steps = [parse_step(s) for s in fields[2].split(",") if s]
                paths.append(PathRecord(fields[1], steps, fields[3], fields[4:]))
    return headers, segments, links, paths


def oriented_sequence(segment: Segment, orient: str) -> str:
    return segment.seq if orient == "+" else revcomp(segment.seq)


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def clean_node_part(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text.upper()).strip("_")
    return text[:40] if text else "EMPTY"


def make_node_id(kind: str, motif: str, seq: str, edits: int) -> str:
    if kind == "exact":
        return f"M_{motif}"
    if kind == "variant":
        return f"V_{motif}_e{edits}_{clean_node_part(seq)}"
    return f"U_{clean_node_part(seq)}"


def candidate_tokens(seq: str, i: int, motifs: tuple[str, ...], max_edits: int) -> list[tuple[float, int, Token]]:
    out: list[tuple[float, int, Token]] = []
    for motif_index, motif in enumerate(motifs):
        m = len(motif)
        for length in range(max(1, m - 1), m + 2):
            if i + length > len(seq):
                continue
            piece = seq[i : i + length]
            edits = edit_distance(piece, motif)
            if edits > max_edits:
                continue
            if edits == 0 and length == m:
                kind = "exact"
                score = 100.0 + length
            else:
                kind = "variant"
                score = 58.0 + length - edits * 15.0 - abs(length - m) * 4.0 - motif_index * 0.01
            token = Token(make_node_id(kind, motif, piece, edits), kind, motif, piece, edits)
            out.append((score, length, token))
    return out


def merge_nonrepeat(tokens: list[Token]) -> list[Token]:
    merged: list[Token] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            seq = "".join(buffer)
            merged.append(Token(make_node_id("nonrepeat", "-", seq, 0), "nonrepeat", "-", seq, 0))
            buffer.clear()

    for token in tokens:
        if token.kind == "nonrepeat":
            buffer.append(token.seq)
        else:
            flush()
            merged.append(token)
    flush()
    return merged


def decompose_with_dp(seq: str, motifs: tuple[str, ...], max_edits: int) -> list[Token]:
    n = len(seq)
    best: list[tuple[float, int, int, list[Token]]] = [(-1e18, 10**9, 10**9, []) for _ in range(n + 1)]
    best[0] = (0.0, 0, 0, [])
    for i in range(n):
        score, edit_sum, token_count, tokens = best[i]
        if score < -1e17:
            continue
        nonrepeat = Token(make_node_id("nonrepeat", "-", seq[i], 0), "nonrepeat", "-", seq[i], 0)
        proposals = [(0.0, 1, nonrepeat)] + candidate_tokens(seq, i, motifs, max_edits)
        for add_score, length, token in proposals:
            j = i + length
            cand = (score + add_score, edit_sum + token.edits, token_count + 1, tokens + [token])
            old = best[j]
            if (cand[0], -cand[1], -cand[2]) > (old[0], -old[1], -old[2]):
                best[j] = cand
    return merge_nonrepeat(best[n][3])


def loop_key(token: Token) -> tuple[str, str, str]:
    if token.kind == "nonrepeat":
        return ("U_NONREPEAT", "nonrepeat", "-")
    return (f"R_{token.motif}", "repeat_family", token.motif)


def compress_loop_runs(tokens: list[Token]) -> list[LoopRun]:
    runs: list[LoopRun] = []
    for token in tokens:
        node_id, kind, motif = loop_key(token)
        if runs and runs[-1].node_id == node_id:
            runs[-1].tokens.append(token)
        else:
            seq = motif if kind == "repeat_family" else "*"
            runs.append(LoopRun(node_id, kind, motif, seq, [token]))
    return runs


def build_paths(
    segments: dict[str, Segment],
    paths: list[PathRecord],
    motifs: tuple[str, ...],
    max_edits: int,
) -> tuple[dict[str, list[Token]], dict[str, list[LoopRun]], dict[str, LoopRun]]:
    path_tokens: dict[str, list[Token]] = {}
    path_runs: dict[str, list[LoopRun]] = {}
    node_defs: dict[str, LoopRun] = {}

    for path in paths:
        sequence = "".join(oriented_sequence(segments[name], orient) for name, orient in path.steps)
        tokens = decompose_with_dp(sequence, motifs, max_edits)
        path_tokens[path.name] = tokens
        path_runs[path.name] = compress_loop_runs(tokens)
        for run in path_runs[path.name]:
            node_defs.setdefault(run.node_id, LoopRun(run.node_id, run.kind, run.motif, run.seq, []))

    return path_tokens, path_runs, node_defs


def write_loop_gfa(
    out_path: Path,
    node_defs: dict[str, LoopRun],
    path_runs: dict[str, list[LoopRun]],
) -> None:
    node_use = Counter()
    loop_use = Counter()
    edge_use = Counter()
    exact_use = Counter()
    variant_use = Counter()
    for runs in path_runs.values():
        for run in runs:
            node_use[run.node_id] += 1
            if run.kind == "repeat_family" and run.loop_count > 1:
                loop_use[run.node_id] += run.loop_count - 1
            for token in run.tokens:
                if token.kind == "exact":
                    exact_use[run.node_id] += 1
                elif token.kind == "variant":
                    variant_use[run.node_id] += 1
        for left, right in zip(runs, runs[1:]):
            edge_use[(left.node_id, right.node_id)] += 1

    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("H\tVN:Z:1.0\n")
        handle.write("H\tCL:Z:20260709_VNTR_approximate_loop_graph\n")
        for node_id in sorted(node_defs):
            run = node_defs[node_id]
            handle.write(
                "\t".join(
                    [
                        "S",
                        node_id,
                        run.seq,
                        f"LN:i:{0 if run.seq == '*' else len(run.seq)}",
                        f"RC:i:{node_use[node_id]}",
                        f"LP:i:{loop_use[node_id]}",
                        f"TP:Z:{run.kind}",
                        f"MT:Z:{run.motif}",
                        f"EX:i:{exact_use[node_id]}",
                        f"VA:i:{variant_use[node_id]}",
                    ]
                )
                + "\n"
            )
        for node_id, count in sorted(loop_use.items()):
            handle.write("\t".join(["L", node_id, "+", node_id, "+", "0M", f"LC:i:{count}", "TP:Z:self_loop"]) + "\n")
        for (left, right), count in sorted(edge_use.items()):
            handle.write("\t".join(["L", left, "+", right, "+", "0M", f"RC:i:{count}", "TP:Z:transition"]) + "\n")
        for path_name, runs in sorted(path_runs.items()):
            steps = ",".join(f"{run.node_id}+" for run in runs)
            loop_counts = ",".join(str(run.loop_count) for run in runs)
            handle.write("\t".join(["P", path_name, steps, "*", f"LC:Z:{loop_counts}"]) + "\n")


def write_tsv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(col, "")) for col in columns) + "\n")


def summarize_path_runs(path_runs: dict[str, list[LoopRun]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    run_rows: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []
    for path_name, runs in sorted(path_runs.items()):
        exact = Counter()
        variant = Counter()
        nonrepeat_len = 0
        loop_items: list[str] = []
        for idx, run in enumerate(runs, 1):
            run_exact = Counter()
            run_variant = Counter()
            for token in run.tokens:
                if token.kind == "exact":
                    exact[token.motif] += 1
                    run_exact[token.motif] += 1
                elif token.kind == "variant":
                    key = f"{token.motif}:{token.seq}"
                    variant[key] += 1
                    run_variant[key] += 1
            if run.kind == "nonrepeat":
                nonrepeat_len += run.nonrepeat_bp
            loop_items.append(f"{run.node_id}x{run.loop_count}")
            run_rows.append(
                {
                    "path": path_name,
                    "run_index": idx,
                    "node": run.node_id,
                    "kind": run.kind,
                    "motif": run.motif,
                    "loop_count": run.loop_count,
                    "nonrepeat_bp": run.nonrepeat_bp,
                    "exact_motif_counts_in_run": ";".join(f"{k}={v}" for k, v in sorted(run_exact.items())),
                    "variant_unit_counts_in_run": ";".join(f"{k}={v}" for k, v in sorted(run_variant.items())),
                }
            )
        path_rows.append(
            {
                "path": path_name,
                "run_count": len(runs),
                "exact_motif_counts": ";".join(f"{k}={v}" for k, v in sorted(exact.items())),
                "variant_unit_counts": ";".join(f"{k}={v}" for k, v in sorted(variant.items())),
                "nonrepeat_total_length": nonrepeat_len,
                "compressed_loop_path": " ".join(loop_items),
            }
        )
    return run_rows, path_rows


def node_color(run: LoopRun) -> str:
    if run.kind == "repeat_family":
        palette = {"CAG": "#f4a261", "GCC": "#2a9d8f", "GCG": "#457b9d", "GGC": "#e9c46a", "CAA": "#b56576"}
        return palette.get(run.motif, "#cdb4db")
    return "#adb5bd"


def short_label(run: LoopRun) -> str:
    if run.kind == "repeat_family":
        return run.motif
    return "nonrepeat"


def make_png(
    out_path: Path,
    node_defs: dict[str, LoopRun],
    path_runs: dict[str, list[LoopRun]],
    max_runs_shown: int,
) -> None:
    node_run_use = Counter()
    loop_total = Counter()
    transition_use = Counter()
    for runs in path_runs.values():
        node_run_use.update(run.node_id for run in runs)
        for run in runs:
            if run.kind == "repeat_family":
                loop_total[run.node_id] += run.loop_count
        for left, right in zip(runs, runs[1:]):
            transition_use[(left.node_id, right.node_id)] += 1

    top_nodes = [node for node, _count in node_run_use.most_common(18)]
    motif_nodes = [node for node in sorted(node_defs) if node_defs[node].kind == "repeat_family"]
    ordered_nodes = []
    for node in motif_nodes + top_nodes:
        if node not in ordered_nodes:
            ordered_nodes.append(node)
    ordered_nodes = ordered_nodes[:20]

    n_paths = len(path_runs)
    fig_h = max(10.5, 3.5 + n_paths * 0.33)
    fig = plt.figure(figsize=(20, fig_h), dpi=220)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, max(6, n_paths * 0.33)], hspace=0.18)
    ax_graph = fig.add_subplot(gs[0])
    ax_paths = fig.add_subplot(gs[1])

    ax_graph.set_title("Global loop graph: nodes are repeat units; each node can be traversed multiple times by each path", fontsize=13)
    ax_graph.set_xlim(-0.6, max(6, len(ordered_nodes)) - 0.4)
    ax_graph.set_ylim(-1.2, 1.55)
    ax_graph.axis("off")

    positions: dict[str, tuple[float, float]] = {}
    for idx, node in enumerate(ordered_nodes):
        positions[node] = (idx, 0)

    for (left, right), count in transition_use.most_common(80):
        if left not in positions or right not in positions or left == right:
            continue
        x1, y1 = positions[left]
        x2, y2 = positions[right]
        rad = 0.18 if x2 >= x1 else -0.18
        patch = FancyArrowPatch(
            (x1, y1 - 0.12),
            (x2, y2 - 0.12),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=min(2.8, 0.7 + math.log2(count + 1) * 0.35),
            color="#667085",
            alpha=0.35,
        )
        ax_graph.add_patch(patch)

    for node in ordered_nodes:
        run = node_defs[node]
        x, y = positions[node]
        ax_graph.add_patch(Circle((x, y), 0.24, color=node_color(run), ec="#1f2937", lw=1.0, zorder=3))
        ax_graph.add_patch(Arc((x, y + 0.34), 0.58, 0.48, theta1=25, theta2=330, color="#111827", lw=1.2))
        ax_graph.annotate("", xy=(x + 0.20, y + 0.50), xytext=(x + 0.28, y + 0.43), arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#111827"))
        ax_graph.text(x, y - 0.01, short_label(run)[:10], ha="center", va="center", fontsize=8)
        ax_graph.text(x, y - 0.48, f"runs:{node_run_use[node]} repeats:{loop_total[node]}", ha="center", va="center", fontsize=7, color="#4b5563")

    path_names = sorted(path_runs)
    ax_paths.set_title("Per-path loop tracks: number means this path's loop count at that repeat-unit node", fontsize=13)
    ax_paths.set_xlim(0, max_runs_shown)
    ax_paths.set_ylim(-1, n_paths)
    ax_paths.set_xlabel("compressed run order along each path")
    ax_paths.set_yticks(range(n_paths))
    ax_paths.set_yticklabels(path_names, fontsize=6)
    ax_paths.invert_yaxis()
    ax_paths.grid(axis="x", color="#e5e7eb", lw=0.5)

    for row, path_name in enumerate(path_names):
        runs = path_runs[path_name][:max_runs_shown]
        for col, run in enumerate(runs):
            if run.kind == "nonrepeat":
                height = 0.22
                y = row - height / 2
            else:
                height = 0.56
                y = row - height / 2
            ax_paths.add_patch(Rectangle((col + 0.05, y), 0.9, height, facecolor=node_color(run), edgecolor="#344054", linewidth=0.25))
            if run.kind != "nonrepeat":
                text = str(run.loop_count)
                ax_paths.text(col + 0.5, row, text, ha="center", va="center", fontsize=5.5, color="#111827")
        if len(path_runs[path_name]) > max_runs_shown:
            ax_paths.text(max_runs_shown - 0.2, row, "...", va="center", ha="right", fontsize=6, color="#6b7280")

    legend_tokens = [node_defs[node] for node in ordered_nodes[:10]]
    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=node_color(token), edgecolor="#344054", label=short_label(token)[:18])
        for token in legend_tokens
    ]
    ax_paths.legend(handles=legend_handles, ncol=5, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build approximate VNTR repeat-unit loop graph from PGGB GFA.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--motifs", default=",".join(DEFAULT_MOTIFS))
    parser.add_argument("--max-edits", type=int, default=1)
    parser.add_argument("--max-runs-shown", type=int, default=120)
    args = parser.parse_args()

    motifs = tuple(m.strip().upper() for m in args.motifs.split(",") if m.strip())
    args.outdir.mkdir(parents=True, exist_ok=True)

    _headers, segments, links, paths = read_gfa(args.input)
    path_tokens, path_runs, node_defs = build_paths(segments, paths, motifs, args.max_edits)
    run_rows, path_rows = summarize_path_runs(path_runs)

    loop_gfa = args.outdir / f"{args.prefix}_VNTR近似重复单元环图.gfa"
    run_tsv = args.outdir / f"{args.prefix}_路径环循环次数.tsv"
    path_tsv = args.outdir / f"{args.prefix}_路径环结构摘要.tsv"
    png = args.outdir / f"{args.prefix}_路径环结构可视化.png"
    summary_json = args.outdir / f"{args.prefix}_运行摘要.json"

    write_loop_gfa(loop_gfa, node_defs, path_runs)
    write_tsv(
        run_tsv,
        run_rows,
        [
            "path",
            "run_index",
            "node",
            "kind",
            "motif",
            "loop_count",
            "nonrepeat_bp",
            "exact_motif_counts_in_run",
            "variant_unit_counts_in_run",
        ],
    )
    write_tsv(path_tsv, path_rows, ["path", "run_count", "exact_motif_counts", "variant_unit_counts", "nonrepeat_total_length", "compressed_loop_path"])
    make_png(png, node_defs, path_runs, args.max_runs_shown)

    summary = {
        "input": str(args.input),
        "motifs": motifs,
        "max_edits": args.max_edits,
        "segments": len(segments),
        "links": len(links),
        "paths": len(paths),
        "token_nodes": len(node_defs),
        "total_tokens": sum(len(tokens) for tokens in path_tokens.values()),
        "total_runs": sum(len(runs) for runs in path_runs.values()),
        "outputs": {
            "loop_gfa": str(loop_gfa),
            "run_tsv": str(run_tsv),
            "path_tsv": str(path_tsv),
            "png": str(png),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
