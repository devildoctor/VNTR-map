#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PGGB GFA 后处理原型：
1. 给原始 PGGB 图添加节点/边/path 使用统计标签，输出增强版 GFA。
2. 按候选 VNTR motif 将每条 path 拆成 token，输出 motif/token 解释层 GFA。
3. 输出节点、边、path 的 TSV 统计和一个无需额外依赖的 HTML/SVG 可视化。
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT = Path("测试数据") / "IRF2BPL.HPRC.fa.3034e58.11fba48.904d69e.smooth.final.gfa"
DEFAULT_OUTDIR = Path("outputs") / "IRF2BPL" / "20260709_PGGB图后处理"
DEFAULT_PREFIX = "20260709_IRF2BPL"
DEFAULT_MOTIFS = ("CAG", "GCC", "GCG", "GGC", "CAA")


@dataclass
class Segment:
    name: str
    seq: str
    tags: list[str]


@dataclass
class Link:
    from_name: str
    from_orient: str
    to_name: str
    to_orient: str
    overlap: str
    tags: list[str]


@dataclass
class PathRecord:
    name: str
    steps: list[tuple[str, str]]
    overlaps: str
    tags: list[str]


def revcomp(seq: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(table)[::-1].upper()


def parse_step(step: str) -> tuple[str, str]:
    step = step.strip()
    if not step:
        raise ValueError("empty path step")
    orient = step[-1]
    if orient not in "+-":
        raise ValueError(f"path step lacks orientation: {step}")
    return step[:-1], orient


def read_gfa(path: Path) -> tuple[list[str], dict[str, Segment], list[Link], list[PathRecord], list[str]]:
    headers: list[str] = []
    segments: dict[str, Segment] = {}
    links: list[Link] = []
    paths: list[PathRecord] = []
    others: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n\r")
            if not line:
                continue
            fields = line.split("\t")
            rec_type = fields[0]
            if rec_type == "H":
                headers.append(line)
            elif rec_type == "S":
                if len(fields) < 3:
                    raise ValueError(f"bad S line: {line}")
                segments[fields[1]] = Segment(fields[1], fields[2].upper(), fields[3:])
            elif rec_type == "L":
                if len(fields) < 6:
                    raise ValueError(f"bad L line: {line}")
                links.append(Link(fields[1], fields[2], fields[3], fields[4], fields[5], fields[6:]))
            elif rec_type == "P":
                if len(fields) < 4:
                    raise ValueError(f"bad P line: {line}")
                steps = [parse_step(step) for step in fields[2].split(",") if step]
                paths.append(PathRecord(fields[1], steps, fields[3], fields[4:]))
            else:
                others.append(line)

    return headers, segments, links, paths, others


def oriented_sequence(segment: Segment, orient: str) -> str:
    if orient == "+":
        return segment.seq
    if orient == "-":
        return revcomp(segment.seq)
    raise ValueError(f"bad orientation: {orient}")


def decompose_sequence(seq: str, motifs: tuple[str, ...]) -> list[tuple[str, str]]:
    motifs_by_length = sorted((m.upper() for m in motifs), key=len, reverse=True)
    tokens: list[tuple[str, str]] = []
    i = 0
    unknown: list[str] = []

    def flush_unknown() -> None:
        nonlocal unknown
        if unknown:
            tokens.append(("非重复", "".join(unknown)))
            unknown = []

    while i < len(seq):
        matched = None
        for motif in motifs_by_length:
            if seq.startswith(motif, i):
                matched = motif
                break
        if matched:
            flush_unknown()
            tokens.append(("motif", matched))
            i += len(matched)
        else:
            unknown.append(seq[i])
            i += 1
    flush_unknown()
    return tokens


def compress_runs(items: list[str]) -> list[tuple[str, int]]:
    runs: list[tuple[str, int]] = []
    for item in items:
        if runs and runs[-1][0] == item:
            runs[-1] = (item, runs[-1][1] + 1)
        else:
            runs.append((item, 1))
    return runs


def safe_tag_value(value: str) -> str:
    return re.sub(r"[\t\r\n ]+", "_", value)


def collect_original_stats(
    segments: dict[str, Segment],
    links: list[Link],
    paths: list[PathRecord],
    motifs: tuple[str, ...],
) -> tuple[Counter[str], Counter[tuple[str, str]], dict[str, Counter[str]]]:
    node_use: Counter[str] = Counter()
    edge_use: Counter[tuple[str, str]] = Counter()
    segment_motifs: dict[str, Counter[str]] = {}

    for segment in segments.values():
        counter: Counter[str] = Counter()
        for kind, value in decompose_sequence(segment.seq, motifs):
            if kind == "motif":
                counter[value] += 1
        segment_motifs[segment.name] = counter

    for path in paths:
        names = [name for name, _orient in path.steps]
        node_use.update(names)
        for left, right in zip(path.steps, path.steps[1:]):
            edge_use[(left[0], right[0])] += 1

    return node_use, edge_use, segment_motifs


def write_annotated_gfa(
    out_path: Path,
    headers: list[str],
    segments: dict[str, Segment],
    links: list[Link],
    paths: list[PathRecord],
    others: list[str],
    node_use: Counter[str],
    edge_use: Counter[tuple[str, str]],
    segment_motifs: dict[str, Counter[str]],
) -> None:
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        if headers:
            for header in headers:
                handle.write(header + "\n")
        else:
            handle.write("H\tVN:Z:1.0\n")
        handle.write("H\tCL:Z:20260709_PGGB图后处理与VNTR可视化.py\n")

        for name in sorted(segments, key=lambda x: int(x) if x.isdigit() else x):
            segment = segments[name]
            motif_text = ";".join(f"{k}={v}" for k, v in sorted(segment_motifs[name].items()))
            tags = list(segment.tags)
            tags.extend(
                [
                    f"LN:i:{len(segment.seq)}",
                    f"RC:i:{node_use[name]}",
                    f"RN:i:{sum(segment_motifs[name].values())}",
                    f"MC:Z:{safe_tag_value(motif_text) if motif_text else '-'}",
                ]
            )
            handle.write("\t".join(["S", segment.name, segment.seq, *tags]) + "\n")

        for link in links:
            tags = list(link.tags)
            tags.append(f"RC:i:{edge_use[(link.from_name, link.to_name)]}")
            handle.write(
                "\t".join(
                    [
                        "L",
                        link.from_name,
                        link.from_orient,
                        link.to_name,
                        link.to_orient,
                        link.overlap,
                        *tags,
                    ]
                )
                + "\n"
            )

        for path in paths:
            step_text = ",".join(f"{name}{orient}" for name, orient in path.steps)
            tags = list(path.tags)
            tags.extend([f"SN:i:{len(path.steps)}"])
            handle.write("\t".join(["P", path.name, step_text, path.overlaps, *tags]) + "\n")

        for other in others:
            handle.write(other + "\n")


def build_token_graph(
    segments: dict[str, Segment],
    paths: list[PathRecord],
    motifs: tuple[str, ...],
) -> tuple[dict[str, str], Counter[tuple[str, str]], dict[str, list[str]], list[dict[str, str]]]:
    token_to_node: dict[str, str] = {}
    node_to_seq: dict[str, str] = {}
    path_tokens: dict[str, list[str]] = {}
    token_rows: list[dict[str, str]] = []
    next_unknown = 1

    def node_for(kind: str, value: str) -> str:
        nonlocal next_unknown
        key = f"{kind}:{value}"
        if key in token_to_node:
            return token_to_node[key]
        if kind == "motif":
            node_id = f"M_{value}"
        else:
            node_id = f"U_{next_unknown:03d}"
            next_unknown += 1
        token_to_node[key] = node_id
        node_to_seq[node_id] = value
        return node_id

    for path in paths:
        expanded: list[str] = []
        for segment_name, orient in path.steps:
            seq = oriented_sequence(segments[segment_name], orient)
            for kind, value in decompose_sequence(seq, motifs):
                node_id = node_for(kind, value)
                expanded.append(node_id)
                token_rows.append(
                    {
                        "path": path.name,
                        "segment": segment_name,
                        "orient": orient,
                        "kind": kind,
                        "token_node": node_id,
                        "token_seq": value,
                        "token_len": str(len(value)),
                    }
                )
        path_tokens[path.name] = expanded

    edge_counts: Counter[tuple[str, str]] = Counter()
    for tokens in path_tokens.values():
        for left, right in zip(tokens, tokens[1:]):
            edge_counts[(left, right)] += 1

    return node_to_seq, edge_counts, path_tokens, token_rows


def write_token_gfa(
    out_path: Path,
    node_to_seq: dict[str, str],
    edge_counts: Counter[tuple[str, str]],
    path_tokens: dict[str, list[str]],
) -> None:
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("H\tVN:Z:1.0\n")
        handle.write("H\tCL:Z:20260709_motif_token_graph\n")

        node_use = Counter()
        for tokens in path_tokens.values():
            node_use.update(tokens)

        for node_id in sorted(node_to_seq):
            kind = "motif" if node_id.startswith("M_") else "nonrepeat"
            handle.write(
                "\t".join(
                    [
                        "S",
                        node_id,
                        node_to_seq[node_id],
                        f"LN:i:{len(node_to_seq[node_id])}",
                        f"RC:i:{node_use[node_id]}",
                        f"TP:Z:{kind}",
                    ]
                )
                + "\n"
            )

        for (left, right), count in sorted(edge_counts.items()):
            handle.write("\t".join(["L", left, "+", right, "+", "0M", f"RC:i:{count}"]) + "\n")

        for path_name, tokens in sorted(path_tokens.items()):
            handle.write("\t".join(["P", path_name, ",".join(f"{token}+" for token in tokens), "*"]) + "\n")


def write_tsv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(col, "")) for col in columns) + "\n")


def build_stats_rows(
    segments: dict[str, Segment],
    links: list[Link],
    paths: list[PathRecord],
    node_use: Counter[str],
    edge_use: Counter[tuple[str, str]],
    segment_motifs: dict[str, Counter[str]],
    path_tokens: dict[str, list[str]],
    node_to_seq: dict[str, str],
    motifs: tuple[str, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    node_rows: list[dict[str, object]] = []
    for name in sorted(segments, key=lambda x: int(x) if x.isdigit() else x):
        motif_counts = segment_motifs[name]
        node_rows.append(
            {
                "segment": name,
                "length": len(segments[name].seq),
                "path_count": node_use[name],
                "motif_token_count": sum(motif_counts.values()),
                "motif_counts": ";".join(f"{k}={v}" for k, v in sorted(motif_counts.items())),
                "sequence_preview": segments[name].seq[:80],
            }
        )

    edge_rows: list[dict[str, object]] = []
    seen_links = {(link.from_name, link.to_name) for link in links}
    for link in links:
        edge_rows.append(
            {
                "from": link.from_name,
                "to": link.to_name,
                "from_orient": link.from_orient,
                "to_orient": link.to_orient,
                "path_count": edge_use[(link.from_name, link.to_name)],
                "in_original_gfa": "yes",
            }
        )
    for (left, right), count in edge_use.items():
        if (left, right) not in seen_links:
            edge_rows.append(
                {
                    "from": left,
                    "to": right,
                    "from_orient": "+",
                    "to_orient": "+",
                    "path_count": count,
                    "in_original_gfa": "path_only",
                }
            )

    token_seq = {node: seq for node, seq in node_to_seq.items()}
    path_rows: list[dict[str, object]] = []
    for path in paths:
        sequence = "".join(oriented_sequence(segments[name], orient) for name, orient in path.steps)
        motif_counter = Counter()
        for token in path_tokens[path.name]:
            seq = token_seq[token]
            if token.startswith("M_") and seq in motifs:
                motif_counter[seq] += 1
        motif_items = [token_seq[token] if token.startswith("M_") else "非重复" for token in path_tokens[path.name]]
        compressed = " ".join(f"{item}x{count}" if count > 1 else item for item, count in compress_runs(motif_items))
        path_rows.append(
            {
                "path": path.name,
                "segment_count": len(path.steps),
                "sequence_length": len(sequence),
                "token_count": len(path_tokens[path.name]),
                "motif_token_count": sum(motif_counter.values()),
                "nonrepeat_token_count": sum(1 for token in path_tokens[path.name] if token.startswith("U_")),
                "motif_counts": ";".join(f"{k}={v}" for k, v in sorted(motif_counter.items())),
                "compressed_token_pattern": compressed,
            }
        )
    return node_rows, edge_rows, path_rows


def make_html_visualization(
    out_path: Path,
    node_to_seq: dict[str, str],
    edge_counts: Counter[tuple[str, str]],
    path_tokens: dict[str, list[str]],
    path_rows: list[dict[str, object]],
) -> None:
    node_use = Counter()
    for tokens in path_tokens.values():
        node_use.update(tokens)

    motif_nodes = sorted([n for n in node_to_seq if n.startswith("M_")])
    other_nodes = sorted([n for n in node_to_seq if n.startswith("U_")], key=lambda n: (-node_use[n], n))
    nodes = motif_nodes + other_nodes[:80]
    node_set = set(nodes)

    width = 1200
    row_gap = 88
    col_gap = 155
    positions: dict[str, tuple[float, float]] = {}
    for idx, node in enumerate(motif_nodes):
        positions[node] = (110 + idx * col_gap, 95)
    for idx, node in enumerate(other_nodes[:80]):
        row = idx // 6
        col = idx % 6
        positions[node] = (110 + col * 175, 220 + row * row_gap)
    height = max(460, 300 + math.ceil(max(1, len(other_nodes[:80])) / 6) * row_gap)

    edge_svg = []
    for (left, right), count in edge_counts.most_common(300):
        if left not in node_set or right not in node_set:
            continue
        x1, y1 = positions[left]
        x2, y2 = positions[right]
        if left == right:
            edge_svg.append(
                f'<path class="edge loop" d="M{x1+24:.1f},{y1-18:.1f} C{x1+70:.1f},{y1-70:.1f} {x1-70:.1f},{y1-70:.1f} {x1-24:.1f},{y1-18:.1f}" />'
                f'<text class="edge-label" x="{x1+34:.1f}" y="{y1-48:.1f}">{count}</text>'
            )
        else:
            stroke_width = min(8, 1 + math.log2(count + 1))
            edge_svg.append(
                f'<line class="edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke-width="{stroke_width:.2f}" />'
            )

    node_svg = []
    for node in nodes:
        x, y = positions[node]
        is_motif = node.startswith("M_")
        radius = min(34, 14 + math.sqrt(node_use[node]) * 1.8)
        seq = node_to_seq[node]
        label = seq if is_motif else node
        css = "motif" if is_motif else "nonrepeat"
        title = f"{node} | {seq[:120]} | path/token count={node_use[node]}"
        node_svg.append(
            f'<g class="node {css}" data-node="{html.escape(node)}">'
            f'<title>{html.escape(title)}</title>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" />'
            f'<text x="{x:.1f}" y="{y+4:.1f}">{html.escape(label[:12])}</text>'
            f'</g>'
        )

    top_paths = sorted(path_rows, key=lambda r: (-int(r["motif_token_count"]), str(r["path"])))[:20]
    table_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['path']))}</td>"
        f"<td>{row['sequence_length']}</td>"
        f"<td>{row['token_count']}</td>"
        f"<td>{row['motif_token_count']}</td>"
        f"<td>{html.escape(str(row['motif_counts']))}</td>"
        f"<td>{html.escape(str(row['compressed_token_pattern'])[:180])}</td>"
        "</tr>"
        for row in top_paths
    )

    payload = {
        "nodes": [
            {
                "id": node,
                "seq": node_to_seq[node],
                "kind": "motif" if node.startswith("M_") else "nonrepeat",
                "count": node_use[node],
            }
            for node in nodes
        ],
        "edges": [
            {"from": left, "to": right, "count": count}
            for (left, right), count in edge_counts.most_common(300)
            if left in node_set and right in node_set
        ],
    }

    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>IRF2BPL VNTR motif/token 图 - 20260709</title>
  <style>
    body {{ margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; color: #202124; background: #f7f7f4; }}
    header {{ padding: 18px 28px; background: #263238; color: white; }}
    h1 {{ margin: 0 0 6px 0; font-size: 22px; }}
    main {{ padding: 22px 28px 42px; }}
    .panel {{ background: white; border: 1px solid #d8d8d2; border-radius: 6px; margin-bottom: 18px; padding: 16px; }}
    .summary {{ display: flex; gap: 18px; flex-wrap: wrap; font-size: 14px; }}
    .summary strong {{ font-size: 20px; display: block; }}
    svg {{ width: 100%; height: auto; background: #fff; border: 1px solid #ddd; }}
    .edge {{ stroke: #87939a; stroke-opacity: .45; }}
    .loop {{ fill: none; stroke: #c45b36; stroke-width: 2.4; }}
    .edge-label {{ font-size: 11px; fill: #8a3d25; }}
    .node circle {{ stroke: #1f2933; stroke-width: 1.2; }}
    .node text {{ font-size: 12px; fill: #111; text-anchor: middle; pointer-events: none; }}
    .motif circle {{ fill: #ffc857; }}
    .nonrepeat circle {{ fill: #8ecae6; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e1e1dc; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f0ea; }}
    code {{ background: #efefea; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>IRF2BPL VNTR motif/token 图 - 20260709</h1>
    <div>黄色节点是候选重复单元，蓝色节点是非候选 motif 片段；自环表示同一 token 连续重复。</div>
  </header>
  <main>
    <section class="panel summary">
      <div><strong>{len(path_tokens)}</strong>paths</div>
      <div><strong>{len(node_to_seq)}</strong>token nodes</div>
      <div><strong>{len(edge_counts)}</strong>token edges</div>
      <div><strong>{sum(1 for n in node_to_seq if n.startswith("M_"))}</strong>motif nodes</div>
    </section>
    <section class="panel">
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="motif token graph">
        <defs>
          <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
            <path d="M0,0 L0,6 L7,3 z" fill="#87939a"></path>
          </marker>
        </defs>
        <g marker-end="url(#arrow)">
          {''.join(edge_svg)}
        </g>
        <g>
          {''.join(node_svg)}
        </g>
      </svg>
    </section>
    <section class="panel">
      <h2>motif token 最多的 path</h2>
      <table>
        <thead>
          <tr><th>path</th><th>序列长度</th><th>token数</th><th>motif数</th><th>motif计数</th><th>压缩token模式</th></tr>
        </thead>
        <tbody>{table_rows}</tbody>
      </table>
    </section>
    <script type="application/json" id="graph-data">{html.escape(json.dumps(payload, ensure_ascii=False))}</script>
  </main>
</body>
</html>
"""
    out_path.write_text(doc, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="PGGB GFA 后处理并生成 VNTR motif/token 图。")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入 PGGB GFA 文件")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="输出目录")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="输出文件前缀")
    parser.add_argument("--motifs", default=",".join(DEFAULT_MOTIFS), help="候选 motif，逗号分隔")
    args = parser.parse_args()

    motifs = tuple(m.strip().upper() for m in args.motifs.split(",") if m.strip())
    args.outdir.mkdir(parents=True, exist_ok=True)

    headers, segments, links, paths, others = read_gfa(args.input)
    node_use, edge_use, segment_motifs = collect_original_stats(segments, links, paths, motifs)
    node_to_seq, token_edge_counts, path_tokens, token_rows = build_token_graph(segments, paths, motifs)

    annotated_gfa = args.outdir / f"{args.prefix}_PGGB节点统计增强.gfa"
    token_gfa = args.outdir / f"{args.prefix}_VNTR重复单元解释图.gfa"
    node_tsv = args.outdir / f"{args.prefix}_节点统计.tsv"
    edge_tsv = args.outdir / f"{args.prefix}_边统计.tsv"
    path_tsv = args.outdir / f"{args.prefix}_路径重复单元统计.tsv"
    token_tsv = args.outdir / f"{args.prefix}_路径token明细.tsv"
    html_path = args.outdir / f"{args.prefix}_VNTR重复单元图可视化.html"

    write_annotated_gfa(annotated_gfa, headers, segments, links, paths, others, node_use, edge_use, segment_motifs)
    write_token_gfa(token_gfa, node_to_seq, token_edge_counts, path_tokens)

    node_rows, edge_rows, path_rows = build_stats_rows(
        segments, links, paths, node_use, edge_use, segment_motifs, path_tokens, node_to_seq, motifs
    )
    write_tsv(node_tsv, node_rows, ["segment", "length", "path_count", "motif_token_count", "motif_counts", "sequence_preview"])
    write_tsv(edge_tsv, edge_rows, ["from", "to", "from_orient", "to_orient", "path_count", "in_original_gfa"])
    write_tsv(
        path_tsv,
        path_rows,
        [
            "path",
            "segment_count",
            "sequence_length",
            "token_count",
            "motif_token_count",
            "nonrepeat_token_count",
            "motif_counts",
            "compressed_token_pattern",
        ],
    )
    write_tsv(token_tsv, token_rows, ["path", "segment", "orient", "kind", "token_node", "token_seq", "token_len"])
    make_html_visualization(html_path, node_to_seq, token_edge_counts, path_tokens, path_rows)

    manifest = {
        "input": str(args.input),
        "motifs": motifs,
        "segments": len(segments),
        "links": len(links),
        "paths": len(paths),
        "token_nodes": len(node_to_seq),
        "token_edges": len(token_edge_counts),
        "outputs": {
            "annotated_gfa": str(annotated_gfa),
            "token_gfa": str(token_gfa),
            "node_tsv": str(node_tsv),
            "edge_tsv": str(edge_tsv),
            "path_tsv": str(path_tsv),
            "token_tsv": str(token_tsv),
            "html": str(html_path),
        },
    }
    (args.outdir / f"{args.prefix}_运行摘要.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
