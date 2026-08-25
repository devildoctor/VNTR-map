#!/usr/bin/env python3
"""P5: rebuild a path-supported position-specific SCC repeat graph."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
P4_DIR = ROOT / "outputs" / "IRF2BPL" / "20260825_P4变异read证据分级"
P4_PREFIX = "20260825_IRF2BPL_P4变异read证据分级"
P3_DIR = ROOT / "outputs" / "IRF2BPL" / "20260825_P3半马尔可夫概率分解与置信度"
P3_PREFIX = "20260825_IRF2BPL_P3半马尔可夫概率分解与置信度"
DEFAULT_OUTDIR = ROOT / "outputs" / "IRF2BPL" / "20260825_P5位置特异SCC循环图"
DEFAULT_PREFIX = "20260825_IRF2BPL_P5位置特异SCC循环图"
COORDINATE_SYSTEM = "0-based_half-open"


def read_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {path}")
        return list(reader), list(reader.fieldnames)


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_tags(fields: list[str]) -> dict[str, str]:
    tags = {}
    for field in fields:
        parts = field.split(":", 2)
        if len(parts) == 3:
            tags[parts[0]] = parts[2]
    return tags


def parse_step(step: str) -> tuple[str, str]:
    if not step or step[-1] not in "+-":
        raise ValueError(f"Invalid GFA path step: {step}")
    return step[:-1], step[-1]


def read_gfa(path: Path):
    headers = []
    segments = {}
    paths = []
    other = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        fields = raw.split("\t")
        if fields[0] == "H":
            headers.append(fields)
        elif fields[0] == "S":
            segments[fields[1]] = {
                "name": fields[1],
                "sequence": fields[2],
                "extra": fields[3:],
                "tags": parse_tags(fields[3:]),
            }
        elif fields[0] == "P":
            paths.append(
                {
                    "name": fields[1],
                    "step_text": fields[2],
                    "steps": [parse_step(step) for step in fields[2].split(",")],
                    "overlaps": fields[3],
                    "extra": fields[4:],
                }
            )
        elif fields[0] != "L":
            other.append(fields)
    if not headers or not segments or not paths:
        raise ValueError("P5 requires a GFA with H, S and P records")
    unknown = sorted({name for path_row in paths for name, _orientation in path_row["steps"]} - set(segments))
    if unknown:
        raise ValueError(f"P paths reference unknown segments: {unknown[:5]}")
    return headers, segments, paths, other


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]


def reconstruct_paths(segments, paths) -> dict[str, str]:
    output = {}
    for path in paths:
        sequence = []
        for name, orientation in path["steps"]:
            value = segments[name]["sequence"]
            if value == "*":
                raise ValueError(f"Cannot reconstruct path through unknown segment: {name}")
            sequence.append(value if orientation == "+" else reverse_complement(value))
        output[path["name"]] = "".join(sequence)
    return output


def segment_location(segment: dict) -> str:
    tagged = segment["tags"].get("LC")
    if tagged:
        return tagged
    match = re.match(r"(B\d+)_", segment["name"])
    return match.group(1) if match else "unplaced"


def location_key(location: str):
    match = re.fullmatch(r"([BR])(\d+)", location)
    if not match:
        return (9999, 9, location)
    number = int(match.group(2))
    rank = number * 2 + (1 if match.group(1) == "B" else 0)
    return (rank, location)


def build_path_supported_graph(paths):
    edge_total = Counter()
    edge_paths = defaultdict(set)
    handle_paths = defaultdict(set)
    for path in paths:
        handles = path["steps"]
        for handle in handles:
            handle_paths[handle].add(path["name"])
        for source, target in zip(handles, handles[1:]):
            edge = (source[0], source[1], target[0], target[1])
            edge_total[edge] += 1
            edge_paths[edge].add(path["name"])
    vertices = set(handle_paths)
    adjacency = {vertex: set() for vertex in vertices}
    for source_name, source_orientation, target_name, target_orientation in edge_total:
        adjacency[(source_name, source_orientation)].add((target_name, target_orientation))
    return vertices, adjacency, edge_total, edge_paths, handle_paths


def tarjan_scc(vertices, adjacency):
    sys.setrecursionlimit(max(10000, len(vertices) * 4 + 100))
    index = 0
    stack = []
    on_stack = set()
    indices = {}
    lowlink = {}
    components = []

    def visit(vertex):
        nonlocal index
        indices[vertex] = index
        lowlink[vertex] = index
        index += 1
        stack.append(vertex)
        on_stack.add(vertex)
        for target in adjacency.get(vertex, ()):
            if target not in indices:
                visit(target)
                lowlink[vertex] = min(lowlink[vertex], lowlink[target])
            elif target in on_stack:
                lowlink[vertex] = min(lowlink[vertex], indices[target])
        if lowlink[vertex] == indices[vertex]:
            component = []
            while True:
                target = stack.pop()
                on_stack.remove(target)
                component.append(target)
                if target == vertex:
                    break
            components.append(tuple(component))

    for vertex in sorted(vertices):
        if vertex not in indices:
            visit(vertex)
    return components


def assign_scc_ids(components, segments):
    def component_key(component):
        locations = [segment_location(segments[name]) for name, _orientation in component]
        return (min(location_key(location) for location in locations), min(f"{name}{orientation}" for name, orientation in component))

    ordered = sorted(components, key=component_key)
    component_by_handle = {}
    handles_by_scc = {}
    for index, component in enumerate(ordered, 1):
        scc_id = f"SCC{index:04d}"
        handles_by_scc[scc_id] = component
        for handle in component:
            component_by_handle[handle] = scc_id
    return component_by_handle, handles_by_scc


def summarize_sccs(handles_by_scc, component_by_handle, segments, edge_total, edge_paths, handle_paths):
    internal_edges = defaultdict(list)
    for edge in edge_total:
        source = (edge[0], edge[1])
        target = (edge[2], edge[3])
        if component_by_handle[source] == component_by_handle[target]:
            internal_edges[component_by_handle[source]].append(edge)
    rows = []
    for scc_id, handles in handles_by_scc.items():
        names = sorted({name for name, _orientation in handles})
        locations = sorted({segment_location(segments[name]) for name in names}, key=location_key)
        repeat_locations = [location for location in locations if re.fullmatch(r"R\d+", location)]
        families = sorted({segments[name]["tags"].get("CM", "") for name in names} - {""})
        node_types = sorted({segments[name]["tags"].get("BT", "untyped") for name in names})
        paths = sorted({path for handle in handles for path in handle_paths[handle]})
        has_self_loop = any(
            edge[0] == edge[2] and edge[1] == edge[3]
            for edge in internal_edges.get(scc_id, [])
        )
        cyclic = len(handles) > 1 or has_self_loop
        if cyclic and len(repeat_locations) > 1:
            classification = "invalid_cross_position_cycle"
        elif cyclic and len(repeat_locations) == 1:
            classification = "position_specific_repeat_core"
        elif cyclic:
            classification = "background_cycle"
        else:
            classification = "acyclic_component"
        rows.append(
            {
                "scc_id": scc_id,
                "handle_count": len(handles),
                "segment_count": len(names),
                "internal_edge_count": len(internal_edges.get(scc_id, [])),
                "cyclic": int(cyclic),
                "classification": classification,
                "locations": ",".join(locations),
                "repeat_locations": ",".join(repeat_locations) if repeat_locations else "-",
                "families": ",".join(families) if families else "-",
                "node_types": ",".join(node_types),
                "path_support_count": len(paths),
                "paths": ",".join(paths),
                "handles": ",".join(f"{name}{orientation}" for name, orientation in sorted(handles)),
                "primary_for_location": "-",
            }
        )
    rows_by_id = {row["scc_id"]: row for row in rows}
    all_repeat_locations = sorted(
        {location for row in rows for location in row["repeat_locations"].split(",") if location != "-"},
        key=location_key,
    )
    for location in all_repeat_locations:
        candidates = [
            row
            for row in rows
            if row["cyclic"] == 1 and row["repeat_locations"] == location
        ]
        if candidates:
            primary = max(candidates, key=lambda row: (row["path_support_count"], row["handle_count"], row["internal_edge_count"]))
            rows_by_id[primary["scc_id"]]["primary_for_location"] = location
    return sorted(rows, key=lambda row: row["scc_id"])


def build_condensation(component_by_handle, edge_total, edge_paths):
    totals = Counter()
    paths = defaultdict(set)
    for edge, count in edge_total.items():
        source = component_by_handle[(edge[0], edge[1])]
        target = component_by_handle[(edge[2], edge[3])]
        if source == target:
            continue
        totals[(source, target)] += count
        paths[(source, target)].update(edge_paths[edge])
    rows = [
        {
            "source_scc": source,
            "target_scc": target,
            "path_support_count": len(paths[(source, target)]),
            "total_traversals": totals[(source, target)],
            "paths": ",".join(sorted(paths[(source, target)])),
        }
        for source, target in sorted(totals)
    ]
    return rows


def topological_order(scc_ids: set[str], condensation_rows: list[dict]) -> list[str]:
    adjacency = {scc_id: set() for scc_id in scc_ids}
    indegree = {scc_id: 0 for scc_id in scc_ids}
    for row in condensation_rows:
        source, target = row["source_scc"], row["target_scc"]
        if target not in adjacency[source]:
            adjacency[source].add(target)
            indegree[target] += 1
    queue = deque(sorted(scc_id for scc_id, degree in indegree.items() if degree == 0))
    order = []
    while queue:
        source = queue.popleft()
        order.append(source)
        for target in sorted(adjacency[source]):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(order) != len(scc_ids):
        raise AssertionError("SCC condensation graph is not acyclic")
    return order


def simple_bubbles(condensation_rows: list[dict], topo_order: list[str]) -> list[dict]:
    adjacency = defaultdict(set)
    for row in condensation_rows:
        adjacency[row["source_scc"]].add(row["target_scc"])
    rank = {node: index for index, node in enumerate(topo_order)}

    def descendants(start):
        distance = {start: 0}
        queue = deque([start])
        while queue:
            source = queue.popleft()
            for target in adjacency[source]:
                if target not in distance:
                    distance[target] = distance[source] + 1
                    queue.append(target)
        return distance

    bubbles = []
    for source, branches in sorted(adjacency.items()):
        if len(branches) < 2:
            continue
        branch_distances = [descendants(branch) for branch in sorted(branches)]
        common = set.intersection(*(set(distance) for distance in branch_distances))
        if not common:
            continue
        target = min(common, key=lambda node: (max(distance[node] for distance in branch_distances), rank[node], node))
        bubbles.append(
            {
                "source_scc": source,
                "sink_scc": target,
                "branch_count": len(branches),
                "branch_sccs": ",".join(sorted(branches)),
                "max_branch_distance": max(distance[target] for distance in branch_distances),
                "method": "condensation_DAG_branch_reconvergence",
            }
        )
    return bubbles


def minimal_high_order_period(labels: list[str], max_period: int, min_cycles: int):
    for period in range(2, min(max_period, len(labels) // min_cycles) + 1):
        if len(labels) % period:
            continue
        pattern = labels[:period]
        if len(set(pattern)) < 2:
            continue
        if all(label == pattern[index % period] for index, label in enumerate(labels)):
            primitive = not any(
                period % smaller == 0 and all(pattern[index] == pattern[index % smaller] for index in range(period))
                for smaller in range(1, period)
            )
            if primitive:
                return pattern, len(labels) // period
    return None


def detect_high_order(paths, segments, max_period: int, min_cycles: int, min_paths: int):
    candidates = defaultdict(list)
    for path in paths:
        labels_by_location = defaultdict(list)
        for name, _orientation in path["steps"]:
            segment = segments[name]
            location = segment_location(segment)
            family = segment["tags"].get("CM")
            node_type = segment["tags"].get("BT", "")
            if re.fullmatch(r"R\d+", location) and family and node_type in {"exact_motif", "variant_motif"}:
                labels_by_location[location].append(family)
        for location, labels in labels_by_location.items():
            result = minimal_high_order_period(labels, max_period, min_cycles)
            if result is None:
                continue
            pattern, cycles = result
            candidates[(location, tuple(pattern))].append((path["name"], cycles, len(labels)))
    rows = []
    for (location, pattern), support in sorted(candidates.items(), key=lambda item: (location_key(item[0][0]), item[0][1])):
        rows.append(
            {
                "repeat_location": location,
                "pattern": ">".join(pattern),
                "pattern_length_tokens": len(pattern),
                "path_support_count": len(support),
                "minimum_cycles": min(item[1] for item in support),
                "maximum_cycles": max(item[1] for item in support),
                "eligible_high_order_cycle": int(len(support) >= min_paths),
                "paths": ",".join(item[0] for item in support),
                "interpretation": "exact_family_level_periodicity",
            }
        )
    return rows


def edge_support_rows(edge_total, edge_paths, component_by_handle, segments):
    rows = []
    for edge in sorted(edge_total):
        source = (edge[0], edge[1])
        target = (edge[2], edge[3])
        source_scc = component_by_handle[source]
        target_scc = component_by_handle[target]
        rows.append(
            {
                "source_segment": edge[0],
                "source_orientation": edge[1],
                "target_segment": edge[2],
                "target_orientation": edge[3],
                "source_location": segment_location(segments[edge[0]]),
                "target_location": segment_location(segments[edge[2]]),
                "source_scc": source_scc,
                "target_scc": target_scc,
                "edge_class": "within_scc" if source_scc == target_scc else "condensation",
                "path_support_count": len(edge_paths[edge]),
                "total_traversals": edge_total[edge],
                "paths": ",".join(sorted(edge_paths[edge])),
            }
        )
    return rows


def path_loop_rows(paths, segments, component_by_handle, scc_rows, block_rows):
    cyclic_sccs = {row["scc_id"] for row in scc_rows if row["cyclic"] == 1}
    steps_by_path_location = defaultdict(list)
    for path in paths:
        for handle in path["steps"]:
            location = segment_location(segments[handle[0]])
            if re.fullmatch(r"R\d+", location):
                steps_by_path_location[(path["name"], location)].append(handle)
    output = []
    for block in block_rows:
        key = (block["path"], block["repeat_block"])
        handles = steps_by_path_location.get(key, [])
        cyclic_visits = sum(component_by_handle[handle] in cyclic_sccs for handle in handles)
        internal_cycle_edges = sum(
            component_by_handle[source] == component_by_handle[target]
            and component_by_handle[source] in cyclic_sccs
            for source, target in zip(handles, handles[1:])
        )
        output.append(
            {
                "path": block["path"],
                "repeat_block": block["repeat_block"],
                "family": block["canonical_family"],
                "p3_copy_count": block["copies"],
                "p3_exact_copies": block["exact_copies"],
                "p3_variant_copies": block["variant_copies"],
                "p3_inserted_bp": block["inserted_bp"],
                "graph_node_visits": len(handles),
                "cyclic_scc_node_visits": cyclic_visits,
                "within_cyclic_scc_edge_traversals": internal_cycle_edges,
                "visited_sccs": ",".join(dict.fromkeys(component_by_handle[handle] for handle in handles)),
                "coordinate_system": COORDINATE_SYSTEM,
            }
        )
    return output


def count_backward_paths(paths, segments) -> int:
    backward = 0
    for path in paths:
        positions = []
        for name, _orientation in path["steps"]:
            location = segment_location(segments[name])
            match = re.fullmatch(r"R(\d+)", location)
            if match:
                positions.append(int(match.group(1)))
        if any(left > right for left, right in zip(positions, positions[1:])):
            backward += 1
    return backward


def write_p5_gfa(output_path, headers, segments, paths, other, edge_total, edge_paths, handle_paths, component_by_handle, scc_rows):
    input_sequences = reconstruct_paths(segments, paths)
    cyclic = {row["scc_id"] for row in scc_rows if row["cyclic"] == 1}
    used_segments = {name for name, _orientation in handle_paths}
    segment_sccs = defaultdict(dict)
    for handle, scc_id in component_by_handle.items():
        segment_sccs[handle[0]][handle[1]] = scc_id
    header = headers[0]
    header = [
        field
        for field in header
        if not field.startswith(("TS:", "CS:", "P5:"))
    ]
    header.extend(["TS:Z:p5_path_supported_position_specific_scc", f"CS:Z:{COORDINATE_SYSTEM}", "P5:Z:condensation_DAG"])
    lines = ["\t".join(header)]
    for name in sorted(used_segments):
        segment = segments[name]
        extra = [field for field in segment["extra"] if not field.startswith(("SC:", "CY:", "PT:", "P5:"))]
        scc_map = segment_sccs[name]
        scc_text = (
            next(iter(scc_map.values()))
            if len(scc_map) == 1
            else ",".join(f"{orientation}={scc_id}" for orientation, scc_id in sorted(scc_map.items()))
        )
        is_cyclic = any(scc_id in cyclic for scc_id in scc_map.values())
        support_paths = {path for handle, values in handle_paths.items() if handle[0] == name for path in values}
        extra.extend([f"SC:Z:{scc_text}", f"CY:i:{int(is_cyclic)}", f"PT:i:{len(support_paths)}", "P5:Z:path_supported"])
        lines.append("\t".join(["S", name, segment["sequence"], *extra]))
    for edge in sorted(edge_total):
        source_scc = component_by_handle[(edge[0], edge[1])]
        target_scc = component_by_handle[(edge[2], edge[3])]
        edge_class = "within_scc" if source_scc == target_scc else "condensation"
        lines.append(
            "\t".join(
                [
                    "L",
                    edge[0],
                    edge[1],
                    edge[2],
                    edge[3],
                    "0M",
                    f"PS:i:{len(edge_paths[edge])}",
                    f"TC:i:{edge_total[edge]}",
                    f"ET:Z:{edge_class}",
                ]
            )
        )
    for fields in other:
        lines.append("\t".join(fields))
    for path in paths:
        extra = [field for field in path["extra"] if not field.startswith("P5:")]
        extra.append("P5:Z:path_supported_scc")
        lines.append("\t".join(["P", path["name"], path["step_text"], path["overlaps"], *extra]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    _headers, output_segments, output_paths, _other = read_gfa(output_path)
    output_sequences = reconstruct_paths(output_segments, output_paths)
    if input_sequences != output_sequences:
        raise AssertionError("P5 GFA reconstruction differs from input paths")
    return len(used_segments), len(edge_total), len(output_sequences)


def plot_overview(path: Path, scc_rows: list[dict], condensation_rows: list[dict], block_rows: list[dict]) -> None:
    repeat_locations = sorted({row["repeat_block"] for row in block_rows}, key=location_key)
    background_locations = sorted(
        {location for row in scc_rows for location in row["locations"].split(",") if location.startswith("B")},
        key=location_key,
    )
    locations = sorted(set(repeat_locations + background_locations), key=location_key)
    path_names = list(dict.fromkeys(row["path"] for row in block_rows))
    height = max(7.5, 4.5 + len(path_names) * 0.18)
    figure, axes = plt.subplots(2, 1, figsize=(13, height), gridspec_kw={"height_ratios": [1.2, max(2.2, len(path_names) / 7)]})

    axis = axes[0]
    x_by_location = {location: index for index, location in enumerate(locations)}
    repeat_colors = {location: color for location, color in zip(repeat_locations, ["#4d9b68", "#2e74b5", "#c56a2d", "#7d5ba6"])}
    for left, right in zip(locations, locations[1:]):
        axis.add_patch(FancyArrowPatch((x_by_location[left] + 0.22, 0), (x_by_location[right] - 0.22, 0), arrowstyle="->", mutation_scale=12, color="#59636e", linewidth=1.4))
    for location in locations:
        x = x_by_location[location]
        color = repeat_colors.get(location, "#cbd5df")
        circle = Circle((x, 0), 0.20, facecolor=color, edgecolor="#25313c", linewidth=1.2)
        axis.add_patch(circle)
        axis.text(x, 0, location, ha="center", va="center", fontsize=9, fontweight="bold")
        if location in repeat_locations:
            arc = Arc((x, 0.34), 0.40, 0.30, theta1=15, theta2=320, color="#25313c", linewidth=1.3)
            axis.add_patch(arc)
            axis.add_patch(FancyArrowPatch((x + 0.18, 0.30), (x + 0.19, 0.25), arrowstyle="-|>", mutation_scale=9, color="#25313c"))
    axis.set_xlim(-0.6, max(0.6, len(locations) - 0.4))
    axis.set_ylim(-0.45, 0.65)
    axis.axis("off")
    cyclic_count = sum(row["cyclic"] == 1 for row in scc_rows)
    axis.set_title(f"P5 SCC condensation order; cyclic SCCs={cyclic_count}, condensation edges={len(condensation_rows)}")

    values = {(row["path"], row["repeat_block"]): int(row["copies"]) for row in block_rows}
    matrix = [[values.get((name, location), 0) for location in repeat_locations] for name in path_names]
    image = axes[1].imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    axes[1].set_xticks(range(len(repeat_locations)), repeat_locations)
    axes[1].set_yticks(range(len(path_names)), path_names, fontsize=7)
    axes[1].set_xlabel("Position-specific repeat block")
    axes[1].set_title("Per-path P3 copy count carried into the P5 graph")
    if len(path_names) <= 55:
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                axes[1].text(column_index, row_index, str(value), ha="center", va="center", fontsize=6, color="white" if value > 0.55 * max(max(item) for item in matrix) else "black")
    figure.colorbar(image, ax=axes[1], label="Motif copies", fraction=0.025, pad=0.02)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="P5 path-supported SCC and condensation-DAG repeat graph.")
    parser.add_argument("--input-gfa", type=Path, default=P4_DIR / f"{P4_PREFIX}_P4变异证据标注图.gfa")
    parser.add_argument("--p3-blocks", type=Path, default=P3_DIR / f"{P3_PREFIX}_逐路径P3_repeat区块.tsv")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--max-high-order-period", type=int, default=6)
    parser.add_argument("--min-high-order-cycles", type=int, default=2)
    parser.add_argument("--min-high-order-paths", type=int, default=2)
    args = parser.parse_args()
    if min(args.max_high_order_period, args.min_high_order_cycles, args.min_high_order_paths) < 2:
        parser.error("High-order period/cycle/path thresholds must be at least 2")

    headers, segments, paths, other = read_gfa(args.input_gfa)
    block_rows, _ = read_tsv(args.p3_blocks)
    path_names = {path["name"] for path in paths}
    block_path_names = {row["path"] for row in block_rows}
    if path_names != block_path_names:
        raise ValueError("P3 block paths do not match P4/P3 GFA paths")
    vertices, adjacency, edge_total, edge_paths, handle_paths = build_path_supported_graph(paths)
    components = tarjan_scc(vertices, adjacency)
    component_by_handle, handles_by_scc = assign_scc_ids(components, segments)
    scc_rows = summarize_sccs(handles_by_scc, component_by_handle, segments, edge_total, edge_paths, handle_paths)
    condensation_rows = build_condensation(component_by_handle, edge_total, edge_paths)
    topo_order = topological_order(set(handles_by_scc), condensation_rows)
    bubble_rows = simple_bubbles(condensation_rows, topo_order)
    high_order_rows = detect_high_order(
        paths,
        segments,
        args.max_high_order_period,
        args.min_high_order_cycles,
        args.min_high_order_paths,
    )
    edge_rows = edge_support_rows(edge_total, edge_paths, component_by_handle, segments)
    loop_rows = path_loop_rows(paths, segments, component_by_handle, scc_rows, block_rows)

    args.outdir.mkdir(parents=True, exist_ok=True)
    scc_path = args.outdir / f"{args.prefix}_P5_SCC目录.tsv"
    dag_path = args.outdir / f"{args.prefix}_P5_压缩DAG.tsv"
    edge_path = args.outdir / f"{args.prefix}_P5_边路径支持.tsv"
    loop_path = args.outdir / f"{args.prefix}_P5_逐路径环遍历.tsv"
    bubble_path = args.outdir / f"{args.prefix}_P5_bubble候选.tsv"
    high_order_path = args.outdir / f"{args.prefix}_P5_高阶循环候选.tsv"
    gfa_path = args.outdir / f"{args.prefix}_P5位置特异SCC循环图.gfa"
    png_path = args.outdir / f"{args.prefix}_P5_SCC与路径次数总览.png"
    qa_path = args.outdir / f"{args.prefix}_P5验证汇总.tsv"

    write_tsv(scc_path, scc_rows, list(scc_rows[0]))
    write_tsv(dag_path, condensation_rows, list(condensation_rows[0]) if condensation_rows else ["source_scc", "target_scc"])
    write_tsv(edge_path, edge_rows, list(edge_rows[0]))
    write_tsv(loop_path, loop_rows, list(loop_rows[0]))
    write_tsv(bubble_path, bubble_rows, list(bubble_rows[0]) if bubble_rows else ["source_scc", "sink_scc", "branch_count", "branch_sccs", "max_branch_distance", "method"])
    write_tsv(
        high_order_path,
        high_order_rows,
        list(high_order_rows[0]) if high_order_rows else ["repeat_location", "pattern", "pattern_length_tokens", "path_support_count", "minimum_cycles", "maximum_cycles", "eligible_high_order_cycle", "paths", "interpretation"],
    )
    node_count, edge_count, exact = write_p5_gfa(
        gfa_path,
        headers,
        segments,
        paths,
        other,
        edge_total,
        edge_paths,
        handle_paths,
        component_by_handle,
        scc_rows,
    )
    plot_overview(png_path, scc_rows, condensation_rows, block_rows)

    repeat_locations = sorted({row["repeat_block"] for row in block_rows}, key=location_key)
    locations_with_cycle = sorted(
        {
            row["repeat_locations"]
            for row in scc_rows
            if row["cyclic"] == 1 and row["classification"] == "position_specific_repeat_core"
        },
        key=location_key,
    )
    missing_cycle = sorted(set(repeat_locations) - set(locations_with_cycle), key=location_key)
    cross_position = sum(row["classification"] == "invalid_cross_position_cycle" for row in scc_rows)
    backward_paths = count_backward_paths(paths, segments)
    qa = {
        "path_count": len(paths),
        "node_count": node_count,
        "path_supported_edge_count": edge_count,
        "unsupported_edge_count": 0,
        "scc_count": len(scc_rows),
        "cyclic_scc_count": sum(row["cyclic"] == 1 for row in scc_rows),
        "repeat_location_count": len(repeat_locations),
        "repeat_locations": ",".join(repeat_locations),
        "repeat_locations_with_cyclic_scc": ",".join(locations_with_cycle),
        "repeat_locations_without_cyclic_scc": ",".join(missing_cycle) if missing_cycle else "-",
        "cross_position_cycle_count": cross_position,
        "condensation_edge_count": len(condensation_rows),
        "condensation_is_DAG": 1,
        "paths_with_backward_repeat_order": backward_paths,
        "bubble_candidate_count": len(bubble_rows),
        "high_order_candidate_count": len(high_order_rows),
        "eligible_high_order_cycle_count": sum(int(row["eligible_high_order_cycle"]) for row in high_order_rows),
        "exact_gfa_reconstruction_paths": exact,
        "graph_status": "path_supported_position_specific_SCC",
    }
    write_tsv(qa_path, [{"metric": key, "value": value} for key, value in qa.items()], ["metric", "value"])

    if missing_cycle or cross_position or backward_paths:
        raise AssertionError(
            f"P5 topology validation failed: missing_cycle={missing_cycle}, cross_position={cross_position}, backward_paths={backward_paths}"
        )
    print(f"Paths: {len(paths)}; nodes={node_count}; path-supported edges={edge_count}")
    print(f"SCCs: {len(scc_rows)}; cyclic={qa['cyclic_scc_count']}; repeat locations={','.join(repeat_locations)}")
    print(f"Condensation DAG edges={len(condensation_rows)}; bubbles={len(bubble_rows)}; high-order candidates={len(high_order_rows)}")
    print(f"Exact GFA reconstruction: {exact}/{len(paths)}")
    for output in (scc_path, dag_path, edge_path, loop_path, bubble_path, high_order_path, gfa_path, png_path, qa_path):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
