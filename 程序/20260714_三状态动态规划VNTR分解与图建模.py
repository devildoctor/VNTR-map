#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


DEFAULT_INPUT = Path("测试数据") / "IRF2BPL.HPRC.fa.3034e58.11fba48.904d69e.smooth.final.gfa"
DEFAULT_OUTDIR = Path("outputs") / "IRF2BPL" / "20260714_三状态动态规划VNTR分解"
DEFAULT_PREFIX = "20260714_IRF2BPL_三状态动态规划"

STATE_INSERT = "I"
STATE_CAG = "C"
STATE_GCC = "G"
STATE_TO_GROUP = {STATE_CAG: "CAG_like", STATE_GCC: "GCC_like"}
GROUP_TO_STATE = {group: state for state, group in STATE_TO_GROUP.items()}
CANONICAL = {
    "CAG_like": ("CAG",),
    "GCC_like": ("GCC", "GCG"),
}

MIN_CORE_UNITS = 5
MIN_REPEAT_UNITS = 3
MIN_LOCAL_EXACT_UNITS = 1
MIN_REPEAT_DENSITY = 0.70
MAX_LOCAL_INSERT_BP = 12
MAX_CORE_GAP_BP = 150
MAX_REGION_BP = 460
LEFT_CONTEXT_BP = 27
RIGHT_CONTEXT_BP = 27

EXACT_REWARD = 8
VARIANT_REWARD = 3
ENTER_REPEAT_PENALTY = 7
LEAVE_REPEAT_PENALTY = 1
SWITCH_REPEAT_PENALTY = 5


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


@dataclass(frozen=True)
class MotifCall:
    group: str
    canonical: str
    observed: str
    kind: str


@dataclass
class Anchor:
    start: int
    end: int
    group: str
    calls: list[MotifCall]
    score: int

    @property
    def units(self) -> int:
        return len(self.calls)

    @property
    def exact_units(self) -> int:
        return sum(call.kind == "exact" for call in self.calls)


@dataclass
class Atom:
    state: str
    start: int
    end: int
    observed: str
    kind: str
    canonical: str = ""

    @property
    def group(self) -> str:
        return STATE_TO_GROUP.get(self.state, "nonrepeat")


@dataclass
class Block:
    path: str
    index: int
    start: int
    end: int
    role: str
    group: str
    seq: str
    nodes: list[str]
    exact_counts: Counter[str]
    variant_counts: Counter[str]
    insertion_parts: list[tuple[int, str]]
    anchor_supported: bool

    @property
    def exact_units(self) -> int:
        return sum(self.exact_counts.values())

    @property
    def variant_units(self) -> int:
        return sum(self.variant_counts.values())

    @property
    def repeat_units(self) -> int:
        return self.exact_units + self.variant_units

    @property
    def inserted_bp(self) -> int:
        return sum(len(seq) for _pos, seq in self.insertion_parts)

    @property
    def repeat_density(self) -> float:
        return 3 * self.repeat_units / max(1, self.end - self.start)


def read_gfa(path: Path) -> tuple[dict[str, Segment], list[PathRecord]]:
    segments: dict[str, Segment] = {}
    paths: list[PathRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            fields = raw.rstrip("\r\n").split("\t")
            if not fields or not fields[0]:
                continue
            if fields[0] == "S":
                segments[fields[1]] = Segment(fields[1], fields[2].upper())
            elif fields[0] == "P":
                steps = [step[:-1] for step in fields[2].split(",") if step]
                paths.append(PathRecord(fields[1], steps))

    for record in paths:
        parts: list[str] = []
        offset = 0
        for node in record.steps:
            seq = segments[node].seq
            parts.append(seq)
            record.spans.append(NodeSpan(node, offset, offset + len(seq), seq))
            offset += len(seq)
        record.seq = "".join(parts)
    return segments, paths


def hamming(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right)) if len(left) == len(right) else max(len(left), len(right))


def call_for_group(observed: str, group: str) -> MotifCall | None:
    candidates = [(motif, hamming(observed, motif)) for motif in CANONICAL[group]]
    canonical, distance = min(candidates, key=lambda item: (item[1], CANONICAL[group].index(item[0])))
    if distance == 0:
        return MotifCall(group, canonical, observed, "exact")
    if distance == 1:
        return MotifCall(group, canonical, observed, "variant")
    return None


def best_triplet_call(observed: str) -> MotifCall | None:
    calls = [call_for_group(observed, group) for group in CANONICAL]
    calls = [call for call in calls if call is not None]
    if not calls:
        return None
    return min(calls, key=lambda call: (call.kind != "exact", hamming(call.observed, call.canonical), call.group))


def anchor_candidates(seq: str) -> list[Anchor]:
    candidates: list[Anchor] = []
    for frame in range(3):
        run_start: int | None = None
        run_group: str | None = None
        run_calls: list[MotifCall] = []

        def flush() -> None:
            nonlocal run_start, run_group, run_calls
            exact = sum(call.kind == "exact" for call in run_calls)
            if (
                run_start is not None
                and run_group is not None
                and len(run_calls) >= MIN_CORE_UNITS
                and exact >= 3
                and exact / len(run_calls) >= 0.50
            ):
                score = 10 * exact + 3 * (len(run_calls) - exact)
                candidates.append(Anchor(run_start, run_start + 3 * len(run_calls), run_group, list(run_calls), score))
            run_start = None
            run_group = None
            run_calls = []

        for pos in range(frame, len(seq) - 2, 3):
            call = best_triplet_call(seq[pos : pos + 3])
            if call is None:
                flush()
            elif run_group is None:
                run_start = pos
                run_group = call.group
                run_calls = [call]
            elif call.group == run_group:
                run_calls.append(call)
            else:
                flush()
                run_start = pos
                run_group = call.group
                run_calls = [call]
        flush()
    return candidates


def select_nonoverlapping_anchors(candidates: list[Anchor]) -> list[Anchor]:
    ordered = sorted(candidates, key=lambda anchor: (anchor.end, anchor.start, -anchor.score))
    scores = [0] * (len(ordered) + 1)
    selections: list[list[Anchor]] = [[] for _ in range(len(ordered) + 1)]
    for index, anchor in enumerate(ordered, start=1):
        previous = 0
        for prior in range(index - 1, 0, -1):
            if ordered[prior - 1].end <= anchor.start:
                previous = prior
                break
        include = scores[previous] + anchor.score
        exclude = scores[index - 1]
        if include > exclude:
            scores[index] = include
            selections[index] = selections[previous] + [anchor]
        else:
            scores[index] = exclude
            selections[index] = list(selections[index - 1])
    return sorted(selections[-1], key=lambda anchor: anchor.start)


def locate_primary_region(seq: str) -> tuple[int, int, list[Anchor]]:
    anchors = select_nonoverlapping_anchors(anchor_candidates(seq))
    if not anchors:
        return 0, len(seq), []

    clusters: list[list[Anchor]] = []
    for anchor in anchors:
        if (
            not clusters
            or anchor.start - clusters[-1][-1].end > MAX_CORE_GAP_BP
            or anchor.end - clusters[-1][0].start > MAX_REGION_BP
        ):
            clusters.append([anchor])
        else:
            clusters[-1].append(anchor)
    core = max(clusters, key=lambda cluster: (sum(anchor.score for anchor in cluster), sum(anchor.units for anchor in cluster)))
    start = max(0, core[0].start - LEFT_CONTEXT_BP)
    end = min(len(seq), core[-1].end + RIGHT_CONTEXT_BP)
    return start, end, core


def transition_score(previous: str, current: str) -> int:
    if previous == current:
        return 0
    if previous == STATE_INSERT and current != STATE_INSERT:
        return -ENTER_REPEAT_PENALTY
    if previous != STATE_INSERT and current == STATE_INSERT:
        return -LEAVE_REPEAT_PENALTY
    return -SWITCH_REPEAT_PENALTY


def update_dp(
    scores: list[dict[str, int]],
    back: dict[tuple[int, str], tuple[int, str, Atom]],
    end: int,
    state: str,
    score: int,
    previous_pos: int,
    previous_state: str,
    atom: Atom,
) -> None:
    if score > scores[end].get(state, -10**12):
        scores[end][state] = score
        back[(end, state)] = (previous_pos, previous_state, atom)


def viterbi_three_state(region: str, offset: int) -> list[Atom]:
    length = len(region)
    scores: list[dict[str, int]] = [dict() for _ in range(length + 1)]
    back: dict[tuple[int, str], tuple[int, str, Atom]] = {}
    scores[0][STATE_INSERT] = 0

    for pos in range(length):
        if not scores[pos]:
            continue
        for previous_state, previous_score in list(scores[pos].items()):
            observed_base = region[pos : pos + 1]
            atom = Atom(STATE_INSERT, offset + pos, offset + pos + 1, observed_base, "insertion")
            update_dp(
                scores,
                back,
                pos + 1,
                STATE_INSERT,
                previous_score + transition_score(previous_state, STATE_INSERT),
                pos,
                previous_state,
                atom,
            )

            if pos + 3 > length:
                continue
            observed = region[pos : pos + 3]
            for group, state in GROUP_TO_STATE.items():
                call = call_for_group(observed, group)
                if call is None:
                    continue
                reward = EXACT_REWARD if call.kind == "exact" else VARIANT_REWARD
                atom = Atom(state, offset + pos, offset + pos + 3, observed, call.kind, call.canonical)
                update_dp(
                    scores,
                    back,
                    pos + 3,
                    state,
                    previous_score + transition_score(previous_state, state) + reward,
                    pos,
                    previous_state,
                    atom,
                )

    state = max(scores[length], key=scores[length].get)
    pos = length
    atoms: list[Atom] = []
    while pos > 0:
        previous_pos, previous_state, atom = back[(pos, state)]
        atoms.append(atom)
        pos = previous_pos
        state = previous_state
    atoms.reverse()
    return atoms


def distance_to_anchor(start: int, end: int, anchor: Anchor) -> int:
    if start < anchor.end and end > anchor.start:
        return 0
    return min(abs(start - anchor.end), abs(anchor.start - end))


def collapse_insertions(atoms: list[Atom]) -> list[Atom]:
    collapsed: list[Atom] = []
    for atom in atoms:
        if collapsed and atom.state == STATE_INSERT and collapsed[-1].state == STATE_INSERT and collapsed[-1].end == atom.start:
            collapsed[-1].end = atom.end
            collapsed[-1].observed += atom.observed
        else:
            collapsed.append(atom)
    return collapsed


def filter_unsupported_repeat_runs(atoms: list[Atom], anchors: list[Anchor]) -> list[Atom]:
    filtered: list[Atom] = []
    index = 0
    while index < len(atoms):
        atom = atoms[index]
        if atom.state == STATE_INSERT:
            filtered.append(atom)
            index += 1
            continue
        end_index = index + 1
        while end_index < len(atoms) and atoms[end_index].state == atom.state:
            end_index += 1
        run = atoms[index:end_index]
        exact = sum(item.kind == "exact" for item in run)
        supported = any(
            anchor.group == atom.group
            and distance_to_anchor(run[0].start, run[-1].end, anchor) <= MAX_LOCAL_INSERT_BP
            for anchor in anchors
        )
        valid = len(run) >= MIN_REPEAT_UNITS and exact >= MIN_LOCAL_EXACT_UNITS and supported
        if valid:
            filtered.extend(run)
        else:
            for item in run:
                filtered.append(Atom(STATE_INSERT, item.start, item.end, item.observed, "insertion"))
        index = end_index
    return collapse_insertions(filtered)


def nodes_for_interval(path: PathRecord, start: int, end: int) -> list[str]:
    return [span.node for span in path.spans if span.path_start < end and span.path_end > start]


def block_from_atoms(path: PathRecord, atoms: list[Atom], anchors: list[Anchor]) -> Block:
    start = atoms[0].start
    end = atoms[-1].end
    repeat_atoms = [atom for atom in atoms if atom.state != STATE_INSERT]
    insertion_atoms = [atom for atom in atoms if atom.state == STATE_INSERT]
    exact_counts: Counter[str] = Counter(atom.observed for atom in repeat_atoms if atom.kind == "exact")
    variant_counts: Counter[str] = Counter(
        f"{atom.canonical}>{atom.observed}" for atom in repeat_atoms if atom.kind == "variant"
    )
    groups = Counter(atom.group for atom in repeat_atoms)
    group = groups.most_common(1)[0][0] if groups else "nonrepeat"
    insertion_parts = [(atom.start, atom.observed) for atom in insertion_atoms]
    anchor_supported = any(anchor.group == group and distance_to_anchor(start, end, anchor) == 0 for anchor in anchors)
    if not repeat_atoms:
        role = "local_insertion" if end - start <= MAX_LOCAL_INSERT_BP else "complex_sequence"
    elif insertion_parts and variant_counts:
        role = "variant_and_interrupted_repeat"
    elif insertion_parts:
        role = "interrupted_repeat"
    elif variant_counts:
        role = "variant_repeat"
    else:
        role = "primary_repeat"
    return Block(
        path=path.name,
        index=0,
        start=start,
        end=end,
        role=role,
        group=group,
        seq=path.seq[start:end],
        nodes=nodes_for_interval(path, start, end),
        exact_counts=exact_counts,
        variant_counts=variant_counts,
        insertion_parts=insertion_parts,
        anchor_supported=anchor_supported,
    )


def atoms_to_blocks(path: PathRecord, atoms: list[Atom], anchors: list[Anchor]) -> list[Block]:
    provisional: list[Block] = []
    index = 0
    while index < len(atoms):
        atom = atoms[index]
        end_index = index + 1
        if atom.state == STATE_INSERT:
            while end_index < len(atoms) and atoms[end_index].state == STATE_INSERT:
                end_index += 1
        else:
            while end_index < len(atoms) and atoms[end_index].state == atom.state:
                end_index += 1
        provisional.append(block_from_atoms(path, atoms[index:end_index], anchors))
        index = end_index

    blocks: list[Block] = []
    index = 0
    while index < len(provisional):
        current = provisional[index]
        while (
            current.group != "nonrepeat"
            and index + 2 < len(provisional)
            and provisional[index + 1].group == "nonrepeat"
            and provisional[index + 1].end - provisional[index + 1].start <= MAX_LOCAL_INSERT_BP
            and provisional[index + 2].group == current.group
        ):
            insertion = provisional[index + 1]
            right = provisional[index + 2]
            units = current.repeat_units + right.repeat_units
            density = 3 * units / max(1, right.end - current.start)
            if density < MIN_REPEAT_DENSITY:
                break
            merged_atoms: list[Atom] = []
            for block in (current, insertion, right):
                if block.group == "nonrepeat":
                    merged_atoms.append(Atom(STATE_INSERT, block.start, block.end, block.seq, "insertion"))
                else:
                    state = GROUP_TO_STATE[block.group]
                    pos = block.start
                    while pos < block.end:
                        if any(ins_pos == pos for ins_pos, _seq in block.insertion_parts):
                            ins_seq = next(seq for ins_pos, seq in block.insertion_parts if ins_pos == pos)
                            merged_atoms.append(Atom(STATE_INSERT, pos, pos + len(ins_seq), ins_seq, "insertion"))
                            pos += len(ins_seq)
                            continue
                        observed = path.seq[pos : pos + 3]
                        call = call_for_group(observed, block.group)
                        if call is None:
                            merged_atoms.append(Atom(STATE_INSERT, pos, pos + 1, observed[:1], "insertion"))
                            pos += 1
                        else:
                            merged_atoms.append(Atom(state, pos, pos + 3, observed, call.kind, call.canonical))
                            pos += 3
            current = block_from_atoms(path, collapse_insertions(merged_atoms), anchors)
            index += 2
        blocks.append(current)
        index += 1

    for block_index, block in enumerate(blocks, start=1):
        block.index = block_index
    return blocks


def motif_text(counter: Counter[str]) -> str:
    return ";".join(f"{key}:{counter[key]}" for key in sorted(counter)) or "-"


def insertion_text(parts: list[tuple[int, str]]) -> str:
    return ";".join(f"{pos}:{seq}" for pos, seq in parts) or "-"


def path_suffix_value(name: str) -> str:
    match = re.search(r"_([0-9]+(?:\.[0-9]+)?)$", name)
    return match.group(1) if match else ""


def analyze(
    paths: list[PathRecord],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, list[Block]],
    dict[str, list[Atom]],
]:
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    blocks_by_path: dict[str, list[Block]] = {}
    atoms_by_path: dict[str, list[Atom]] = {}
    evidence_units: Counter[tuple[str, str, str, str]] = Counter()
    evidence_paths: defaultdict[tuple[str, str, str, str], set[str]] = defaultdict(set)

    for path in paths:
        start, end, anchors = locate_primary_region(path.seq)
        for anchor in anchors:
            for call in anchor.calls:
                key = (call.group, call.canonical, call.observed, call.kind)
                evidence_units[key] += 1
                evidence_paths[key].add(path.name)

        raw_atoms = viterbi_three_state(path.seq[start:end], start)
        atoms = filter_unsupported_repeat_runs(raw_atoms, anchors)
        blocks = atoms_to_blocks(path, atoms, anchors)
        blocks_by_path[path.name] = blocks
        atoms_by_path[path.name] = atoms

        group_units: Counter[str] = Counter()
        for block in blocks:
            group_units[block.group] += block.repeat_units
        repeat_groups = [block.group for block in blocks if block.group != "nonrepeat"]
        role_counts = Counter(block.role for block in blocks)
        summary_rows.append(
            {
                "path": path.name,
                "path_suffix_label": path_suffix_value(path.name),
                "region_start_bp": start,
                "region_end_bp": end,
                "region_bp": end - start,
                "primary_motifs": "CAG_like=CAG;GCC_like=GCC,GCG",
                "block_count": len(blocks),
                "CAG_like_units": group_units["CAG_like"],
                "GCC_like_units": group_units["GCC_like"],
                "exact_units": sum(block.exact_units for block in blocks),
                "variant_units": sum(block.variant_units for block in blocks),
                "inserted_bp": sum(block.inserted_bp for block in blocks),
                "switches": sum(left != right for left, right in zip(repeat_groups, repeat_groups[1:])),
                "primary_blocks": role_counts["primary_repeat"],
                "variant_blocks": role_counts["variant_repeat"] + role_counts["variant_and_interrupted_repeat"],
                "interrupted_blocks": role_counts["interrupted_repeat"] + role_counts["variant_and_interrupted_repeat"],
                "complex_blocks": role_counts["complex_sequence"] + role_counts["local_insertion"],
                "block_signature": "|".join(f"{block.group}:{block.repeat_units}:{block.role}" for block in blocks),
            }
        )
        for block in blocks:
            detail_rows.append(
                {
                    "path": block.path,
                    "block_index": block.index,
                    "role": block.role,
                    "group": block.group,
                    "start_bp": block.start,
                    "end_bp": block.end,
                    "bp": block.end - block.start,
                    "nodes": ",".join(block.nodes),
                    "anchor_supported": int(block.anchor_supported),
                    "repeat_units": block.repeat_units,
                    "exact_units": block.exact_units,
                    "variant_units": block.variant_units,
                    "inserted_bp": block.inserted_bp,
                    "repeat_density": f"{block.repeat_density:.4f}",
                    "motif_counts": motif_text(block.exact_counts),
                    "variant_counts": motif_text(block.variant_counts),
                    "insertions": insertion_text(block.insertion_parts),
                    "sequence": block.seq,
                }
            )

    evidence_rows = [
        {
            "group": key[0],
            "canonical_motif": key[1],
            "observed_motif": key[2],
            "type": key[3],
            "anchor_path_count": len(evidence_paths[key]),
            "anchor_units": evidence_units[key],
        }
        for key in sorted(evidence_units)
    ]
    return summary_rows, detail_rows, evidence_rows, blocks_by_path, atoms_by_path


def write_tsv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row.get(column, "")) for column in columns) + "\n")


def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def atom_node(atom: Atom) -> tuple[str, str, list[str]]:
    if atom.state == STATE_INSERT:
        digest = hashlib.sha1(atom.observed.encode("ascii")).hexdigest()[:12]
        return f"INS_{digest}", atom.observed, ["BT:Z:insertion_or_complex", "GR:Z:nonrepeat"]
    group = atom.group
    if atom.kind == "exact":
        node_id = f"M_{group}_{atom.observed}"
        tags = ["BT:Z:exact_motif", f"GR:Z:{group}", f"CA:Z:{atom.canonical}"]
    else:
        node_id = f"V_{group}_{atom.canonical}_to_{atom.observed}"
        tags = ["BT:Z:variant_motif", f"GR:Z:{group}", f"CA:Z:{atom.canonical}", "ED:i:1"]
    return node_id, atom.observed, tags


def write_state_gfa(
    path: Path,
    atoms_by_path: dict[str, list[Atom]],
    summary_rows: list[dict[str, object]],
) -> None:
    nodes: dict[str, tuple[str, list[str]]] = {}
    path_steps: dict[str, list[str]] = {}
    edges: set[tuple[str, str]] = set()
    for path_name, atoms in atoms_by_path.items():
        steps: list[str] = []
        for atom in atoms:
            node_id, seq, tags = atom_node(atom)
            nodes[node_id] = (seq, tags)
            steps.append(node_id)
        path_steps[path_name] = steps
        edges.update(zip(steps, steps[1:]))

    summary_by_path = {str(row["path"]): row for row in summary_rows}
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("H\tVN:Z:1.0\tTS:Z:three_state_dp_vntr\n")
        for node_id in sorted(nodes):
            seq, tags = nodes[node_id]
            handle.write(f"S\t{node_id}\t{seq}\t" + "\t".join(tags) + "\n")
        for left, right in sorted(edges):
            handle.write(f"L\t{left}\t+\t{right}\t+\t0M\n")
        for path_name, steps in path_steps.items():
            overlaps = ",".join("0M" for _ in steps[1:]) if len(steps) > 1 else "*"
            summary = summary_by_path[path_name]
            tags = (
                f"CU:i:{summary['CAG_like_units']}\tGU:i:{summary['GCC_like_units']}"
                f"\tVU:i:{summary['variant_units']}\tIB:i:{summary['inserted_bp']}"
                f"\tBS:Z:{summary['block_signature']}"
            )
            handle.write(f"P\t{path_name}\t{','.join(step + '+' for step in steps)}\t{overlaps}\t{tags}\n")


def plot_blocks(path: Path, summary_rows: list[dict[str, object]], detail_rows: list[dict[str, object]]) -> None:
    by_path: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        by_path[str(row["path"])].append(row)
    sorted_paths = [str(row["path"]) for row in sorted(summary_rows, key=lambda row: (int(row["region_bp"]), str(row["path"])))]
    fig_height = max(9, 0.27 * len(sorted_paths) + 2.8)
    fig, ax = plt.subplots(figsize=(15, fig_height))
    colors = {"CAG_like": "#3478b8", "GCC_like": "#4d9b68", "nonrepeat": "#9aa1ad"}
    hatches = {
        "primary_repeat": "",
        "variant_repeat": "//",
        "interrupted_repeat": "xx",
        "variant_and_interrupted_repeat": "xx",
        "local_insertion": "//",
        "complex_sequence": "..",
    }
    max_x = 1.0
    for y, path_name in enumerate(sorted_paths):
        x = 0.0
        for block in by_path[path_name]:
            units = int(block["exact_units"]) + int(block["variant_units"])
            width = max(0.45, units if units else int(block["bp"]) / 9)
            rect = Rectangle(
                (x, y - 0.34),
                width,
                0.68,
                facecolor=colors[str(block["group"])],
                edgecolor="white",
                linewidth=0.7,
            )
            rect.set_hatch(hatches[str(block["role"])])
            ax.add_patch(rect)
            if width >= 1.4:
                variants = int(block["variant_units"])
                label = f"{units}" if units else f"{block['bp']}bp"
                if variants:
                    label += f" v{variants}"
                ax.text(x + width / 2, y, label, ha="center", va="center", fontsize=6, color="white")
            x += width
        max_x = max(max_x, x)
    ax.set_yticks(range(len(sorted_paths)))
    ax.set_yticklabels(sorted_paths, fontsize=6)
    ax.set_xlim(0, max_x + 1)
    ax.set_ylim(len(sorted_paths) - 0.4, -1.9)
    ax.set_xlabel("Repeat blocks use motif-unit counts; gray blocks use bp/9")
    ax.set_title("IRF2BPL three-state dynamic-programming VNTR decomposition")
    ax.text(
        0,
        -1.3,
        "blue=CAG-like, green=GCC-like, gray=insertion/complex; //=variant, xx=interrupted.",
        fontsize=8,
    )
    ax.grid(axis="x", color="#d0d5dd", linewidth=0.5, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-state DP decomposition of an irregular VNTR from a PGGB GFA.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    _segments, paths = read_gfa(args.input)
    summary, details, evidence, _blocks, atoms = analyze(paths)
    summary_path = args.outdir / f"{args.prefix}_路径汇总.tsv"
    detail_path = args.outdir / f"{args.prefix}_block明细.tsv"
    evidence_path = args.outdir / f"{args.prefix}_主体motif证据.tsv"
    gfa_path = args.outdir / f"{args.prefix}_状态路径图.gfa"
    png_path = args.outdir / f"{args.prefix}_block图.png"

    write_tsv(
        summary_path,
        summary,
        [
            "path", "path_suffix_label", "region_start_bp", "region_end_bp", "region_bp", "primary_motifs",
            "block_count", "CAG_like_units", "GCC_like_units", "exact_units", "variant_units", "inserted_bp",
            "switches", "primary_blocks", "variant_blocks", "interrupted_blocks", "complex_blocks", "block_signature",
        ],
    )
    write_tsv(
        detail_path,
        details,
        [
            "path", "block_index", "role", "group", "start_bp", "end_bp", "bp", "nodes", "anchor_supported",
            "repeat_units", "exact_units", "variant_units", "inserted_bp", "repeat_density", "motif_counts", "variant_counts",
            "insertions", "sequence",
        ],
    )
    write_tsv(
        evidence_path,
        evidence,
        ["group", "canonical_motif", "observed_motif", "type", "anchor_path_count", "anchor_units"],
    )
    write_state_gfa(gfa_path, atoms, summary)
    plot_blocks(png_path, summary, details)

    cag = [int(row["CAG_like_units"]) for row in summary]
    gcc = [int(row["GCC_like_units"]) for row in summary]
    variants = [int(row["variant_units"]) for row in summary]
    print(f"Paths: {len(paths)}")
    print(f"CAG-like units: {min(cag)}-{max(cag)}")
    print(f"GCC-like units: {min(gcc)}-{max(gcc)}")
    print(f"Variant units: {min(variants)}-{max(variants)}")
    for output in (summary_path, detail_path, evidence_path, gfa_path, png_path):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
