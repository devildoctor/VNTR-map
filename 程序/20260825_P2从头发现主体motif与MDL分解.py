#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


DEFAULT_INPUT = Path("测试数据") / "IRF2BPL.HPRC.fa.3034e58.11fba48.904d69e.smooth.final.gfa"
DEFAULT_BOUNDARIES = (
    Path("outputs")
    / "IRF2BPL"
    / "20260825_P1唯一侧翼锚点与共识边界"
    / "20260825_IRF2BPL_P1唯一侧翼锚点与共识边界_路径边界共识.tsv"
)
DEFAULT_OUTDIR = Path("outputs") / "IRF2BPL" / "20260825_P2从头发现主体motif与MDL分解"
DEFAULT_PREFIX = "20260825_IRF2BPL_P2从头发现主体motif与MDL分解"

DEFAULT_MAX_MOTIF_BP = 18
DEFAULT_MIN_COPIES = 4
DEFAULT_MIN_TANDEM_BP = 12
DEFAULT_MIN_SEED_DENSITY = 0.35
DEFAULT_MIN_PATH_FRACTION = 0.10
DEFAULT_MIN_BLOCK_DENSITY = 0.70
DEFAULT_MAX_LOCAL_INSERT_BP = 12
DEFAULT_MAX_FAMILIES = 6
DEFAULT_MAX_MDL_CANDIDATES = 8
MAX_CONSECUTIVE_VARIANT_TOKENS = 2
MIN_CORE_EXACT_FRACTION = 0.60
MIN_BRIDGED_RUN_EXACT_FRACTION = 0.70

BACKGROUND_BITS_PER_BP = 2.0
EXACT_MOTIF_BITS_PER_BP = 0.25
VARIANT_EDIT_BITS = 4.0
ENTER_MOTIF_BITS = 8.0
EXIT_MOTIF_BITS = 4.0
SWITCH_MOTIF_BITS = 10.0
PHASE_SWITCH_BITS = 10.0
FAMILY_FIXED_BITS = 12.0
FAMILY_BASE_BITS = 2.0
MIN_MDL_GAIN_BITS = 1.0
LOCAL_INSERT_RUN_BITS = 5.0
LOCAL_INSERT_BASE_BITS = 1.5

DNA_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")
COORDINATE_SYSTEM = "0-based_half-open"


@dataclass(frozen=True)
class PathRecord:
    name: str
    sequence: str


@dataclass(frozen=True)
class Region:
    path: str
    sequence: str
    start: int
    end: int


@dataclass(frozen=True)
class SeedEvidence:
    path: str
    canonical: str
    period: int
    representative: str
    start: int
    end: int
    copies: int
    density: float


@dataclass
class FamilyCandidate:
    canonical: str
    period: int
    representative: str
    path_support: int
    total_seed_bp: int
    max_seed_copies: int
    mean_seed_density: float
    orientations: tuple[str, ...]
    composite_of: tuple[str, ...] = ()
    shadowed_by: tuple[str, ...] = ()
    selected_step: int = 0
    mdl_gain_bits: float = 0.0


@dataclass(frozen=True)
class Action:
    path: str
    start: int
    end: int
    kind: str
    observed: str
    canonical: str = ""
    oriented_motif: str = ""
    edit_distance: int = 0


@dataclass
class RepeatBlock:
    path: str
    order: int
    start: int
    end: int
    canonical: str
    oriented_motif: str
    copies: int
    exact_copies: int
    variant_copies: int
    inserted_bp: int
    insertion_parts: tuple[tuple[int, str], ...]
    density: float


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load program: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reverse_complement(sequence: str) -> str:
    return sequence.translate(DNA_COMPLEMENT)[::-1]


def rotations(sequence: str) -> tuple[str, ...]:
    return tuple(sequence[index:] + sequence[:index] for index in range(len(sequence)))


def equivalent_motifs(sequence: str) -> tuple[str, ...]:
    values = set(rotations(sequence))
    values.update(rotations(reverse_complement(sequence)))
    return tuple(sorted(values))


def primitive_root(sequence: str) -> str:
    for period in range(1, len(sequence) + 1):
        if len(sequence) % period == 0 and sequence == sequence[:period] * (len(sequence) // period):
            return sequence[:period]
    return sequence


def canonical_motif(sequence: str) -> str:
    root = primitive_root(sequence.upper())
    return min(equivalent_motifs(root))


def hamming(left: str, right: str) -> int:
    if len(left) != len(right):
        return max(len(left), len(right))
    return sum(a != b for a, b in zip(left, right))


def parse_walk(walk: str) -> list[tuple[str, str]]:
    return [(match.group(2), "+" if match.group(1) == ">" else "-") for match in re.finditer(r"([><])([^><]+)", walk)]


def read_gfa_paths(path: Path) -> list[PathRecord]:
    segments: dict[str, str] = {}
    path_steps: list[tuple[str, list[tuple[str, str]]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            fields = raw.rstrip("\r\n").split("\t")
            if not fields or not fields[0]:
                continue
            if fields[0] == "S":
                segments[fields[1]] = fields[2].upper()
            elif fields[0] == "P":
                steps = [(token[:-1], token[-1]) for token in fields[2].split(",") if token]
                path_steps.append((fields[1], steps))
            elif fields[0] == "W":
                sample, haplotype, sequence_id = fields[1], fields[2], fields[3]
                name = sequence_id if sequence_id != "*" else f"{sample}#{haplotype}"
                path_steps.append((name, parse_walk(fields[6])))
    paths = []
    for name, steps in path_steps:
        parts = []
        for node, orientation in steps:
            if node not in segments:
                raise ValueError(f"Path {name} references missing segment {node}")
            sequence = segments[node]
            parts.append(sequence if orientation == "+" else reverse_complement(sequence))
        paths.append(PathRecord(name, "".join(parts)))
    if not paths:
        raise ValueError(f"No P or W paths found in {path}")
    return paths


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_regions(paths: list[PathRecord], boundary_path: Path) -> tuple[list[Region], list[Region]]:
    boundaries = {row["path"]: row for row in read_tsv(boundary_path)}
    if set(boundaries) != {path.name for path in paths}:
        missing = sorted({path.name for path in paths} - set(boundaries))
        extra = sorted(set(boundaries) - {path.name for path in paths})
        raise ValueError(f"P1 boundaries do not match GFA paths; missing={missing}, extra={extra}")
    homology_regions = []
    repeat_regions = []
    for path in paths:
        row = boundaries[path.name]
        homology_start = int(row["homology_window_start_bp"])
        homology_end = int(row["homology_window_end_bp"])
        repeat_start = int(row["consensus_start_bp"])
        repeat_end = int(row["consensus_end_bp"])
        if not 0 <= homology_start <= repeat_start < repeat_end <= homology_end <= len(path.sequence):
            raise ValueError(
                f"Invalid P1 windows for {path.name}: "
                f"homology={homology_start}-{homology_end}, repeat={repeat_start}-{repeat_end}"
            )
        homology_regions.append(
            Region(path.name, path.sequence[homology_start:homology_end], homology_start, homology_end)
        )
        repeat_regions.append(Region(path.name, path.sequence[repeat_start:repeat_end], repeat_start, repeat_end))
    return homology_regions, repeat_regions


def minimum_copies(period: int, min_copies: int, min_tandem_bp: int) -> int:
    return max(min_copies, math.ceil(min_tandem_bp / period))


def best_dense_seed(
    positions: list[tuple[int, str]],
    period: int,
    min_copies: int,
    min_density: float,
) -> tuple[int, int, int, float, str] | None:
    best = None
    for left in range(len(positions)):
        for right in range(left + min_copies - 1, len(positions)):
            start = positions[left][0]
            end = positions[right][0] + period
            copies = right - left + 1
            density = copies * period / max(1, end - start)
            if density < min_density:
                continue
            orientations = Counter(value for _position, value in positions[left : right + 1])
            representative = min(orientations, key=lambda value: (-orientations[value], value))
            score = (copies * period, density, -start, representative)
            if best is None or score > best[0]:
                best = (score, start, end, copies, density, representative)
    if best is None:
        return None
    return best[1], best[2], best[3], best[4], best[5]


def discover_seed_evidence(
    regions: list[Region],
    max_motif_bp: int,
    min_copies: int,
    min_tandem_bp: int,
    min_density: float,
) -> list[SeedEvidence]:
    best_by_path_family: dict[tuple[str, str], SeedEvidence] = {}
    for region in regions:
        sequence = region.sequence
        for period in range(1, min(max_motif_bp, len(sequence)) + 1):
            required = minimum_copies(period, min_copies, min_tandem_bp)
            for phase in range(period):
                by_family: dict[str, list[tuple[int, str]]] = defaultdict(list)
                for position in range(phase, len(sequence) - period + 1, period):
                    observed = sequence[position : position + period]
                    if primitive_root(observed) != observed:
                        continue
                    by_family[canonical_motif(observed)].append((position, observed))
                for canonical, positions in by_family.items():
                    seed = best_dense_seed(positions, period, required, min_density)
                    if seed is None:
                        continue
                    start, end, copies, density, representative = seed
                    evidence = SeedEvidence(
                        region.path,
                        canonical,
                        period,
                        representative,
                        region.start + start,
                        region.start + end,
                        copies,
                        density,
                    )
                    key = (region.path, canonical)
                    previous = best_by_path_family.get(key)
                    if previous is None or (copies * period, density, -start) > (
                        previous.copies * previous.period,
                        previous.density,
                        -previous.start,
                    ):
                        best_by_path_family[key] = evidence
    return sorted(best_by_path_family.values(), key=lambda row: (row.canonical, row.path))


def composite_components(candidate: FamilyCandidate, available: dict[str, FamilyCandidate]) -> tuple[str, ...]:
    sequence = candidate.representative
    # Single-base composition reflects nucleotide frequency, not an independently
    # supported biological repeat-unit alternation such as CAG/GCC.
    for period in range(2, candidate.period):
        if candidate.period % period:
            continue
        chunks = [sequence[index : index + period] for index in range(0, candidate.period, period)]
        components = tuple(canonical_motif(chunk) for chunk in chunks)
        if len(set(components)) < 2:
            continue
        if all(component in available and available[component].period == period for component in components):
            return components
    return ()


def aggregate_candidates(
    evidence: list[SeedEvidence], path_count: int, min_path_fraction: float
) -> list[FamilyCandidate]:
    grouped: dict[str, list[SeedEvidence]] = defaultdict(list)
    for row in evidence:
        grouped[row.canonical].append(row)
    required_paths = max(2, math.ceil(path_count * min_path_fraction))
    candidates = []
    for canonical, rows in grouped.items():
        if len(rows) < required_paths:
            continue
        orientation_counts = Counter(row.representative for row in rows)
        representative = min(orientation_counts, key=lambda value: (-orientation_counts[value], value))
        candidates.append(
            FamilyCandidate(
                canonical=canonical,
                period=len(canonical),
                representative=representative,
                path_support=len(rows),
                total_seed_bp=sum(row.copies * row.period for row in rows),
                max_seed_copies=max(row.copies for row in rows),
                mean_seed_density=sum(row.density for row in rows) / len(rows),
                orientations=tuple(sorted(orientation_counts)),
            )
        )
    available = {candidate.canonical: candidate for candidate in candidates}
    for candidate in candidates:
        candidate.composite_of = composite_components(candidate, available)
    return sorted(
        candidates,
        key=lambda row: (-row.path_support, -row.total_seed_bp, row.period, row.canonical),
    )


def allowed_edits(period: int) -> int:
    if period == 1:
        return 0
    return min(2, max(1, round(period * 0.15)))


def transition_cost(previous: str, current: str) -> float:
    if previous == current:
        return 0.0
    if previous == "BG":
        return ENTER_MOTIF_BITS
    if current == "BG":
        return EXIT_MOTIF_BITS
    if previous.split("|", 1)[0] == current.split("|", 1)[0]:
        return PHASE_SWITCH_BITS
    return SWITCH_MOTIF_BITS


def motif_states(families: list[FamilyCandidate]) -> list[tuple[str, str, str]]:
    states = []
    for family in sorted(families, key=lambda row: (row.period, row.canonical)):
        for oriented in equivalent_motifs(family.canonical):
            states.append((f"{family.canonical}|{oriented}", family.canonical, oriented))
    return states


def segment_region(region: Region, families: list[FamilyCandidate], traceback: bool = True):
    sequence = region.sequence
    states = motif_states(families)
    scores: list[dict[str, float]] = [dict() for _ in range(len(sequence) + 1)]
    scores[0]["BG"] = 0.0
    back: dict[tuple[int, str], tuple[int, str, tuple[str, str, str, int]]] = {}

    def update(end: int, state: str, score: float, start: int, previous: str, payload) -> None:
        existing = scores[end].get(state)
        if existing is None or score < existing - 1e-9:
            scores[end][state] = score
            if traceback:
                back[(end, state)] = (start, previous, payload)

    for position in range(len(sequence)):
        if not scores[position]:
            continue
        for previous, previous_score in sorted(scores[position].items()):
            update(
                position + 1,
                "BG",
                previous_score + transition_cost(previous, "BG") + BACKGROUND_BITS_PER_BP,
                position,
                previous,
                ("background", sequence[position : position + 1], "", 0),
            )
            for state, canonical, oriented in states:
                period = len(oriented)
                if position + period > len(sequence):
                    continue
                observed = sequence[position : position + period]
                distance = hamming(observed, oriented)
                if distance > allowed_edits(period):
                    continue
                emission = EXACT_MOTIF_BITS_PER_BP * period + VARIANT_EDIT_BITS * distance
                update(
                    position + period,
                    state,
                    previous_score + transition_cost(previous, state) + emission,
                    position,
                    previous,
                    (canonical, observed, oriented, distance),
                )

    if not scores[-1]:
        raise AssertionError(f"MDL segmentation failed for {region.path}")
    state, score = min(
        scores[-1].items(),
        key=lambda item: (item[1] + (EXIT_MOTIF_BITS if item[0] != "BG" else 0.0), item[0]),
    )
    score += EXIT_MOTIF_BITS if state != "BG" else 0.0
    if not traceback:
        return score, []

    position = len(sequence)
    actions = []
    while position > 0:
        previous_position, previous_state, payload = back[(position, state)]
        label, observed, oriented, distance = payload
        if label == "background":
            action = Action(
                region.path,
                region.start + previous_position,
                region.start + position,
                "background",
                observed,
            )
        else:
            action = Action(
                region.path,
                region.start + previous_position,
                region.start + position,
                "exact_motif" if distance == 0 else "variant_motif",
                observed,
                label,
                oriented,
                distance,
            )
        actions.append(action)
        position = previous_position
        state = previous_state
    actions.reverse()
    collapsed = []
    for action in actions:
        if collapsed and action.kind == "background" and collapsed[-1].kind == "background":
            previous = collapsed[-1]
            collapsed[-1] = Action(
                action.path,
                previous.start,
                action.end,
                "background",
                previous.observed + action.observed,
            )
        else:
            collapsed.append(action)
    if "".join(action.observed for action in collapsed) != sequence:
        raise AssertionError(f"Token reconstruction failed for {region.path}")
    return score, collapsed


def dictionary_cost(regions: list[Region], families: list[FamilyCandidate]) -> float:
    sequence_cost = sum(segment_region(region, families, traceback=False)[0] for region in regions)
    model_cost = sum(FAMILY_FIXED_BITS + FAMILY_BASE_BITS * family.period for family in families)
    return sequence_cost + model_cost


def shorter_family_shadows(candidate: FamilyCandidate, selected: list[FamilyCandidate]) -> tuple[str, ...]:
    shadows = []
    for family in selected:
        if family.period >= candidate.period:
            continue
        matched = False
        for oriented in equivalent_motifs(family.canonical):
            repeated = oriented * (math.ceil((candidate.period + family.period) / family.period) + 1)
            for phase in range(family.period):
                expected = repeated[phase : phase + candidate.period]
                if hamming(candidate.representative, expected) <= allowed_edits(candidate.period):
                    matched = True
                    break
            if matched:
                break
        if matched:
            shadows.append(family.canonical)
    return tuple(sorted(shadows))


def select_dictionary(
    regions: list[Region],
    candidates: list[FamilyCandidate],
    max_families: int,
    max_mdl_candidates: int,
    cost_cache: dict[tuple[str, ...], float] | None = None,
) -> tuple[list[FamilyCandidate], float, float]:
    eligible = [candidate for candidate in candidates if not candidate.composite_of][:max_mdl_candidates]
    cache = cost_cache if cost_cache is not None else {}

    def cached_cost(families: list[FamilyCandidate]) -> float:
        key = tuple(sorted(family.canonical for family in families))
        if key not in cache:
            cache[key] = dictionary_cost(regions, families)
        return cache[key]

    selected: list[FamilyCandidate] = []
    baseline = cached_cost([])
    current = baseline
    while len(selected) < max_families:
        trials = []
        for candidate in eligible:
            if candidate in selected:
                continue
            shadows = shorter_family_shadows(candidate, selected)
            if shadows:
                candidate.shadowed_by = shadows
                continue
            score = cached_cost(selected + [candidate])
            trials.append((current - score, candidate.canonical, candidate, score))
        if not trials:
            break
        gain, _canonical, candidate, score = max(trials, key=lambda item: (item[0], -len(item[1]), item[1]))
        if gain < MIN_MDL_GAIN_BITS:
            break
        selected.append(candidate)
        candidate.selected_step = len(selected)
        candidate.mdl_gain_bits = gain
        current = score
    selected.sort(key=lambda row: (row.period, row.canonical))
    return selected, baseline, current


def family_ids(families: list[FamilyCandidate]) -> dict[str, str]:
    return {family.canonical: f"M{index}" for index, family in enumerate(families, start=1)}


def repeat_blocks(
    actions_by_path: dict[str, list[Action]],
    min_copies: int,
    min_tandem_bp: int,
    min_density: float,
    max_insert_bp: int,
) -> dict[str, list[RepeatBlock]]:
    output: dict[str, list[RepeatBlock]] = {}
    for path, actions in actions_by_path.items():
        blocks = []
        index = 0
        while index < len(actions):
            if actions[index].kind == "background":
                index += 1
                continue
            canonical = actions[index].canonical
            selected_indexes = [index]
            last_motif_index = index
            cursor = index + 1
            while cursor < len(actions):
                action = actions[cursor]
                if action.kind == "background":
                    if len(action.observed) > max_insert_bp:
                        break
                    if cursor + 1 >= len(actions) or actions[cursor + 1].canonical != canonical:
                        break
                    next_end = cursor + 1
                    while (
                        next_end < len(actions)
                        and actions[next_end].kind != "background"
                        and actions[next_end].canonical == canonical
                    ):
                        next_end += 1
                    next_run = actions[cursor + 1 : next_end]
                    next_exact_fraction = sum(
                        item.kind == "exact_motif" for item in next_run
                    ) / max(1, len(next_run))
                    if next_exact_fraction < MIN_BRIDGED_RUN_EXACT_FRACTION:
                        break
                    selected_indexes.append(cursor)
                    cursor += 1
                    continue
                if action.canonical != canonical:
                    break
                selected_indexes.append(cursor)
                last_motif_index = cursor
                cursor += 1
            motif_actions = [actions[position] for position in selected_indexes if actions[position].kind != "background"]
            start = motif_actions[0].start
            end = motif_actions[-1].end
            period = len(canonical)
            copies = len(motif_actions)
            required = minimum_copies(period, min_copies, min_tandem_bp)
            density = copies * period / max(1, end - start)
            if copies >= required and density >= min_density:
                insertion_actions = [
                    actions[position]
                    for position in selected_indexes
                    if actions[position].kind == "background" and position < last_motif_index
                ]
                orientations = Counter(action.oriented_motif for action in motif_actions)
                oriented = min(orientations, key=lambda value: (-orientations[value], value))
                blocks.append(
                    RepeatBlock(
                        path=path,
                        order=len(blocks) + 1,
                        start=start,
                        end=end,
                        canonical=canonical,
                        oriented_motif=oriented,
                        copies=copies,
                        exact_copies=sum(action.kind == "exact_motif" for action in motif_actions),
                        variant_copies=sum(action.kind == "variant_motif" for action in motif_actions),
                        inserted_bp=sum(len(action.observed) for action in insertion_actions),
                        insertion_parts=tuple((action.start, action.observed) for action in insertion_actions),
                        density=density,
                    )
                )
                index = last_motif_index + 1
            else:
                index += 1
        output[path] = blocks
    return output


def phase_anchored_actions(region: Region, families: list[FamilyCandidate]) -> list[Action]:
    periods = {family.period for family in families}
    if len(periods) != 1:
        _score, actions = segment_region(region, families, traceback=True)
        return actions
    period = next(iter(periods))
    actions = []
    previous_family = ""
    consecutive_variants = 0
    position = 0
    while position + period <= len(region.sequence):
        observed = region.sequence[position : position + period]
        calls = []
        for family in families:
            distance, oriented = best_family_call(observed, family.canonical)
            if distance <= allowed_edits(period):
                calls.append((distance, family.canonical != previous_family, family.canonical, oriented))
        if calls:
            distance, _switched, canonical, oriented = min(calls)
            if distance > 0 and canonical == previous_family and consecutive_variants >= MAX_CONSECUTIVE_VARIANT_TOKENS:
                calls = []
        if calls:
            actions.append(
                Action(
                    region.path,
                    region.start + position,
                    region.start + position + period,
                    "exact_motif" if distance == 0 else "variant_motif",
                    observed,
                    canonical,
                    oriented,
                    distance,
                )
            )
            consecutive_variants = consecutive_variants + 1 if distance > 0 and canonical == previous_family else int(distance > 0)
            previous_family = canonical
        else:
            actions.append(
                Action(
                    region.path,
                    region.start + position,
                    region.start + position + period,
                    "background",
                    observed,
                )
            )
            previous_family = ""
            consecutive_variants = 0
        position += period
    if position < len(region.sequence):
        actions.append(
            Action(
                region.path,
                region.start + position,
                region.end,
                "background",
                region.sequence[position:],
            )
        )
    collapsed = []
    for action in actions:
        if collapsed and action.kind == "background" and collapsed[-1].kind == "background":
            previous = collapsed[-1]
            collapsed[-1] = Action(
                action.path,
                previous.start,
                action.end,
                "background",
                previous.observed + action.observed,
            )
        else:
            collapsed.append(action)
    return collapsed


def denoise_phase_actions(
    actions: list[Action], min_copies: int, min_tandem_bp: int
) -> list[Action]:
    cleaned = list(actions)
    index = 0
    while index < len(actions):
        if actions[index].kind == "background":
            index += 1
            continue
        canonical = actions[index].canonical
        end = index + 1
        while end < len(actions) and actions[end].kind != "background" and actions[end].canonical == canonical:
            end += 1
        period = len(canonical)
        required = minimum_copies(period, min_copies, min_tandem_bp)
        best = None
        for left in range(index, end):
            score = 0
            exact = 0
            variants = 0
            for right in range(left, end):
                if actions[right].kind == "exact_motif":
                    score += 2
                    exact += 1
                else:
                    score -= 1
                    variants += 1
                length = right - left + 1
                if length < required:
                    continue
                key = (score, exact, -variants, length, -left)
                if best is None or key > best[0]:
                    best = (key, left, right + 1)
        if best is not None:
            best_length = best[2] - best[1]
            best_exact_fraction = best[0][1] / max(1, best_length)
        else:
            best_exact_fraction = 0.0
        keep_start, keep_end = (
            (best[1], best[2])
            if best is not None and best[0][0] > 0 and best_exact_fraction >= MIN_CORE_EXACT_FRACTION
            else (end, end)
        )
        left_variants = 0
        while keep_start > index and actions[keep_start - 1].kind == "variant_motif":
            if left_variants >= MAX_CONSECUTIVE_VARIANT_TOKENS:
                break
            keep_start -= 1
            left_variants += 1
        for position in range(index, end):
            if keep_start <= position < keep_end:
                continue
            action = actions[position]
            cleaned[position] = Action(
                action.path,
                action.start,
                action.end,
                "background",
                action.observed,
            )
        index = end

    collapsed = []
    for action in cleaned:
        if collapsed and action.kind == "background" and collapsed[-1].kind == "background":
            previous = collapsed[-1]
            collapsed[-1] = Action(
                action.path,
                previous.start,
                action.end,
                "background",
                previous.observed + action.observed,
            )
        else:
            collapsed.append(action)
    return collapsed


def best_family_call(observed: str, canonical: str) -> tuple[int, str]:
    return min((hamming(observed, oriented), oriented) for oriented in equivalent_motifs(canonical))


def align_family_interval(region: Region, start: int, end: int, canonical: str):
    sequence = region.sequence[start - region.start : end - region.start]
    period = len(canonical)
    scores: list[dict[str, tuple[float, int, int, int]]] = [dict() for _ in range(len(sequence) + 1)]
    scores[0]["M"] = (0.0, 0, 0, 0)
    back: dict[tuple[int, str], tuple[int, str, str, str, str, int]] = {}

    def update(position: int, state: str, score, previous: int, previous_state: str, kind: str, observed: str, oriented: str, distance: int):
        current = scores[position].get(state)
        if current is None or score < current:
            scores[position][state] = score
            back[(position, state)] = (previous, previous_state, kind, observed, oriented, distance)

    for position in range(len(sequence)):
        if not scores[position]:
            continue
        for previous_state, score in sorted(scores[position].items()):
            run_cost = 0.0 if previous_state == "BG" else LOCAL_INSERT_RUN_BITS
            update(
                position + 1,
                "BG",
                (score[0] + LOCAL_INSERT_BASE_BITS + run_cost, score[1] + 1, score[2], score[3]),
                position,
                previous_state,
                "background",
                sequence[position : position + 1],
                "",
                0,
            )
            if position + period <= len(sequence):
                observed = sequence[position : position + period]
                distance, oriented = best_family_call(observed, canonical)
                if distance > allowed_edits(period):
                    continue
                update(
                    position + period,
                    "M",
                    (
                        score[0] + VARIANT_EDIT_BITS * distance,
                        score[1],
                        score[2] + distance,
                        score[3] - 1,
                    ),
                    position,
                    previous_state,
                    "exact_motif" if distance == 0 else "variant_motif",
                    observed,
                    oriented,
                    distance,
                )
    if not scores[-1]:
        raise AssertionError(f"Local family alignment failed for {region.path}: {start}-{end}")
    state, final_score = min(scores[-1].items(), key=lambda item: (item[1], item[0]))
    actions = []
    position = len(sequence)
    while position > 0:
        previous, previous_state, kind, observed, oriented, distance = back[(position, state)]
        actions.append(
            Action(
                region.path,
                start + previous,
                start + position,
                kind,
                observed,
                canonical if kind != "background" else "",
                oriented,
                distance,
            )
        )
        position = previous
        state = previous_state
    actions.reverse()
    collapsed = []
    for action in actions:
        if collapsed and action.kind == "background" and collapsed[-1].kind == "background":
            previous = collapsed[-1]
            collapsed[-1] = Action(
                action.path,
                previous.start,
                action.end,
                "background",
                previous.observed + action.observed,
            )
        else:
            collapsed.append(action)
    return final_score, collapsed


def refine_core_blocks(
    repeat_regions: list[Region],
    core_blocks_by_path: dict[str, list[RepeatBlock]],
    min_copies: int,
    min_tandem_bp: int,
    min_density: float,
) -> tuple[dict[str, list[Action]], dict[str, list[RepeatBlock]]]:
    refined_actions: dict[str, list[Action]] = {}
    refined_blocks: dict[str, list[RepeatBlock]] = {}
    for region in repeat_regions:
        cores = core_blocks_by_path[region.path]
        blocks = []
        aligned_by_block: list[list[Action]] = []
        for index, core in enumerate(cores):
            period = len(core.canonical)
            start = region.start if index == 0 else core.start
            end = region.end if index == len(cores) - 1 else core.end
            if index > 0 and core.start - cores[index - 1].end > period:
                candidate_start = core.start - period
                observed = region.sequence[candidate_start - region.start : core.start - region.start]
                if len(observed) == period and best_family_call(observed, core.canonical)[0] <= allowed_edits(period):
                    start = candidate_start
            if start >= end:
                raise AssertionError(f"Invalid core interval for {region.path} R{index + 1}: {start}-{end}")
            _score, actions = align_family_interval(region, start, end, core.canonical)
            motif_actions = [action for action in actions if action.kind != "background"]
            insertion_actions = [action for action in actions if action.kind == "background"]
            copies = len(motif_actions)
            required = minimum_copies(period, min_copies, min_tandem_bp)
            density = copies * period / max(1, end - start)
            if copies < required or density < min_density:
                raise AssertionError(f"Local refinement lost repeat support for {region.path} R{index + 1}")
            orientations = Counter(action.oriented_motif for action in motif_actions)
            oriented = min(orientations, key=lambda value: (-orientations[value], value))
            block = RepeatBlock(
                path=region.path,
                order=index + 1,
                start=actions[0].start,
                end=actions[-1].end,
                canonical=core.canonical,
                oriented_motif=oriented,
                copies=len(motif_actions),
                exact_copies=sum(action.kind == "exact_motif" for action in motif_actions),
                variant_copies=sum(action.kind == "variant_motif" for action in motif_actions),
                inserted_bp=sum(len(action.observed) for action in insertion_actions),
                insertion_parts=tuple((action.start, action.observed) for action in insertion_actions),
                density=density,
            )
            blocks.append(block)
            aligned_by_block.append(actions)

        actions = []
        cursor = region.start
        for block, aligned in zip(blocks, aligned_by_block):
            if cursor < block.start:
                actions.append(
                    Action(
                        region.path,
                        cursor,
                        block.start,
                        "background",
                        region.sequence[cursor - region.start : block.start - region.start],
                    )
                )
            actions.extend(aligned)
            cursor = block.end
        if cursor < region.end:
            actions.append(
                Action(
                    region.path,
                    cursor,
                    region.end,
                    "background",
                    region.sequence[cursor - region.start :],
                )
            )
        if "".join(action.observed for action in actions) != region.sequence:
            raise AssertionError(f"Refined token reconstruction failed for {region.path}")
        refined_actions[region.path] = actions
        refined_blocks[region.path] = blocks
    return refined_actions, refined_blocks


def higher_order_rows(
    actions_by_path: dict[str, list[Action]], family_lookup: dict[str, str]
) -> list[dict[str, object]]:
    rows = []
    for path, actions in actions_by_path.items():
        chains: list[list[Action]] = []
        current = []
        for action in actions:
            if action.kind == "background":
                if current:
                    chains.append(current)
                    current = []
                continue
            current.append(action)
        if current:
            chains.append(current)
        seen: set[tuple[int, int, tuple[str, ...]]] = set()
        for chain in chains:
            labels = [action.canonical for action in chain]
            for start in range(len(chain)):
                for pattern_length in (2, 3):
                    if start + pattern_length * 3 > len(chain):
                        continue
                    pattern = tuple(labels[start : start + pattern_length])
                    if len(set(pattern)) < 2:
                        continue
                    cycles = 1
                    while labels[
                        start + cycles * pattern_length : start + (cycles + 1) * pattern_length
                    ] == list(pattern):
                        cycles += 1
                    if cycles < 3:
                        continue
                    end_index = start + cycles * pattern_length
                    key = (chain[start].start, chain[end_index - 1].end, pattern)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "path": path,
                            "start_bp": chain[start].start,
                            "end_bp": chain[end_index - 1].end,
                            "primitive_family_pattern": ">".join(family_lookup[value] for value in pattern),
                            "canonical_pattern": ">".join(pattern),
                            "pattern_token_length": pattern_length,
                            "cycle_count": cycles,
                            "status": "higher_order_candidate_not_primary_override",
                        }
                    )
    return sorted(rows, key=lambda row: (str(row["path"]), int(row["start_bp"]), int(row["pattern_token_length"])))


def candidate_rows(candidates: list[FamilyCandidate], max_mdl_candidates: int) -> list[dict[str, object]]:
    shortlist = {
        candidate.canonical
        for candidate in [row for row in candidates if not row.composite_of][:max_mdl_candidates]
    }
    return [
        {
            "canonical_family": candidate.canonical,
            "period_bp": candidate.period,
            "representative_orientation": candidate.representative,
            "equivalent_orientations": ";".join(equivalent_motifs(candidate.canonical)),
            "path_support": candidate.path_support,
            "total_seed_bp": candidate.total_seed_bp,
            "max_seed_copies": candidate.max_seed_copies,
            "mean_seed_density": f"{candidate.mean_seed_density:.4f}",
            "composite_of_primitive_families": ">".join(candidate.composite_of) or "-",
            "shadowed_by_selected_shorter_family": ";".join(candidate.shadowed_by) or "-",
            "eligible_primary": int(not candidate.composite_of),
            "MDL_shortlisted": int(candidate.canonical in shortlist),
            "selected": int(candidate.selected_step > 0),
            "selection_step": candidate.selected_step,
            "marginal_MDL_gain_bits": f"{candidate.mdl_gain_bits:.4f}",
        }
        for candidate in candidates
    ]


def dictionary_rows(
    selected: list[FamilyCandidate], ids: dict[str, str], path_count: int
) -> list[dict[str, object]]:
    return [
        {
            "family_id": ids[family.canonical],
            "canonical_family": family.canonical,
            "primitive_period_bp": family.period,
            "representative_orientation": family.representative,
            "equivalent_orientations": ";".join(equivalent_motifs(family.canonical)),
            "path_support": family.path_support,
            "path_count": path_count,
            "path_support_fraction": f"{family.path_support / path_count:.4f}",
            "total_seed_bp": family.total_seed_bp,
            "selection_step": family.selected_step,
            "marginal_MDL_gain_bits": f"{family.mdl_gain_bits:.4f}",
            "selection_model": "cohort_de_novo_kmer_plus_MDL",
        }
        for family in selected
    ]


def token_rows(actions_by_path: dict[str, list[Action]], ids: dict[str, str]) -> list[dict[str, object]]:
    rows = []
    for path, actions in actions_by_path.items():
        for index, action in enumerate(actions, start=1):
            rows.append(
                {
                    "path": path,
                    "token_index": index,
                    "start_bp": action.start,
                    "end_bp": action.end,
                    "length_bp": action.end - action.start,
                    "token_type": action.kind,
                    "family_id": ids.get(action.canonical, "-"),
                    "canonical_family": action.canonical or "-",
                    "oriented_motif": action.oriented_motif or "-",
                    "observed": action.observed,
                    "edit_distance": action.edit_distance,
                    "coordinate_system": COORDINATE_SYSTEM,
                }
            )
    return rows


def block_rows(blocks_by_path: dict[str, list[RepeatBlock]], ids: dict[str, str]) -> list[dict[str, object]]:
    rows = []
    for path, blocks in blocks_by_path.items():
        for block in blocks:
            rows.append(
                {
                    "path": path,
                    "repeat_block": f"R{block.order}",
                    "start_bp": block.start,
                    "end_bp": block.end,
                    "span_bp": block.end - block.start,
                    "family_id": ids[block.canonical],
                    "canonical_family": block.canonical,
                    "oriented_motif": block.oriented_motif,
                    "primitive_period_bp": len(block.canonical),
                    "copies": block.copies,
                    "exact_copies": block.exact_copies,
                    "variant_copies": block.variant_copies,
                    "inserted_bp": block.inserted_bp,
                    "insertions": ";".join(f"{position}:{sequence}" for position, sequence in block.insertion_parts) or "-",
                    "repeat_density": f"{block.density:.4f}",
                    "interpretation_status": "provisional_sequence_only",
                    "coordinate_system": COORDINATE_SYSTEM,
                }
            )
    return rows


def matrix_rows(blocks_by_path: dict[str, list[RepeatBlock]], ids: dict[str, str]) -> list[dict[str, object]]:
    max_blocks = max((len(blocks) for blocks in blocks_by_path.values()), default=0)
    rows = []
    for path, blocks in blocks_by_path.items():
        row: dict[str, object] = {"path": path, "repeat_block_count": len(blocks)}
        for order in range(1, max_blocks + 1):
            if order <= len(blocks):
                block = blocks[order - 1]
                row[f"R{order}_family"] = ids[block.canonical]
                row[f"R{order}_motif"] = block.oriented_motif
                row[f"R{order}_copies"] = block.copies
                row[f"R{order}_variants"] = block.variant_copies
                row[f"R{order}_inserted_bp"] = block.inserted_bp
            else:
                for suffix in ("family", "motif", "copies", "variants", "inserted_bp"):
                    row[f"R{order}_{suffix}"] = "-"
        rows.append(row)
    return rows


def short_hash(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("ascii")).hexdigest()[:10]


def write_gfa(
    path: Path,
    paths: list[PathRecord],
    homology_regions: list[Region],
    repeat_regions: list[Region],
    actions_by_path: dict[str, list[Action]],
    blocks_by_path: dict[str, list[RepeatBlock]],
    ids: dict[str, str],
    mdl_baseline: float,
    mdl_selected: float,
):
    nodes: dict[str, tuple[str, tuple[str, ...]]] = {}
    edges: Counter[tuple[str, str]] = Counter()
    path_steps: dict[str, list[str]] = {}
    homology_by_path = {region.path: region for region in homology_regions}
    repeat_by_path = {region.path: region for region in repeat_regions}

    for record in paths:
        actions = actions_by_path[record.name]
        blocks = blocks_by_path[record.name]
        steps = []
        cursor = 0
        action_by_start = {action.start: action for action in actions}
        for block in blocks:
            if cursor < block.start:
                sequence = record.sequence[cursor : block.start]
                node_id = f"B{block.order - 1}_{short_hash(sequence)}"
                nodes[node_id] = (sequence, ("BT:Z:background_or_flank",))
                steps.append(node_id)
            position = block.start
            while position < block.end:
                action = action_by_start.get(position)
                if action is None or action.end > block.end:
                    raise AssertionError(f"Missing P2 token coverage for {record.name} at {position}")
                if action.kind == "background":
                    node_id = f"R{block.order}_INS_{short_hash(action.observed)}"
                    tags = (f"LC:Z:R{block.order}", "BT:Z:explicit_insertion_or_complex")
                else:
                    family_id = ids[action.canonical]
                    token_kind = "exact" if action.kind == "exact_motif" else "variant"
                    node_id = f"R{block.order}_{family_id}_{token_kind}_{action.observed}"
                    tags = (
                        f"LC:Z:R{block.order}",
                        f"MF:Z:{family_id}",
                        f"CM:Z:{action.canonical}",
                        f"OM:Z:{action.oriented_motif}",
                        f"BT:Z:{action.kind}",
                        f"ED:i:{action.edit_distance}",
                    )
                nodes[node_id] = (action.observed, tags)
                steps.append(node_id)
                position = action.end
            cursor = block.end
        if cursor < len(record.sequence):
            sequence = record.sequence[cursor:]
            node_id = f"B{len(blocks)}_{short_hash(sequence)}"
            nodes[node_id] = (sequence, ("BT:Z:background_or_flank",))
            steps.append(node_id)
        path_steps[record.name] = steps
        edges.update(zip(steps, steps[1:]))

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "H\tVN:Z:1.0\tTS:Z:p2_de_novo_primitive_motif_mdl"
            f"\tMB:f:{mdl_baseline:.4f}\tMS:f:{mdl_selected:.4f}\tCS:Z:{COORDINATE_SYSTEM}\n"
        )
        for node_id in sorted(nodes):
            sequence, tags = nodes[node_id]
            handle.write(f"S\t{node_id}\t{sequence}\t" + "\t".join(tags) + "\n")
        for (left, right), support in sorted(edges.items()):
            loop_tag = "\tRL:i:1" if left == right and left.startswith("R") else ""
            handle.write(f"L\t{left}\t+\t{right}\t+\t0M\tPS:i:{support}{loop_tag}\n")
        for record in paths:
            steps = path_steps[record.name]
            overlaps = ",".join("0M" for _ in steps[1:]) if len(steps) > 1 else "*"
            homology = homology_by_path[record.name]
            repeat = repeat_by_path[record.name]
            tags = [
                f"WS:i:{homology.start}",
                f"WE:i:{homology.end}",
                f"RS:i:{repeat.start}",
                f"RE:i:{repeat.end}",
                f"RB:i:{len(blocks_by_path[record.name])}",
                "VS:Z:provisional_sequence_only",
            ]
            for block in blocks_by_path[record.name]:
                tags.append(
                    f"R{block.order}:Z:family={ids[block.canonical]};motif={block.oriented_motif};"
                    f"copies={block.copies};variants={block.variant_copies};inserted_bp={block.inserted_bp}"
                )
            handle.write(
                f"P\t{record.name}\t{','.join(step + '+' for step in steps)}\t{overlaps}\t"
                + "\t".join(tags)
                + "\n"
            )
    return nodes, edges, path_steps


def validate_graph(paths: list[PathRecord], nodes, path_steps) -> int:
    reconstructed = 0
    by_name = {path.name: path.sequence for path in paths}
    for path, steps in path_steps.items():
        sequence = "".join(nodes[node][0] for node in steps)
        if sequence != by_name[path]:
            raise AssertionError(f"P2 GFA reconstruction failed for {path}")
        reconstructed += 1
    return reconstructed


def plot_overview(
    path: Path,
    regions: list[Region],
    selected: list[FamilyCandidate],
    ids: dict[str, str],
    blocks_by_path: dict[str, list[RepeatBlock]],
) -> None:
    colors = ["#2e74b5", "#4d9b68", "#c56a2d", "#7d5ba6", "#b64d6b", "#548a8a"]
    color_by_family = {family.canonical: colors[index % len(colors)] for index, family in enumerate(selected)}
    fig = plt.figure(figsize=(14, max(9.0, 0.24 * len(regions) + 4.4)))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.5, 8.5), hspace=0.24)
    top = fig.add_subplot(grid[0])
    bottom = fig.add_subplot(grid[1])

    labels = [f"{ids[family.canonical]}  {family.canonical} ({family.period} bp)" for family in selected]
    supports = [family.path_support for family in selected]
    top.barh(range(len(selected)), supports, color=[color_by_family[family.canonical] for family in selected])
    top.set_yticks(range(len(selected)))
    top.set_yticklabels(labels, fontsize=8)
    top.set_xlim(0, max(1, len(regions)))
    top.set_xlabel("Paths supporting de novo seed")
    top.grid(axis="x", color="#e5e7eb", linewidth=0.7)
    top.spines[["top", "right"]].set_visible(False)

    for index, region in enumerate(regions):
        y = len(regions) - index - 1
        bottom.hlines(y, region.start, region.end, color="#d7dde5", linewidth=7, zorder=1)
        for block in blocks_by_path[region.path]:
            bottom.add_patch(
                Rectangle(
                    (block.start, y - 0.30),
                    block.end - block.start,
                    0.60,
                    facecolor=color_by_family[block.canonical],
                    edgecolor="#263746",
                    linewidth=0.5,
                    hatch="//" if block.variant_copies or block.inserted_bp else None,
                    zorder=2,
                )
            )
    bottom.set_yticks(range(len(regions)))
    bottom.set_yticklabels([region.path for region in reversed(regions)], fontsize=6.5)
    bottom.set_xlabel("Path coordinate (bp; 0-based half-open)")
    bottom.set_title("Position-specific repeat blocks from the de novo motif dictionary", fontsize=12)
    bottom.grid(axis="x", color="#e5e7eb", linewidth=0.7)
    bottom.spines[["top", "right"]].set_visible(False)
    handles = [Patch(facecolor="#d7dde5", label="P1 homology window")]
    handles.extend(
        Patch(facecolor=color_by_family[family.canonical], label=f"{ids[family.canonical]} {family.canonical}")
        for family in selected
    )
    handles.append(Patch(facecolor="white", edgecolor="#263746", hatch="//", label="variant or insertion"))
    bottom.legend(handles=handles, loc="upper right", fontsize=8, ncol=min(4, len(handles)))
    fig.suptitle("P2 de novo primitive motif discovery and MDL decomposition", fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.06)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="P2 de novo primitive motif discovery and MDL path decomposition.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--p1-boundaries", type=Path, default=DEFAULT_BOUNDARIES)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--max-motif-bp", type=int, default=DEFAULT_MAX_MOTIF_BP)
    parser.add_argument("--min-copies", type=int, default=DEFAULT_MIN_COPIES)
    parser.add_argument("--min-tandem-bp", type=int, default=DEFAULT_MIN_TANDEM_BP)
    parser.add_argument("--min-seed-density", type=float, default=DEFAULT_MIN_SEED_DENSITY)
    parser.add_argument("--min-path-fraction", type=float, default=DEFAULT_MIN_PATH_FRACTION)
    parser.add_argument("--min-block-density", type=float, default=DEFAULT_MIN_BLOCK_DENSITY)
    parser.add_argument("--max-local-insert-bp", type=int, default=DEFAULT_MAX_LOCAL_INSERT_BP)
    parser.add_argument("--max-families", type=int, default=DEFAULT_MAX_FAMILIES)
    parser.add_argument("--max-mdl-candidates", type=int, default=DEFAULT_MAX_MDL_CANDIDATES)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    paths = read_gfa_paths(args.input)
    homology_regions, repeat_regions = load_regions(paths, args.p1_boundaries)
    evidence = discover_seed_evidence(
        homology_regions,
        args.max_motif_bp,
        args.min_copies,
        args.min_tandem_bp,
        args.min_seed_density,
    )
    candidates = aggregate_candidates(evidence, len(paths), args.min_path_fraction)
    cost_cache: dict[tuple[str, ...], float] = {}
    selected, baseline_cost, selected_cost = select_dictionary(
        repeat_regions,
        candidates,
        args.max_families,
        args.max_mdl_candidates,
        cost_cache,
    )
    if not selected:
        raise AssertionError("P2 did not select any de novo motif family")
    ids = family_ids(selected)

    strict_actions_by_path = {}
    for region in repeat_regions:
        anchored = phase_anchored_actions(region, selected)
        strict_actions_by_path[region.path] = denoise_phase_actions(
            anchored, args.min_copies, args.min_tandem_bp
        )
    core_blocks_by_path = repeat_blocks(
        strict_actions_by_path,
        args.min_copies,
        args.min_tandem_bp,
        args.min_block_density,
        args.max_local_insert_bp,
    )
    actions_by_path, blocks_by_path = refine_core_blocks(
        repeat_regions,
        core_blocks_by_path,
        args.min_copies,
        args.min_tandem_bp,
        args.min_block_density,
    )
    high_order = higher_order_rows(actions_by_path, ids)

    reversed_candidates = aggregate_candidates(
        discover_seed_evidence(
            list(reversed(homology_regions)),
            args.max_motif_bp,
            args.min_copies,
            args.min_tandem_bp,
            args.min_seed_density,
        ),
        len(paths),
        args.min_path_fraction,
    )
    reversed_selected, _reversed_baseline, _reversed_cost = select_dictionary(
        list(reversed(repeat_regions)),
        reversed_candidates,
        args.max_families,
        args.max_mdl_candidates,
        cost_cache,
    )
    order_stable = [family.canonical for family in selected] == [family.canonical for family in reversed_selected]
    if not order_stable:
        raise AssertionError("P2 motif dictionary changed when path order was reversed")

    candidate_path = args.outdir / f"{args.prefix}_de_novo_motif候选.tsv"
    dictionary_path = args.outdir / f"{args.prefix}_共享motif字典.tsv"
    token_path = args.outdir / f"{args.prefix}_逐路径motif_token.tsv"
    block_path = args.outdir / f"{args.prefix}_逐路径repeat区块.tsv"
    higher_order_path = args.outdir / f"{args.prefix}_高阶重复候选.tsv"
    matrix_path = args.outdir / f"{args.prefix}_逐路径motif矩阵.tsv"
    gfa_path = args.outdir / f"{args.prefix}_P2_de_novo_motif图.gfa"
    png_path = args.outdir / f"{args.prefix}_P2_motif与区块总览.png"
    qa_path = args.outdir / f"{args.prefix}_P2验证汇总.tsv"

    candidate_output = candidate_rows(candidates, args.max_mdl_candidates)
    dictionary_output = dictionary_rows(selected, ids, len(paths))
    tokens_output = token_rows(actions_by_path, ids)
    blocks_output = block_rows(blocks_by_path, ids)
    matrix_output = matrix_rows(blocks_by_path, ids)
    write_tsv(candidate_path, candidate_output, list(candidate_output[0]))
    write_tsv(dictionary_path, dictionary_output, list(dictionary_output[0]))
    write_tsv(token_path, tokens_output, list(tokens_output[0]))
    write_tsv(
        block_path,
        blocks_output,
        list(blocks_output[0]) if blocks_output else ["path", "repeat_block"],
    )
    write_tsv(
        higher_order_path,
        high_order,
        list(high_order[0])
        if high_order
        else [
            "path",
            "start_bp",
            "end_bp",
            "primitive_family_pattern",
            "canonical_pattern",
            "pattern_token_length",
            "cycle_count",
            "status",
        ],
    )
    write_tsv(matrix_path, matrix_output, list(matrix_output[0]))
    nodes, edges, path_steps = write_gfa(
        gfa_path,
        paths,
        homology_regions,
        repeat_regions,
        actions_by_path,
        blocks_by_path,
        ids,
        baseline_cost,
        selected_cost,
    )
    reconstructed = validate_graph(paths, nodes, path_steps)
    plot_overview(png_path, homology_regions, selected, ids, blocks_by_path)

    block_counts = Counter(len(blocks) for blocks in blocks_by_path.values())
    modal_block_count = min(block_counts, key=lambda value: (-block_counts[value], value))
    self_loop_positions = sorted(
        {
            left.split("_", 1)[0]
            for left, right in edges
            if left == right and left.startswith("R")
        }
    )
    qa = {
        "path_count": len(paths),
        "homology_window_paths": len(homology_regions),
        "de_novo_candidate_family_count": len(candidates),
        "selected_primitive_family_count": len(selected),
        "selected_primitive_families": ",".join(family.canonical for family in selected),
        "selected_periods_bp": ",".join(str(family.period) for family in selected),
        "selected_composite_family_count": sum(bool(family.composite_of) for family in selected),
        "MDL_background_bits": f"{baseline_cost:.4f}",
        "MDL_selected_bits": f"{selected_cost:.4f}",
        "MDL_gain_bits": f"{baseline_cost - selected_cost:.4f}",
        "path_order_stable_dictionary": int(order_stable),
        "repeat_block_count": sum(len(blocks) for blocks in blocks_by_path.values()),
        "modal_repeat_block_count_per_path": modal_block_count,
        "paths_with_modal_repeat_block_count": block_counts[modal_block_count],
        "higher_order_candidate_count": len(high_order),
        "sequence_only_provisional_paths": len(paths),
        "gfa_node_count": len(nodes),
        "gfa_edge_count": len(edges),
        "explicit_loop_positions": ",".join(self_loop_positions),
        "exact_gfa_reconstruction_paths": reconstructed,
    }
    write_tsv(qa_path, [{"metric": key, "value": value} for key, value in qa.items()], ["metric", "value"])

    print(f"Paths: {len(paths)}")
    print(
        "Selected primitive motif families: "
        + ", ".join(
            f"{ids[family.canonical]}={family.canonical} period={family.period} support={family.path_support}/{len(paths)}"
            for family in selected
        )
    )
    print(f"MDL: {baseline_cost:.2f} -> {selected_cost:.2f} bits; gain={baseline_cost - selected_cost:.2f}")
    print(
        f"Repeat blocks: {qa['repeat_block_count']}; modal/path={modal_block_count} "
        f"for {block_counts[modal_block_count]}/{len(paths)} paths"
    )
    print(f"Higher-order candidates: {len(high_order)}")
    print(f"GFA nodes={len(nodes)} edges={len(edges)} loops={','.join(self_loop_positions) or '-'}")
    print(f"Exact GFA reconstruction: {reconstructed}/{len(paths)}")
    for output in (
        candidate_path,
        dictionary_path,
        token_path,
        block_path,
        higher_order_path,
        matrix_path,
        gfa_path,
        png_path,
        qa_path,
    ):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
