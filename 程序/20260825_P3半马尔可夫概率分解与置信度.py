#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


P2_SCRIPT = Path(__file__).with_name("20260825_P2从头发现主体motif与MDL分解.py")
DEFAULT_INPUT = Path("测试数据") / "IRF2BPL.HPRC.fa.3034e58.11fba48.904d69e.smooth.final.gfa"
DEFAULT_P1 = (
    Path("outputs")
    / "IRF2BPL"
    / "20260825_P1唯一侧翼锚点与共识边界"
    / "20260825_IRF2BPL_P1唯一侧翼锚点与共识边界_路径边界共识.tsv"
)
DEFAULT_P2_DIR = Path("outputs") / "IRF2BPL" / "20260825_P2从头发现主体motif与MDL分解"
DEFAULT_P2_PREFIX = "20260825_IRF2BPL_P2从头发现主体motif与MDL分解"
DEFAULT_OUTDIR = Path("outputs") / "IRF2BPL" / "20260825_P3半马尔可夫概率分解与置信度"
DEFAULT_PREFIX = "20260825_IRF2BPL_P3半马尔可夫概率分解与置信度"

COORDINATE_SYSTEM = "0-based_half-open"
SPLIT_SEED = "vntrmap-p3-v1"
LOG_QUARTER = math.log(0.25)
NEG_INF = float("-inf")
MIN_PROBABILITY = 1e-12


@dataclass(frozen=True)
class Family:
    family_id: str
    canonical: str
    period: int
    representative: str


@dataclass(frozen=True)
class Observation:
    path: str
    index: int
    start: int
    end: int
    kind: str
    observed: str
    canonical: str
    oriented_motif: str
    edit_distance: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class BaselineBlock:
    path: str
    order: int
    start: int
    end: int
    canonical: str
    copies: int


@dataclass
class ProbabilityModel:
    exact_probability: dict[str, float]
    variant_probability: dict[str, float]
    duration_mean: dict[str, float]
    duration_sd: dict[str, float]
    insertion_run_mean: dict[str, float]
    complex_run_mean: float
    transition_probability: dict[tuple[str, str], float]
    training_paths: tuple[str, ...]


@dataclass(frozen=True)
class Segment:
    start_index: int
    end_index: int
    state: str
    start_bp: int
    end_bp: int
    motif_copies: int
    exact_copies: int
    variant_copies: int
    inserted_bp: int
    insertion_parts: tuple[tuple[int, str], ...]
    oriented_motif: str
    density: float
    log_emission: float
    log_duration: float

    @property
    def log_score(self) -> float:
        return self.log_emission + self.log_duration


@dataclass
class BlockCall:
    path: str
    order: int
    segment: Segment
    segment_posterior: float
    mean_token_posterior: float
    min_token_posterior: float
    start_ci_low_bp: int
    start_ci_high_bp: int
    end_ci_low_bp: int
    end_ci_high_bp: int


@dataclass
class DecodedPath:
    path: str
    split: str
    best_segments: tuple[Segment, ...]
    second_segments: tuple[Segment, ...]
    best_log_score: float
    second_log_score: float
    log_evidence: float
    segment_posterior: dict[Segment, float]
    token_posterior: list[dict[str, float]]
    blocks: list[BlockCall]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def logsumexp(values) -> float:
    values = [value for value in values if value != NEG_INF]
    if not values:
        return NEG_INF
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def safe_log(value: float) -> float:
    return math.log(max(MIN_PROBABILITY, value))


def weighted_quantile(values: list[tuple[int, float]], quantile: float) -> int:
    totals: defaultdict[int, float] = defaultdict(float)
    for value, weight in values:
        totals[value] += max(0.0, weight)
    if not totals:
        raise ValueError("Cannot calculate a weighted quantile without values")
    total = sum(totals.values())
    if total <= 0:
        return sorted(totals)[0]
    target = quantile * total
    running = 0.0
    for value in sorted(totals):
        running += totals[value]
        if running >= target:
            return value
    return max(totals)


def load_families(path: Path) -> list[Family]:
    families = [
        Family(
            family_id=row["family_id"],
            canonical=row["canonical_family"],
            period=int(row["primitive_period_bp"]),
            representative=row["representative_orientation"],
        )
        for row in read_tsv(path)
    ]
    if not families:
        raise AssertionError("P3 requires at least one P2 motif family")
    if len({family.canonical for family in families}) != len(families):
        raise AssertionError("P2 motif dictionary contains duplicate canonical families")
    return families


def load_observations(path: Path) -> dict[str, list[Observation]]:
    output: defaultdict[str, list[Observation]] = defaultdict(list)
    for row in read_tsv(path):
        output[row["path"]].append(
            Observation(
                path=row["path"],
                index=int(row["token_index"]),
                start=int(row["start_bp"]),
                end=int(row["end_bp"]),
                kind=row["token_type"],
                observed=row["observed"],
                canonical="" if row["canonical_family"] == "-" else row["canonical_family"],
                oriented_motif="" if row["oriented_motif"] == "-" else row["oriented_motif"],
                edit_distance=int(row["edit_distance"]),
            )
        )
    for observations in output.values():
        observations.sort(key=lambda row: row.index)
    return dict(output)


def refine_short_motif_like_background(
    p2,
    observations_by_path: dict[str, list[Observation]],
    families: list[Family],
    max_refine_bp: int,
) -> dict[str, list[Observation]]:
    periods = {family.period for family in families}
    if len(periods) != 1:
        return observations_by_path
    period = next(iter(periods))
    output = {}
    for path, observations in observations_by_path.items():
        refined = []
        for observation in observations:
            if (
                observation.kind != "background"
                or observation.length > max_refine_bp
                or observation.length < period
                or observation.length % period
            ):
                refined.append(observation)
                continue
            calls = []
            for offset in range(0, observation.length, period):
                observed = observation.observed[offset : offset + period]
                exact = []
                for family in families:
                    distance, oriented = p2.best_family_call(observed, family.canonical)
                    if distance == 0:
                        exact.append((family.canonical, oriented))
                if not exact:
                    calls = []
                    break
                canonical, oriented = min(exact)
                calls.append((offset, observed, canonical, oriented))
            if not calls:
                refined.append(observation)
                continue
            for offset, observed, canonical, oriented in calls:
                refined.append(
                    Observation(
                        path=path,
                        index=0,
                        start=observation.start + offset,
                        end=observation.start + offset + period,
                        kind="exact_motif",
                        observed=observed,
                        canonical=canonical,
                        oriented_motif=oriented,
                        edit_distance=0,
                    )
                )
        output[path] = [
            Observation(
                path=row.path,
                index=index,
                start=row.start,
                end=row.end,
                kind=row.kind,
                observed=row.observed,
                canonical=row.canonical,
                oriented_motif=row.oriented_motif,
                edit_distance=row.edit_distance,
            )
            for index, row in enumerate(refined, start=1)
        ]
    return output


def load_baseline_blocks(path: Path) -> dict[str, list[BaselineBlock]]:
    output: defaultdict[str, list[BaselineBlock]] = defaultdict(list)
    for row in read_tsv(path):
        output[row["path"]].append(
            BaselineBlock(
                path=row["path"],
                order=int(row["repeat_block"].removeprefix("R")),
                start=int(row["start_bp"]),
                end=int(row["end_bp"]),
                canonical=row["canonical_family"],
                copies=int(row["copies"]),
            )
        )
    for blocks in output.values():
        blocks.sort(key=lambda row: row.order)
    return dict(output)


def validate_inputs(paths, repeat_regions, observations_by_path, blocks_by_path, families) -> None:
    path_names = [path.name for path in paths]
    region_by_path = {region.path: region for region in repeat_regions}
    family_names = {family.canonical for family in families}
    if set(path_names) != set(observations_by_path):
        missing = sorted(set(path_names) - set(observations_by_path))
        extra = sorted(set(observations_by_path) - set(path_names))
        raise AssertionError(f"P2 token/path mismatch; missing={missing}, extra={extra}")
    if set(path_names) != set(blocks_by_path):
        raise AssertionError("P2 block table does not cover every GFA path")
    for name in path_names:
        region = region_by_path[name]
        observations = observations_by_path[name]
        if not observations:
            raise AssertionError(f"No P2 tokens for {name}")
        if observations[0].start != region.start or observations[-1].end != region.end:
            raise AssertionError(f"P2 token boundaries do not match P1 consensus for {name}")
        for left, right in zip(observations, observations[1:]):
            if left.end != right.start:
                raise AssertionError(f"P2 token coverage gap for {name}: {left.end}-{right.start}")
        if "".join(row.observed for row in observations) != region.sequence:
            raise AssertionError(f"P2 token reconstruction failed before P3 for {name}")
        unknown = {row.canonical for row in observations if row.canonical} - family_names
        if unknown:
            raise AssertionError(f"Unknown P2 motif families for {name}: {sorted(unknown)}")


def split_paths(
    path_names: list[str], weights: dict[str, float] | None = None
) -> tuple[dict[str, str], dict[str, str]]:
    keyed = sorted(
        (hashlib.sha256(f"{SPLIT_SEED}|{name}".encode("utf-8")).hexdigest(), name)
        for name in path_names
    )
    count = len(keyed)
    if count < 3:
        return ({name: "train" for _digest, name in keyed}, {name: digest for digest, name in keyed})
    hashes = {name: digest for digest, name in keyed}
    train_count = max(1, int(round(count * 0.60)))
    tune_count = max(1, int(round(count * 0.20)))
    if train_count + tune_count >= count:
        train_count = max(1, count - 2)
        tune_count = 1
    target_counts = {"train": train_count, "tune": tune_count, "test": count - train_count - tune_count}
    if weights is not None and len({round(weights[name], 12) for name in path_names}) > 1:
        target_fractions = {"train": 0.60, "tune": 0.20, "test": 0.20}
        total_weight = sum(weights[name] for name in path_names)
        targets = {role: total_weight * fraction for role, fraction in target_fractions.items()}
        assigned = {role: 0.0 for role in target_fractions}
        assigned_counts = {role: 0 for role in target_fractions}
        roles = {}
        ranked = sorted(path_names, key=lambda name: (-weights[name], hashes[name]))
        role_order = ("train", "tune", "test")
        for name in ranked:
            available = [role for role in role_order if assigned_counts[role] < target_counts[role]]
            role = max(available, key=lambda candidate: targets[candidate] - assigned[candidate])
            roles[name] = role
            assigned[role] += weights[name]
            assigned_counts[role] += 1
        return roles, hashes
    roles = {}
    for index, (digest, name) in enumerate(keyed):
        roles[name] = "train" if index < train_count else "tune" if index < train_count + tune_count else "test"
    return roles, hashes


def path_weights(path_names: list[str], mode: str) -> dict[str, float]:
    if mode == "equal":
        return {name: 1.0 for name in path_names}
    output = {}
    for name in path_names:
        try:
            weight = float(name.rsplit("_", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"Path does not contain a numeric suffix weight: {name}") from error
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Path weight must be positive and finite: {name}={weight}")
        output[name] = weight
    return output


def weighted_mean_and_sd(values: list[tuple[float, float]], default: float) -> tuple[float, float]:
    if not values:
        return default, 0.0
    total_weight = sum(weight for _value, weight in values)
    mean = sum(value * weight for value, weight in values) / total_weight
    variance = sum(weight * (value - mean) ** 2 for value, weight in values) / total_weight
    return mean, math.sqrt(max(0.0, variance))


def blocks_to_state_spans(observations: list[Observation], blocks: list[BaselineBlock]):
    spans = []
    cursor = observations[0].start
    for block in blocks:
        if cursor < block.start:
            spans.append(("C", cursor, block.start))
        spans.append((block.canonical, block.start, block.end))
        cursor = block.end
    if cursor < observations[-1].end:
        spans.append(("C", cursor, observations[-1].end))
    return spans


def estimate_model(
    families: list[Family],
    observations_by_path: dict[str, list[Observation]],
    blocks_by_path: dict[str, list[BaselineBlock]],
    roles: dict[str, str],
    weights: dict[str, float],
) -> ProbabilityModel:
    family_names = [family.canonical for family in families]
    exact = Counter()
    variant = Counter()
    durations: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
    insertions: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
    complex_lengths: list[tuple[float, float]] = []
    transitions = Counter()
    outgoing = Counter()
    training_paths = tuple(sorted(path for path, role in roles.items() if role == "train"))

    for path in training_paths:
        path_weight = weights[path]
        observations = observations_by_path[path]
        blocks = blocks_by_path[path]
        spans = blocks_to_state_spans(observations, blocks)
        states = [state for state, _start, _end in spans]
        previous = "START"
        for state in states:
            if state == previous:
                continue
            transitions[(previous, state)] += path_weight
            outgoing[previous] += path_weight
            previous = state
        transitions[(previous, "END")] += path_weight
        outgoing[previous] += path_weight

        for state, start, end in spans:
            if state == "C":
                complex_lengths.append((end - start, path_weight))
                continue
            block = next(block for block in blocks if block.canonical == state and block.start == start and block.end == end)
            durations[state].append((block.copies, path_weight))
            for observation in observations:
                if not (start <= observation.start and observation.end <= end):
                    continue
                if observation.canonical == state:
                    if observation.kind == "exact_motif":
                        exact[state] += path_weight
                    else:
                        variant[state] += path_weight
                elif observation.kind == "background":
                    insertions[state].append((observation.length, path_weight))

    exact_probability = {}
    variant_probability = {}
    duration_mean = {}
    duration_sd = {}
    insertion_run_mean = {}
    for family in families:
        canonical = family.canonical
        total = exact[canonical] + variant[canonical]
        exact_probability[canonical] = (exact[canonical] + 2.0) / (total + 4.0)
        variant_probability[canonical] = (variant[canonical] + 2.0) / (total + 4.0)
        learned_mean, observed_sd = weighted_mean_and_sd(durations[canonical], 4.0)
        duration_mean[canonical] = learned_mean
        duration_sd[canonical] = max(2.0, observed_sd, math.sqrt(duration_mean[canonical]) * 0.25)
        learned_insert_mean, _insert_sd = weighted_mean_and_sd(insertions[canonical], 4.0)
        insertion_run_mean[canonical] = max(2.0, learned_insert_mean)

    states = ["C", *family_names]
    transition_probability = {}
    pseudocount = 0.5
    for previous in ["START", *states]:
        allowed = [state for state in [*states, "END"] if state != previous and not (previous == "START" and state == "END")]
        denominator = outgoing[previous] + pseudocount * len(allowed)
        for state in allowed:
            transition_probability[(previous, state)] = (transitions[(previous, state)] + pseudocount) / denominator

    return ProbabilityModel(
        exact_probability=exact_probability,
        variant_probability=variant_probability,
        duration_mean=duration_mean,
        duration_sd=duration_sd,
        insertion_run_mean=insertion_run_mean,
        complex_run_mean=max(6.0, weighted_mean_and_sd(complex_lengths, 30.0)[0]),
        transition_probability=transition_probability,
        training_paths=training_paths,
    )


def gaussian_duration_log_probability(value: int, mean: float, sd: float) -> float:
    z = (value - mean) / sd
    return -0.5 * z * z - math.log(sd * math.sqrt(2.0 * math.pi))


def geometric_duration_log_probability(value: int, mean: float) -> float:
    probability = min(0.5, max(1e-4, 1.0 / max(1.0, mean)))
    return safe_log(probability) + max(0, value - 1) * safe_log(1.0 - probability)


def transition_log_probability(model: ProbabilityModel, previous: str, current: str) -> float:
    if previous == current:
        return NEG_INF
    return safe_log(model.transition_probability.get((previous, current), MIN_PROBABILITY))


def build_segments(
    observations: list[Observation],
    families: list[Family],
    model: ProbabilityModel,
    duration_weight: float,
    min_copies: int,
    min_density: float,
    max_insert_bp: int,
    max_total_insert_bp: int,
) -> list[list[Segment]]:
    by_start: list[list[Segment]] = [[] for _ in range(len(observations))]

    for start_index in range(len(observations)):
        total_bp = 0
        for end_index in range(start_index + 1, len(observations) + 1):
            total_bp += observations[end_index - 1].length
            by_start[start_index].append(
                Segment(
                    start_index=start_index,
                    end_index=end_index,
                    state="C",
                    start_bp=observations[start_index].start,
                    end_bp=observations[end_index - 1].end,
                    motif_copies=0,
                    exact_copies=0,
                    variant_copies=0,
                    inserted_bp=total_bp,
                    insertion_parts=(),
                    oriented_motif="",
                    density=0.0,
                    log_emission=total_bp * LOG_QUARTER,
                    log_duration=duration_weight
                    * geometric_duration_log_probability(total_bp, model.complex_run_mean),
                )
            )

    for family in families:
        canonical = family.canonical
        orientations = Counter()
        for start_index, first in enumerate(observations):
            if first.canonical != canonical:
                continue
            copies = 0
            exact_copies = 0
            variant_copies = 0
            inserted_bp = 0
            insertion_parts = []
            log_emission = 0.0
            orientations.clear()
            for end_index in range(start_index, len(observations)):
                observation = observations[end_index]
                if observation.canonical == canonical:
                    copies += 1
                    orientations[observation.oriented_motif] += 1
                    if observation.kind == "exact_motif":
                        exact_copies += 1
                        orientation_count = max(1, family.period * 2)
                        log_emission += safe_log(model.exact_probability[canonical]) - math.log(orientation_count)
                    else:
                        variant_copies += 1
                        alternatives = max(1, family.period * 3)
                        log_emission += safe_log(model.variant_probability[canonical]) - math.log(alternatives)
                else:
                    if observation.length > max_insert_bp:
                        break
                    inserted_bp += observation.length
                    if inserted_bp > max_total_insert_bp:
                        break
                    insertion_parts.append((observation.start, observation.observed))
                    log_emission += observation.length * LOG_QUARTER
                    log_emission += geometric_duration_log_probability(
                        observation.length, model.insertion_run_mean[canonical]
                    )
                if copies < min_copies or observation.canonical != canonical:
                    continue
                start_bp = first.start
                end_bp = observation.end
                density = copies * family.period / max(1, end_bp - start_bp)
                if density < min_density:
                    continue
                oriented = min(orientations, key=lambda value: (-orientations[value], value))
                by_start[start_index].append(
                    Segment(
                        start_index=start_index,
                        end_index=end_index + 1,
                        state=canonical,
                        start_bp=start_bp,
                        end_bp=end_bp,
                        motif_copies=copies,
                        exact_copies=exact_copies,
                        variant_copies=variant_copies,
                        inserted_bp=inserted_bp,
                        insertion_parts=tuple(insertion_parts),
                        oriented_motif=oriented,
                        density=density,
                        log_emission=log_emission,
                        log_duration=duration_weight
                        * gaussian_duration_log_probability(
                            copies, model.duration_mean[canonical], model.duration_sd[canonical]
                        ),
                    )
                )
    for segments in by_start:
        segments.sort(key=lambda row: (row.end_index, row.state, -row.motif_copies, row.start_bp))
    return by_start


def k_best_decode(segments_by_start: list[list[Segment]], model: ProbabilityModel):
    count = len(segments_by_start)
    dp: list[dict[str, list[tuple[float, tuple[Segment, ...]]]]] = [defaultdict(list) for _ in range(count + 1)]
    dp[0]["START"] = [(0.0, ())]

    def add_candidate(end: int, state: str, score: float, segments: tuple[Segment, ...]) -> None:
        candidates = dp[end][state]
        signature = tuple((row.start_index, row.end_index, row.state) for row in segments)
        if any(tuple((row.start_index, row.end_index, row.state) for row in existing) == signature for _score, existing in candidates):
            return
        candidates.append((score, segments))
        candidates.sort(key=lambda row: (-row[0], tuple((item.state, item.start_index, item.end_index) for item in row[1])))
        del candidates[2:]

    for start in range(count):
        if not dp[start]:
            continue
        for segment in segments_by_start[start]:
            for previous, candidates in sorted(dp[start].items()):
                transition = transition_log_probability(model, previous, segment.state)
                if transition == NEG_INF:
                    continue
                for previous_score, previous_segments in candidates:
                    add_candidate(
                        segment.end_index,
                        segment.state,
                        previous_score + transition + segment.log_score,
                        previous_segments + (segment,),
                    )

    finals = []
    for previous, candidates in dp[count].items():
        end_score = transition_log_probability(model, previous, "END")
        for score, segments in candidates:
            finals.append((score + end_score, segments))
    finals.sort(key=lambda row: (-row[0], tuple((item.state, item.start_index, item.end_index) for item in row[1])))
    if not finals:
        raise AssertionError("P3 semi-Markov decoding produced no complete path")
    if len(finals) == 1:
        finals.append((NEG_INF, ()))
    return finals[0], finals[1]


def forward_backward(segments_by_start: list[list[Segment]], model: ProbabilityModel):
    count = len(segments_by_start)
    states = sorted({segment.state for segments in segments_by_start for segment in segments})
    alpha: list[dict[str, float]] = [dict() for _ in range(count + 1)]
    alpha[0]["START"] = 0.0
    for start in range(count):
        for segment in segments_by_start[start]:
            values = []
            for previous, previous_score in alpha[start].items():
                transition = transition_log_probability(model, previous, segment.state)
                if transition != NEG_INF:
                    values.append(previous_score + transition)
            if not values:
                continue
            value = logsumexp(values) + segment.log_score
            alpha[segment.end_index][segment.state] = logsumexp(
                [alpha[segment.end_index].get(segment.state, NEG_INF), value]
            )
    log_evidence = logsumexp(
        score + transition_log_probability(model, state, "END") for state, score in alpha[count].items()
    )

    beta: list[dict[str, float]] = [dict() for _ in range(count + 1)]
    for previous in ["START", *states]:
        beta[count][previous] = transition_log_probability(model, previous, "END")
    for start in range(count - 1, -1, -1):
        for previous in ["START", *states]:
            beta[start][previous] = logsumexp(
                transition_log_probability(model, previous, segment.state)
                + segment.log_score
                + beta[segment.end_index][segment.state]
                for segment in segments_by_start[start]
                if segment.state != previous
            )

    posterior = {}
    for start, segments in enumerate(segments_by_start):
        for segment in segments:
            prefix = logsumexp(
                score + transition_log_probability(model, previous, segment.state)
                for previous, score in alpha[start].items()
                if previous != segment.state
            )
            if prefix == NEG_INF or beta[segment.end_index].get(segment.state, NEG_INF) == NEG_INF:
                probability = 0.0
            else:
                probability = math.exp(
                    min(
                        0.0,
                        prefix
                        + segment.log_score
                        + beta[segment.end_index][segment.state]
                        - log_evidence,
                    )
                )
            posterior[segment] = probability

    token_posterior = [defaultdict(float) for _ in range(count)]
    for segment, probability in posterior.items():
        for index in range(segment.start_index, segment.end_index):
            token_posterior[index][segment.state] += probability
    normalized = []
    for row in token_posterior:
        total = sum(row.values())
        normalized.append({state: value / total for state, value in row.items()} if total else {"C": 1.0})
    return log_evidence, posterior, normalized


def local_boundary_ci(selected: Segment, posterior: dict[Segment, float]):
    candidates = []
    selected_span = selected.end_index - selected.start_index
    for segment, probability in posterior.items():
        if segment.state != selected.state or probability <= 0:
            continue
        overlap = max(0, min(selected.end_index, segment.end_index) - max(selected.start_index, segment.start_index))
        if overlap < 0.5 * min(selected_span, segment.end_index - segment.start_index):
            continue
        candidates.append((segment, probability))
    if not candidates:
        candidates = [(selected, 1.0)]
    return (
        weighted_quantile([(segment.start_bp, weight) for segment, weight in candidates], 0.025),
        weighted_quantile([(segment.start_bp, weight) for segment, weight in candidates], 0.975),
        weighted_quantile([(segment.end_bp, weight) for segment, weight in candidates], 0.025),
        weighted_quantile([(segment.end_bp, weight) for segment, weight in candidates], 0.975),
    )


def decode_path(
    path: str,
    split: str,
    observations: list[Observation],
    families: list[Family],
    model: ProbabilityModel,
    duration_weight: float,
    min_copies: int,
    min_density: float,
    max_insert_bp: int,
    max_total_insert_bp: int,
) -> DecodedPath:
    segments_by_start = build_segments(
        observations,
        families,
        model,
        duration_weight,
        min_copies,
        min_density,
        max_insert_bp,
        max_total_insert_bp,
    )
    (best_score, best_segments), (second_score, second_segments) = k_best_decode(segments_by_start, model)
    log_evidence, posterior, token_posterior = forward_backward(segments_by_start, model)
    blocks = []
    for segment in best_segments:
        if segment.state == "C":
            continue
        start_low, start_high, end_low, end_high = local_boundary_ci(segment, posterior)
        token_values = [token_posterior[index].get(segment.state, 0.0) for index in range(segment.start_index, segment.end_index)]
        blocks.append(
            BlockCall(
                path=path,
                order=len(blocks) + 1,
                segment=segment,
                segment_posterior=posterior.get(segment, 0.0),
                mean_token_posterior=statistics.mean(token_values),
                min_token_posterior=min(token_values),
                start_ci_low_bp=start_low,
                start_ci_high_bp=start_high,
                end_ci_low_bp=end_low,
                end_ci_high_bp=end_high,
            )
        )
    return DecodedPath(
        path=path,
        split=split,
        best_segments=best_segments,
        second_segments=second_segments,
        best_log_score=best_score,
        second_log_score=second_score,
        log_evidence=log_evidence,
        segment_posterior=posterior,
        token_posterior=token_posterior,
        blocks=blocks,
    )


def family_sequence(segments) -> tuple[str, ...]:
    return tuple(segment.state for segment in segments if segment.state != "C")


def tune_duration_weight(
    candidates: list[float],
    roles: dict[str, str],
    observations_by_path: dict[str, list[Observation]],
    blocks_by_path: dict[str, list[BaselineBlock]],
    families: list[Family],
    model: ProbabilityModel,
    min_copies: int,
    min_density: float,
    max_insert_bp: int,
    max_total_insert_bp: int,
):
    tune_paths = sorted(path for path, role in roles.items() if role == "tune")
    rows = []
    for weight in candidates:
        count_error = 0
        family_error = 0
        boundary_error = 0
        boundary_pairs = 0
        for path in tune_paths:
            decoded = decode_path(
                path,
                "tune",
                observations_by_path[path],
                families,
                model,
                weight,
                min_copies,
                min_density,
                max_insert_bp,
                max_total_insert_bp,
            )
            predicted = decoded.blocks
            baseline = blocks_by_path[path]
            count_error += abs(len(predicted) - len(baseline))
            family_error += sum(a.segment.state != b.canonical for a, b in zip(predicted, baseline))
            family_error += abs(len(predicted) - len(baseline))
            for left, right in zip(predicted, baseline):
                if left.segment.state == right.canonical:
                    boundary_error += abs(left.segment.start_bp - right.start) + abs(left.segment.end_bp - right.end)
                    boundary_pairs += 2
        mean_boundary_error = boundary_error / max(1, boundary_pairs)
        objective = count_error * 20.0 + family_error * 10.0 + mean_boundary_error
        rows.append(
            {
                "duration_weight": weight,
                "tune_path_count": len(tune_paths),
                "block_count_absolute_error": count_error,
                "family_sequence_error": family_error,
                "mean_boundary_absolute_error_bp": f"{mean_boundary_error:.4f}",
                "selection_objective": f"{objective:.4f}",
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            float(row["selection_objective"]),
            abs(float(row["duration_weight"]) - 1.0),
            float(row["duration_weight"]),
        ),
    )
    for row in rows:
        row["selected"] = int(row is selected)
    return float(selected["duration_weight"]), rows


def confidence_label(segment_probability: float, token_probability: float, ci_width: int, period: int) -> str:
    probability = min(segment_probability, token_probability)
    if probability >= 0.90 and ci_width <= period:
        return "high"
    if probability >= 0.50 and ci_width <= 3 * period:
        return "moderate"
    return "low"


def state_path_text(segments: tuple[Segment, ...]) -> str:
    parts = []
    for segment in segments:
        label = "COMPLEX" if segment.state == "C" else f"{segment.state}x{segment.motif_copies}"
        parts.append(f"{label}[{segment.start_bp},{segment.end_bp})")
    return " -> ".join(parts)


def parameter_rows(model: ProbabilityModel, families: list[Family]) -> list[dict[str, object]]:
    rows = []
    for family in families:
        canonical = family.canonical
        for name, value, unit in (
            ("exact_emission_probability", model.exact_probability[canonical], "probability"),
            ("variant_emission_probability", model.variant_probability[canonical], "probability"),
            ("duration_mean", model.duration_mean[canonical], "motif_copies"),
            ("duration_sd", model.duration_sd[canonical], "motif_copies"),
            ("insertion_run_mean", model.insertion_run_mean[canonical], "bp"),
        ):
            rows.append(
                {
                    "parameter_group": "emission_or_duration",
                    "state_from": canonical,
                    "state_to": "-",
                    "parameter": name,
                    "estimate": f"{value:.8f}",
                    "unit": unit,
                    "learned_from": "training_paths_only",
                }
            )
    rows.append(
        {
            "parameter_group": "duration",
            "state_from": "C",
            "state_to": "-",
            "parameter": "complex_run_mean",
            "estimate": f"{model.complex_run_mean:.8f}",
            "unit": "bp",
            "learned_from": "training_paths_only",
        }
    )
    for (previous, current), probability in sorted(model.transition_probability.items()):
        rows.append(
            {
                "parameter_group": "transition",
                "state_from": previous,
                "state_to": current,
                "parameter": "transition_probability",
                "estimate": f"{probability:.8f}",
                "unit": "probability",
                "learned_from": "training_paths_only",
            }
        )
    return rows


def split_rows(roles, hashes, weights, baseline_blocks, decoded_by_path):
    rows = []
    for path in sorted(roles):
        decoded = decoded_by_path[path]
        rows.append(
            {
                "path": path,
                "split": roles[path],
                "split_hash": hashes[path],
                "path_weight": f"{weights[path]:.8f}",
                "p2_block_count": len(baseline_blocks[path]),
                "p3_block_count": len(decoded.blocks),
                "p2_family_order": ">".join(block.canonical for block in baseline_blocks[path]),
                "p3_family_order": ">".join(block.segment.state for block in decoded.blocks),
                "changed_from_p2": int(
                    len(baseline_blocks[path]) != len(decoded.blocks)
                    or tuple(block.canonical for block in baseline_blocks[path])
                    != tuple(block.segment.state for block in decoded.blocks)
                ),
            }
        )
    return rows


def token_rows(observations_by_path, decoded_by_path, families):
    rows = []
    states = ["C", *(family.canonical for family in families)]
    for path, observations in observations_by_path.items():
        decoded = decoded_by_path[path]
        selected_by_index = {}
        for segment in decoded.best_segments:
            for index in range(segment.start_index, segment.end_index):
                selected_by_index[index] = segment.state
        for index, observation in enumerate(observations):
            best_state = selected_by_index[index]
            if best_state == "C":
                emission_state = "complex"
            elif observation.canonical == best_state and observation.kind == "exact_motif":
                emission_state = "exact_motif"
            elif observation.canonical == best_state:
                emission_state = "variant_motif"
            else:
                emission_state = "local_insertion"
            probabilities = decoded.token_posterior[index]
            entropy = -sum(value * math.log2(value) for value in probabilities.values() if value > 0)
            row = {
                "path": path,
                "split": decoded.split,
                "token_index": observation.index,
                "start_bp": observation.start,
                "end_bp": observation.end,
                "observed": observation.observed,
                "p2_token_type": observation.kind,
                "p2_family": observation.canonical or "-",
                "p3_block_state": best_state,
                "p3_emission_state": emission_state,
                "selected_state_posterior": f"{probabilities.get(best_state, 0.0):.8f}",
                "posterior_entropy_bits": f"{entropy:.8f}",
                "coordinate_system": COORDINATE_SYSTEM,
            }
            for state in states:
                row[f"posterior_{'complex' if state == 'C' else state}"] = f"{probabilities.get(state, 0.0):.8f}"
            rows.append(row)
    return rows


def block_rows(decoded_by_path, family_ids):
    rows = []
    for path, decoded in decoded_by_path.items():
        for block in decoded.blocks:
            segment = block.segment
            ci_width = max(
                block.start_ci_high_bp - block.start_ci_low_bp,
                block.end_ci_high_bp - block.end_ci_low_bp,
            )
            rows.append(
                {
                    "path": path,
                    "split": decoded.split,
                    "repeat_block": f"R{block.order}",
                    "start_bp": segment.start_bp,
                    "end_bp": segment.end_bp,
                    "start_ci_low_bp": block.start_ci_low_bp,
                    "start_ci_high_bp": block.start_ci_high_bp,
                    "end_ci_low_bp": block.end_ci_low_bp,
                    "end_ci_high_bp": block.end_ci_high_bp,
                    "family_id": family_ids[segment.state],
                    "canonical_family": segment.state,
                    "oriented_motif": segment.oriented_motif,
                    "copies": segment.motif_copies,
                    "exact_copies": segment.exact_copies,
                    "variant_copies": segment.variant_copies,
                    "inserted_bp": segment.inserted_bp,
                    "insertions": ";".join(f"{position}:{sequence}" for position, sequence in segment.insertion_parts) or "-",
                    "repeat_density": f"{segment.density:.8f}",
                    "segment_posterior": f"{block.segment_posterior:.8f}",
                    "mean_token_posterior": f"{block.mean_token_posterior:.8f}",
                    "min_token_posterior": f"{block.min_token_posterior:.8f}",
                    "confidence": confidence_label(
                        block.segment_posterior,
                        block.mean_token_posterior,
                        ci_width,
                        len(segment.state),
                    ),
                    "interpretation_status": "assembly_only_probabilistic",
                    "coordinate_system": COORDINATE_SYSTEM,
                }
            )
    return rows


def solution_rows(decoded_by_path, baseline_blocks):
    rows = []
    for path, decoded in decoded_by_path.items():
        best_posterior = math.exp(min(0.0, decoded.best_log_score - decoded.log_evidence))
        second_posterior = (
            0.0
            if decoded.second_log_score == NEG_INF
            else math.exp(min(0.0, decoded.second_log_score - decoded.log_evidence))
        )
        gap_bits = (
            float("inf")
            if decoded.second_log_score == NEG_INF
            else (decoded.best_log_score - decoded.second_log_score) / math.log(2.0)
        )
        baseline_order = tuple(block.canonical for block in baseline_blocks[path])
        rows.append(
            {
                "path": path,
                "split": decoded.split,
                "best_log2_score": f"{decoded.best_log_score / math.log(2.0):.8f}",
                "second_log2_score": "-inf" if decoded.second_log_score == NEG_INF else f"{decoded.second_log_score / math.log(2.0):.8f}",
                "best_second_gap_bits": "inf" if math.isinf(gap_bits) else f"{gap_bits:.8f}",
                "log2_evidence": f"{decoded.log_evidence / math.log(2.0):.8f}",
                "best_path_posterior": f"{best_posterior:.8f}",
                "second_path_posterior": f"{second_posterior:.8f}",
                "best_decomposition": state_path_text(decoded.best_segments),
                "second_decomposition": state_path_text(decoded.second_segments) if decoded.second_segments else "-",
                "p2_family_order": ">".join(baseline_order),
                "p3_family_order": ">".join(family_sequence(decoded.best_segments)),
                "changed_from_p2": int(baseline_order != family_sequence(decoded.best_segments)),
                "interpretation_status": "assembly_only_probabilistic",
            }
        )
    return rows


def evaluation_rows(decoded_by_path, baseline_blocks):
    rows = []
    for split in ("train", "tune", "test", "all"):
        paths = [path for path, decoded in decoded_by_path.items() if split == "all" or decoded.split == split]
        if not paths:
            continue
        exact_order = 0
        count_error = 0
        boundary_error = 0
        boundary_pairs = 0
        changed = 0
        for path in paths:
            decoded = decoded_by_path[path]
            baseline = baseline_blocks[path]
            predicted_order = tuple(block.segment.state for block in decoded.blocks)
            baseline_order = tuple(block.canonical for block in baseline)
            exact_order += predicted_order == baseline_order
            changed += predicted_order != baseline_order
            count_error += abs(len(decoded.blocks) - len(baseline))
            for predicted, expected in zip(decoded.blocks, baseline):
                if predicted.segment.state == expected.canonical:
                    boundary_error += abs(predicted.segment.start_bp - expected.start)
                    boundary_error += abs(predicted.segment.end_bp - expected.end)
                    boundary_pairs += 2
        metrics = {
            "path_count": len(paths),
            "exact_P2_family_order_paths": exact_order,
            "changed_from_P2_paths": changed,
            "mean_absolute_block_count_difference": f"{count_error / len(paths):.8f}",
            "mean_absolute_boundary_difference_bp": f"{boundary_error / max(1, boundary_pairs):.8f}",
        }
        rows.extend({"split": split, "metric": key, "value": value} for key, value in metrics.items())
    return rows


def make_p2_actions(p2, observations_by_path):
    output = {}
    for path, observations in observations_by_path.items():
        output[path] = [
            p2.Action(
                path,
                row.start,
                row.end,
                row.kind,
                row.observed,
                row.canonical,
                row.oriented_motif,
                row.edit_distance,
            )
            for row in observations
        ]
    return output


def make_p2_blocks(p2, decoded_by_path):
    output = {}
    for path, decoded in decoded_by_path.items():
        output[path] = [
            p2.RepeatBlock(
                path=path,
                order=block.order,
                start=block.segment.start_bp,
                end=block.segment.end_bp,
                canonical=block.segment.state,
                oriented_motif=block.segment.oriented_motif,
                copies=block.segment.motif_copies,
                exact_copies=block.segment.exact_copies,
                variant_copies=block.segment.variant_copies,
                inserted_bp=block.segment.inserted_bp,
                insertion_parts=block.segment.insertion_parts,
                density=block.segment.density,
            )
            for block in decoded.blocks
        ]
    return output


def write_p3_gfa(
    p2,
    path: Path,
    paths,
    homology_regions,
    repeat_regions,
    actions_by_path,
    blocks_by_path,
    family_ids,
    decoded_by_path,
    duration_weight: float,
):
    nodes, edges, path_steps = p2.write_gfa(
        path,
        paths,
        homology_regions,
        repeat_regions,
        actions_by_path,
        blocks_by_path,
        family_ids,
        0.0,
        0.0,
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = (
        "H\tVN:Z:1.0\tTS:Z:p3_hidden_semi_markov_assembly_only"
        f"\tDW:f:{duration_weight:.4f}\tCS:Z:{COORDINATE_SYSTEM}"
    )
    rewritten = []
    for line in lines:
        if not line.startswith("P\t"):
            rewritten.append(line)
            continue
        fields = line.split("\t")
        path_name = fields[1]
        fields = [field for field in fields if field != "VS:Z:provisional_sequence_only"]
        fields.append("VS:Z:assembly_only_probabilistic")
        decoded = decoded_by_path[path_name]
        gap = (
            999.0
            if decoded.second_log_score == NEG_INF
            else (decoded.best_log_score - decoded.second_log_score) / math.log(2.0)
        )
        fields.append(f"PG:f:{gap:.4f}")
        for block in decoded.blocks:
            fields.append(f"Q{block.order}:f:{block.mean_token_posterior:.6f}")
        rewritten.append("\t".join(fields))
    path.write_text("\n".join(rewritten) + "\n", encoding="utf-8", newline="\n")
    return nodes, edges, path_steps


def plot_overview(
    path: Path,
    repeat_regions,
    decoded_by_path,
    observations_by_path,
    families,
    family_ids,
    model,
) -> None:
    colors = ["#2e74b5", "#4d9b68", "#c56a2d", "#7d5ba6", "#b64d6b", "#548a8a"]
    color_by_family = {family.canonical: colors[index % len(colors)] for index, family in enumerate(families)}
    fig = plt.figure(figsize=(14, max(9.0, 0.25 * len(repeat_regions) + 5.0)))
    grid = fig.add_gridspec(2, 1, height_ratios=(2.0, 8.0), hspace=0.28)
    top = fig.add_subplot(grid[0])
    bottom = fig.add_subplot(grid[1])

    labels = [f"{family_ids[family.canonical]} {family.canonical}" for family in families]
    means = [model.duration_mean[family.canonical] for family in families]
    errors = [model.duration_sd[family.canonical] for family in families]
    top.barh(range(len(families)), means, xerr=errors, color=[color_by_family[row.canonical] for row in families])
    top.set_yticks(range(len(families)))
    top.set_yticklabels(labels)
    top.set_xlabel("Learned block duration (motif copies; mean +/- SD)")
    top.grid(axis="x", color="#e5e7eb", linewidth=0.7)
    top.spines[["top", "right"]].set_visible(False)

    for index, region in enumerate(repeat_regions):
        y = len(repeat_regions) - index - 1
        bottom.hlines(y, region.start, region.end, color="#d7dde5", linewidth=7, zorder=1)
        decoded = decoded_by_path[region.path]
        for block in decoded.blocks:
            segment = block.segment
            alpha = max(0.28, min(1.0, block.mean_token_posterior))
            bottom.add_patch(
                Rectangle(
                    (segment.start_bp, y - 0.30),
                    segment.end_bp - segment.start_bp,
                    0.60,
                    facecolor=color_by_family[segment.state],
                    edgecolor="#263746",
                    linewidth=0.5,
                    alpha=alpha,
                    hatch="//" if segment.variant_copies or segment.inserted_bp else None,
                    zorder=2,
                )
            )
            if block.start_ci_low_bp != block.start_ci_high_bp:
                bottom.hlines(y, block.start_ci_low_bp, block.start_ci_high_bp, color="#111827", linewidth=1.2, zorder=3)
            if block.end_ci_low_bp != block.end_ci_high_bp:
                bottom.hlines(y, block.end_ci_low_bp, block.end_ci_high_bp, color="#111827", linewidth=1.2, zorder=3)
            for observation in observations_by_path[region.path]:
                if observation.start < segment.start_bp or observation.end > segment.end_bp:
                    continue
                if observation.canonical and observation.canonical != segment.state:
                    interruption_color = color_by_family[observation.canonical]
                elif observation.kind == "background":
                    interruption_color = "#111827"
                else:
                    continue
                bottom.add_patch(
                    Rectangle(
                        (observation.start, y - 0.18),
                        observation.end - observation.start,
                        0.36,
                        facecolor=interruption_color,
                        edgecolor="#ffffff",
                        linewidth=0.35,
                        zorder=4,
                    )
                )
    bottom.set_yticks(range(len(repeat_regions)))
    bottom.set_yticklabels([region.path for region in reversed(repeat_regions)], fontsize=6.5)
    bottom.set_xlabel("Path coordinate (bp; 0-based half-open)")
    bottom.set_title("P3 position-specific repeat blocks; opacity represents posterior support", fontsize=11)
    bottom.grid(axis="x", color="#e5e7eb", linewidth=0.7)
    bottom.spines[["top", "right"]].set_visible(False)
    handles = [Patch(facecolor="#d7dde5", label="P1 repeat span")]
    handles.extend(
        Patch(facecolor=color_by_family[family.canonical], label=f"{family_ids[family.canonical]} {family.canonical}")
        for family in families
    )
    handles.append(Patch(facecolor="white", edgecolor="#263746", hatch="//", label="variant or insertion"))
    handles.append(Patch(facecolor="#111827", label="non-motif insertion"))
    bottom.legend(handles=handles, loc="upper right", fontsize=8, ncol=min(4, len(handles)))
    fig.suptitle("P3 hidden semi-Markov decomposition and uncertainty", fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.06)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="P3 hidden semi-Markov repeat decomposition with posterior uncertainty.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--p1-boundaries", type=Path, default=DEFAULT_P1)
    parser.add_argument("--p2-dictionary", type=Path, default=DEFAULT_P2_DIR / f"{DEFAULT_P2_PREFIX}_共享motif字典.tsv")
    parser.add_argument("--p2-tokens", type=Path, default=DEFAULT_P2_DIR / f"{DEFAULT_P2_PREFIX}_逐路径motif_token.tsv")
    parser.add_argument("--p2-blocks", type=Path, default=DEFAULT_P2_DIR / f"{DEFAULT_P2_PREFIX}_逐路径repeat区块.tsv")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--min-copies", type=int, default=4)
    parser.add_argument("--min-density", type=float, default=0.60)
    parser.add_argument("--max-insert-bp", type=int, default=12)
    parser.add_argument("--max-total-insert-bp", type=int, default=18)
    parser.add_argument("--duration-weights", default="0.5,1.0,1.5,2.0,3.0")
    parser.add_argument("--path-weight-mode", choices=("equal", "suffix"), default="equal")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    p2 = load_module(P2_SCRIPT, "vntrmap_p3_p2")
    paths = p2.read_gfa_paths(args.input)
    homology_regions, repeat_regions = p2.load_regions(paths, args.p1_boundaries)
    families = load_families(args.p2_dictionary)
    family_ids = {family.canonical: family.family_id for family in families}
    observations_by_path = load_observations(args.p2_tokens)
    observations_by_path = refine_short_motif_like_background(
        p2,
        observations_by_path,
        families,
        args.max_total_insert_bp,
    )
    baseline_blocks = load_baseline_blocks(args.p2_blocks)
    validate_inputs(paths, repeat_regions, observations_by_path, baseline_blocks, families)

    path_names = [path.name for path in paths]
    weights = path_weights(path_names, args.path_weight_mode)
    roles, hashes = split_paths(path_names, weights)
    model = estimate_model(families, observations_by_path, baseline_blocks, roles, weights)
    duration_candidates = [float(value) for value in args.duration_weights.split(",") if value.strip()]
    selected_weight, tuning = tune_duration_weight(
        duration_candidates,
        roles,
        observations_by_path,
        baseline_blocks,
        families,
        model,
        args.min_copies,
        args.min_density,
        args.max_insert_bp,
        args.max_total_insert_bp,
    )

    decoded_by_path = {}
    for path in path_names:
        decoded_by_path[path] = decode_path(
            path,
            roles[path],
            observations_by_path[path],
            families,
            model,
            selected_weight,
            args.min_copies,
            args.min_density,
            args.max_insert_bp,
            args.max_total_insert_bp,
        )

    parameter_output = parameter_rows(model, families)
    split_output = split_rows(roles, hashes, weights, baseline_blocks, decoded_by_path)
    token_output = token_rows(observations_by_path, decoded_by_path, families)
    block_output = block_rows(decoded_by_path, family_ids)
    solution_output = solution_rows(decoded_by_path, baseline_blocks)
    evaluation_output = evaluation_rows(decoded_by_path, baseline_blocks)

    parameter_path = args.outdir / f"{args.prefix}_学习参数.tsv"
    tuning_path = args.outdir / f"{args.prefix}_调参选择.tsv"
    split_path = args.outdir / f"{args.prefix}_训练调参与留出划分.tsv"
    token_path = args.outdir / f"{args.prefix}_逐路径token后验.tsv"
    block_path = args.outdir / f"{args.prefix}_逐路径P3_repeat区块.tsv"
    solution_path = args.outdir / f"{args.prefix}_最优与第二优分解.tsv"
    evaluation_path = args.outdir / f"{args.prefix}_分组评估.tsv"
    gfa_path = args.outdir / f"{args.prefix}_P3半马尔可夫多环图.gfa"
    png_path = args.outdir / f"{args.prefix}_P3概率区块总览.png"
    qa_path = args.outdir / f"{args.prefix}_P3验证汇总.tsv"

    write_tsv(parameter_path, parameter_output, list(parameter_output[0]))
    write_tsv(tuning_path, tuning, list(tuning[0]))
    write_tsv(split_path, split_output, list(split_output[0]))
    write_tsv(token_path, token_output, list(token_output[0]))
    write_tsv(block_path, block_output, list(block_output[0]) if block_output else ["path", "repeat_block"])
    write_tsv(solution_path, solution_output, list(solution_output[0]))
    write_tsv(evaluation_path, evaluation_output, ["split", "metric", "value"])

    p2_actions = make_p2_actions(p2, observations_by_path)
    p2_blocks = make_p2_blocks(p2, decoded_by_path)
    nodes, edges, path_steps = write_p3_gfa(
        p2,
        gfa_path,
        paths,
        homology_regions,
        repeat_regions,
        p2_actions,
        p2_blocks,
        family_ids,
        decoded_by_path,
        selected_weight,
    )
    reconstructed = p2.validate_graph(paths, nodes, path_steps)
    plot_overview(
        png_path,
        repeat_regions,
        decoded_by_path,
        observations_by_path,
        families,
        family_ids,
        model,
    )

    block_counts = Counter(len(decoded.blocks) for decoded in decoded_by_path.values())
    modal_count = min(block_counts, key=lambda value: (-block_counts[value], value))
    finite_gaps = [
        (decoded.best_log_score - decoded.second_log_score) / math.log(2.0)
        for decoded in decoded_by_path.values()
        if decoded.second_log_score != NEG_INF
    ]
    low_confidence = sum(
        confidence_label(
            block.segment_posterior,
            block.mean_token_posterior,
            max(
                block.start_ci_high_bp - block.start_ci_low_bp,
                block.end_ci_high_bp - block.end_ci_low_bp,
            ),
            len(block.segment.state),
        )
        == "low"
        for decoded in decoded_by_path.values()
        for block in decoded.blocks
    )
    self_loops = sorted({left.split("_", 1)[0] for left, right in edges if left == right and left.startswith("R")})
    qa = {
        "path_count": len(paths),
        "training_path_count": sum(role == "train" for role in roles.values()),
        "tuning_path_count": sum(role == "tune" for role in roles.values()),
        "heldout_path_count": sum(role == "test" for role in roles.values()),
        "path_weight_mode": args.path_weight_mode,
        "total_effective_path_weight": f"{sum(weights.values()):.8f}",
        "training_effective_path_weight": f"{sum(weights[path] for path, role in roles.items() if role == 'train'):.8f}",
        "tuning_effective_path_weight": f"{sum(weights[path] for path, role in roles.items() if role == 'tune'):.8f}",
        "heldout_effective_path_weight": f"{sum(weights[path] for path, role in roles.items() if role == 'test'):.8f}",
        "selected_duration_weight": f"{selected_weight:.4f}",
        "motif_family_count": len(families),
        "repeat_block_count": sum(len(decoded.blocks) for decoded in decoded_by_path.values()),
        "modal_repeat_block_count_per_path": modal_count,
        "paths_with_modal_repeat_block_count": block_counts[modal_count],
        "paths_changed_from_P2_family_order": sum(int(row["changed_from_p2"]) for row in split_output),
        "low_confidence_block_count": low_confidence,
        "median_best_second_gap_bits": f"{statistics.median(finite_gaps):.8f}" if finite_gaps else "inf",
        "posterior_available_paths": len(decoded_by_path),
        "second_best_available_paths": sum(decoded.second_log_score != NEG_INF for decoded in decoded_by_path.values()),
        "assembly_only_probabilistic_paths": len(decoded_by_path),
        "gfa_node_count": len(nodes),
        "gfa_edge_count": len(edges),
        "explicit_loop_positions": ",".join(self_loops),
        "exact_gfa_reconstruction_paths": reconstructed,
    }
    write_tsv(qa_path, [{"metric": key, "value": value} for key, value in qa.items()], ["metric", "value"])

    print(f"Paths: {len(paths)}; split train/tune/test=" + "/".join(str(sum(role == value for role in roles.values())) for value in ("train", "tune", "test")))
    print(f"Selected duration weight: {selected_weight:.2f}")
    print(
        "Learned durations: "
        + ", ".join(
            f"{family_ids[family.canonical]}={model.duration_mean[family.canonical]:.2f}+/-{model.duration_sd[family.canonical]:.2f} copies"
            for family in families
        )
    )
    print(
        f"Repeat blocks: {qa['repeat_block_count']}; modal/path={modal_count} "
        f"for {block_counts[modal_count]}/{len(paths)} paths"
    )
    print(f"Changed from P2 family order: {qa['paths_changed_from_P2_family_order']} paths")
    print(f"Low-confidence blocks: {low_confidence}")
    print(f"GFA nodes={len(nodes)} edges={len(edges)} loops={','.join(self_loops) or '-'}")
    print(f"Exact GFA reconstruction: {reconstructed}/{len(paths)}")
    for output in (
        parameter_path,
        tuning_path,
        split_path,
        token_path,
        block_path,
        solution_path,
        evaluation_path,
        gfa_path,
        png_path,
        qa_path,
    ):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
