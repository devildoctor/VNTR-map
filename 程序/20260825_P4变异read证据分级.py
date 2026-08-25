#!/usr/bin/env python3
"""P4: integrate assembly events with optional per-read validation evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
P3_DIR = ROOT / "outputs" / "IRF2BPL" / "20260825_P3半马尔可夫概率分解与置信度"
P3_PREFIX = "20260825_IRF2BPL_P3半马尔可夫概率分解与置信度"
DEFAULT_OUTDIR = ROOT / "outputs" / "IRF2BPL" / "20260825_P4变异read证据分级"
DEFAULT_PREFIX = "20260825_IRF2BPL_P4变异read证据分级"
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


def gfa_path_sequences(lines: list[str]) -> dict[str, str]:
    segments = {}
    paths = {}
    for line in lines:
        fields = line.split("\t")
        if fields[0] == "S":
            segments[fields[1]] = fields[2]
        elif fields[0] == "P":
            sequence = []
            for step in fields[2].split(","):
                name, orientation = parse_step(step)
                value = segments[name]
                if value == "*":
                    raise ValueError(f"Cannot reconstruct path through unknown segment: {name}")
                if orientation == "-":
                    complement = str.maketrans("ACGTNacgtn", "TGCANtgcan")
                    value = value.translate(complement)[::-1]
                sequence.append(value)
            paths[fields[1]] = "".join(sequence)
    return paths


def alignment_edit_script(expected: str, observed: str) -> tuple[int, str]:
    rows = len(expected) + 1
    columns = len(observed) + 1
    score = [[0] * columns for _ in range(rows)]
    trace = [[""] * columns for _ in range(rows)]
    for i in range(1, rows):
        score[i][0] = i
        trace[i][0] = "D"
    for j in range(1, columns):
        score[0][j] = j
        trace[0][j] = "I"
    priority = {"M": 0, "S": 1, "D": 2, "I": 3}
    for i in range(1, rows):
        for j in range(1, columns):
            diagonal = "M" if expected[i - 1] == observed[j - 1] else "S"
            candidates = [
                (score[i - 1][j - 1] + (diagonal == "S"), diagonal),
                (score[i - 1][j] + 1, "D"),
                (score[i][j - 1] + 1, "I"),
            ]
            score[i][j], trace[i][j] = min(candidates, key=lambda item: (item[0], priority[item[1]]))
    operations = []
    i, j = len(expected), len(observed)
    while i or j:
        operation = trace[i][j]
        if operation == "M":
            i -= 1
            j -= 1
        elif operation == "S":
            operations.append(f"S{i}:{expected[i - 1]}>{observed[j - 1]}")
            i -= 1
            j -= 1
        elif operation == "D":
            operations.append(f"D{i}:{expected[i - 1]}")
            i -= 1
        elif operation == "I":
            operations.append(f"I{i}:{observed[j - 1]}")
            j -= 1
        else:
            raise AssertionError("Incomplete edit traceback")
    operations.reverse()
    return score[-1][-1], ";".join(operations) if operations else "-"


def stable_event_id(path: str, repeat_block: str, start: int, end: int, event_type: str, observed: str) -> str:
    payload = f"{path}|{repeat_block}|{start}|{end}|{event_type}|{observed}"
    return "EV_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:14]


def extract_events(token_rows: list[dict[str, str]], block_rows: list[dict[str, str]]) -> list[dict]:
    blocks_by_path = defaultdict(list)
    for row in block_rows:
        blocks_by_path[row["path"]].append(row)
    for rows in blocks_by_path.values():
        rows.sort(key=lambda row: int(row["start_bp"]))

    events = []
    for token in token_rows:
        emission = token["p3_emission_state"]
        if emission not in {"variant_motif", "local_insertion"}:
            continue
        start = int(token["start_bp"])
        end = int(token["end_bp"])
        containing = [
            block
            for block in blocks_by_path[token["path"]]
            if int(block["start_bp"]) <= start and end <= int(block["end_bp"])
        ]
        if len(containing) != 1:
            raise AssertionError(f"P4 event does not map to one P3 block: {token['path']}:{start}-{end}")
        block = containing[0]
        expected = block["oriented_motif"]
        observed = token["observed"]
        if emission == "variant_motif":
            edit_distance, edit_script = alignment_edit_script(expected, observed)
            event_type = "motif_substitution" if len(expected) == len(observed) else "motif_indel"
        else:
            edit_distance = len(observed)
            edit_script = f"INS:{observed}"
            event_type = (
                "foreign_family_interruption"
                if token["p2_family"] not in {"", "-", block["canonical_family"]}
                else "local_insertion"
            )
        repeat_block = block["repeat_block"]
        offset = start - int(block["start_bp"])
        signature = f"{repeat_block}|{offset}|{event_type}|{expected}|{observed}"
        events.append(
            {
                "event_id": stable_event_id(token["path"], repeat_block, start, end, event_type, observed),
                "path": token["path"],
                "split": token["split"],
                "repeat_block": repeat_block,
                "event_type": event_type,
                "start_bp": start,
                "end_bp": end,
                "block_offset_bp": offset,
                "expected_motif": expected,
                "observed": observed,
                "edit_distance": edit_distance,
                "edit_script": edit_script,
                "p2_family": token["p2_family"],
                "p3_family": block["canonical_family"],
                "token_posterior": float(token["selected_state_posterior"]),
                "block_confidence": block["confidence"],
                "cohort_signature": signature,
                "coordinate_system": COORDINATE_SYSTEM,
            }
        )
    signature_paths = defaultdict(set)
    for event in events:
        signature_paths[event["cohort_signature"]].add(event["path"])
    for event in events:
        event["cohort_path_count"] = len(signature_paths[event["cohort_signature"]])
    return events


def parse_bool(value: str, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean in {field}: {value}")


EVIDENCE_FIELDS = [
    "event_id",
    "path",
    "repeat_block",
    "start_bp",
    "end_bp",
    "event_type",
    "read_id",
    "platform",
    "strand",
    "mapq",
    "base_quality",
    "spanning",
    "supports_event",
    "caller",
]


def evidence_template(events: list[dict]) -> list[dict]:
    return [
        {
            "event_id": event["event_id"],
            "path": event["path"],
            "repeat_block": event["repeat_block"],
            "start_bp": event["start_bp"],
            "end_bp": event["end_bp"],
            "event_type": event["event_type"],
            "read_id": ".",
            "platform": ".",
            "strand": ".",
            "mapq": ".",
            "base_quality": ".",
            "spanning": ".",
            "supports_event": ".",
            "caller": ".",
        }
        for event in events
    ]


def load_evidence(path: Path | None, events_by_id: dict[str, dict]) -> tuple[list[dict], int]:
    if path is None:
        return [], 0
    rows, fields = read_tsv(path)
    missing = [field for field in EVIDENCE_FIELDS if field not in fields]
    if missing:
        raise ValueError(f"Read evidence is missing columns: {','.join(missing)}")
    parsed = []
    seen = set()
    unknown = 0
    for row in rows:
        if row["read_id"].strip().lower() in {"", ".", "na"}:
            continue
        if row["event_id"] not in events_by_id:
            unknown += 1
            continue
        event = events_by_id[row["event_id"]]
        expected_identity = (
            event["path"],
            event["repeat_block"],
            str(event["start_bp"]),
            str(event["end_bp"]),
            event["event_type"],
        )
        observed_identity = (
            row["path"],
            row["repeat_block"],
            row["start_bp"],
            row["end_bp"],
            row["event_type"],
        )
        if observed_identity != expected_identity:
            raise ValueError(f"Read evidence identity does not match event catalog: {row['event_id']}")
        key = (row["event_id"], row["read_id"])
        if key in seen:
            raise ValueError(f"Duplicate event/read evidence row: {key[0]} {key[1]}")
        seen.add(key)
        strand = row["strand"].strip()
        if strand not in {"+", "-"}:
            raise ValueError(f"Invalid strand for {key[1]}: {strand}")
        mapq = float(row["mapq"])
        base_quality = float(row["base_quality"])
        if not math.isfinite(mapq) or not math.isfinite(base_quality) or mapq < 0 or base_quality < 0:
            raise ValueError(f"Invalid read quality for {key[1]}")
        parsed.append(
            {
                **row,
                "strand": strand,
                "mapq": mapq,
                "base_quality": base_quality,
                "spanning": parse_bool(row["spanning"], "spanning"),
                "supports_event": parse_bool(row["supports_event"], "supports_event"),
            }
        )
    return parsed, unknown


def classify_events(events: list[dict], evidence: list[dict], args) -> tuple[list[dict], list[dict]]:
    evidence_by_event = defaultdict(list)
    for row in evidence:
        evidence_by_event[row["event_id"]].append(row)
    output = []
    summaries = []
    for event in events:
        rows = evidence_by_event[event["event_id"]]
        usable = [
            row
            for row in rows
            if row["spanning"] and row["mapq"] >= args.min_mapq and row["base_quality"] >= args.min_base_quality
        ]
        supporting = [row for row in usable if row["supports_event"]]
        raw_supporting = [row for row in rows if row["supports_event"]]
        total = len(usable)
        support = len(supporting)
        fraction = support / total if total else 0.0
        strands = sorted({row["strand"] for row in supporting})
        callers = sorted({row["caller"].strip() for row in supporting if row["caller"].strip()})
        platforms = sorted({row["platform"].strip() for row in rows if row["platform"].strip()})
        both_strands = set(strands) == {"+", "-"}
        strand_pass = both_strands or not args.require_both_strands
        if not rows:
            status = "assembly_only_unvalidated"
            reason = "no_read_evidence"
        elif (
            total >= args.min_coverage
            and support >= args.min_support_reads
            and fraction >= args.min_alt_fraction
            and strand_pass
        ):
            status = "read_backed_high_confidence"
            reason = "support_quality_spanning_fraction_and_strand_pass"
        elif (
            total >= args.min_coverage
            and fraction < args.min_alt_fraction
        ) or (len(raw_supporting) > 0 and support == 0 and len(rows) >= args.min_coverage):
            status = "error_like"
            reason = "adequate_coverage_but_support_is_absent_low_fraction_or_low_quality"
        else:
            status = "uncertain"
            reason = "read_evidence_present_but_thresholds_not_all_met"
        evidence_summary = {
            "read_evidence_rows": len(rows),
            "hq_spanning_reads": total,
            "hq_supporting_reads": support,
            "supporting_strands": ",".join(strands) if strands else "-",
            "both_strands_supported": int(both_strands),
            "read_support_fraction": f"{fraction:.8f}",
            "platforms": ",".join(platforms) if platforms else "-",
            "supporting_callers": ",".join(callers) if callers else "-",
            "validation_status": status,
            "validation_reason": reason,
        }
        summaries.append({"event_id": event["event_id"], **evidence_summary})
        output.append(
            {
                **event,
                "token_posterior": f"{event['token_posterior']:.8f}",
                **evidence_summary,
            }
        )
    return output, summaries


def status_summary(statuses: set[str]) -> str:
    if not statuses:
        return "not_an_event_node"
    if "error_like" in statuses:
        return "contains_error_like"
    if "uncertain" in statuses:
        return "contains_uncertain"
    if statuses == {"read_backed_high_confidence"}:
        return "read_backed_high_confidence"
    if "read_backed_high_confidence" in statuses:
        return "mixed_read_backed_and_assembly_only"
    return "assembly_only_unvalidated"


def write_annotated_gfa(input_path: Path, output_path: Path, events: list[dict]) -> tuple[int, int, int]:
    original = input_path.read_text(encoding="utf-8").splitlines()
    before = gfa_path_sequences(original)
    event_by_node_key = defaultdict(list)
    event_by_path = defaultdict(list)
    for event in events:
        event_by_node_key[(event["repeat_block"], event["observed"])].append(event)
        event_by_path[event["path"]].append(event)
    rewritten = []
    node_count = 0
    edge_count = 0
    for line in original:
        fields = line.split("\t")
        if fields[0] == "H":
            fields = [field for field in fields if not field.startswith(("P4:", "CS:"))]
            fields.extend(["P4:Z:assembly_and_optional_read_evidence", f"CS:Z:{COORDINATE_SYSTEM}"])
        elif fields[0] == "S":
            node_count += 1
            tags = parse_tags(fields[3:])
            matched = event_by_node_key.get((tags.get("LC", ""), fields[2]), [])
            fields = [field for field in fields if not field.startswith(("EV:", "RB:", "P4:"))]
            if matched:
                statuses = {event["validation_status"] for event in matched}
                read_backed = sum(event["validation_status"] == "read_backed_high_confidence" for event in matched)
                fields.extend(
                    [
                        f"EV:i:{len(matched)}",
                        f"RB:i:{read_backed}",
                        f"P4:Z:{status_summary(statuses)}",
                    ]
                )
        elif fields[0] == "L":
            edge_count += 1
        elif fields[0] == "P":
            statuses = {event["validation_status"] for event in event_by_path.get(fields[1], [])}
            fields = [field for field in fields if not field.startswith("P4:")]
            fields.append(f"P4:Z:{status_summary(statuses)}")
        rewritten.append("\t".join(fields))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8", newline="\n")
    after = gfa_path_sequences(rewritten)
    exact = sum(path in before and sequence == before[path] for path, sequence in after.items())
    if before != after:
        raise AssertionError("P4 GFA annotation changed one or more path sequences")
    return node_count, edge_count, exact


def plot_events(path: Path, rows: list[dict]) -> None:
    statuses = [
        "read_backed_high_confidence",
        "uncertain",
        "error_like",
        "assembly_only_unvalidated",
    ]
    colors = {
        "read_backed_high_confidence": "#2f855a",
        "uncertain": "#d69e2e",
        "error_like": "#c53030",
        "assembly_only_unvalidated": "#718096",
    }
    counts = Counter(row["validation_status"] for row in rows)
    path_names = list(dict.fromkeys(row["path"] for row in rows))
    height = max(7.0, 3.5 + 0.22 * len(path_names))
    figure, axes = plt.subplots(2, 1, figsize=(15, height), gridspec_kw={"height_ratios": [1, max(2, len(path_names) / 8)]})
    axes[0].barh(statuses, [counts[status] for status in statuses], color=[colors[status] for status in statuses])
    axes[0].set_xlabel("Assembly event count")
    axes[0].set_title("P4 event validation status")
    axes[0].grid(axis="x", color="#d7dde5", linewidth=0.8)
    path_index = {name: index for index, name in enumerate(path_names)}
    marker_by_type = {
        "motif_substitution": "o",
        "motif_indel": "s",
        "local_insertion": "D",
        "foreign_family_interruption": "^",
    }
    for event_type, marker in marker_by_type.items():
        subset = [row for row in rows if row["event_type"] == event_type]
        if not subset:
            continue
        axes[1].scatter(
            [(int(row["start_bp"]) + int(row["end_bp"])) / 2 for row in subset],
            [path_index[row["path"]] for row in subset],
            c=[colors[row["validation_status"]] for row in subset],
            marker=marker,
            s=25,
            linewidths=0.4,
            edgecolors="#1f2933",
            label=event_type,
        )
    axes[1].set_yticks(range(len(path_names)), path_names, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Path coordinate (bp; 0-based half-open)")
    axes[1].set_title("Per-path assembly events; color is P4 validation status")
    axes[1].grid(axis="x", color="#d7dde5", linewidth=0.7)
    axes[1].legend(loc="upper right", ncol=2, fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="P4 assembly-event catalog and optional per-read validation.")
    parser.add_argument("--p3-gfa", type=Path, default=P3_DIR / f"{P3_PREFIX}_P3半马尔可夫多环图.gfa")
    parser.add_argument("--p3-tokens", type=Path, default=P3_DIR / f"{P3_PREFIX}_逐路径token后验.tsv")
    parser.add_argument("--p3-blocks", type=Path, default=P3_DIR / f"{P3_PREFIX}_逐路径P3_repeat区块.tsv")
    parser.add_argument("--read-evidence", type=Path)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--min-coverage", type=int, default=5)
    parser.add_argument("--min-support-reads", type=int, default=3)
    parser.add_argument("--min-mapq", type=float, default=20.0)
    parser.add_argument("--min-base-quality", type=float, default=20.0)
    parser.add_argument("--min-alt-fraction", type=float, default=0.20)
    parser.add_argument("--require-both-strands", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-unknown-evidence", action="store_true")
    args = parser.parse_args()
    if args.min_coverage < 1 or args.min_support_reads < 1:
        parser.error("Read count thresholds must be positive")
    if not 0.0 <= args.min_alt_fraction <= 1.0:
        parser.error("--min-alt-fraction must be between 0 and 1")

    token_rows, _ = read_tsv(args.p3_tokens)
    block_rows, _ = read_tsv(args.p3_blocks)
    events = extract_events(token_rows, block_rows)
    if not events:
        raise ValueError("P3 outputs contain no variant or insertion events")
    evidence, unknown_evidence = load_evidence(args.read_evidence, {event["event_id"]: event for event in events})
    if unknown_evidence and not args.allow_unknown_evidence:
        raise ValueError(f"Read evidence contains {unknown_evidence} event IDs absent from the P4 catalog")
    classified, evidence_summaries = classify_events(events, evidence, args)

    args.outdir.mkdir(parents=True, exist_ok=True)
    event_path = args.outdir / f"{args.prefix}_P4组装事件与read判定.tsv"
    template_path = args.outdir / f"{args.prefix}_P4_read证据模板.tsv"
    summary_path = args.outdir / f"{args.prefix}_P4_read证据汇总.tsv"
    gfa_path = args.outdir / f"{args.prefix}_P4变异证据标注图.gfa"
    png_path = args.outdir / f"{args.prefix}_P4事件证据总览.png"
    qa_path = args.outdir / f"{args.prefix}_P4验证汇总.tsv"

    write_tsv(event_path, classified, list(classified[0]))
    write_tsv(template_path, evidence_template(events), EVIDENCE_FIELDS)
    write_tsv(summary_path, evidence_summaries, list(evidence_summaries[0]))
    node_count, edge_count, exact = write_annotated_gfa(args.p3_gfa, gfa_path, classified)
    plot_events(png_path, classified)

    path_count = len({row["path"] for row in token_rows})
    statuses = Counter(row["validation_status"] for row in classified)
    event_types = Counter(row["event_type"] for row in classified)
    qa = {
        "path_count": path_count,
        "assembly_event_count": len(classified),
        "motif_substitution_event_count": event_types["motif_substitution"],
        "motif_indel_event_count": event_types["motif_indel"],
        "local_insertion_event_count": event_types["local_insertion"],
        "foreign_family_interruption_event_count": event_types["foreign_family_interruption"],
        "read_evidence_row_count": len(evidence),
        "unknown_read_evidence_event_count": unknown_evidence,
        "read_backed_high_confidence_event_count": statuses["read_backed_high_confidence"],
        "uncertain_event_count": statuses["uncertain"],
        "error_like_event_count": statuses["error_like"],
        "assembly_only_unvalidated_event_count": statuses["assembly_only_unvalidated"],
        "gfa_node_count": node_count,
        "gfa_edge_count": edge_count,
        "exact_gfa_reconstruction_paths": exact,
        "interpretation_status": "read_backed" if evidence else "assembly_only_unvalidated",
    }
    write_tsv(qa_path, [{"metric": key, "value": value} for key, value in qa.items()], ["metric", "value"])

    print(f"Paths: {path_count}; assembly events: {len(classified)}")
    print(
        "P4 status: "
        f"high={statuses['read_backed_high_confidence']} uncertain={statuses['uncertain']} "
        f"error_like={statuses['error_like']} assembly_only={statuses['assembly_only_unvalidated']}"
    )
    print(f"Read evidence rows: {len(evidence)}; exact GFA reconstruction: {exact}/{path_count}")
    for output in (event_path, template_path, summary_path, gfa_path, png_path, qa_path):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
