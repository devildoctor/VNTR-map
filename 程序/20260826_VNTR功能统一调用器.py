#!/usr/bin/env python3
"""Cross-platform registry-driven launcher for the VNTR-map workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_REGISTRY = SCRIPT_PATH.with_name("20260826_VNTR功能注册表.json")
PLACEHOLDER = re.compile(r"\$\{([^{}]+)\}")


class LauncherError(RuntimeError):
    """A user-facing launcher or registry error."""


@dataclass
class StageResult:
    stage_id: str
    status: str
    started_utc: str
    finished_utc: str
    elapsed_seconds: float
    return_code: int | None
    log: str
    command: list[str]
    outputs: dict[str, str] = field(default_factory=dict)
    missing_outputs: list[str] = field(default_factory=list)
    metrics: dict[str, str] = field(default_factory=dict)
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage_id,
            "status": self.status,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "return_code": self.return_code,
            "log": self.log,
            "command": self.command,
            "outputs": self.outputs,
            "missing_outputs": self.missing_outputs,
            "metrics": self.metrics,
            "message": self.message,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def load_registry(path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LauncherError(f"功能注册表不存在: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LauncherError(f"功能注册表 JSON 无效: {path}: {exc}") from exc

    if data.get("schema_version") != 1:
        raise LauncherError("当前调用器只支持 schema_version=1")
    stages = data.get("stages")
    if not isinstance(stages, list) or not stages:
        raise LauncherError("功能注册表必须包含非空 stages 数组")

    by_id: dict[str, dict[str, Any]] = {}
    declaration_order: list[str] = []
    required_fields = {"id", "name", "command", "arguments", "outputs", "required_outputs"}
    for stage in stages:
        if not isinstance(stage, dict):
            raise LauncherError("每个 stage 必须是 JSON 对象")
        missing = required_fields - set(stage)
        if missing:
            raise LauncherError(f"stage 缺少字段 {sorted(missing)}: {stage.get('id', '<unknown>')}")
        stage_id = str(stage["id"]).upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9_-]*", stage_id):
            raise LauncherError(f"非法 stage id: {stage_id}")
        if stage_id in by_id:
            raise LauncherError(f"重复 stage id: {stage_id}")
        stage["id"] = stage_id
        stage.setdefault("depends_on", [])
        stage["depends_on"] = [str(item).upper() for item in stage["depends_on"]]
        if not isinstance(stage["command"], list) or not stage["command"]:
            raise LauncherError(f"{stage_id}.command 必须是非空数组")
        if not isinstance(stage["arguments"], list) or not isinstance(stage["outputs"], dict):
            raise LauncherError(f"{stage_id} 的 arguments/outputs 类型错误")
        unknown_required = set(stage["required_outputs"]) - set(stage["outputs"])
        if unknown_required:
            raise LauncherError(f"{stage_id}.required_outputs 引用了未声明产物: {sorted(unknown_required)}")
        by_id[stage_id] = stage
        declaration_order.append(stage_id)

    for stage_id, stage in by_id.items():
        unknown = set(stage["depends_on"]) - set(by_id)
        if unknown:
            raise LauncherError(f"{stage_id} 依赖未注册功能: {sorted(unknown)}")

    order = topological_order(by_id, declaration_order)
    data["stages"] = [by_id[stage_id] for stage_id in order]
    return data, order


def topological_order(by_id: dict[str, dict[str, Any]], declaration_order: list[str]) -> list[str]:
    state: dict[str, int] = {}
    result: list[str] = []

    def visit(stage_id: str) -> None:
        marker = state.get(stage_id, 0)
        if marker == 1:
            raise LauncherError(f"功能依赖存在循环，涉及: {stage_id}")
        if marker == 2:
            return
        state[stage_id] = 1
        for dependency in by_id[stage_id].get("depends_on", []):
            visit(dependency)
        state[stage_id] = 2
        if stage_id not in result:
            result.append(stage_id)

    for stage_id in declaration_order:
        visit(stage_id)
    return result


def stage_map(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {stage["id"]: stage for stage in registry["stages"]}


def expand_template(template: str, context: dict[str, str], *, optional: bool = False) -> str | None:
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        value = context.get(key)
        if value is None or value == "":
            missing.append(key)
            return ""
        return value

    expanded = PLACEHOLDER.sub(replace, str(template))
    if missing:
        if optional:
            return None
        raise LauncherError(f"缺少模板变量: {', '.join(sorted(set(missing)))}")
    return expanded


def normalize_path_string(value: str) -> str:
    return str(Path(value).expanduser().resolve())


def parse_key_values(items: Iterable[str], label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise LauncherError(f"{label} 必须使用 KEY=VALUE 格式: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise LauncherError(f"{label} 的 KEY 和 VALUE 不能为空: {item}")
        parsed[key] = value
    return parsed


def parse_stage_params(items: Iterable[str]) -> dict[str, list[tuple[str, str]]]:
    parsed: dict[str, list[tuple[str, str]]] = {}
    for item in items:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise LauncherError(f"--set-param 必须使用 STAGE.PARAM=VALUE 格式: {item}")
        left, value = item.split("=", 1)
        stage_id, parameter = left.split(".", 1)
        stage_id = stage_id.upper()
        flag = parameter if parameter.startswith("--") else f"--{parameter}"
        parsed.setdefault(stage_id, []).append((flag, value))
    return parsed


def parse_stage_flags(items: Iterable[str]) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for item in items:
        if "." not in item:
            raise LauncherError(f"--set-flag 必须使用 STAGE.FLAG 格式: {item}")
        stage_id, flag_name = item.split(".", 1)
        stage_id = stage_id.upper()
        flag = flag_name if flag_name.startswith("--") else f"--{flag_name}"
        parsed.setdefault(stage_id, []).append(flag)
    return parsed


def validate_override_stages(
    valid_ids: set[str],
    params: dict[str, list[tuple[str, str]]],
    flags: dict[str, list[str]],
) -> None:
    unknown = (set(params) | set(flags)) - valid_ids
    if unknown:
        raise LauncherError(f"参数覆盖引用未注册功能: {', '.join(sorted(unknown))}")


def select_stages(order: list[str], args: argparse.Namespace) -> list[str]:
    if args.stages:
        requested = [item.strip().upper() for item in args.stages.split(",") if item.strip()]
        unknown = set(requested) - set(order)
        if unknown:
            raise LauncherError(f"--stages 包含未注册功能: {', '.join(sorted(unknown))}")
        requested_set = set(requested)
        return [stage_id for stage_id in order if stage_id in requested_set]

    start = 0
    end = len(order) - 1
    if args.from_stage:
        stage_id = args.from_stage.upper()
        if stage_id not in order:
            raise LauncherError(f"未知 --from-stage: {stage_id}")
        start = order.index(stage_id)
    if args.to_stage:
        stage_id = args.to_stage.upper()
        if stage_id not in order:
            raise LauncherError(f"未知 --to-stage: {stage_id}")
        end = order.index(stage_id)
    if start > end:
        raise LauncherError("--from-stage 不能位于 --to-stage 之后")
    return order[start : end + 1]


def git_value(repo: Path, *arguments: str) -> str:
    command = ["git", "-c", f"safe.directory={repo}", "-C", str(repo), *arguments]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError:
        return "unavailable"
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_status_tsv(path: Path, results: list[StageResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["stage", "status", "started_utc", "finished_utc", "elapsed_seconds", "return_code", "log", "message"])
        for result in results:
            writer.writerow(
                [
                    result.stage_id,
                    result.status,
                    result.started_utc,
                    result.finished_utc,
                    f"{result.elapsed_seconds:.3f}",
                    "" if result.return_code is None else result.return_code,
                    result.log,
                    result.message,
                ]
            )


def read_metrics(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    metrics: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) >= 2 and row[0] not in {"metric", "key"}:
                metrics[row[0]] = row[1]
    return metrics


def file_ready(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def checksum_outputs(run_dir: Path) -> None:
    destination = run_dir / "SHA256SUMS"
    rows: list[str] = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file() and item != destination):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append(f"{digest.hexdigest()}  {path.relative_to(run_dir).as_posix()}")
    destination.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def build_stage(
    stage: dict[str, Any],
    base_context: dict[str, str],
    artifacts: dict[str, str],
    params: dict[str, list[tuple[str, str]]],
    flags: dict[str, list[str]],
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    stage_id = stage["id"]
    context = dict(base_context)
    context.update({f"artifact.{key}": value for key, value in artifacts.items()})
    command: list[str] = []
    for token in stage["command"]:
        expanded = expand_template(str(token), context)
        assert expanded is not None
        command.append(expanded)
    for argument in stage["arguments"]:
        if not isinstance(argument, dict) or "flag" not in argument or "value" not in argument:
            raise LauncherError(f"{stage_id}.arguments 中存在无效参数定义")
        expanded = expand_template(
            str(argument["value"]),
            context,
            optional=bool(argument.get("optional", False)),
        )
        if expanded is None:
            continue
        command.extend([str(argument["flag"]), expanded])
    for flag, value in params.get(stage_id, []):
        command.extend([flag, value])
    command.extend(flags.get(stage_id, []))

    outputs: dict[str, str] = {}
    for name, template in stage["outputs"].items():
        expanded = expand_template(str(template), context)
        assert expanded is not None
        outputs[name] = normalize_path_string(expanded)
    required = {name: outputs[name] for name in stage["required_outputs"]}
    return command, outputs, required


def quote_command(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def run_command(command: list[str], log_path: Path, cwd: Path, environment: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as log_handle:
        log_handle.write(f"COMMAND\t{quote_command(command)}\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        return process.wait()


def pipeline(args: argparse.Namespace, registry_path: Path) -> int:
    registry, order = load_registry(registry_path)
    by_id = stage_map(registry)
    selected = select_stages(order, args)
    if not selected:
        raise LauncherError("没有选择任何功能")

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise LauncherError(f"输入 GFA 不存在: {input_path}")
    run_dir = args.outdir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or re.sub(r"[^0-9A-Za-z._-]+", "_", input_path.stem).strip("_") or "vntr"

    artifact_overrides = parse_key_values(args.artifact, "--artifact")
    artifacts = {key: normalize_path_string(value) for key, value in artifact_overrides.items()}
    params = parse_stage_params(args.set_param)
    flags = parse_stage_flags(args.set_flag)
    validate_override_stages(set(order), params, flags)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "attempt_id": utc_stamp(),
        "dataset": prefix,
        "created_utc": utc_now(),
        "repository": str(REPO_ROOT),
        "git_commit": git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "git_branch": git_value(REPO_ROOT, "branch", "--show-current"),
        "input": str(input_path),
        "run_dir": str(run_dir),
        "registry": str(registry_path),
        "selected_stages": selected,
        "dry_run": bool(args.dry_run),
        "resume": bool(args.resume),
        "artifact_overrides": artifacts,
        "stage_results": [],
    }
    manifest_path = run_dir / "run_manifest.json"
    attempt_manifest_path = run_dir / "run_attempts" / f"{manifest['attempt_id']}.json"
    status_path = run_dir / "stage_status.tsv"
    write_json(manifest_path, manifest)

    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    matplotlib_dir = run_dir / ".matplotlib"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    environment["MPLCONFIGDIR"] = str(matplotlib_dir)

    results: list[StageResult] = []
    failed_stages: set[str] = set()
    for stage_id in selected:
        stage = by_id[stage_id]
        blocked_by = [dependency for dependency in stage["depends_on"] if dependency in failed_stages]
        if blocked_by:
            now = utc_now()
            result = StageResult(
                stage_id=stage_id,
                status="blocked",
                started_utc=now,
                finished_utc=now,
                elapsed_seconds=0.0,
                return_code=None,
                log="",
                command=[],
                message=f"依赖阶段失败: {','.join(blocked_by)}",
            )
            results.append(result)
            failed_stages.add(stage_id)
            if not args.keep_going:
                break
            continue

        stage_dir = run_dir / stage_id
        stage_prefix = f"{prefix}_{stage_id}"
        base_context = {
            "repo": str(REPO_ROOT),
            "python": str(args.python.expanduser().resolve()) if any(separator in str(args.python) for separator in ("/", "\\")) else str(args.python),
            "input": str(input_path),
            "run_dir": str(run_dir),
            "stage_dir": str(stage_dir),
            "stage_prefix": stage_prefix,
            "prefix": prefix,
            "read_evidence": str(args.read_evidence.expanduser().resolve()) if args.read_evidence else "",
            "external_evidence": str(args.external_evidence.expanduser().resolve()) if args.external_evidence else "",
        }
        try:
            command, outputs, required = build_stage(stage, base_context, artifacts, params, flags)
        except LauncherError as exc:
            now = utc_now()
            result = StageResult(
                stage_id=stage_id,
                status="configuration_error",
                started_utc=now,
                finished_utc=now,
                elapsed_seconds=0.0,
                return_code=None,
                log="",
                command=[],
                message=str(exc),
            )
            results.append(result)
            failed_stages.add(stage_id)
            if not args.keep_going:
                break
            continue

        log_path = run_dir / f"{stage_id}.log"
        started = utc_now()
        start_clock = time.monotonic()
        print(f"\n[{stage_id}] {stage['name']}")
        print(f"[{stage_id}] {quote_command(command)}")

        if args.dry_run:
            finished = utc_now()
            result = StageResult(
                stage_id=stage_id,
                status="planned",
                started_utc=started,
                finished_utc=finished,
                elapsed_seconds=time.monotonic() - start_clock,
                return_code=None,
                log=str(log_path),
                command=command,
                outputs=outputs,
            )
            results.append(result)
            artifacts.update({f"{stage_id}.{key}": value for key, value in outputs.items()})
            continue

        if args.resume and all(file_ready(Path(path)) for path in required.values()):
            finished = utc_now()
            validation_path = Path(outputs["validation"]) if "validation" in outputs else Path()
            result = StageResult(
                stage_id=stage_id,
                status="skipped_existing",
                started_utc=started,
                finished_utc=finished,
                elapsed_seconds=time.monotonic() - start_clock,
                return_code=0,
                log=str(log_path),
                command=command,
                outputs=outputs,
                metrics=read_metrics(validation_path),
                message="required outputs already exist",
            )
            results.append(result)
            artifacts.update({f"{stage_id}.{key}": value for key, value in outputs.items()})
            print(f"[{stage_id}] SKIP: required outputs already exist")
            continue

        missing_inputs = []
        for dependency in stage["depends_on"]:
            dependency_outputs = by_id[dependency]["required_outputs"]
            for output_name in dependency_outputs:
                key = f"{dependency}.{output_name}"
                if key in artifacts and not file_ready(Path(artifacts[key])):
                    missing_inputs.append(f"{key}={artifacts[key]}")
        if missing_inputs:
            finished = utc_now()
            result = StageResult(
                stage_id=stage_id,
                status="missing_input",
                started_utc=started,
                finished_utc=finished,
                elapsed_seconds=time.monotonic() - start_clock,
                return_code=None,
                log=str(log_path),
                command=command,
                outputs=outputs,
                message="; ".join(missing_inputs),
            )
            results.append(result)
            failed_stages.add(stage_id)
            if not args.keep_going:
                break
            continue

        stage_dir.mkdir(parents=True, exist_ok=True)
        try:
            return_code = run_command(command, log_path, REPO_ROOT, environment)
            execution_message = ""
        except OSError as exc:
            return_code = 127
            execution_message = str(exc)

        missing_outputs = [name for name, path in required.items() if not file_ready(Path(path))]
        if return_code == 0 and not missing_outputs:
            status = "completed"
            artifacts.update({f"{stage_id}.{key}": value for key, value in outputs.items()})
        elif return_code != 0:
            status = "failed"
            failed_stages.add(stage_id)
        else:
            status = "missing_output"
            failed_stages.add(stage_id)
        validation_path = Path(outputs["validation"]) if "validation" in outputs else Path()
        finished = utc_now()
        result = StageResult(
            stage_id=stage_id,
            status=status,
            started_utc=started,
            finished_utc=finished,
            elapsed_seconds=time.monotonic() - start_clock,
            return_code=return_code,
            log=str(log_path),
            command=command,
            outputs=outputs,
            missing_outputs=missing_outputs,
            metrics=read_metrics(validation_path),
            message=execution_message,
        )
        results.append(result)
        print(f"[{stage_id}] {status.upper()} ({result.elapsed_seconds:.1f}s)")
        if status != "completed" and not args.keep_going:
            break

        manifest["stage_results"] = [item.as_dict() for item in results]
        write_json(manifest_path, manifest)
        write_status_tsv(status_path, results)

    manifest["finished_utc"] = utc_now()
    manifest["artifacts"] = artifacts
    manifest["stage_results"] = [item.as_dict() for item in results]
    manifest["success"] = bool(results) and all(item.status in {"completed", "skipped_existing", "planned"} for item in results) and len(results) == len(selected)
    write_json(manifest_path, manifest)
    write_json(attempt_manifest_path, manifest)
    write_status_tsv(status_path, results)
    if not args.dry_run:
        checksum_outputs(run_dir)

    if manifest["success"]:
        print(f"\nPASS: {len(results)} 个功能已处理")
        print(f"结果目录: {run_dir}")
        return 0
    failed = [f"{item.stage_id}:{item.status}" for item in results if item.status not in {"completed", "skipped_existing", "planned"}]
    print(f"\nFAIL: {', '.join(failed) or '流水线未完整执行'}", file=sys.stderr)
    print(f"运行清单: {manifest_path}", file=sys.stderr)
    return 1


def direct_stage(args: argparse.Namespace, registry_path: Path) -> int:
    registry, _order = load_registry(registry_path)
    by_id = stage_map(registry)
    stage_id = args.stage_id.upper()
    if stage_id not in by_id:
        raise LauncherError(f"未知功能: {stage_id}")
    stage = by_id[stage_id]
    context = {
        "python": str(args.python),
        "repo": str(REPO_ROOT),
    }
    command = []
    for token in stage["command"]:
        expanded = expand_template(str(token), context)
        assert expanded is not None
        command.append(expanded)
    remainder = list(args.arguments)
    if remainder and remainder[0] == "--":
        remainder = remainder[1:]
    command.extend(remainder)
    print(quote_command(command))
    if args.dry_run:
        return 0
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vntrmap-matplotlib"))
    return subprocess.run(command, cwd=str(REPO_ROOT), env=environment).returncode


def print_stage_list(registry: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(registry["stages"], ensure_ascii=False, indent=2))
        return
    print("ID\t名称\t依赖\t说明")
    for stage in registry["stages"]:
        dependencies = ",".join(stage["depends_on"]) or "-"
        print(f"{stage['id']}\t{stage['name']}\t{dependencies}\t{stage.get('description', '')}")


def print_stage_description(stage: dict[str, Any]) -> None:
    print(f"{stage['id']}  {stage['name']}")
    print(stage.get("description", ""))
    print(f"依赖: {', '.join(stage['depends_on']) or '-'}")
    print("命令:")
    print("  " + " ".join(stage["command"]))
    print("参数模板:")
    for argument in stage["arguments"]:
        optional = " (可选)" if argument.get("optional") else ""
        print(f"  {argument['flag']} {argument['value']}{optional}")
    print("产物:")
    required = set(stage["required_outputs"])
    for name, template in stage["outputs"].items():
        marker = "*" if name in required else " "
        print(f" {marker} {name}: {template}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VNTR-map P1-P5 统一调用与可扩展流水线")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY, help="功能注册表 JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="列出所有已注册功能")
    list_parser.add_argument("--json", action="store_true", help="输出完整 JSON")

    describe_parser = subparsers.add_parser("describe", help="查看一个功能的依赖、参数和产物")
    describe_parser.add_argument("stage_id")

    stage_parser = subparsers.add_parser("stage", help="直接调用一个功能，并把后续参数原样传给它")
    stage_parser.add_argument("stage_id")
    stage_parser.add_argument("--python", default=sys.executable)
    stage_parser.add_argument("--dry-run", action="store_true")
    stage_parser.add_argument("arguments", nargs=argparse.REMAINDER)

    pipeline_parser = subparsers.add_parser("pipeline", help="按依赖运行一个或多个功能")
    pipeline_parser.add_argument("--input", type=Path, required=True, help="输入 smooth.final.gfa")
    pipeline_parser.add_argument("--outdir", type=Path, required=True, help="本次运行根目录")
    pipeline_parser.add_argument("--prefix", help="数据集输出前缀；默认由输入文件名生成")
    pipeline_parser.add_argument("--python", type=Path, default=Path(sys.executable), help="执行各阶段的 Python")
    selection = pipeline_parser.add_mutually_exclusive_group()
    selection.add_argument("--stages", help="逗号分隔的功能 ID，例如 P1,P2,P3")
    selection.add_argument("--from-stage", help="从此功能开始，默认运行到最后")
    pipeline_parser.add_argument("--to-stage", help="运行到此功能为止")
    pipeline_parser.add_argument("--read-evidence", type=Path, help="P4 可选逐 read 证据 TSV")
    pipeline_parser.add_argument("--external-evidence", type=Path, help="P1 可选外部边界证据 TSV")
    pipeline_parser.add_argument("--artifact", action="append", default=[], metavar="KEY=PATH", help="为非连续运行提供已有产物，如 P3.blocks=文件")
    pipeline_parser.add_argument("--set-param", action="append", default=[], metavar="STAGE.PARAM=VALUE", help="覆盖阶段参数，可重复")
    pipeline_parser.add_argument("--set-flag", action="append", default=[], metavar="STAGE.FLAG", help="追加无值开关，可重复")
    pipeline_parser.add_argument("--resume", action="store_true", help="必需产物已存在时跳过该阶段")
    pipeline_parser.add_argument("--dry-run", action="store_true", help="只生成并显示命令，不执行")
    pipeline_parser.add_argument("--keep-going", action="store_true", help="失败后继续尝试不受影响的后续功能")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    registry_path = args.registry.expanduser().resolve()
    try:
        if args.command == "list":
            registry, _order = load_registry(registry_path)
            print_stage_list(registry, args.json)
            return 0
        if args.command == "describe":
            registry, _order = load_registry(registry_path)
            by_id = stage_map(registry)
            stage_id = args.stage_id.upper()
            if stage_id not in by_id:
                raise LauncherError(f"未知功能: {stage_id}")
            print_stage_description(by_id[stage_id])
            return 0
        if args.command == "stage":
            return direct_stage(args, registry_path)
        if args.command == "pipeline":
            return pipeline(args, registry_path)
        raise LauncherError(f"未知命令: {args.command}")
    except LauncherError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
