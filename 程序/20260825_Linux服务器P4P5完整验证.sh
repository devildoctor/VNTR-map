#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

P3_DIR="${1:-outputs/IRF2BPL/20260825_P3半马尔可夫概率分解与置信度}"
P3_PREFIX="${2:-20260825_IRF2BPL_P3半马尔可夫概率分解与置信度}"
READ_EVIDENCE="${3:-}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="${4:-outputs/IRF2BPL/20260825_linux_P4P5_full_validation/$RUN_ID}"
PYTHON_BIN="${PYTHON_BIN:-python}"
P4_PREFIX="server_P4"
P5_PREFIX="server_P5"

mkdir -p "$RUN_ROOT/P4" "$RUN_ROOT/P5"

metric() {
    awk -F '\t' -v key="$2" '$1 == key {print $2}' "$1"
}

require_metric() {
    local file="$1"
    local key="$2"
    local expected="$3"
    local observed
    observed="$(metric "$file" "$key")"
    if [[ "$observed" != "$expected" ]]; then
        printf 'FAIL %s: expected %s, observed %s\n' "$key" "$expected" "${observed:-missing}" >&2
        exit 1
    fi
    printf 'PASS %s=%s\n' "$key" "$observed"
}

{
    printf 'run_id\t%s\n' "$RUN_ID"
    printf 'git_commit\t%s\n' "$(git -c safe.directory="$ROOT" rev-parse HEAD)"
    printf 'git_branch\t%s\n' "$(git -c safe.directory="$ROOT" branch --show-current)"
    printf 'python\t%s\n' "$($PYTHON_BIN --version 2>&1)"
    printf 'matplotlib\t%s\n' "$($PYTHON_BIN -c 'import matplotlib; print(matplotlib.__version__)')"
    printf 'read_evidence\t%s\n' "${READ_EVIDENCE:-none}"
} > "$RUN_ROOT/software_versions.tsv"

P4_ARGS=(
    --p3-gfa "$P3_DIR/${P3_PREFIX}_P3半马尔可夫多环图.gfa"
    --p3-tokens "$P3_DIR/${P3_PREFIX}_逐路径token后验.tsv"
    --p3-blocks "$P3_DIR/${P3_PREFIX}_逐路径P3_repeat区块.tsv"
    --outdir "$RUN_ROOT/P4"
    --prefix "$P4_PREFIX"
)
if [[ -n "$READ_EVIDENCE" ]]; then
    P4_ARGS+=(--read-evidence "$READ_EVIDENCE")
fi

"$PYTHON_BIN" -u "程序/20260825_P4变异read证据分级.py" "${P4_ARGS[@]}" \
    2>&1 | tee "$RUN_ROOT/P4.log"

"$PYTHON_BIN" -u "程序/20260825_P5位置特异SCC循环图.py" \
    --input-gfa "$RUN_ROOT/P4/${P4_PREFIX}_P4变异证据标注图.gfa" \
    --p3-blocks "$P3_DIR/${P3_PREFIX}_逐路径P3_repeat区块.tsv" \
    --outdir "$RUN_ROOT/P5" \
    --prefix "$P5_PREFIX" \
    2>&1 | tee "$RUN_ROOT/P5.log"

"$PYTHON_BIN" "tests/20260825_P4P5回归测试.py" \
    2>&1 | tee "$RUN_ROOT/tests.log"

P4_QA="$RUN_ROOT/P4/${P4_PREFIX}_P4验证汇总.tsv"
require_metric "$P4_QA" path_count 47
require_metric "$P4_QA" assembly_event_count 755
require_metric "$P4_QA" unknown_read_evidence_event_count 0
require_metric "$P4_QA" exact_gfa_reconstruction_paths 47
if [[ -z "$READ_EVIDENCE" ]]; then
    require_metric "$P4_QA" assembly_only_unvalidated_event_count 755
    require_metric "$P4_QA" interpretation_status assembly_only_unvalidated
fi

P5_QA="$RUN_ROOT/P5/${P5_PREFIX}_P5验证汇总.tsv"
require_metric "$P5_QA" path_count 47
require_metric "$P5_QA" node_count 34
require_metric "$P5_QA" path_supported_edge_count 47
require_metric "$P5_QA" unsupported_edge_count 0
require_metric "$P5_QA" cyclic_scc_count 3
require_metric "$P5_QA" repeat_locations R1,R2,R3
require_metric "$P5_QA" repeat_locations_with_cyclic_scc R1,R2,R3
require_metric "$P5_QA" repeat_locations_without_cyclic_scc -
require_metric "$P5_QA" cross_position_cycle_count 0
require_metric "$P5_QA" condensation_is_DAG 1
require_metric "$P5_QA" paths_with_backward_repeat_order 0
require_metric "$P5_QA" exact_gfa_reconstruction_paths 47

find "$RUN_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/SHA256SUMS"

printf 'PASS: P4/P5 full validation completed\n'
printf 'Results: %s\n' "$RUN_ROOT"
