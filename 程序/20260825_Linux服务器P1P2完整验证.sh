#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INPUT="${1:-测试数据/IRF2BPL.HPRC.fa.3034e58.11fba48.904d69e.smooth.final.gfa}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="${2:-outputs/IRF2BPL/20260825_linux_P1P2_full_validation/$RUN_ID}"
PYTHON_BIN="${PYTHON_BIN:-python}"
P1_OUT="$RUN_ROOT/P1"
P2_OUT="$RUN_ROOT/P2"
P1_PREFIX="server_P1"
P2_PREFIX="server_P2"

mkdir -p "$P1_OUT" "$P2_OUT"

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
    printf 'git_commit\t%s\n' "$(git rev-parse HEAD)"
    printf 'git_branch\t%s\n' "$(git branch --show-current)"
    printf 'python\t%s\n' "$($PYTHON_BIN --version 2>&1)"
    printf 'matplotlib\t%s\n' "$($PYTHON_BIN -c 'import matplotlib; print(matplotlib.__version__)')"
    if command -v conda >/dev/null 2>&1; then
        printf 'conda_prefix\t%s\n' "${CONDA_PREFIX:-not-active}"
        if conda env list | awk '{print $1}' | grep -qx pggb; then
            printf 'pggb\t%s\n' "$(conda run -n pggb pggb --version 2>&1 | tail -n 1)"
        fi
    fi
} > "$RUN_ROOT/software_versions.tsv"

"$PYTHON_BIN" -u "程序/20260825_P1唯一侧翼锚点与共识边界.py" \
    --input "$INPUT" \
    --outdir "$P1_OUT" \
    --prefix "$P1_PREFIX" \
    2>&1 | tee "$RUN_ROOT/P1.log"

"$PYTHON_BIN" -u "程序/20260825_P2从头发现主体motif与MDL分解.py" \
    --input "$INPUT" \
    --p1-boundaries "$P1_OUT/${P1_PREFIX}_路径边界共识.tsv" \
    --outdir "$P2_OUT" \
    --prefix "$P2_PREFIX" \
    2>&1 | tee "$RUN_ROOT/P2.log"

"$PYTHON_BIN" -m unittest \
    "tests/20260825_P1边界回归测试.py" \
    "tests/20260825_P2从头motif回归测试.py" \
    -v 2>&1 | tee "$RUN_ROOT/tests.log"

P1_QA="$P1_OUT/${P1_PREFIX}_P1验证汇总.tsv"
P2_QA="$P2_OUT/${P2_PREFIX}_P2验证汇总.tsv"

require_metric "$P1_QA" path_count 47
require_metric "$P1_QA" unique_left_anchor_paths 47
require_metric "$P1_QA" unique_right_anchor_paths 47
require_metric "$P1_QA" exact_locus_reconstruction_paths 47

require_metric "$P2_QA" path_count 47
require_metric "$P2_QA" selected_primitive_family_count 2
require_metric "$P2_QA" selected_periods_bp 3,3
require_metric "$P2_QA" path_order_stable_dictionary 1
require_metric "$P2_QA" repeat_block_count 141
require_metric "$P2_QA" modal_repeat_block_count_per_path 3
require_metric "$P2_QA" paths_with_modal_repeat_block_count 47
require_metric "$P2_QA" explicit_loop_positions R1,R2,R3
require_metric "$P2_QA" exact_gfa_reconstruction_paths 47

find "$RUN_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/SHA256SUMS"

printf 'PASS: P1 and P2 full validation completed\n'
printf 'Results: %s\n' "$RUN_ROOT"
