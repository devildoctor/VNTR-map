#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INPUT="${1:-测试数据/IRF2BPL.HPRC.fa.3034e58.11fba48.904d69e.smooth.final.gfa}"
P1_BOUNDARIES="${2:-outputs/IRF2BPL/20260825_P1唯一侧翼锚点与共识边界/20260825_IRF2BPL_P1唯一侧翼锚点与共识边界_路径边界共识.tsv}"
P2_DIR="${3:-outputs/IRF2BPL/20260825_P2从头发现主体motif与MDL分解}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ROOT="${4:-outputs/IRF2BPL/20260825_linux_P3_full_validation/$RUN_ID}"
PYTHON_BIN="${PYTHON_BIN:-python}"
P2_PREFIX="20260825_IRF2BPL_P2从头发现主体motif与MDL分解"
PREFIX="server_P3"

mkdir -p "$RUN_ROOT"

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
    printf 'model_status\tassembly_only_probabilistic\n'
} > "$RUN_ROOT/software_versions.tsv"

"$PYTHON_BIN" -u "程序/20260825_P3半马尔可夫概率分解与置信度.py" \
    --input "$INPUT" \
    --p1-boundaries "$P1_BOUNDARIES" \
    --p2-dictionary "$P2_DIR/${P2_PREFIX}_共享motif字典.tsv" \
    --p2-tokens "$P2_DIR/${P2_PREFIX}_逐路径motif_token.tsv" \
    --p2-blocks "$P2_DIR/${P2_PREFIX}_逐路径repeat区块.tsv" \
    --outdir "$RUN_ROOT/P3" \
    --prefix "$PREFIX" \
    2>&1 | tee "$RUN_ROOT/P3.log"

"$PYTHON_BIN" -m unittest "tests/20260825_P3半马尔可夫回归测试.py" -v \
    2>&1 | tee "$RUN_ROOT/tests.log"

QA="$RUN_ROOT/P3/${PREFIX}_P3验证汇总.tsv"
require_metric "$QA" path_count 47
require_metric "$QA" training_path_count 28
require_metric "$QA" tuning_path_count 9
require_metric "$QA" heldout_path_count 10
require_metric "$QA" selected_duration_weight 1.0000
require_metric "$QA" repeat_block_count 141
require_metric "$QA" modal_repeat_block_count_per_path 3
require_metric "$QA" paths_with_modal_repeat_block_count 47
require_metric "$QA" posterior_available_paths 47
require_metric "$QA" second_best_available_paths 47
require_metric "$QA" explicit_loop_positions R1,R2,R3
require_metric "$QA" exact_gfa_reconstruction_paths 47

find "$RUN_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/SHA256SUMS"

printf 'PASS: P3 hidden semi-Markov full validation completed\n'
printf 'Results: %s\n' "$RUN_ROOT"
