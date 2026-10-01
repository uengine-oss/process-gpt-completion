#!/usr/bin/env bash
# 취약점 점검 — 로컬과 GitHub Actions(.github/workflows/security-scan.yml)가 같은 스크립트를 쓴다.
#
#   scripts/security/scan.sh [deps|secrets|sast|iac|all]   (기본: all)
#
# 게이트: Critical/High 가 하나라도 있으면 종료 코드 1. 결과 파일: reports/security/
# 필요 도구: bash, docker
#
# 환경변수
#   SECRETS_LOG_OPTS  gitleaks 가 이력에서 검사할 커밋 범위(git log 옵션). 비우면 작업 트리만 검사.
# `sh scan.sh` 처럼 bash 가 아닌 셸로 실행되면 bash 로 다시 실행한다 (배열 등 bash 문법 사용).
# 이 줄까지는 POSIX sh 문법만 쓴다.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/reports/security"
mkdir -p "$OUT"

# 스캐너 이미지는 버전을 고정한다 (태그 변조·예고 없는 동작 변경 방지).
TRIVY_IMAGE="aquasec/trivy:0.74.0"
GITLEAKS_IMAGE="zricethezav/gitleaks:v8.30.1"
SEMGREP_IMAGE="semgrep/semgrep:1.177.0"
TRIVY_CACHE="${TRIVY_CACHE:-$ROOT/.cache/trivy}"
mkdir -p "$TRIVY_CACHE"

log() { printf '\n\033[1;36m[security] %s\033[0m\n' "$*"; }

# git 이 추적하는(또는 추적 대상인) 파일만 임시 디렉터리로 복사 — .venv·.env 등 무시 파일 제외
tracked_tree() {
    local dir
    dir="$(mktemp -d)"
    (cd "$ROOT" && git ls-files -z --cached --others --exclude-standard | xargs -0 tar -cf - 2>/dev/null) | tar -xf - -C "$dir" 2>/dev/null || true
    echo "$dir"
}

run_deps() {
    log "SCA: Python(uv.lock · requirements.txt) / npm 의존성 (Trivy)"
    local tree rc=0
    tree="$(tracked_tree)"
    # uv.lock 은 .gitignore 대상이지만 실제 실행(uv run)이 쓰는 해석 결과라 함께 검사한다
    [ -f "$ROOT/uv.lock" ] && cp "$ROOT/uv.lock" "$tree/"
    docker run --rm -v "$tree:/src:ro" -v "$TRIVY_CACHE:/root/.cache" -v "$OUT:/out" "$TRIVY_IMAGE" \
        fs --scanners vuln --ignorefile /src/.trivyignore.yaml -q \
        --format sarif --output /out/trivy-deps.sarif /src || true
    docker run --rm -v "$tree:/src:ro" -v "$TRIVY_CACHE:/root/.cache" "$TRIVY_IMAGE" \
        fs --scanners vuln --ignorefile /src/.trivyignore.yaml -q \
        --severity CRITICAL,HIGH --exit-code 1 /src || rc=$?
    rm -rf "$tree"
    return $rc
}

run_secrets() {
    log "Secrets: gitleaks (작업 트리${SECRETS_LOG_OPTS:+ + 커밋 범위 $SECRETS_LOG_OPTS})"
    local tree rc=0
    tree="$(tracked_tree)"
    docker run --rm -v "$tree:/repo:ro" -v "$ROOT/.gitleaks.toml:/config.toml:ro" -v "$OUT:/out" "$GITLEAKS_IMAGE" \
        dir /repo --config /config.toml --no-banner --redact --max-target-megabytes 5 \
        --report-format sarif --report-path /out/gitleaks-tree.sarif --exit-code 1 || rc=$?
    rm -rf "$tree"
    if [ -n "${SECRETS_LOG_OPTS:-}" ]; then
        docker run --rm -v "$ROOT:/repo:ro" -v "$ROOT/.gitleaks.toml:/config.toml:ro" -v "$OUT:/out" "$GITLEAKS_IMAGE" \
            git /repo --config /config.toml --no-banner --redact --log-opts="$SECRETS_LOG_OPTS" \
            --report-format sarif --report-path /out/gitleaks-commits.sarif --exit-code 1 || rc=$?
    fi
    return $rc
}

run_sast() {
    log "SAST: Semgrep (KISA Python 룰 + p/python · p/javascript, ERROR 등급 차단)"
    local common=(--metrics off --config semgrep-rules --config p/python --config p/javascript
        --exclude .venv --exclude node_modules --exclude frontend --exclude tests --exclude reports
        --timeout 30 --quiet)
    docker run --rm -v "$ROOT:/src" -w /src "$SEMGREP_IMAGE" \
        semgrep scan "${common[@]}" --sarif --output reports/security/semgrep.sarif . >/dev/null || true
    local rc=0
    docker run --rm -v "$ROOT:/src" -w /src "$SEMGREP_IMAGE" \
        semgrep scan "${common[@]}" --severity ERROR --error . || rc=$?
    return $rc
}

run_iac() {
    log "IaC: Dockerfile / Kubernetes (Trivy config)"
    local tree rc=0
    tree="$(tracked_tree)"
    docker run --rm -v "$tree:/src:ro" -v "$TRIVY_CACHE:/root/.cache" -v "$OUT:/out" "$TRIVY_IMAGE" \
        config -q --format sarif --output /out/trivy-iac.sarif /src || true
    docker run --rm -v "$tree:/src:ro" -v "$TRIVY_CACHE:/root/.cache" "$TRIVY_IMAGE" \
        config -q --severity CRITICAL,HIGH --exit-code 1 /src || rc=$?
    rm -rf "$tree"
    return $rc
}

target="${1:-all}"
case "$target" in
    deps|secrets|sast|iac) "run_$target" ;;
    all)
        failed=()
        for t in deps secrets sast iac; do
            "run_$t" || failed+=("$t")
        done
        if [ ${#failed[@]} -gt 0 ]; then
            log "FAIL — Critical/High 발견: ${failed[*]}"
            exit 1
        fi
        log "PASS — 모든 영역 Critical/High 0건"
        ;;
    *) echo "usage: $0 [deps|secrets|sast|iac|all]" >&2; exit 2 ;;
esac
