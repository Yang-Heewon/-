#!/usr/bin/env bash
# LEGACY reconstruction/S1 visual-only sweep. 현재 dual-prefill 실험은
# chain_core_delta_fullkv.sh를 사용한다. 아래 경로는 과거 결과 재현용이다.
# GPU 2·3만 사용 (safe_launch 정책), 판마다 순차 실행 후 분석 스크립트로 판정 파일 생성.
#
#   bash vlm_diagnosis/scripts/chain_core_delta.sh                 # 4판 전부, 5%
#   BUDGETS=0.02,0.05,0.1 PANELS="q25:sqa" bash .../chain_core_delta.sh   # 일부·예산 변경
set -u
cd /root/research/heewon/VLM
export GPUS="${GPUS:-2 3}" STAGGER="${STAGGER:-45}"
BUDGETS="${BUDGETS:-0.05}"
ALPHAS="${ALPHAS:-0,0.1,0.25,0.5,0.75,1}"
PANELS="${PANELS:-q25:sqa q25:gqa q3:sqa q3:gqa}"
TAGSUFFIX="${TAGSUFFIX:-}"
EXTRA="${EXTRA:---wsum}"

wait_mod () {  # $1=pgrep 패턴(브래킷), $2=monitor tag
  sleep 90
  while pgrep -f "$1" >/dev/null; do sleep 60; done
  kill $(cat /tmp/${2}_monitor.pid) 2>/dev/null
}

for P in $PANELS; do
  M=${P%%:*}; DOM=${P##*:}
  case "$M" in q25) MODEL=qwen25vl ;; q3) MODEL=qwen3vl ;; *) echo "bad model $M"; exit 2 ;; esac
  # D6(원저 정면비교) 동결 4판과 같은 승격 표본: discovery manifest (SQA 172장 / GQA 300장)
  case "$DOM" in
    sqa) MAN=experiments/manifests/screenqa_discovery.jsonl ;;
    gqa) MAN=experiments/manifests/gqa_discovery.jsonl ;;
    *) echo "bad domain $DOM"; exit 2 ;;
  esac
  TAG="cd_${M}_${DOM}${TAGSUFFIX}"
  echo "[cd] $TAG 시작: $(date +%H:%M)  (model=$MODEL budgets=$BUDGETS alphas=$ALPHAS)"
  bash vlm_diagnosis/scripts/safe_launch.sh vlm_diagnosis.exps.core_delta_sweep \
    results/smoke/${TAG}.jsonl --manifest "$MAN" --model "$MODEL" \
    --budgets "$BUDGETS" --alphas "$ALPHAS" $EXTRA
  wait_mod "python -m vlm_diagnosis.exps.core_delta_swee[p]" "$TAG"
  echo "[cd] $TAG 완료: $(date +%H:%M)"
  python -m vlm_diagnosis.scripts.core_delta_analysis \
    --pattern "results/smoke/${TAG}.shard*.jsonl" --out "results/smoke/${TAG}_summary" \
    --title "Core–Delta Phase A — ${M} × ${DOM} (budgets ${BUDGETS})" \
    > "results/smoke/${TAG}_analysis.log" 2>&1
  echo "[cd] $TAG 분석 저장: results/smoke/${TAG}_summary.md"
done
echo "[cd] 연쇄 종료: $(date +%H:%M)"
