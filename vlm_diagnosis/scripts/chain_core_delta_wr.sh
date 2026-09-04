#!/usr/bin/env bash
# Dual-prefill 쓰기/읽기 판: image-only core(C) + image/text-prefix delta(D) 격자.
# GPU 2·3만 사용 (safe_launch 정책), 판마다 순차 실행 후 분석.
#
#   bash vlm_diagnosis/scripts/chain_core_delta_wr.sh
#   PANELS="q25:sqa" bash vlm_diagnosis/scripts/chain_core_delta_wr.sh
set -u
cd /root/research/heewon/VLM
export GPUS="${GPUS:-2 3}" STAGGER="${STAGGER:-45}"
CORES="${CORES:-0,0.025,0.05,0.1}"
DELTAS="${DELTAS:-0,0.01,0.025,0.05}"
PANELS="${PANELS:-q25:sqa q25:gqa q3:sqa q3:gqa}"
TAGSUFFIX="${TAGSUFFIX:-}"
EXTRA="${EXTRA:-}"

wait_mod () {  # $1=pgrep 패턴(브래킷), $2=monitor tag
  sleep 90
  while pgrep -f "$1" >/dev/null; do sleep 60; done
  kill $(cat /tmp/${2}_monitor.pid) 2>/dev/null
}

for P in $PANELS; do
  M=${P%%:*}; DOM=${P##*:}
  case "$M" in q25) MODEL=qwen25vl ;; q3) MODEL=qwen3vl ;; *) echo "bad model $M"; exit 2 ;; esac
  case "$DOM" in
    sqa) MAN=experiments/manifests/screenqa_discovery.jsonl ;;
    gqa) MAN=experiments/manifests/gqa_discovery.jsonl ;;
    *) echo "bad domain $DOM"; exit 2 ;;
  esac
  TAG="wr_${M}_${DOM}${TAGSUFFIX}"
  echo "[wr] $TAG 시작: $(date +%H:%M)  (model=$MODEL C=$CORES D=$DELTAS extra=$EXTRA)"
  bash vlm_diagnosis/scripts/safe_launch.sh vlm_diagnosis.exps.core_delta_write_read \
    results/smoke/${TAG}.jsonl --manifest "$MAN" --model "$MODEL" \
    --core-sizes "$CORES" --delta-sizes "$DELTAS" $EXTRA
  wait_mod "python -m vlm_diagnosis.exps.core_delta_write_rea[d]" "$TAG"
  echo "[wr] $TAG 완료: $(date +%H:%M)"
  python -m vlm_diagnosis.scripts.core_delta_wr_analysis \
    --pattern "results/smoke/${TAG}.shard*.jsonl" --out "results/smoke/${TAG}_summary" \
    --title "Dual-prefill 쓰기/읽기 — ${M} × ${DOM}" > "results/smoke/${TAG}_analysis.log" 2>&1
  echo "[wr] $TAG 분석 저장: results/smoke/${TAG}_summary.md"
done
echo "[wr] 연쇄 종료: $(date +%H:%M)"
