#!/usr/bin/env bash
# Core–Delta 누적 선택 판 (core_delta_incremental.py): 같은 화면에 질문 4개가 차례로 올 때,
# keep(20% 유지·재선택) / grow(20%씩 추가: 20→40→60→80%) × alpha {1,.75,.5,.25,0}. 전체 KV는 DRAM.
# GPU 2·3만 사용. smoke 통과 후 4판 순차 실행, 판마다 분석.
#
#   bash vlm_diagnosis/scripts/chain_core_delta_incremental.sh
#   PANELS="q25:sqa" bash vlm_diagnosis/scripts/chain_core_delta_incremental.sh
set -u
cd /root/research/heewon/VLM
export GPUS="${GPUS:-2 3}" STAGGER="${STAGGER:-45}"
STEPS="${STEPS:-4}"
BSTEP="${BSTEP:-0.2}"
ALPHAS="${ALPHAS:-1,0.75,0.5,0.25,0}"
SCHEMES="${SCHEMES:-keep,grow}"
SIGNAL="${SIGNAL:-kvzip}"
HIST="${HIST:-episode}"
PANELS="${PANELS:-q25:sqa q25:gqa q3:sqa q3:gqa}"
TAGSUFFIX="${TAGSUFFIX:-}"
ARGS=(--steps "$STEPS" --budget-step "$BSTEP" --alphas "$ALPHAS" --schemes "$SCHEMES"
      --image-signal "$SIGNAL" --hist-rows "$HIST")

wait_mod () {  # $1=pgrep 패턴(브래킷), $2=monitor tag
  sleep 90
  while pgrep -f "$1" >/dev/null; do sleep 60; done
  kill $(cat /tmp/${2}_monitor.pid) 2>/dev/null
}

echo "[inc] smoke 시작: $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=2 python -m vlm_diagnosis.exps.core_delta_incremental --limit 1 --device cuda:0 \
  "${ARGS[@]}" --out results/smoke/inc_smoke.jsonl > results/smoke/inc_smoke.log 2>&1
if [ $? -ne 0 ] || ! grep -q "\[saved\]" results/smoke/inc_smoke.log; then
  echo "[inc] smoke 실패 — 연쇄 중단 (results/smoke/inc_smoke.log 확인)"; exit 1
fi
echo "[inc] smoke 통과: $(date +%H:%M)"

for P in $PANELS; do
  M=${P%%:*}; DOM=${P##*:}
  case "$M" in q25) MODEL=qwen25vl ;; q3) MODEL=qwen3vl ;; *) echo "bad model $M"; exit 2 ;; esac
  case "$DOM" in
    sqa) MAN=experiments/manifests/screenqa_discovery.jsonl ;;
    gqa) MAN=experiments/manifests/gqa_discovery.jsonl ;;
    *) echo "bad domain $DOM"; exit 2 ;;
  esac
  TAG="inc_${M}_${DOM}${TAGSUFFIX}"
  echo "[inc] $TAG 시작: $(date +%H:%M)  (model=$MODEL steps=$STEPS step=$BSTEP alphas=$ALPHAS schemes=$SCHEMES signal=$SIGNAL)"
  bash vlm_diagnosis/scripts/safe_launch.sh vlm_diagnosis.exps.core_delta_incremental \
    results/smoke/${TAG}.jsonl --manifest "$MAN" --model "$MODEL" "${ARGS[@]}"
  wait_mod "python -m vlm_diagnosis.exps.core_delta_incrementa[l]" "$TAG"
  echo "[inc] $TAG 완료: $(date +%H:%M)"
  python -m vlm_diagnosis.scripts.core_delta_incremental_analysis \
    --pattern "results/smoke/${TAG}.shard*.jsonl" --out "results/smoke/${TAG}_summary" \
    --title "Core–Delta 누적 선택 판 — ${M} × ${DOM}" > "results/smoke/${TAG}_analysis.log" 2>&1
  echo "[inc] $TAG 분석 저장: results/smoke/${TAG}_summary.md"
done
echo "[inc] 연쇄 종료: $(date +%H:%M)"
