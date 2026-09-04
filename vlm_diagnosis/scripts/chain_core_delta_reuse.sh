#!/usr/bin/env bash
# Core–Delta 재사용 판 (core_delta_reuse.py): 쓰기 시점에 '이미지만' + '이미지+과거 질문' 두 기준으로
# 예산 B만 남기고 삭제, 새 질문 q1..q3으로 평가 (KVzip 규약). GPU 2·3만 사용. smoke 통과 후 4판 순차.
#
#   bash vlm_diagnosis/scripts/chain_core_delta_reuse.sh
#   PANELS="q25:sqa" BUDGETS=0.05,0.1 bash vlm_diagnosis/scripts/chain_core_delta_reuse.sh
set -u
cd /root/research/heewon/VLM
export GPUS="${GPUS:-2 3}" STAGGER="${STAGGER:-45}"
BUDGETS="${BUDGETS:-0.05,0.1,0.2}"
ALPHAS="${ALPHAS:-0,0.25,0.5,0.75,1}"
SIGNALS="${SIGNALS:-kvzip,image}"
TOK="${TOK:-0.1:0.5,0.1:1}"
PAST_ROWS="${PAST_ROWS:-episode}"
PANELS="${PANELS:-q25:sqa q25:gqa q3:sqa q3:gqa}"
TAGSUFFIX="${TAGSUFFIX:-}"
ARGS=(--budgets "$BUDGETS" --alphas "$ALPHAS" --image-signals "$SIGNALS" --token-cells "$TOK" --past-rows "$PAST_ROWS")

wait_mod () {  # $1=pgrep 패턴(브래킷), $2=monitor tag
  sleep 90
  while pgrep -f "$1" >/dev/null; do sleep 60; done
  kill $(cat /tmp/${2}_monitor.pid) 2>/dev/null
}

echo "[reuse] smoke 시작: $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=2 python -m vlm_diagnosis.exps.core_delta_reuse --limit 1 --device cuda:0 \
  "${ARGS[@]}" --out results/smoke/ru_smoke.jsonl > results/smoke/ru_smoke.log 2>&1
if [ $? -ne 0 ] || ! grep -q "\[saved\]" results/smoke/ru_smoke.log; then
  echo "[reuse] smoke 실패 — 연쇄 중단 (results/smoke/ru_smoke.log 확인)"; exit 1
fi
echo "[reuse] smoke 통과: $(date +%H:%M)"

for P in $PANELS; do
  M=${P%%:*}; DOM=${P##*:}
  case "$M" in q25) MODEL=qwen25vl ;; q3) MODEL=qwen3vl ;; *) echo "bad model $M"; exit 2 ;; esac
  case "$DOM" in
    sqa) MAN=experiments/manifests/screenqa_discovery.jsonl ;;
    gqa) MAN=experiments/manifests/gqa_discovery.jsonl ;;
    *) echo "bad domain $DOM"; exit 2 ;;
  esac
  TAG="ru_${M}_${DOM}${TAGSUFFIX}"
  echo "[reuse] $TAG 시작: $(date +%H:%M)  (model=$MODEL B=$BUDGETS alphas=$ALPHAS signals=$SIGNALS)"
  bash vlm_diagnosis/scripts/safe_launch.sh vlm_diagnosis.exps.core_delta_reuse \
    results/smoke/${TAG}.jsonl --manifest "$MAN" --model "$MODEL" "${ARGS[@]}"
  wait_mod "python -m vlm_diagnosis.exps.core_delta_reus[e]" "$TAG"
  echo "[reuse] $TAG 완료: $(date +%H:%M)"
  python -m vlm_diagnosis.scripts.core_delta_reuse_analysis \
    --pattern "results/smoke/${TAG}.shard*.jsonl" --out "results/smoke/${TAG}_summary" \
    --title "Core–Delta 재사용 판 — ${M} × ${DOM}" > "results/smoke/${TAG}_analysis.log" 2>&1
  echo "[reuse] $TAG 분석 저장: results/smoke/${TAG}_summary.md"
done
echo "[reuse] 연쇄 종료: $(date +%H:%M)"
