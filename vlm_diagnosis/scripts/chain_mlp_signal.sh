#!/usr/bin/env bash
# MLP 신호 probe: 화면 172장(Qwen2.5), 5% 예산, 신호 9개 비교. GPU는 GPUS(기본 "0 2").
#   GPUS="0 2" bash vlm_diagnosis/scripts/chain_mlp_signal.sh
set -u
cd /root/research/heewon/VLM
export GPUS="${GPUS:-0 2}" STAGGER="${STAGGER:-45}"
PANELS="${PANELS:-q25:sqa}"
BUDGETS="${BUDGETS:-0.05}"
for P in $PANELS; do
  M=${P%%:*}; DOM=${P##*:}
  case "$M" in q25) MODEL=qwen25vl ;; q3) MODEL=qwen3vl ;; *) exit 2 ;; esac
  case "$DOM" in sqa) MAN=experiments/manifests/screenqa_discovery.jsonl ;; gqa) MAN=experiments/manifests/gqa_discovery.jsonl ;; *) exit 2 ;; esac
  TAG="mlp_${M}_${DOM}"
  echo "[mlp] $TAG 시작: $(date +%H:%M)"
  bash vlm_diagnosis/scripts/safe_launch.sh vlm_diagnosis.exps.mlp_signal_probe \
    results/smoke/${TAG}.jsonl --manifest "$MAN" --model "$MODEL" --budgets "$BUDGETS"
  sleep 90
  while pgrep -f "python -m vlm_diagnosis.exps.mlp_signal_prob[e]" >/dev/null; do sleep 60; done
  kill $(cat /tmp/${TAG}_monitor.pid) 2>/dev/null
  echo "[mlp] $TAG 완료: $(date +%H:%M)"
  python -m vlm_diagnosis.scripts.mlp_signal_analysis --pattern "results/smoke/${TAG}.shard*.jsonl" \
    --out "results/smoke/${TAG}_summary" > "results/smoke/${TAG}_analysis.log" 2>&1
  echo "[mlp] $TAG 분석 저장: results/smoke/${TAG}_summary.md"
done
echo "[mlp] 종료: $(date +%H:%M)"
