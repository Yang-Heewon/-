#!/usr/bin/env bash
# 1-pass 변형 판: 앞의 MLP 판이 끝난 뒤, 이미지만 한 번 읽은 신호의 변형(최대 집계, 순위 정규화 MLP)을
# kvzip·attn1 기준과 함께 화면 172장 5%에서 비교. GPU 0·2.
set -u
cd /root/research/heewon/VLM
export GPUS="${GPUS:-0 2}" STAGGER="${STAGGER:-45}"
while pgrep -f "python -m vlm_diagnosis.exps.mlp_signal_prob[e]" >/dev/null || pgrep -f "^bash vlm_diagnosis/scripts/chain_mlp_signal.sh" >/dev/null; do sleep 60; done
TAG=mlp1p_q25_sqa
echo "[mlp1p] $TAG 시작: $(date +%H:%M)"
bash vlm_diagnosis/scripts/safe_launch.sh vlm_diagnosis.exps.mlp_signal_probe \
  results/smoke/${TAG}.jsonl --manifest experiments/manifests/screenqa_discovery.jsonl --model qwen25vl \
  --budgets 0.05 --signals kvzip,attn1,attn1_max,attn1_max_x_mlp,attn1_x_mlprank,attn1_max_x_mlprank
sleep 90
while pgrep -f "python -m vlm_diagnosis.exps.mlp_signal_prob[e]" >/dev/null; do sleep 60; done
kill $(cat /tmp/${TAG}_monitor.pid) 2>/dev/null
echo "[mlp1p] $TAG 완료: $(date +%H:%M)"
python -m vlm_diagnosis.scripts.mlp_signal_analysis --pattern "results/smoke/${TAG}.shard*.jsonl" \
  --out "results/smoke/${TAG}_summary" > "results/smoke/${TAG}_analysis.log" 2>&1
echo "[mlp1p] 분석 저장: results/smoke/${TAG}_summary.md"
