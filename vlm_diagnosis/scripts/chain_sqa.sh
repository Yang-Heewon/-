#!/usr/bin/env bash
# ScreenQA 실험 연쇄 실행: held-out → 사다리 (GPU 2·3, 안전 정책 적용)
set -u
cd /root/research/heewon/VLM
export GPUS="2 3" STAGGER=45

echo "[chain] 1/2 held-out 시작: $(date +%H:%M)"
bash vlm_diagnosis/scripts/safe_launch.sh vlm_diagnosis.exps.m3_pilot_cross_eval \
  results/smoke/sqa_heldout.jsonl \
  --manifest experiments/manifests/screenqa_transfer.jsonl \
  --eval-mode heldout --budgets 0.05,0.2
sleep 90
while pgrep -f "exps.m3_pilot_cross_eva[l]" >/dev/null; do sleep 60; done
kill $(cat /tmp/sqa_heldout_monitor.pid) 2>/dev/null
echo "[chain] held-out 완료: $(date +%H:%M)"

echo "[chain] 2/2 사다리 시작: $(date +%H:%M)"
bash vlm_diagnosis/scripts/safe_launch.sh vlm_diagnosis.exps.m2a_fixed_budget \
  results/smoke/sqa_ladder.jsonl \
  --manifest experiments/manifests/screenqa_transfer.jsonl --budgets 0.05,0.2
sleep 90
while pgrep -f "exps.m2a_fixed_budge[t]" >/dev/null; do sleep 60; done
kill $(cat /tmp/sqa_ladder_monitor.pid) 2>/dev/null
echo "[chain] 사다리 완료: $(date +%H:%M)"
