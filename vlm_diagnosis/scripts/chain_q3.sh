#!/usr/bin/env bash
# 두 번째 모델(Qwen3-VL-8B) 재현 연쇄: GQA→ScreenQA × 기준선(gate)→사다리→교차→held-out
set -u
cd /root/research/heewon/VLM
export GPUS="2 3" STAGGER=45

wait_mod () {  # $1=pgrep 패턴(브래킷), $2=monitor tag
  sleep 90
  while pgrep -f "$1" >/dev/null; do sleep 60; done
  kill $(cat /tmp/${2}_monitor.pid) 2>/dev/null
}

for DOM in gqa sqa; do
  if [ "$DOM" = gqa ]; then MAN=experiments/manifests/gqa_transfer.jsonl
  else MAN=experiments/manifests/screenqa_transfer.jsonl; fi

  echo "[q3] $DOM 기준선 시작: $(date +%H:%M)"
  bash vlm_diagnosis/scripts/safe_launch.sh vlm_diagnosis.scripts.base_accuracy \
    results/smoke/q3_${DOM}_base.jsonl --manifest "$MAN" --model qwen3vl
  wait_mod "python -m vlm_diagnosis.scripts.base_accurac[y]" "q3_${DOM}_base"
  EM=$(python3 -c "
import json,glob
n=ok=0
for f in glob.glob('results/smoke/q3_${DOM}_base.shard*.jsonl'):
    for l in open(f):
        try: r=json.loads(l)
        except Exception: continue
        n+=1; ok+=r.get('em',0)
print(f'{ok/max(n,1):.4f}')")
  echo "[q3] $DOM 기준선 EM=$EM"
  if ! python3 -c "import sys; sys.exit(0 if float('$EM')>=0.30 else 1)"; then
    echo "[q3] $DOM gate 실패 — 이후 단계 건너뜀"; continue
  fi

  echo "[q3] $DOM 사다리 시작: $(date +%H:%M)"
  bash vlm_diagnosis/scripts/safe_launch.sh vlm_diagnosis.exps.m2a_fixed_budget \
    results/smoke/q3_${DOM}_ladder.jsonl --manifest "$MAN" --model qwen3vl --budgets 0.05,0.2
  wait_mod "python -m vlm_diagnosis.exps.m2a_fixed_budge[t]" "q3_${DOM}_ladder"

  echo "[q3] $DOM 교차 시작: $(date +%H:%M)"
  bash vlm_diagnosis/scripts/safe_launch.sh vlm_diagnosis.exps.m3_pilot_cross_eval \
    results/smoke/q3_${DOM}_cross.jsonl --manifest "$MAN" --model qwen3vl --budgets 0.05,0.2
  wait_mod "python -m vlm_diagnosis.exps.m3_pilot_cross_eva[l]" "q3_${DOM}_cross"

  echo "[q3] $DOM held-out 시작: $(date +%H:%M)"
  bash vlm_diagnosis/scripts/safe_launch.sh vlm_diagnosis.exps.m3_pilot_cross_eval \
    results/smoke/q3_${DOM}_heldout.jsonl --manifest "$MAN" --model qwen3vl \
    --eval-mode heldout --budgets 0.05,0.2
  wait_mod "python -m vlm_diagnosis.exps.m3_pilot_cross_eva[l]" "q3_${DOM}_heldout"
  echo "[q3] $DOM 전체 완료: $(date +%H:%M)"
done
echo "[q3] 연쇄 종료: $(date +%H:%M)"
