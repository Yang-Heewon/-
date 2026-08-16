#!/usr/bin/env bash
# GQA(자연 이미지) 실험 연쇄: 사다리 종료 대기 → 기준선(gate) → 교차 → held-out
# GPU 2·3, 전력 180W 캡, 45초 stagger (safe_launch 기본 정책)
set -u
cd /root/research/heewon/VLM
export GPUS="2 3" STAGGER=45
MAN=experiments/manifests/gqa_transfer.jsonl

echo "[chain-gqa] 사다리 종료 대기 시작: $(date +%H:%M)"
while pgrep -f "python -m vlm_diagnosis.exps.m2a_fixed_budge[t]" >/dev/null; do sleep 60; done
kill $(cat /tmp/sqa_ladder_monitor.pid) 2>/dev/null
echo "[chain-gqa] 사다리 종료 확인: $(date +%H:%M)"
sleep 30

echo "[chain-gqa] 1/3 기준선(base_accuracy) 시작: $(date +%H:%M)"
bash vlm_diagnosis/scripts/safe_launch.sh vlm_diagnosis.scripts.base_accuracy \
  results/smoke/gqa_base.jsonl --manifest "$MAN"
sleep 90
while pgrep -f "python -m vlm_diagnosis.scripts.base_accurac[y]" >/dev/null; do sleep 60; done
kill $(cat /tmp/gqa_base_monitor.pid) 2>/dev/null

# M0-A gate: 전체 EM이 0.35 미만이면 이후 단계 낭비이므로 중단
EM=$(python3 - <<'PY'
import json, glob
n=ok=0
for f in glob.glob('results/smoke/gqa_base.shard*.jsonl'):
    for l in open(f):
        try: r=json.loads(l)
        except Exception: continue
        n+=1; ok+=r.get('em',0)
print(f"{ok/max(n,1):.4f} {n}")
PY
)
echo "[chain-gqa] 기준선 EM=$EM : $(date +%H:%M)"
if python3 -c "import sys; sys.exit(0 if float('$EM'.split()[0]) >= 0.35 else 1)"; then
  echo "[chain-gqa] gate 통과"
else
  echo "[chain-gqa] gate 실패 (EM<0.35) — 교차/held-out 실행하지 않음"; exit 1
fi

echo "[chain-gqa] 2/3 교차(cross) 시작: $(date +%H:%M)"
bash vlm_diagnosis/scripts/safe_launch.sh vlm_diagnosis.exps.m3_pilot_cross_eval \
  results/smoke/gqa_cross.jsonl --manifest "$MAN" --budgets 0.05,0.2
sleep 90
while pgrep -f "python -m vlm_diagnosis.exps.m3_pilot_cross_eva[l]" >/dev/null; do sleep 60; done
kill $(cat /tmp/gqa_cross_monitor.pid) 2>/dev/null
echo "[chain-gqa] 교차 완료: $(date +%H:%M)"

echo "[chain-gqa] 3/3 held-out 시작: $(date +%H:%M)"
bash vlm_diagnosis/scripts/safe_launch.sh vlm_diagnosis.exps.m3_pilot_cross_eval \
  results/smoke/gqa_heldout.jsonl --manifest "$MAN" --eval-mode heldout --budgets 0.05,0.2
sleep 90
while pgrep -f "python -m vlm_diagnosis.exps.m3_pilot_cross_eva[l]" >/dev/null; do sleep 60; done
kill $(cat /tmp/gqa_heldout_monitor.pid) 2>/dev/null
echo "[chain-gqa] 전체 완료: $(date +%H:%M)"
