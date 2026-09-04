#!/usr/bin/env bash
# 누적 선택 판 후속 드라이버: 진행 중인 Qwen2.5×화면(20% 단계) shard가 끝나면 분석하고,
# 천장 효과(20%에서 이미 0.93~1.0)를 피하기 위한 5% 단계 변형(keep 5% / grow 5→10→15→20%)과
# 나머지 판을 순서대로 돌린다. GPU는 GPUS 환경변수(기본 "0 2").
#
#   GPUS="0 2" bash vlm_diagnosis/scripts/chain_core_delta_incremental_driver.sh
set -u
cd /root/research/heewon/VLM
export GPUS="${GPUS:-0 2}" STAGGER="${STAGGER:-45}"

while pgrep -f "python -m vlm_diagnosis.exps.core_delta_incrementa[l]" >/dev/null; do sleep 60; done
kill $(cat /tmp/inc_q25_sqa_monitor.pid 2>/dev/null) 2>/dev/null
if ls results/smoke/inc_q25_sqa.shard*.jsonl >/dev/null 2>&1 && [ ! -f results/smoke/inc_q25_sqa_summary.md ]; then
  echo "[driver] inc_q25_sqa(20% 단계) 분석: $(date +%H:%M)"
  python -m vlm_diagnosis.scripts.core_delta_incremental_analysis \
    --pattern "results/smoke/inc_q25_sqa.shard*.jsonl" --out results/smoke/inc_q25_sqa_summary \
    --title "Core–Delta 누적 선택 판 (20% 단계) — q25 × sqa" > results/smoke/inc_q25_sqa_analysis.log 2>&1
  echo "[driver] 분석 저장: results/smoke/inc_q25_sqa_summary.md"
fi

run () {  # $1=PANELS $2=BSTEP $3=TAGSUFFIX
  echo "[driver] 시작 $1 step=$2 tag=$3: $(date +%H:%M)"
  PANELS="$1" BSTEP="$2" TAGSUFFIX="$3" bash vlm_diagnosis/scripts/chain_core_delta_incremental.sh
  echo "[driver] 끝 $1 step=$2 tag=$3: $(date +%H:%M)"
}
run "q25:sqa" 0.05 "_b5"
run "q25:gqa" 0.2  ""
run "q25:gqa" 0.05 "_b5"
run "q3:sqa"  0.2  ""
run "q3:sqa"  0.05 "_b5"
run "q3:gqa"  0.2  ""
run "q3:gqa"  0.05 "_b5"
echo "[driver] 전체 종료: $(date +%H:%M)"
