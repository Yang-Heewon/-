#!/usr/bin/env bash
# Core–Delta DRAM 판 (core_delta_dram.py): 쓰기 시점에 DRAM 보관분 B(10/20/40%)만 남기고 나머지 삭제,
# core C는 GPU 상주, delta D는 질문 시 DRAM→GPU. 두 core 신호(kvzip 재구성 / image-only prefill)를
# 같은 표본·같은 격자에서 나란히 돌린다. KVzip 기준선(core만 B / core만 C+D) 동시 실행.
# 먼저 돌고 있는 전부-보관 판(wr_q25_sqa, KVzip 신호)이 끝나기를 기다린 뒤 분석해 두고 시작한다.
#
#   bash vlm_diagnosis/scripts/chain_core_delta_dram.sh
set -u
cd /root/research/heewon/VLM
export GPUS="${GPUS:-2 3}" STAGGER="${STAGGER:-45}"
SIGNALS="${SIGNALS:-kvzip,image}"
COLD="${COLD:-0.1,0.2,0.4}"
CORES="${CORES:-0,0.025,0.05}"
DELTAS="${DELTAS:-0.01,0.025}"
CORE_ONLY="${CORE_ONLY:-0.025,0.035,0.05,0.06,0.075,0.1,0.2,0.4}"
RAND="${RAND:-0.2:0.05:0.01}"
TOK="${TOK:-0.2:0:0.01,0.2:0.05:0.01}"
PANELS="${PANELS:-q25:sqa q25:gqa q3:sqa q3:gqa}"
TAGSUFFIX="${TAGSUFFIX:-}"
ARGS=(--core-signals "$SIGNALS" --cold-budgets "$COLD" --core-sizes "$CORES" --delta-sizes "$DELTAS"
      --core-only-sizes "$CORE_ONLY" --random-core-cells "$RAND" --token-cells "$TOK")

wait_mod () {  # $1=pgrep 패턴(브래킷), $2=monitor tag
  sleep 90
  while pgrep -f "$1" >/dev/null; do sleep 60; done
  kill $(cat /tmp/${2}_monitor.pid) 2>/dev/null
}

# 0) 진행 중인 전부-보관 판이 끝나면 분석해 둔다 (참고치)
while pgrep -f "python -m vlm_diagnosis.exps.core_delta_write_rea[d]" >/dev/null; do sleep 60; done
kill $(cat /tmp/wr_q25_sqa_monitor.pid 2>/dev/null) 2>/dev/null
if ls results/smoke/wr_q25_sqa.shard*.jsonl >/dev/null 2>&1 && [ ! -f results/smoke/wr_q25_sqa_summary.md ]; then
  echo "[dram] 전부-보관 판(wr_q25_sqa, kvzip 신호) 분석: $(date +%H:%M)"
  python -m vlm_diagnosis.scripts.core_delta_wr_analysis \
    --pattern "results/smoke/wr_q25_sqa.shard*.jsonl" --out results/smoke/wr_q25_sqa_summary \
    --title "Core–Delta 쓰기/읽기 (전부 보관, kvzip 신호, 참고) — q25 × sqa" > results/smoke/wr_q25_sqa_analysis.log 2>&1
fi

# 1) smoke (1장, GPU 2) — 실패하면 연쇄 중단
echo "[dram] smoke 시작: $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=2 python -m vlm_diagnosis.exps.core_delta_dram --limit 1 --device cuda:0 \
  "${ARGS[@]}" --out results/smoke/wd_smoke.jsonl > results/smoke/wd_smoke.log 2>&1
if [ $? -ne 0 ] || ! grep -q "\[saved\]" results/smoke/wd_smoke.log; then
  echo "[dram] smoke 실패 — 연쇄 중단 (results/smoke/wd_smoke.log 확인)"; exit 1
fi
echo "[dram] smoke 통과: $(date +%H:%M)"

for P in $PANELS; do
  M=${P%%:*}; DOM=${P##*:}
  case "$M" in q25) MODEL=qwen25vl ;; q3) MODEL=qwen3vl ;; *) echo "bad model $M"; exit 2 ;; esac
  case "$DOM" in
    sqa) MAN=experiments/manifests/screenqa_discovery.jsonl ;;
    gqa) MAN=experiments/manifests/gqa_discovery.jsonl ;;
    *) echo "bad domain $DOM"; exit 2 ;;
  esac
  TAG="wd_${M}_${DOM}${TAGSUFFIX}"
  echo "[dram] $TAG 시작: $(date +%H:%M)  (model=$MODEL signals=$SIGNALS B=$COLD C=$CORES D=$DELTAS)"
  bash vlm_diagnosis/scripts/safe_launch.sh vlm_diagnosis.exps.core_delta_dram \
    results/smoke/${TAG}.jsonl --manifest "$MAN" --model "$MODEL" "${ARGS[@]}"
  wait_mod "python -m vlm_diagnosis.exps.core_delta_dra[m]" "$TAG"
  echo "[dram] $TAG 완료: $(date +%H:%M)"
  python -m vlm_diagnosis.scripts.core_delta_dram_analysis \
    --pattern "results/smoke/${TAG}.shard*.jsonl" --out "results/smoke/${TAG}_summary" \
    --title "Core–Delta DRAM 판 — ${M} × ${DOM}" > "results/smoke/${TAG}_analysis.log" 2>&1
  echo "[dram] $TAG 분석 저장: results/smoke/${TAG}_summary.md"
done
echo "[dram] 연쇄 종료: $(date +%H:%M)"
