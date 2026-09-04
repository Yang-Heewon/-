#!/usr/bin/env bash
# Dual-prefill 전체 KV 판: image+기존 text prefix 1회와 image-only 1회의 중요 집합을 합쳐
# 프롬프트 전체 KV를 (층, KV head, token) 단위로 압축한다. GPU 2·3만 사용.
#
#   bash vlm_diagnosis/scripts/chain_core_delta_fullkv.sh
#   PANELS="q25:sqa" BUDGETS=0.05 bash vlm_diagnosis/scripts/chain_core_delta_fullkv.sh
set -u
cd /root/research/heewon/VLM
export GPUS="${GPUS:-2 3}" STAGGER="${STAGGER:-45}"
BUDGETS="${BUDGETS:-0.05,0.1}"
ALPHAS="${ALPHAS:-0,0.25,0.5,0.75,1}"
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
  TAG="fk_${M}_${DOM}${TAGSUFFIX}"
  echo "[fk] $TAG 시작: $(date +%H:%M)  (model=$MODEL budgets=$BUDGETS alphas=$ALPHAS extra=$EXTRA)"
  bash vlm_diagnosis/scripts/safe_launch.sh vlm_diagnosis.exps.core_delta_full_kv \
    results/smoke/${TAG}.jsonl --manifest "$MAN" --model "$MODEL" \
    --budgets "$BUDGETS" --alphas "$ALPHAS" $EXTRA
  wait_mod "python -m vlm_diagnosis.exps.core_delta_full_k[v]" "$TAG"
  echo "[fk] $TAG 완료: $(date +%H:%M)"
  python -m vlm_diagnosis.scripts.core_delta_analysis \
    --pattern "results/smoke/${TAG}.shard*.jsonl" --out "results/smoke/${TAG}_summary" \
    --title "Dual-prefill 전체 KV — ${M} × ${DOM} (budgets ${BUDGETS})" \
    > "results/smoke/${TAG}_analysis.log" 2>&1
  echo "[fk] $TAG 분석 저장: results/smoke/${TAG}_summary.md"
done
echo "[fk] 연쇄 종료: $(date +%H:%M)"
