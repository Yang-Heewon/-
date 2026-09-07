#!/usr/bin/env bash
# Stage 1 "Hidden → K": 화면 dev 40 (GPU 0, 0C 종료 후) + 사진 dev 40 (GPU 2), 유지율 20%·10%·5%.
set -u
cd /root/research/heewon/VLM
OUT=results/context_only; mkdir -p $OUT
log(){ echo "[hid] $(date +%H:%M) $*"; }
run(){ CUDA_VISIBLE_DEVICES=$1 python -m vlm_diagnosis.exps.context_only_improve --panel hidden --keep-ratios 0.2,0.1,0.05 "${@:2}" --device cuda:0; }
log "GQA 시작 (GPU 2)"
run 2 --manifest experiments/manifests/gqa_discovery.jsonl --split dev --out $OUT/hidden_qwen25vl_gqa_dev.jsonl > $OUT/hidden_gqa_dev.log 2>&1 &
while pgrep -f "hidden_k_complementarit[y]" >/dev/null; do sleep 30; done
log "SQA 시작 (GPU 0)"
run 0 --manifest experiments/manifests/screenqa_discovery.jsonl --split dev --out $OUT/hidden_qwen25vl_screenqa_dev.jsonl > $OUT/hidden_sqa_dev.log 2>&1 &
wait
log "종료"
