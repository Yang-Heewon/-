#!/usr/bin/env bash
# 개선 판 dev pilot: 화면(ScreenQA dev 40, GPU 0) + 자연 사진(GQA dev 40, GPU 2), 유지율 20%·5%.
set -u
cd /root/research/heewon/VLM
OUT=results/context_only; mkdir -p $OUT
log(){ echo "[imp] $(date +%H:%M) $*"; }
run(){ CUDA_VISIBLE_DEVICES=$1 python -m vlm_diagnosis.exps.context_only_improve "${@:2}" --device cuda:0; }
log "dev pilot 시작 (SQA→GPU0, GQA→GPU2)"
run 0 --manifest experiments/manifests/screenqa_discovery.jsonl --split dev --out $OUT/improve_qwen25vl_screenqa_dev.jsonl > $OUT/improve_sqa_dev.log 2>&1 &
sleep 45
run 2 --manifest experiments/manifests/gqa_discovery.jsonl --split dev --out $OUT/improve_qwen25vl_gqa_dev.jsonl > $OUT/improve_gqa_dev.log 2>&1 &
wait
for D in screenqa gqa; do
  python -m vlm_diagnosis.scripts.context_only_analysis --pattern "$OUT/improve_qwen25vl_${D}_dev.jsonl" --out $OUT/improve_${D}_dev_summary > $OUT/improve_${D}_dev_analysis.log 2>&1 || log "분석기(v2) 거부: $D → legacy 분석 사용"
done
log "dev pilot 종료"
