#!/usr/bin/env bash
# Context-only KV (문서 단계 1–5, VLM 화면 표본): full parity(dev 8) → probe(dev 8) → deletion(dev 40, 20%) →
# sweep(eval 132, 0.5/0.2/0.1/0.05, + attn1 + recon_desc) → profile(dev 20; plain/d/r/mlp_norm 각각 별도 실행)
# GPU: 처음엔 GPU 2 만; sweep 은 GPU 0 의 probe 판이 끝나면 0·2 두 장.
set -u
cd /root/research/heewon/VLM
OUT=results/context_only; mkdir -p $OUT
log(){ echo "[co] $(date +%H:%M) $*"; }
run(){ CUDA_VISIBLE_DEVICES=$1 python -m vlm_diagnosis.exps.context_only_kv "${@:2}" --device cuda:0 --resume; }
ana(){ python -m vlm_diagnosis.scripts.context_only_analysis --pattern "$1" --out "$2" > "$2.log" 2>&1; }

log "stage full (parity) 시작"
run 2 --stage full --split dev --limit 8 --out $OUT/full_qwen25vl_dev.jsonl > $OUT/full_dev.log 2>&1
ana "$OUT/full_qwen25vl_dev.jsonl" $OUT/full_dev_summary
log "stage probe 시작"
run 2 --stage probe --split dev --limit 8 --out $OUT/probe_qwen25vl_dev.jsonl > $OUT/probe_dev.log 2>&1
ana "$OUT/probe_qwen25vl_dev.jsonl" $OUT/probe_dev_summary
log "stage deletion (dev 40, keep 0.2) 시작"
run 2 --stage deletion --split dev --keep-ratios 0.2 --out $OUT/deletion_qwen25vl_dev.jsonl > $OUT/deletion_dev.log 2>&1
ana "$OUT/deletion_qwen25vl_dev.jsonl" $OUT/deletion_dev_summary
log "deletion 완료 → sweep 대기 (GPU 0 probe 판 종료 확인)"
while pgrep -f "single_prefill_prob[e]" >/dev/null; do sleep 60; done
log "stage sweep (eval 132) GPU 0·2 시작"
SW="--stage sweep --split eval --keep-ratios 0.5,0.2,0.1,0.05 --with-attn1 --with-reconstruction --out $OUT/sweep_qwen25vl_eval.jsonl"
run 2 $SW --shard 0 --nshards 2 > $OUT/sweep_eval_shard0.log 2>&1 &
sleep 45
run 0 $SW --shard 1 --nshards 2 > $OUT/sweep_eval_shard1.log 2>&1 &
wait
ana "$OUT/sweep_qwen25vl_eval.shard*.jsonl" $OUT/sweep_eval_summary
log "stage profile (dev 20) 시작"
for M in plain d r mlp_norm k_norm; do
  run 2 --stage profile --split dev --limit 20 --keep-ratios 0.2 --profile-method $M --out $OUT/profile_${M}_qwen25vl_dev.jsonl > $OUT/profile_${M}.log 2>&1
done
ana "$OUT/profile_*_qwen25vl_dev.jsonl" $OUT/profile_dev_summary
log "종료"
