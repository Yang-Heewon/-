#!/usr/bin/env bash
# LEGACY Core–Delta 후속 재현 경로 (현재 dual-prefill 방법에는 사용하지 않음):
#   (0) §3 진단 probe — Qwen3-VL(GPU 2)·Qwen2.5-VL(GPU 3) 시각 KV의 텍스트 의존 원인 분리 (길이 vs 내용)
#   (a) 재사용 검정 — 과거 질문 q0의 S1 + KVzip core로 cache를 만들고 q1..q3 평가 (VLM_idea §11)
#   (b) 예산 격자 — 1/2/10% 강압축 구간 (VLM_idea §4.3 "판정의 중심은 1–10%")
# (a)(b)는 Qwen2.5 × ScreenQA(discovery 172장)에서만 먼저 실행한다.
set -u
cd /root/research/heewon/VLM
while pgrep -f "chain_core_delta.s[h]" >/dev/null; do sleep 120; done
echo "[followup] 본 연쇄 종료 확인: $(date +%H:%M)"

echo "[followup] (0) §3 probe 시작: $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=2 python -m vlm_diagnosis.scripts.kv_invariance_probe --model qwen3vl \
  --device cuda:0 --out results/smoke/kv_invariance_probe_q3.json > results/smoke/kv_invariance_probe_q3.log 2>&1 &
P1=$!
sleep 45
CUDA_VISIBLE_DEVICES=3 python -m vlm_diagnosis.scripts.kv_invariance_probe --model qwen25vl \
  --device cuda:0 --out results/smoke/kv_invariance_probe_q25.json > results/smoke/kv_invariance_probe_q25.log 2>&1 &
P2=$!
wait $P1 $P2
echo "[followup] (0) probe 완료: $(date +%H:%M)"
grep -E "^\[|saved" results/smoke/kv_invariance_probe_q3.log results/smoke/kv_invariance_probe_q25.log

echo "[followup] (a) 재사용 검정 시작: $(date +%H:%M)"
PANELS="q25:sqa" BUDGETS=0.05 TAGSUFFIX="_q0" EXTRA="--wsum --query-from q0" \
  bash vlm_diagnosis/scripts/chain_core_delta.sh
echo "[followup] (a) 완료: $(date +%H:%M)"

echo "[followup] (b) 예산 격자 시작: $(date +%H:%M)"
PANELS="q25:sqa" BUDGETS=0.01,0.02,0.1 ALPHAS="0,0.1,0.25,0.5,1" TAGSUFFIX="_grid" EXTRA="--wsum" \
  bash vlm_diagnosis/scripts/chain_core_delta.sh
echo "[followup] (b) 완료: $(date +%H:%M)"
echo "[followup] 종료: $(date +%H:%M)"
