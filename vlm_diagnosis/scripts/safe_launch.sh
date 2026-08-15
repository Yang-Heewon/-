#!/usr/bin/env bash
# 안전 실행기 — DGX Station이 4-GPU 동시 기동에서 전원 차단으로 죽는 문제 대응.
#
#   1) 동시 GPU 수 제한 (기본 2)   2) 프로세스를 STAGGER초 간격으로 순차 기동
#   3) 실행 중 전력·온도를 기록     4) runner에 --resume 전달 (크래시 후 이어서)
#
# 사용:
#   bash vlm_diagnosis/scripts/safe_launch.sh <모듈> <출력파일> [추가인자...]
# 예:
#   bash vlm_diagnosis/scripts/safe_launch.sh vlm_diagnosis.exps.m3_pilot_cross_eval \
#     results/smoke/sqa_cross.jsonl --manifest experiments/manifests/screenqa_transfer.jsonl
set -euo pipefail
MODULE="$1"; OUT="$2"; shift 2
GPUS="${GPUS:-0 2}"          # 기본 2장 (GPU1은 NVLink 오류 이력으로 기본 제외)
STAGGER="${STAGGER:-45}"     # 기동 간격(초)
N=$(echo $GPUS | wc -w)
TAG=$(basename "$OUT" .jsonl)
mkdir -p results/smoke

echo "[safe_launch] GPU $GPUS ($N장) · ${STAGGER}초 간격 기동 · resume 켜짐"
nvidia-smi --query-gpu=index,power.limit --format=csv,noheader | sed 's/^/  전력상한: /'

# 전력·온도 감시 (10초 간격)
( while true; do
    echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=index,power.draw,temperature.gpu,utilization.gpu \
      --format=csv,noheader,nounits | tr '\n' ' | ')"
    sleep 10
  done ) > "results/smoke/${TAG}_power.log" 2>&1 &
echo $! > /tmp/${TAG}_monitor.pid

i=0
for g in $GPUS; do
  nohup python -m "$MODULE" --shard $i --nshards $N --device cuda:$g \
    --out "$OUT" --resume "$@" > "results/smoke/${TAG}_shard${i}.log" 2>&1 &
  echo "  shard $i → cuda:$g (pid $!)"
  i=$((i+1))
  [ $i -lt $N ] && sleep "$STAGGER"
done
echo "[safe_launch] 기동 완료. 감시 로그: results/smoke/${TAG}_power.log"
