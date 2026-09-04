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
GPUS="${GPUS:-2 3}"          # 물리 GPU 2·3만 사용 (GPU0 열, GPU1 NVLink 이력)
STAGGER="${STAGGER:-45}"     # 기동 간격(초)
N=$(echo $GPUS | wc -w)
TAG=$(basename "$OUT" .jsonl)
mkdir -p results/smoke

# 이 머신의 연구 실행은 물리 GPU 2·3으로만 제한한다. 각 프로세스에는 카드 한 장만
# 노출하므로 runner 내부의 cuda:0은 각각 물리 GPU 2 또는 3을 뜻한다.
for g in $GPUS; do
  case "$g" in
    0|2|3) ;;
    *) echo "[safe_launch] 거부: 허용된 물리 GPU는 0,2,3입니다 (2026-09-04 사용자 지시로 0 추가) (요청=$g)" >&2; exit 2 ;;
  esac
done
GPU_CSV=$(echo "$GPUS" | tr ' ' ',')

echo "[safe_launch] GPU $GPUS ($N장) · ${STAGGER}초 간격 기동 · resume 켜짐"
nvidia-smi --id="$GPU_CSV" --query-gpu=index,power.limit --format=csv,noheader | sed 's/^/  전력상한: /'

# 전력·온도 감시 (10초 간격)
( while true; do
    echo "$(date +%H:%M:%S) $(nvidia-smi --id="$GPU_CSV" --query-gpu=index,power.draw,temperature.gpu,utilization.gpu \
      --format=csv,noheader,nounits | tr '\n' ' | ')"
    sleep 10
  done ) > "results/smoke/${TAG}_power.log" 2>&1 &
echo $! > /tmp/${TAG}_monitor.pid

i=0
for g in $GPUS; do
  nohup env CUDA_VISIBLE_DEVICES="$g" python -m "$MODULE" \
    --shard $i --nshards $N --device cuda:0 \
    --out "$OUT" --resume "$@" > "results/smoke/${TAG}_shard${i}.log" 2>&1 &
  echo "  shard $i → physical GPU $g / process cuda:0 (pid $!)"
  i=$((i+1))
  [ $i -lt $N ] && sleep "$STAGGER"
done
echo "[safe_launch] 기동 완료. 감시 로그: results/smoke/${TAG}_power.log"
