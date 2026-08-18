#!/bin/bash
# FT-1: D4 mini를 안전 GPU 2·3의 2개 샤드로 병렬 실행.
cd /root/research/heewon/VLM
mkdir -p results/smoke/legacy/d4_mini
shard=0
for physical_gpu in 2 3; do
  CUDA_VISIBLE_DEVICES=$physical_gpu nohup python -m vlm_diagnosis.exps.d4_mini \
    --shard $shard --nshards 2 --device cuda:0 --seed 42 \
    > results/smoke/legacy/d4_mini/shard$shard.log 2>&1 &
  shard=$((shard+1))
done
echo "2개 샤드(물리 GPU 2·3) 실행됨. 진행: tail -f results/smoke/legacy/d4_mini/shard*.log"
