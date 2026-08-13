#!/bin/bash
# FT-1: D4 mini를 4-GPU 샤드로 병렬 실행. 사용: bash launch_d4_mini.sh
cd /root/research/heewon/VLM
mkdir -p results/smoke/legacy/d4_mini
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i nohup python -m vlm_diagnosis.exps.d4_mini \
    --shard $i --nshards 4 --device cuda:0 --seed 42 \
    > results/smoke/legacy/d4_mini/shard$i.log 2>&1 &
done
echo "4개 샤드 실행됨. 진행: tail -f results/smoke/legacy/d4_mini/shard*.log"
