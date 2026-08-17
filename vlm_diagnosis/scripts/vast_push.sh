#!/usr/bin/env bash
# 병렬 rsync — 파일 목록을 N등분해 rsync N개를 동시 실행 (WAN 단일 스트림 한계 우회).
#
#   bash vlm_diagnosis/scripts/vast_push.sh <HOST> <PORT> <로컬디렉토리> <원격디렉토리> [병렬수=8]
# 예:
#   bash vlm_diagnosis/scripts/vast_push.sh 1.2.3.4 30652 data/screenqa_pilot /workspace/VLM/data/screenqa_pilot 8
set -euo pipefail
HOST=$1; PORT=$2; SRC=${3%/}; DST=${4%/}; N=${5:-8}
KEY=~/.ssh/id_ed25519_vast
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new"

ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new root@$HOST "mkdir -p '$DST'"
cd "$SRC"
# 파일 목록을 크기 순 라운드로빈으로 N등분 (큰 파일이 한 줄에 몰리지 않게)
find . -type f -printf '%s\t%p\n' | sort -rn | cut -f2 > /tmp/vp_files
total=$(wc -l < /tmp/vp_files)
for i in $(seq 0 $((N-1))); do
  awk -v i=$i -v n=$N 'NR % n == i' /tmp/vp_files > /tmp/vp_part$i
done
echo "[vast_push] $total개 파일을 $N개 스트림으로 전송"
t0=$(date +%s)
for i in $(seq 0 $((N-1))); do
  [ -s /tmp/vp_part$i ] || continue
  rsync -az --files-from=/tmp/vp_part$i -e "$SSH" ./ root@$HOST:"$DST"/ &
done
wait
echo "[vast_push] 완료: $(( $(date +%s) - t0 ))초"
rm -f /tmp/vp_files /tmp/vp_part*
