#!/usr/bin/env bash
# vast 인스턴스가 죽었을 때, 새 컨테이너에서 죽은 지점부터 재개하는 복구 스크립트.
# (로컬 DGX에서 실행)
#
#   bash vlm_diagnosis/scripts/vast_restore.sh <HOST> <PORT> [실행명령...]
# 예:
#   bash vlm_diagnosis/scripts/vast_restore.sh 1.2.3.4 30652 \
#     "bash chain_vast_sqa.sh"     # 명령 생략 시 환경 복구까지만
#
# 원리 (체크포인트 = 결과 파일 그 자체):
#   1) 모든 러너는 --resume: 출력 jsonl에 이미 기록된 화면(sample_id)을 건너뜀
#   2) 5분 주기 동기화가 결과를 로컬 results/smoke/vast/ 에 보관
#   3) 이 스크립트가 그 보관본을 새 컨테이너에 되밀어넣고 같은 명령을 재실행
#      → 죽기 전 마지막 flush 지점(화면 단위)부터 이어짐. 유실 상한 = 동기화 주기 5분.
set -euo pipefail
HOST=$1; PORT=$2; shift 2 || true
KEY=~/.ssh/id_ed25519_vast
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new"
R="root@$HOST"
cd /root/research/heewon/VLM

echo "== 1/4 코드+manifest 전송 =="
rsync -az -e "$SSH" --exclude .git --exclude results --exclude data \
  --exclude _backup --exclude archive --exclude "*.xlsx" --exclude __pycache__ \
  ./ $R:/workspace/VLM/
echo "== 2/4 데이터 전송 =="
rsync -az -e "$SSH" data/screenqa_pilot data/gqa_pilot $R:/workspace/VLM/data/
echo "== 3/4 체크포인트(회수해둔 결과) 되밀어넣기 =="
$SSH $R "mkdir -p /workspace/VLM/results/smoke"
rsync -az -e "$SSH" results/smoke/vast/ $R:/workspace/VLM/results/smoke/
echo "== 4/4 환경 부트스트랩 =="
$SSH $R "cd /workspace/VLM && bash vlm_diagnosis/scripts/vast_bootstrap.sh --no-data" \
  | tail -3
if [ $# -gt 0 ]; then
  echo "== 재개 실행: $* =="
  $SSH $R "cd /workspace/VLM && nohup $* > /workspace/chain_vast.log 2>&1 & echo resumed"
fi
echo "복구 완료. 감시/동기화 루프는 로컬 세션에서 새 HOST:PORT로 다시 걸 것."
