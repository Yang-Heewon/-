#!/usr/bin/env bash
# vast.ai 컨테이너 초기화 — 코드/환경/모델/데이터 준비 후 바로 실험 가능 상태로.
#
# 사용 (컨테이너 안에서):
#   bash vast_bootstrap.sh            # 환경+모델+데이터 전부
#   bash vast_bootstrap.sh --no-data  # 데이터 재생성 생략 (rsync로 받을 때)
#
# 전제: 이 repo가 /workspace/VLM 에 있음 (git clone 또는 rsync).
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo 루트

echo "== 1/4 파이썬 환경 (venv — 데비안 시스템 pip 회피) =="
if [ ! -d /workspace/venv ]; then python3 -m venv /workspace/venv; fi
source /workspace/venv/bin/activate
pip install -q --upgrade pip
pip install -q "transformers==4.57.6" "torch>=2.4" accelerate pillow \
  huggingface_hub pyarrow openpyxl

echo "== 2/4 GPU 확인 =="
python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA 없음"
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i)
    print(f"cuda:{i} {p.name} {p.total_memory/2**30:.0f}GB sm_{p.major}{p.minor}")
PY

echo "== 3/4 모델 다운로드 (HF 캐시) =="
python3 - <<'PY'
from huggingface_hub import snapshot_download
for m in ("Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen3-VL-8B-Instruct"):
    print(m, "→", snapshot_download(m))
PY

if [ "${1:-}" != "--no-data" ]; then
  echo "== 4/4 데이터 재생성 (manifest는 repo에 있음, 이미지만 재다운로드) =="
  # 각 prep 스크립트는 결정적(seed 고정)이라 같은 manifest/이미지를 재생성한다.
  python -m vlm_diagnosis.scripts.prep_screenqa_transfer --images-only || \
    python -m vlm_diagnosis.scripts.prep_screenqa_transfer
  python -m vlm_diagnosis.scripts.prep_gqa_transfer --images-only || \
    python -m vlm_diagnosis.scripts.prep_gqa_transfer
else
  echo "== 4/4 데이터 생략 (--no-data) — rsync로 data/ 를 받아올 것 =="
fi

echo "== 스모크 (1분) =="
python3 - <<'PY'
import torch, json
from PIL import Image
from vlm_diagnosis.core.loader import load_vlm
from vlm_diagnosis.core import signals as S
from vlm_diagnosis.core.masked_generate import greedy_generate_masked
model, processor = load_vlm("qwen25vl", device="cuda:0", max_pixels=1280*28*28)
r=json.loads(open('experiments/manifests/gqa_transfer.jsonl').readline())
img=Image.open(r['image']).convert('RGB')
ins=S.vlm_inputs(processor, img, r['questions'][1]['question']+" Answer with a single word or phrase.", "cuda:0")
with torch.no_grad():
    out=model(**{k:ins[k] for k in ('input_ids','pixel_values','image_grid_thw')},
              attention_mask=torch.ones_like(ins['input_ids']))
    assert torch.isfinite(out.logits).all(), "NaN!"
    print("finite OK, greedy:", greedy_generate_masked(model, processor, ins, max_new_tokens=8))
PY
echo "== 준비 완료 =="
echo "예시 실행:"
echo "  python -m vlm_diagnosis.exps.m2a_fixed_budget --manifest experiments/manifests/screenqa_transfer.jsonl \\"
echo "    --model qwen3vl --budgets 0.05,0.2 --shard 0 --nshards 1 --device cuda:0 --out results/smoke/sqa_ladder_q3.jsonl --resume"
