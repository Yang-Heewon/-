# 하드웨어/소프트웨어 제약 — V100 환경에서 지켜야 할 것들

## 환경

`/root/research/heewon/VLM`은 **Tesla V100-DGXS-32GB 4장 (sm_70)**, torch 2.9, transformers 4.57.6 환경에서 돌아갑니다.

## 제약 사항

### 1. bf16 네이티브 지원 없음 → fp16 강제

- `torch.cuda.is_bf16_supported()`는 True를 반환하지만, 이것은 **에뮬레이션 포함** 결과입니다.
- `including_emulation=False`로 확인하면 False가 나옵니다.
- **반드시 `torch_dtype=torch.float16`을 강제**해야 합니다.

### 2. FlashAttention-2 사용 불가

- FA2는 sm_80 이상(A100 세대~)이 필요합니다. V100은 sm_70이므로 **eager 또는 SDPA만** 사용 가능합니다.

### 3. `output_attentions=True`는 VLM 시퀀스 길이에서 사용 불가 (OOM 함정)

- Qwen2.5-VL-7B (레이어 28, 헤드 28) 기준: 전체 레이어의 attention 행렬은
  **4k 토큰에서 26 GB, 6k 토큰에서 56 GB**를 차지합니다.
- 1080p 스크린샷 한 장이 **약 2,584개의 비주얼 토큰**이므로, GUI 이미지 **두 장이면 32 GB 카드가 터집니다(OOM)**.
- **대안**: attention 행렬을 통째로 뽑지 말고, **RoPE 적용 후의 Q, K에 hook을 걸어** 뽑을 것
  (n=6000 기준 총 ~1.4 GB밖에 안 듦), 그리고 query를 청크 단위로 나눠 **column-mass로 축약**해서 계산합니다.

### 4. KV 캐시 자체는 작다 — 동기 설정 시 주의

- 두 모델 모두 GQA(Grouped-Query Attention)를 쓰기 때문에 KV가 작습니다:
  - Qwen2.5-VL-7B: **토큰당 56 KB**
  - Qwen3-VL-8B: **토큰당 144 KB**
- 10스텝 trajectory 전체를 저장해도 **약 1.5–3.7 GB**에 불과합니다.
- 따라서 **"KV 메모리가 폭발한다"는 동기는 단일 세션에서는 성립하지 않습니다.**
  정직한 비용 논거는 **prefill FLOPs / TTFT(첫 토큰까지의 시간) / 배치 동시성**입니다.
  (장기 기억 프레임에서는 저장 용량 논거가 되살아남 —
  [과거 연구목표 노트](../archive/notes-ko/01-연구목표-KV압축.md) 참조)

### 5. fp16 NaN 함정 — **해결됨 (2026-08-13, M0-04)**

- 증상: Qwen2.5-VL-7B fp16에서 일부 입력의 logits가 NaN (greedy 출력 `'!47!!'` 등).
- **근본 원인 (실측 확정)**: HF eager attention은 `matmul(Q, K^T) * scaling` 순서라
  **스케일 전 QK^T가 fp16 matmul 출력 한계(65504)를 넘으면 inf → softmax NaN**.
  실측: doc4733 q1의 layer 0에서 스케일 전 ≈69.5k(입력 의존적), layer 27은 상시
  스케일 전 ≈118k — 과거 "layer 27만 fp32" 완화가 우연히 통했던 이유도 같은 메커니즘.
  mask/position 문제 아님 (무마스크 2D 경로에서도 재현됐음).
- **처방**: 수학적으로 동일한 `(Q*scaling) @ K^T` 순서로 교체 —
  `vlm_diagnosis/core/loader.py`의 `patch_stable_qk_scale()` (기본 ON).
  layer-27 fp32 패치는 retire (`LEGACY_FP32_LAYERS`로 진단 비교용만 보존).
- **검증**: d4_mini 32문서 × 4질문 × 3경로 = 384/384 finite
  (`results/smoke/nan_diagnosis/sweep.jsonl`). 스케일 후 max|logit| ≈ 11k로 fp16 여유.
- `assert_finite_logits` 가드는 계속 모든 실험에 유지한다 (다른 입력 분포 대비).

## 모델 관련 메모

- **Qwen3-VL-8B**: 레이어 36개, `deepstack_visual_indexes=[8,16,24]`
- **Qwen2.5-VL-7B**: 이미 HF 캐시에 다운로드되어 있음
- **Qwen3-VL-8B, UI-TARS-1.5-7B**: 다운로드 필요

## 관련 문서

- [현재 PLAN](../PLAN.md)
- [과거 연구목표 노트](../archive/notes-ko/01-연구목표-KV압축.md)
- [과거 선행연구 요약](../archive/notes-ko/03-선행연구-현황.md)

### 6. GPU 1번 NVLink 고장 (2026-08-14 발생) — 재부팅 필요

- 증상: `nvidia-smi`가 GPU1을 인식 못 함 ("Unable to determine the device handle",
  Xid 74 = NVLink 링크 훈련 실패, dmesg 확인).
- 2차 피해: 종료된 실험 프로세스 3개가 zombie로 남아 GPU 0/2/3에 각각 ~20GB를
  점유한 채 회수 불가 (드라이버가 NVLink 고장으로 CUDA context를 못 놓아줌).
  `nvidia-smi -i 1 -r` 리셋도 실패 ("No devices were found").
- 결과: 남은 여유 ~12GB로는 Qwen2.5-VL-7B(≈17GB)를 못 올림 → **머신 재부팅
  전까지 GPU 실험 전면 불가.**
- 증거: 죽어가는 GPU에서 돌던 shard가 논리적으로 불가능한 값(8칸 텐서의 argmax가
  1262)을 반환하며 죽음 — 하드웨어가 죽는 중에는 커널 결과를 신뢰할 수 없다는
  실례. 해당 shard 데이터는 폐기함.

### 7. 4-GPU 동시 실행 금지 — 전원/열 보호 차단 (2026-08-15 규명)

- **증상**: 4개 shard를 동시에 띄운 직후(17:24:23) 시스템 로그가 17:24:09에서
  아무 경고 없이 끊기고 재부팅. 커널 패닉·OOM·Xid 기록 없음 = **급작스러운 전원 차단**.
  같은 방식으로 최소 2회 발생(08-14, 08-15). 08-14 크래시 때는 git 객체 11개가
  깨져 마지막 커밋이 유실됐다(작업 트리로 복구).
- **원인 후보 1 — GPU0 냉각 결함(실측)**: 97W에서 **83°C**(= GPU Max Operating Temp)
  도달, throttle reason `0x20`(SW thermal slowdown) 상시 작동. 같은 97W의 GPU2는
  55°C. 유휴 상태에서도 GPU0만 62~68°C(타 GPU 47~52°C).
- **원인 후보 2 — 섀시 공기 흐름**: 디스크 SMART 온도 55~58°C(정상 30~45°C).
- **원인 후보 3 — GPU1 하드웨어**: NVLink Xid 74 이력(§6).
- **대응(적용됨)**:
  1. `nvidia-smi -pl 180` — 전력 상한 300W→180W (4장 합계 1200W→720W)
  2. **GPU 2·3만 사용**. GPU0(열)·GPU1(NVLink)은 기본 제외
  3. 프로세스를 45초 간격으로 **순차 기동** (동시 모델 로딩 스파이크 방지)
  4. 실행 중 전력·온도 로깅 + runner `--resume`으로 크래시 후 이어서 실행
  5. 실행기: `vlm_diagnosis/scripts/safe_launch.sh` (기본값이 위 정책)
- **물리적 조치 권고**: 섀시 먼지 청소·팬 점검. GPU0 온도가 정상화되지 않으면
  하드웨어 서비스 대상.
