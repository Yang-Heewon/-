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

### 5. fp16 NaN 함정 (2026-08-13 실측) — 현재 M0 blocker

- Qwen2.5-VL-7B를 fp16으로 돌리면 **logits가 NaN**이 됩니다 (증상: greedy 생성이
  `'!47!!'` 같은 깨진 출력).
- 원인: ViT도 잔차 스트림도 정상(absmax ~6.7k)인데, **LLM 마지막 층(27) 내부 연산**이
  fp16 범위를 초과.
- 1차 완화: 해당 층만 fp32로 실행 (추가 ~1GB). `vlm_diagnosis/core/loader.py`의
  `fp32_layers="auto"`에 구현됨.
- 후속 실측: 이 패치 이후에도 legacy D4의 S0 4D-mask 경로에서 NaN이 재현됐다. 따라서 layer 27
  하나가 전체 원인이라고 확정하지 않는다.
- 다음 조치: full 2D/full 4D/failing mask를 같은 position path로 맞춘 뒤 mask 행, Q/K/V,
  attention probability, hidden state를 layer별로 추적한다. 원인 확인 전 본실험 금지.

## 모델 관련 메모

- **Qwen3-VL-8B**: 레이어 36개, `deepstack_visual_indexes=[8,16,24]`
- **Qwen2.5-VL-7B**: 이미 HF 캐시에 다운로드되어 있음
- **Qwen3-VL-8B, UI-TARS-1.5-7B**: 다운로드 필요

## 관련 문서

- [현재 PLAN](../PLAN.md)
- [과거 연구목표 노트](../archive/notes-ko/01-연구목표-KV압축.md)
- [과거 선행연구 요약](../archive/notes-ko/03-선행연구-현황.md)
