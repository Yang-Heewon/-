# 모델 계보와 M7 선택 계약

이 문서는 “다른 모델 두 개”를 이름만 바꾸는 실수를 막는다. M7 모델은 발견된 현상을 어떤 축에서
일반화할지 먼저 정한 뒤 고른다.

## 1. 독립성 축

| 축 | 질문 | 확인해야 할 metadata |
|---|---|---|
| language backbone | Qwen decoder 특이 현상인가? | base LLM family, attention/cache layout |
| position encoding | mRoPE 이식 문제인가? | mRoPE/1D RoPE/기타, visual position construction |
| visual integration | visual token화·fusion 특이 현상인가? | encoder, projector, token insertion 방식 |
| task tuning | GUI fine-tuning 특이 현상인가? | base checkpoint와 tuning corpus/목표 |

## 2. 현재 후보

| 후보 | language lineage | position·visual 특징 | 적합한 확인 역할 |
|---|---|---|---|
| Qwen3-VL-8B | Qwen | Qwen3-VL visual integration | Qwen 세대 변화 |
| UI-TARS-1.5-7B | Qwen2.5-VL | mRoPE, GUI tuning | 같은 구조의 domain tuning |
| OpenCUA-7B | Qwen2.5-VL 기반 | 공식 구현은 mRoPE 대신 1D RoPE | position encoding 대조 |
| InternVL3-8B | Qwen2 계열 LLM | InternViT+MLP, dynamic tiles | visual integration 대조 |
| LLaVA-OneVision-7B | Qwen2 LLM | SigLIP/projector 계열 | visual integration 대조 |
| strict non-Qwen | TBD | LLM·position·fusion 독립 | architecture-general 주장 |

공식 확인 근거:

- [UI-TARS config](https://huggingface.co/ByteDance-Seed/UI-TARS-1.5-7B/blob/3ed9359ba593fa0a5a001b8403d96daa19877b79/config.json)
- [OpenCUA model documentation](https://github.com/xlang-ai/OpenCUA/blob/main/model/README.md)
- [InternVL3-8B config](https://huggingface.co/OpenGVLab/InternVL3-8B/blob/main/config.json)
- [LLaVA-OneVision model card](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf)

InternVL3와 LLaVA-OneVision은 multimodal architecture가 Qwen-VL과 다르지만 language backbone까지
non-Qwen인 후보는 아니다.

## 3. 발견별 routing

| Phase 1 발견 | 최소 confirmation 조합 |
|---|---|
| M1-C offset/mRoPE failure | Qwen mRoPE 1개 + position encoding이 다른 1개 |
| M2/M3 future relevance failure | write-time selector를 이식할 수 있는 visual integration 2종 |
| M4 OCR/layout 선택 손실 | visual encoder/tokenization이 다른 2종 |
| GUI tuning 특이 실패 | 일반 VLM 1개 + 같은 기반 GUI-tuned 1개 |
| architecture-general 주장 | language backbone까지 non-Qwen 1개 이상 포함 |

## 4. 후보 승격 gate

모델명을 M7 config에 넣기 전에 다음을 smoke한다.

- exact model·processor revision 고정
- discovery task의 IMAGE base 성능
- visual/prefix token span 식별
- cache extraction과 resume 지원
- position metadata 재구성 가능성
- fp16/bf16 finite와 V100/A100 필요 조건
- 동일 official task metric 적용 가능성

cache API가 다른 모델을 억지로 같은 runner에 끼우지 않는다. 공통 결과 schema를 공유하되 adapter는
모델별로 분리한다.

## 5. 사용자가 결정할 것

- G04-A: 확인하려는 독립성 축
- G04-B: confirmation 모델 2개와 각 역할
- G04-C: strict non-Qwen이 필요한 주장인지
- G04-D: V100에서 불가능할 때 A100 사용을 허용할지

결정은 discovery 결과를 본 뒤 하되 M7 데이터를 보기 전에는 동결한다.

