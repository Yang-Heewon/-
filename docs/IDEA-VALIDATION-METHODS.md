# 아이디어 검증 방법론 (Phase-2 후보용)

> **범위**: [idea.md](../idea.md)에 등록된 방법 후보(아이디어 1·2·3)를 어떤 절차로
> 검증할지 정의한다. Phase-1 규율상 방법 개발은 문제 확정 뒤이므로, 이 문서는
> **측정 절차의 사전 정의**이며 결과가 아니다.
>
> **작성 근거**: 파일럿에서 확인된 두 교훈이 모든 절차의 뼈대다 —
> ① 중요도는 낱개가 아니라 **집합의 성질**(probe LGO 실패, FINDINGS §6),
> ② 아는 질문에서의 성공은 **미래 질문 일반화를 보장하지 않는다**(held-out 붕괴, §5).

---

## 0. 공통 원칙 (모든 아이디어에 무조건 적용)

| # | 원칙 | 이유 |
|---|---|---|
| P1 | **held-out 통과가 유일한 성패 기준** | 아는 질문 pool이 안 본 질문에서 무작위였음 (FINDINGS §5) |
| P2 | 주 판정은 **공식 task metric**(ANLS/EM/click). 로짓·logp는 **기제 진단 전용** | SHARED_PROTOCOL §5 |
| P3 | 예산은 **serialized bytes**로 맞춘다 (index·metadata 포함) | 토큰 수 비교는 불공정 |
| P4 | 비교 상대는 random·spatial_uniform·**s5(현 최고 write-time)** | "무작위보다 낫다"는 기준 미달 |
| P5 | 측정 잡음 대역 아래 차이는 해석 금지 (로짓 ~0.15 nat, 예측 40/40 불변) | M0 실측 |
| P6 | 분석 단위 = 이미지/화면, screen-cluster bootstrap | 같은 이미지의 질문은 독립 아님 |
| P7 | 압축 시점 라벨 필수 (write-time / source-aware / read-time / diagnostic) | PLAN §1.2 — 반칙 칸 혼입 방지 |

---

## 1. 방법 A — 층별 스윕: 전체 문맥 대비 로짓 거리 (TSP-style)

### 1.1 원 절차 (사용자 제공 발췌, 2026-08-15)

> "We first investigate the effect of applying Token-Selective Propagation (TSP) at
> different layers. For each candidate TSP layer, we compute the final logits and
> compare them against those obtained from the full-context baseline. Figure 3 reports
> the normalized L2 distance between the two outputs for LLaMA-3.1-8B-Instruct."

즉 **"몇 번째 층부터 토큰을 쳐내기 시작할 것인가"를 정하기 위한 층 스윕**이며,
판정량은 최종 로짓의 **정규화 L2 거리**다.

```text
후보 층 L = 0 … N-1 각각에 대해:
  1. 층 L부터 선택된 토큰만 다음 층으로 전파 (그 이전 층은 전체 문맥 그대로)
  2. 최종 로짓 z_L 계산
  3. d(L) = || z_L − z_full ||₂ / || z_full ||₂     ← 정규화 L2 거리
층 축 위의 d(L) 곡선에서 "거리가 급증하기 시작하는 층" 이전 = 안전한 시작 층
```

> **출처 주의**: 위 인용은 사용자가 제공한 발췌로, 원 논문·figure 번호를 아직 확인하지
> 않았다. 논문에 인용하기 전 원문 확인이 필요하며, 현재는 **절차의 출처 미상 상태로**
> 기록한다.

### 1.2 왜 이 절차가 우리에게 필요한가

아이디어 1(층을 지나며 점진적으로 토큰을 지우는 기억 선택)의 **핵심 설계 변수가 정확히
"어느 층부터 지우기 시작하는가"**이고, 이미 선행연구가 이 변수에서 실패한 전례가 있다:
FEATHER의 경고 — 이른 층의 attention 기반 pruning은 coarse 벤치마크에선 무해해 보이지만
**grounding·잔글씨를 파괴**한다 (MOTIVATION_ANALYSIS 기록). 층 스윕은 그 경계를 값싸게
찾는 절차다.

### 1.3 우리 환경으로의 이식 사양

| 항목 | 원 절차 | 우리 적용 |
|---|---|---|
| 모델 | LLaMA-3.1-8B-Instruct (텍스트) | Qwen2.5-VL-7B (LLM 28층), 시각 토큰만 대상 |
| 대상 토큰 | 전체 문맥 토큰 | **시각 토큰만** (텍스트/시스템 토큰은 항상 보존) |
| 비교 기준 | full-context 로짓 | full visual KV 로짓 (동일 4D-mask·명시 position 경로 — 2D/4D 혼용 금지, legacy D4 무효화 사유) |
| 거리 | 정규화 L2 | 동일 + **정답 구간 logp 차이**를 병기 (우리 프로토콜의 기제 지표) |
| 예산 | (논문 미상) | 예산 B ∈ {5,10,20,40}% 각각에 대해 층 스윕 (층×예산 격자) |

### 1.4 구현 부족분 (현재 코드 기준)

현재 `masked_eval.evict_columns`는 **모든 층에 동일한 4D mask**를 적용한다. 층별 스윕은
"층 L 이전에는 전체, 이후에는 부분"이 필요하므로 **층 인덱스에 따라 mask를 바꾸는 훅**이
있어야 한다. 최소 구현:

```text
- decoder layer forward pre-hook에서 layer_idx ≥ L 일 때만 evict mask 적용
- 또는 layer별 attention mask를 리스트로 받는 래퍼 (attnstat.QKCapture와 동일한 패치 방식)
- 검증: L=0이면 기존 전층 evict와 일치, L=N이면 full과 정확히 일치해야 함 (M0-D와 같은 논리)
```

### 1.5 판정 규칙 (사전 정의)

1. **잡음 기준선 먼저**: L=N(무개입)에서 d=0이어야 하고, 합법적 계산 변형(chunked
   prefill)에서의 d 분포를 재서 **잡음 대역**을 구한다. 이 대역 아래의 d(L) 차이는
   "무해"로 해석하지 않는다 — 판정 불가로 둔다.
2. **로짓 거리는 스크리닝 전용**: d(L)이 작다고 "성능 무손실"이라고 쓰지 않는다.
   후보 층 2~3개로 좁힌 뒤 반드시 **task metric으로 재판정**한다 (원칙 P2).
   파일럿에서 이미 로짓과 task metric이 갈린 사례가 있다(probe의 logp 개선이 EM
   개선으로 항상 이어지지 않음, FINDINGS §7).
3. **정보 유형별 분해 필수**: 평균 d(L)만 보면 FEATHER 함정에 그대로 빠진다.
   최소한 OCR/semantic vs grounding·layout으로 나눠 곡선을 따로 그린다
   (M4 유형 축, T4 파일럿 데이터로 가능).
4. **write-time 변형에서 재실행**: 위 절차를 미래 질문이 문맥에 있는 상태로 돌리면
   read-time 칸이다(P7). 아이디어 1의 합법 변형(generic 설명 prefill / 과거 에피소드
   prefill)에서 다시 스윕해야 저장용 압축 주장에 쓸 수 있다.
5. **최종 관문은 held-out**(P1): 층을 최적화한 결과가 **안 본 질문**에서 s5·random을
   이기는지가 유일한 채택 기준이다.

### 1.6 이 방법이 답하는 것 / 답하지 못하는 것

- 답한다: "몇 번째 층부터 쳐내도 계산이 크게 안 흔들리는가"(안전 시작 층의 상한),
  층 축에서의 민감도 프로파일.
- 답하지 못한다: 어떤 **집합**을 남겨야 하는가(선택 규칙 자체의 품질), 미래 질문
  일반화, 실제 task 성능, 정보 유형별 선택적 손실 — 전부 별도 실험이 필요하다.

---

## 2. 아이디어별 검증 경로 요약

| 아이디어 | 1차 스크리닝 | 본 판정 | 고유 위험 |
|---|---|---|---|
| 1. 층별 점진 pruning | **방법 A 층 스윕** (층×예산 격자) | 사다리 비교(20%·5%) → held-out | FastV/PyramidDrop 계열과의 신규성 구분; 이른 층 grounding 파괴 |
| 2. CPU-side 기억 관리 | CPU 선택 규칙(k-center 등)의 사다리 비교 | held-out + **M6 비용축(GPU-초/CPU-초 분리)** | 값 기반 신호 전례 나쁨(knorm 양방향 모두 random 이하); uniform(0.45)을 이겨야 함 |
| 3. 검증된 토큰 초안 | 무검증 T_visual 대비 개선폭 | held-out + bytes/read 비용 | **검증 기준의 과적합** — 과거 질문으로 검증하면 held-out에서 무너짐 |

---

## 3. 공통 보고 양식

모든 아이디어 검증 결과는 다음을 함께 보고한다.

```text
selector 이름 | 압축 시점 라벨 | 예산 B(bytes) | 실제 bytes/utilization
task metric (ANLS/EM) + 이미지 단위 95% CI (screen-cluster bootstrap)
held-out task metric (안 본 질문) + 같은 크기 random·s5 대조
기제 지표(로짓 거리·logp)는 부록으로만
write 비용(GPU-초/CPU-초 분리) + read 비용
구현 지위: upstream-runtime / vlm_adaptation / quality-simulation
```
