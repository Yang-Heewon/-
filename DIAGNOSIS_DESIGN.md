# DIAGNOSIS_DESIGN.md — 문제의식 확립을 위한 진단 실험 재설계 (v2)

> v1(`EXPERIMENTS.md`)에 대한 비판적 분석 + 실행 가능한 대체 설계.
> **핵심 진단: v1은 "통계"를 재고 "피해"를 재지 않는다.** 리뷰어의 첫 질문인 *"그래서 그게 뭐가 문제인데?"* 에
> v1의 E1~E4는 답할 수 없다. v2는 모든 주장을 **태스크 성능 손실**에 접지시킨다.

---

## 0. 이 환경에서 검증된 사실 (v1 가정과 다름)

실측했다. v1 계획의 전제 3개가 틀렸다.

| 항목 | v1 가정 | **실측값** | 영향 |
|---|---|---|---|
| GPU | A100 | **Tesla V100-DGXS 32GB × 4** (sm_70) | ⚠️ |
| bf16 | 암묵적 사용 | **네이티브 미지원** (`is_bf16_supported(including_emulation=False)=False`) → 에뮬레이션, 극도로 느림 | **fp16 강제** |
| FlashAttention-2 | — | **불가** (sm_80+ 필요) | eager/SDPA만 |
| 시간 예산 | Phase1 ~40분 | V100은 A100 대비 3~5× 느림 | **~3시간으로 재산정** |

### 0.1 치명적 실행 불가 지점 — `output_attentions=True`

E1/E2/E4는 전부 전층 attention 맵을 요구한다. Qwen2.5-VL-7B(L=28, H=28) 기준 실측 계산:

| 시퀀스 길이 | KV 캐시(fp16) | **`output_attentions=True` 전층 attention** |
|---|---|---|
| 2,048 | 0.12 GB | 6.6 GB |
| 4,096 | 0.23 GB | **26.3 GB** |
| 6,000 | 0.34 GB | **56.4 GB** |
| 8,192 | 0.47 GB | **105.2 GB** |

1080p 스크린샷 1장 = **2,584 시각 토큰** (Qwen2.5-VL, 28×28px/token) → 프롬프트 포함 ~2.7k.
**GUI 이미지 2장만 넣어도 32GB V100에서 즉시 OOM.** v1의 E1/E2/E4는 현재 형태로 실행 자체가 안 된다.
→ §3.1의 메모리 안전 커널로 반드시 교체.

### 0.2 C1(메모리 폭증)의 전제가 과장됨 — 동기를 바꿔야 함

두 모델 모두 **GQA**라서 KV가 생각보다 작다:

| 모델 | L / KV-heads / d | **KV/토큰** | 1080p 1장 | 10-step 궤적 |
|---|---|---|---|---|
| Qwen2.5-VL-7B | 28 / 4 / 128 | 56 KB | **0.15 GB** | 1.5 GB |
| Qwen3-VL-8B | 36 / 8 / 128 | 144 KB | **0.37 GB** | 3.7 GB |

즉 *"1080p 스크린샷 1장 = ○○MB KV, k-스텝이면 △GB"* 라는 v1의 C1 서사는 **단일 세션 기준으로는 안 먹힌다.**
10스텝 해봐야 1.5GB이고 32GB 카드에 그냥 들어간다. 이대로 쓰면 리뷰어에게 바로 반박당한다.

**정직한 재구성 (→ D0):** 진짜 병목은 GB가 아니라
1. **프리필 연산량/TTFT** — 궤적이 쌓이면 시퀀스가 25k+ 토큰이 되고 attention이 2차로 증가,
2. **서빙 동시성** — 배치 32 세션 × 3.7GB = 119GB → 여기서 터진다,
3. **에이전트 루프의 반복 프리필** — 매 스텝 재계산.

D0는 이걸 **측정**한다. 산술로 주장하지 않는다.

---

## 1. v1의 구조적 결함 6가지

### ❶ 모든 주장이 "통계"에서 끝나고 "피해"로 이어지지 않는다 (가장 큰 문제)
C2 "GUI attention이 평탄하고 희소하다" → **So what?** 평탄한 게 왜 나쁜지 아무도 안 보여줬다.
E5만 태스크 성능을 보는데 맨 마지막에, 모델 2개, 100건뿐이다.
> **수정:** 순서를 뒤집는다. **피해를 먼저 보이고(D1), 원인을 나중에 설명한다(D2~D3).**

### ❷ C2의 지표가 길이 교란(length confound)에 오염됨
`coverage@0.9` = 어텐션 질량 90%를 덮는 토큰 **비율**. 이건 시퀀스 길이에 강하게 의존한다.
GUI 이미지는 자연 이미지보다 토큰이 **훨씬 많다**(2584 vs ~700). 따라서
*"GUI에서 coverage가 낮다"* 는 결과는 **희소성이 아니라 단순히 토큰이 많아서** 나올 수 있다.
v1은 이 교란을 통제하지 않으므로 C2 판정은 무효다.
> **수정:** ① `--max_pixels`로 **시각 토큰 수를 조건 간 동일하게 고정**, ② 길이 불변 지표
> (정규화 엔트로피 H/log L, Gini, 길이 매칭 null 대비 비율)를 병행. §D2.

### ❸ C3의 "텍스트 과대추출"은 위치·개수·싱크와 교란됨
텍스트 토큰이 시각 토큰보다 토큰당 attention을 많이 받는 건 **거의 자명하다**:
(a) BOS/시스템 프롬프트의 **attention sink**, (b) 텍스트 토큰 수가 압도적으로 적음(토큰당 값이 커짐),
(c) 질문이 이미지 **뒤에** 와서 쿼리와 위치적으로 가까움(recency).
v1의 intra/inter 비율은 이 3개를 modality 효과로 잘못 귀속시킨다.
> **수정:** **싱크 토큰 명시적 제외 + 위치 스왑 통제(image-before-text vs text-before-image) +
> 개수 매칭**. 이 3개 통제 후에도 남는 격차만 modality gap으로 인정. §D2.

### ❹ E4는 **순환논증**이다 — 이게 방법론 전체를 무너뜨린다
E4는 `enc_embed_norm`(인코더 임베딩 노름)과 `value_norm`(V 노름)의 상관을 본다.
그런데 **V = W_v · hidden**이고, **layer 0의 hidden = 인코더 임베딩 그 자체**다.
즉 초기층의 높은 상관은 **선형사영에 의해 구조적으로 보장**된 것이지 발견이 아니다.
v1대로면 **가짜 GREEN**이 나오고, 그 위에 방법을 설계하게 된다.

> **수정 (v2의 핵심 기여):** 상관의 **타깃을 바꾼다.** 노름 vs 노름이 아니라
> **oracle KV 중요도** — 실제로 그 토큰의 KV를 지웠을 때 정답 로그확률이 얼마나 떨어지는가.
> 이게 "중요도"의 유일한 정의다. §D3.

### ❺ C6(query-aware 붕괴)이 가장 강한 카드인데 맨 뒤에 묻혀 있다
그리고 설계도 약하다. "범용 질문으로 점수 매기기"는 에이전트 시나리오가 아니다.
실제 시나리오는 **캐시를 만들 때 미래 지시를 모른다**는 것.
> **수정:** **K×K 쿼리 전이 행렬**로 승격. 대각=쿼리 일치(논문들이 보고하는 값),
> 비대각=재사용(에이전트가 실제로 겪는 값). 여기에 **union coverage 천장 분석**을 붙인다. §D4.

### ❻ 베이스라인이 전부 자작 허수아비다
`random/streaming/value_norm/attn_prefill`은 출판된 방법이 아니다. 문제제기 논문은
**출판된 방법이 실패한다**를 보여야 한다. `attn_prefill`은 사실상 **SnapKV**이니 그렇게 부르고,
**H2O, PyramidKV, StreamingLLM**, 그리고 VLM 전용인 **FastV, VisionZip**을 넣어야 한다.

### 기타
- **통계 없음**: 조건당 이미지 1장, 시드 없음, 신뢰구간 없음. "모든 모델에서 임계값 초과"를 n=1로 주장 불가 → **도메인당 ≥50샘플 + 부트스트랩 CI**.
- **에셋 부재**: `traj_A/B`, `eval.jsonl`, `screenshot.png` 모두 존재하지 않음 → §5 데이터 계획.
- **M4(LLaVA-OneVision)** 는 핵심인 E3/E4에서 빠지므로 "아키텍처 축" 역할을 못 함. 차라리 **InternVL3-8B**(가용 확인됨)로 교체하거나 축을 정직하게 포기.

---

## 2. v2 설계 원칙 — 인과 사슬 하나로 꿴다

문제의식은 **다섯 개의 링크로 된 하나의 사슬**이어야 하고, 각 링크가 하나의 실험이다.

```
D0  비용이 실재한다        →  프리필/동시성이 궤적 길이에 따라 무너진다
D1  기존 압축이 실패한다   →  텍스트에선 되는 20% 예산이 GUI/doc에선 안 된다   ★핵심 그림
D2  왜: 신호가 퇴화했다    →  (길이·위치·싱크 통제 후) attention이 변별력을 잃는다
D3  대안 신호도 안 맞는다  →  oracle 중요도와 어떤 값싼 신호도 정렬되지 않는다  ★방법 분기점
D4  패러다임 내 수리 불가  →  query-aware는 재사용에서 구조적으로 붕괴한다      ★가장 강한 카드
D5  그러나 여지는 있다     →  시간 중복은 실제로 버려도 된다 (인과 검증)        → 방법의 씨앗
D6  기억이 될 수 있다      →  이식성(D6a)·저장 경제성(D6b) — PRIOR_WORK.md §5     → 상위 프레임
```

> **상위 프레임 (2026-08-13 갱신):** 이 진단의 최종 목적은 "압축 시각 KV = 에이전트 장기기억"이다.
> D4는 기억의 R1(query-agnostic), 신규 D6은 R2(위치 이식성)·R3(저장 경제성)을 검증한다.
> D6 명세와 판정 P7은 `PRIOR_WORK.md` §5.3 참조. D6a는 게이트: 실패 시 기억 프레임을 접는다.

**사전등록 규칙(v1에서 유지할 좋은 점):** 모든 임계값은 실행 **전에** 고정. 모든 모델에서 성립 → `C`(공통),
일부만 → `S`(모델 특이). **S도 버리지 않는다.** 반증되면 `✗`로 기록하고 그대로 논문에 쓴다.

---

## 3. 공통 인프라 (이걸 먼저 만들어야 나머지가 돌아간다)

```
vlm_diagnosis/
├── core/
│   ├── loader.py        # fp16 고정, attn_implementation="eager", trust_remote_code
│   ├── spans.py         # 시각/텍스트/싱크 토큰 인덱스 분해 (모델별 image_token_id)
│   ├── attnstat.py      # ★ 메모리 안전 attention 통계 (§3.1)
│   ├── ablate.py        # ★ oracle KV 마스킹 중요도 (§3.2)
│   ├── compress.py      # 압축 정책 레지스트리 (full/random/streaming/H2O/SnapKV/PyramidKV/FastV/value_norm/enc_norm)
│   └── stats.py         # 부트스트랩 CI, Spearman, 길이 매칭 null
├── exps/  d0_cost.py d1_failure.py d2_signal.py d3_oracle.py d4_transfer.py d5_temporal.py
└── results/{M1,M2,M3}/{d0..d5}/
```

### 3.1 메모리 안전 attention 통계 — `output_attentions=True`를 절대 쓰지 마라

전층 맵을 **materialize하지 않고** Q·K만 후킹해서 필요한 축약량(열 합계)만 청크로 계산한다.
Q,K 저장량은 n=6000에서 **총 1.4GB**뿐 (vs 56GB).

```python
# 각 layer.self_attn에 hook을 걸어 post-RoPE q,k만 캡처 → 맵은 만들지 않는다
@torch.no_grad()
def attn_column_mass(q, k, causal_start, chunk=512):
    """q:(H,n,d) k:(Hkv,n,d) → 각 key 위치가 받은 attention 총량 (n,)"""
    H, n, d = q.shape
    k = k.repeat_interleave(H // k.shape[0], dim=0)          # GQA 확장
    recv = torch.zeros(n, device=q.device, dtype=torch.float32)
    for s in range(0, n, chunk):                              # 쿼리 축을 청크로
        e = min(s + chunk, n)
        w = (q[:, s:e] @ k.transpose(-1, -2)) / math.sqrt(d)  # (H, chunk, n)
        idx = torch.arange(s, e, device=q.device)
        w.masked_fill_(torch.arange(n, device=q.device)[None, None, :] > idx[None, :, None], -torch.inf)
        recv += w.softmax(-1).sum(dim=(0, 1)).float()         # 즉시 축약 후 폐기
        del w
    return recv / (H * n)
```
`coverage@k`, 엔트로피, Gini는 전부 이 `recv` 벡터에서 나온다. 맵 저장 불필요.

### 3.2 Oracle KV 중요도 — v2의 심장

"중요도"의 **유일하게 정직한 정의**: 그 KV를 지우면 정답이 얼마나 망가지는가.

```python
@torch.no_grad()
def oracle_group_importance(model, inputs, vis_idx, answer_ids, n_groups=64):
    """1회 프리필 + G회 값싼 teacher-forced 디코드 → 그룹별 Δlogp"""
    out  = model(**inputs, use_cache=True)                    # 프리필 1회 (재사용)
    kv   = out.past_key_values
    base = teacher_forced_logp(model, kv, answer_ids, mask=None)
    groups = spatial_groups(vis_idx, n_groups)                # 시각 그리드 8×8 블록
    imp = []
    for g in groups:
        m = torch.ones(kv_len, dtype=torch.bool); m[g] = False   # 해당 KV만 차단
        imp.append(base - teacher_forced_logp(model, kv, answer_ids, mask=m))
    return torch.tensor(imp)      # 클수록 중요 (제거 시 정답 로그확률 하락폭)
```

- **비용**: 프리필 1회 + G회 × (정답 길이 ~10토큰) 포워드. 프리필 재사용하므로 이미지당 수 초.
- **왜 마스킹이 곧 제거인가**: softmax가 재정규화되므로 attention 관점에서 KV 마스킹 = KV 축출과 동치. 실제 eviction이 하는 일과 같다.
- **⚠️ 구현 주의**: Qwen2.5-VL은 mRoPE + 커스텀 마스킹을 쓴다. 2D `attention_mask`가 무시될 수 있으니
  **4D 마스크를 직접 구성**하고, 위생검사로 *"전체 시각 KV 마스킹 → logp 대폭 하락"* 을 반드시 먼저 확인할 것.
  이 sanity check가 통과 안 되면 D3 결과는 전부 무의미하다.
- **레이어별 변형**: 특정 층 ℓ에서만 마스킹 → 층별 중요도. G×L회라 비싸므로 **부분집합에만** 적용.

---

## 4. 실험 D0 ~ D5

### D0 — 비용 봉투 (C1 대체)
**목적:** 산술이 아니라 측정으로 "비용이 실재한다"를 세운다.
**설계:** 궤적 길이 1→12 스텝을 누적 프리필하며 매 스텝 기록:
KV GB(시각/텍스트 분해), **TTFT(ms)**, peak GPU mem, 실제 OOM 지점, 그리고
**최대 동시 세션 수**(32GB에서 몇 세션까지 얹히는가).
**측정량:** KV GB vs 스텝, TTFT vs 스텝(2차 성장 여부 fit), 시각 KV 점유율 %.
**사전등록 판정 P1:** 모든 모델에서 (a) 시각 KV 점유율 **>90%**, (b) TTFT가 스텝 수에 **초선형(exponent >1.3)** → **P1 확정**.
> 논문 문장: *"에이전트 궤적 k스텝에서 KV의 ○○%가 시각 토큰이고 TTFT는 k^△로 증가한다."*
> ⚠️ (a)만 성립하고 (b)가 선형이면 → 동기를 **동시성/처리량**으로 재구성 (§0.2).

---

### D1 — 기존 압축의 실패 ★핵심 그림
**목적:** *"압축 문헌의 '20% 예산이면 무손실'이 이 워크로드에선 성립하지 않는다"* 를 보인다.
**설계:** 2×2 요인 — {정책} × {도메인} × {예산}
- **정책(출판된 것 위주):** `full`, `random`, `StreamingLLM(sink+recent)`, `H2O`, `SnapKV`, `PyramidKV`, `FastV`(시각 토큰 프루닝), `value_norm`, `enc_norm(ours-v0)`
- **도메인:** `text-only QA`(★대조군, 같은 모델), `natural(VQAv2/TextVQA)`, `doc/chart(DocVQA/ChartQA)`, `GUI(ScreenSpot/Mind2Web)`
- **예산:** {5, 10, 20, 30, 50, 100}%
- **샘플:** 도메인당 **≥200건**, 부트스트랩 95% CI.
**측정량:** budget–accuracy 곡선, **retention@20% = acc(0.2)/acc(1.0)**.
**사전등록 판정 P2:** 모든 모델에서 최고 성능 정책조차
`retention@20%(GUI) 및 (doc)` 가 `retention@20%(text-only)` 보다 **≥10%p 낮음** → **P2 확정**.
> **text-only 대조군이 이 실험의 생명이다.** 같은 모델에서 텍스트는 되고 GUI는 안 된다를 보여야
> "모델이 약해서"가 아니라 "모달리티/워크로드 때문"이라고 귀속할 수 있다.
> ⚠️ 만약 GUI에서도 20%가 멀쩡하면 → **문제의식 자체가 없다.** 그땐 예산을 5%까지 낮춰 붕괴점을 찾고,
> 논문 프레이밍을 "극저예산 영역"으로 재조정해야 한다. **이 실험을 가장 먼저 돌려라.**

---

### D2 — 왜: 신호의 퇴화 (C2+C3 교란 제거판)
**목적:** D1의 실패를 attention 신호의 성질로 설명한다.
**설계(통제가 핵심):**
- **길이 통제:** `--max_pixels`로 모든 도메인의 시각 토큰 수를 **1280개로 고정**. 추가로 native-res 조건 병행.
- **싱크 제외:** 앞 4토큰 + 시스템 프롬프트 구간을 통계에서 제거.
- **위치 스왑:** `[img][text]` vs `[text][img]` 두 순서 모두 측정 → 위치 효과 분리.
- **null 기준선:** 같은 길이의 균등분포/셔플 대비 정규화.
**측정량(전부 길이 불변):**
`H/log L`(정규화 엔트로피), `Gini`, `coverage@0.9 / coverage@0.9(null)`,
**층간 순위상관** `mean_ℓ Spearman(recv_ℓ, recv_{ℓ+1})`, 그리고
**modality gap** = 위치·개수·싱크 통제 후 텍스트/시각 토큰당 attention 비.
**사전등록 판정:**
- **P3a:** 모든 모델에서 GUI의 `H/log L > 0.9` **그리고** `Gini(GUI) < Gini(text)` → *"변별력 붕괴"* 확정.
- **P3b:** 층간 순위상관이 GUI에서 **>0.8** → *"층 특화 없음 = 층별 예산 배분(PyramidKV류)의 전제가 GUI에서 무너짐"*.
  반대로 **<0.2**면 *"층마다 딴소리 = 단일 전역 예산 불가"*. **양 극단 모두 문제이며 둘 다 논문거리다.**
- **P3c:** 위치 스왑·싱크 제외 후에도 텍스트 편향이 **>1.3배 잔존** → modality gap 확정.
  통제 후 사라지면 → **C3는 ✗로 기록하고 정직하게 폐기.**

---

### D3 — Oracle 정렬: 값싼 신호가 진짜 중요도를 맞히는가 ★방법 분기점
**목적:** v1 E4의 순환논증 제거. **방법 설계의 근거를 여기서 얻는다.**
**설계:** §3.2의 oracle Δlogp를 정답으로 두고, 후보 신호들과 **Spearman ρ**를 잰다.

| 후보 신호 | 종류 | 왜 넣는가 |
|---|---|---|
| `attn_received` (SnapKV식) | query-aware | 현 SOTA의 신호 |
| `attn_received_generic` | query-agnostic | 쿼리 제거 시 얼마나 남는가 |
| `value_norm`, `key_norm` | 내부 | v1이 보던 것 |
| `enc_embed_norm` (proj 전/후) | 인코더 | **ours 가설** |
| `ViT last-block attention`, `attention rollout` | 인코더 | PIVOT 후보 |
| **`patch pixel variance` / edge energy** | **무학습** | ★**결정적 대조군** |
| `temporal Δ` (직전 프레임 대비) | 시간 | D5와 연결 |

> **`patch pixel variance`를 반드시 넣어라.** 공짜 이미지 통계가 인코더 saliency만큼 맞히면
> "인코더 신호"라는 기여는 그 자리에서 증발한다. 이 대조군 없이 GREEN 판정하면 리뷰에서 죽는다.

**사전등록 3갈래 판정 P4:**
- **GREEN** — `enc_embed_norm`의 평균 ρ > 0.3 이고 **pixel-variance 대비 ≥0.1 우위** → 인코더 신호 단독 설계.
- **HYBRID** — 특정 층 구간에서만 성립 (특히 M2의 DeepStack 주입층 **0–2**, `deepstack_visual_indexes=[8,16,24]` 확인됨)
  → 층 가중 α(ℓ)·enc + (1−α)·내부신호. **D3가 α의 초기값을 준다.**
- **PIVOT** — 전부 ρ≈0 → *"어떤 값싼 신호도 KV 중요도를 예측하지 못한다"*.
  **이것도 논문거리다** (부정적 결과 + 원인분석: 문맥화가 중요도를 재분배).
- **P4′ (추가 공통문제):** 어떤 단일 신호도 **전 층을 커버 못 함** → *"단일 신호로는 불충분"* = 하이브리드 방법론의 일반 근거.

---

### D4 — Query-agnosticity 스트레스 테스트 ★가장 강한 카드
**목적:** v1 C6를 헤드라인으로 승격. 에이전트 캐시 재사용의 구조적 붕괴를 정량화.
**설계 A — K×K 전이 행렬:** 이미지당 서로 다른 질문 **K=8**개
(DocVQA는 이미지당 다중 QA 존재, GUI는 스크린샷당 다중 지시 사용).
쿼리 `q_i`로 캐시를 압축 → 쿼리 `q_j`로 평가 → K×K 정확도 행렬.
- **대각선** = 쿼리 일치 = *논문들이 보고하는 숫자*
- **비대각** = 재사용 = *에이전트가 실제로 겪는 숫자*
**설계 B — union coverage 천장 (가장 중요한 그림):**
K개 쿼리의 oracle 중요 토큰 집합의 **합집합**을 덮으려면 예산이 얼마나 필요한가를 K=1..8로 플롯.
- 낮은 값에서 **포화** → query-agnostic 압축이 *원리적으로 가능* → **우리 방법의 존재 근거**
- **선형 증가** → 압축의 근본적 한계 → 그것대로 강한 부정적 결과
**설계 C — 멀티턴 드리프트:** 스텝 t에서 압축 → t+1..t+5에서 재사용 (Mind2Web 실궤적).
**사전등록 판정 P5:** 모든 모델에서 query-aware 정책(SnapKV/H2O)의 **대각−비대각 격차 ≥5%p**,
동시에 query-agnostic 정책(value_norm/enc_norm)의 격차는 **≤2%p** → **P5 확정**.
> 논문 문장: *"미래 지시를 모르는 압축에서 query-aware 점수는 붕괴한다."*

---

### D5 — 시간 중복: 상관이 아니라 **인과**로 (C4 수정)
**목적:** v1 C4는 "코사인 유사도 >0.9"라는 **상관**만 보고 "버려도 된다"는 **인과**를 주장한다. 그건 비약이다.
**설계:**
1. **상관부:** 연속 프레임 정렬 후 **pre-RoPE K** 코사인(RoPE는 위치 때문에 비교를 오염시킴 — v1이 이건 맞게 잡음),
   `cos>0.95` 토큰 비율. *정렬 전제: 프레임 해상도/토큰 그리드를 사전 리사이즈로 강제 통일.*
2. **인과부(핵심):** `cos>τ`인 과거 프레임 토큰의 KV를 **실제로 제거** → **행동 예측 정확도** 측정. τ 스윕.
3. **oracle 교차:** 정답 관련 정보가 최신 프레임에 얼마나 몰려 있는가 (D3 oracle을 프레임별로 집계).
**사전등록 판정 P6:** 모든 모델에서 중복 제거로 과거 시각 KV의 **≥40%를 제거해도 정확도 하락 <1%p**
→ **P6 확정** = *"기존 방법들이 시퀀스 내부만 보느라 프레임 간 중복을 못 쓴다"* = 방법의 두 번째 축.

---

## 5. 데이터 계획 (가용성 확인 완료)

v1의 에셋은 **하나도 존재하지 않는다.** 실제로 받을 수 있는 것으로 대체:

| 용도 | 데이터셋 | 상태 |
|---|---|---|
| doc/chart | `lmms-lab/DocVQA`, `lmms-lab/ChartQA` | ✅ 확인 |
| natural | `lmms-lab/VQAv2`, `lmms-lab/textvqa` | ✅ 확인 |
| GUI 그라운딩 | `rootsautomation/ScreenSpot`, `OS-Copilot/OS-Atlas-data` | ✅ 확인 |
| **GUI 궤적(D5 필수)** | **`osunlp/Multimodal-Mind2Web`** (스크린샷+행동 시퀀스) | ✅ 확인 |
| text-only 대조군(D1 생명줄) | 같은 모델에 이미지 없이 QA (예: TriviaQA/HotpotQA 서브셋) | 구성 필요 |
| ❌ | `GUI-Odyssey`, `google/android_control` | **접근 불가** |

**모델:** M1 `Qwen2.5-VL-7B` ✅ **이미 캐시됨** / M2 `Qwen3-VL-8B` ✅ / M3 `UI-TARS-1.5-7B` ✅ (다운로드 필요, 디스크 204GB 여유).
M4는 `InternVL3-8B`(✅ 가용)로 교체하거나 축을 포기 — LLaVA-OneVision은 핵심 실험에 못 들어가므로 의미 없음.

---

## 6. 마스터 판정표 (이것만 채우면 논문 Section 2가 나온다)

| 코드 | 주장 | 실험 | 사전등록 기준 | 판정(C/S/✗) | 논문 문장 |
|---|---|---|---|---|---|
| P1 | 비용이 실재 | D0 | 시각KV>90%, TTFT 초선형(>1.3) | | |
| **P2** | **기존 압축 실패** | **D1** | **retention@20%: GUI/doc가 text 대비 ≥10%p 낮음** | | |
| P3a | 신호 변별력 붕괴 | D2 | H/logL>0.9, Gini(GUI)<Gini(text) | | |
| P3b | 층 구조 붕괴 | D2 | 층간 ρ>0.8 (또는 <0.2) | | |
| P3c | modality gap | D2 | 3중 통제 후 >1.3배 잔존 | | |
| **P4** | **신호 정렬** | **D3** | **GREEN/HYBRID/PIVOT 3갈래 + pixel-var 대조** | | |
| **P5** | **query-aware 붕괴** | **D4** | **대각−비대각 ≥5%p, agnostic은 ≤2%p** | | |
| P6 | 시간 중복 제거 가능 | D5 | 40% 제거 시 하락<1%p | | |

**교차 해석:** P2+P3+P5가 모두 C → *"LLM 내부 attention은 이 워크로드에서 신뢰할 수 없는 보존 신호"* 한 문장으로 묶인다 → 인코더-신호 방법의 1문단 동기.
P4가 GREEN이고 P6가 C → 최종 방법 = **인코더 saliency(공간) × 시간 중복(시간)** 2축 점수가 **데이터에서 도출**된다.

---

## 7. 실행 순서 — 위험이 큰 것부터

**v1처럼 E1→E5 순서로 가면 안 된다.** 가장 반증 위험이 큰 것부터 돌려서 빨리 죽이거나 살린다.

| 순서 | 실험 | 왜 여기인가 | V100 예상 |
|---|---|---|---|
| **1** | **D1 (축소판: M1, 3정책, 3도메인, 50건)** | **P2가 실패하면 연구 자체가 무의미.** 반나절 안에 확인 | ~2h |
| **2** | **D3 sanity + 축소판 (M1, 20이미지)** | oracle 마스킹이 동작 안 하면 D3/D4가 전부 무너짐 | ~2h |
| 3 | D4 (M1, K=8, 30이미지) | 가장 강한 카드의 조기 검증 | ~3h |
| 4 | D0, D2 (전 모델) | 값싸고 안전. 서술 보강용 | ~3h |
| 5 | D1 전체 (전 모델·전 예산·200건) | 확정 수치 | 밤새 |
| 6 | D5 (Mind2Web 궤적) | 방법 설계 입력 | ~4h |

**게이트:** 1·2단계에서 P2가 ✗거나 oracle sanity가 깨지면 **거기서 멈추고 프레이밍을 다시 짠다.**

| 증상 | 조치 |
|---|---|
| OOM | `output_attentions` 쓰고 있는지부터 확인 (§3.1). 그다음 `--max_pixels 600*28*28` |
| M3 로딩 실패 | UI-TARS-1.5는 Qwen2.5-VL 아키텍처 → `trust_remote_code=True` |
| bf16 관련 에러/극저속 | **V100은 bf16 네이티브 미지원.** `torch_dtype=torch.float16` 강제 |
| oracle sanity 실패 | 2D mask가 무시되는 것 — 4D 마스크 직접 구성 |
| D5 프레임 정렬 불가 | 사전 리사이즈로 토큰 그리드 강제 통일 |

---

## 8. v1에서 버리는 것 / 살리는 것

**살린다:** 모델 격자(스케일/파인튜닝/도메인 3축), 사전등록 임계값 문화, C/S/✗ 기록 규칙, pre-RoPE K 비교, 산출물 스태싱 규칙.
**버린다:** `output_attentions` 기반 전층 맵(실행 불가), 길이 미통제 coverage@0.9, 노름-대-노름 상관(순환), 통제 없는 modality gap, 자작 허수아비 베이스라인, n=1 통계, KV GB 중심의 C1 서사.
