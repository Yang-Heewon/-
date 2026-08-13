# G01 — published baseline 후보와 압축 시점 분류 (PROPOSED)

> **상태: PROPOSED.** 최종 목록 확정은 사용자 결정(G01/M2A-01). 표의 `TBV`(to be
> verified)는 웹에서 repo·license·commit 재확인이 필요한 항목이다 (작성 시점에
> subagent 웹 조사가 API 장애로 불가하여 로컬 문서
> [MOTIVATION_ANALYSIS.md](../MOTIVATION_ANALYSIS.md), [PRIOR_WORK.md](../PRIOR_WORK.md),
> [BASELINES.md](../BASELINES.md)와 third_party pin 기준으로 작성).

## 1. 후보 표

| 후보 | 출처 | PLAN §1.2 timing | 선택 신호 (무엇을 소비하나) | 구현 상태/비용 | 검증 |
|---|---|---|---|---|---|
| random | — | write-time/query-agnostic | 없음 (통제 하한) | **로컬 구현 완료** (S0) | ✓ |
| spatial-uniform | — | write-time/query-agnostic | 공간 격자 | 구현 필요 (소) | ✓ |
| **KVzip-VLM 적응** (S5) | KVzip, NeurIPS 2025 | write-time/query-agnostic | 재구성 prompt("화면을 상세히 서술")에 대한 attention — 미래 질문 불사용 | **로컬 구현 완료** (signals.S5) — upstream은 text-LLM이므로 `vlm_adaptation` 라벨 필수 | 논문 수치는 MOTIVATION_ANALYSIS §1 기록 참조; repo/license TBV |
| kvpress **Knorm / ExpectedAttention** press | NVIDIA kvpress (pinned `4e41f14`, Apache-2.0) | write-time/query-agnostic | key norm / 기대 attention (query 무관) | third_party에 **pin 완료**, visual-only 적응 필요 (중) | ✓ pin·license 확정 |
| **H2O-style source-aware** | H2O, NeurIPS 2023 | write-time/**source-aware** | 과거 질문+**decoded 출력**까지의 누적 attention — episode 종료 후 합법 | S1 코드 재사용으로 구현 가능 (중소). **M3의 F_w 대표** | 논문·repo TBV |
| S1 (SnapKV-style) | SnapKV, NeurIPS 2024 | **read-time/query-aware** | 현재(=미래) 질문의 attention | **로컬 구현 완료** — read-time comparator 라벨 고정 (BASELINES.md 규칙) | ✓ |
| GUI-KV | 2025 (스쿱 맵 G 참조) | **read-time/query-aware** — ω=8 최근 토큰 관찰창이 질문을 봄 (MOTIVATION_ANALYSIS의 미발사 공격 지점) | query-window attention + spatial saliency | 적응 비용 중 — read-time comparator로만 | repo 공개 여부 TBV |
| PyramidKV | 2024 | read-time/query-aware | layer별 예산 + query attention | 적응 비용 중 | TBV |
| VisionZip | CVPR 2025 | write-time/query-agnostic (인코더 단) | ViT attention 기반 토큰 선택 — LLM KV가 아니라 **인코더 토큰 pruning** | 비교 축이 달라 M2-B TRANSFORMED 계열 후보로 재분류 권장 | TBV |

## 2. Timing 분류 근거 (§1.2 표와의 대응)

- **S1/SnapKV·GUI-KV가 read-time인 이유**: selector 입력에 현재 질문(또는 질문을 포함한
  최근 토큰 창)의 attention이 들어간다. 저장 시점엔 미래 질문이 없으므로 이 방식으로
  persistent 압축을 하려면 full KV를 read까지 보관해야 한다 → "저장 압축 baseline 아님".
- **KVzip 계열이 write-time인 이유**: 재구성 질의는 이미지 자체에서 만들어지며 미래
  질문과 무관. 단 upstream은 text-LLM 대상이므로 우리 S5는 반드시 `vlm_adaptation`으로
  보고한다.
- **H2O-style이 source-aware인 이유**: episode가 끝난 시점에는 과거 질문·모델 출력이
  "이미 일어난 일"이라 합법. gold answer가 아니라 **자기 decoded 출력**을 쓰는 변형만
  배포 가능 조건이다 (gold 사용 시 diagnostic으로 강등).

## 3. 추천 조합

**안 A — 최소 (권장 시작점)**
- write-time: S5(KVzip-VLM) + kvpress Knorm 적응
- source-aware: H2O-style (M3 F_w)
- read-time comparator: S1
- 통제: random + spatial-uniform

이유: 전부 로컬 pin 또는 기존 코드 재사용이라 구현 리스크 최소이고, §1.2의 세 행
(agnostic/source-aware/read-time)을 각각 대표한다. P1 headline("write-time selector가
probe 대비 ≥15%p 잃는다")에 필요한 F_w가 두 종류(agnostic·source-aware) 확보된다.

**안 B — 안 A + GUI-KV** (repo 접근 확인 시)
- read-time SOTA를 GUI 도메인 대표로 추가 → "query-aware SOTA조차 probe에 못 미친다/미친다"
  비교와 α-분해 공격(MOTIVATION_ANALYSIS)이 가능해짐. ScreenQA 실험 전 확보 권장.

**안 C — 안 B + kvpress ExpectedAttention**
- write-time 축을 2개→3개로. 계산량 증가 대비 P1 대표성 이득은 한계적 — M2-A 결과가
  selection 문제를 가리킬 때만.

## 4. P1 headline 대표성

P1은 "**published write-time policy**가 target probe 대비 얼마나 잃는가"이므로, 목록의
필수 조건은 (a) write-time 후보 ≥ 2 (약한 하나만 있으면 strawman 반론), (b) read-time
comparator ≥ 1 (상한 참조), (c) random/uniform (하한). 안 A가 이 최소 조건을 만족한다.

## 5. 사용자가 확정할 것

- [ ] 안 A/B/C 중 선택 (권장: A로 시작, GUI-KV repo 확인 후 B 승격)
- [ ] TBV 항목의 repo·license·commit 확인 후 third_party pin 추가
- [ ] H2O-style의 "decoded 출력" 범위 (답만 vs 전체 생성)
- [ ] 각 후보의 upstream-runtime / vlm_adaptation / quality_simulation 라벨 확정
