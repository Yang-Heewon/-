# M1 — 무엇을 저장하고 어떻게 재사용할 수 있는가

**상태:** `PLANNED`  
**질문:** 압축하지 않은 상태에서도 memory payload가 어디서 처음 깨지는가?
**실행 계약:** [configs/m1.yaml](configs/m1.yaml)

## 1. 선행 gate

- M0 최소 gate를 통과해야 runner 구현 가능
- 결과 해석 전에는 M0 전체 통과 필요
- [m1.yaml](configs/m1.yaml)과 `m1_canonical.jsonl` 확정

## 2. 비교 이름

- `CACHE-IDENTITY`: M0 구현 gate
- `IMAGE-REENCODE`: read 시 원본 이미지 재인코딩
- `STORED-FULL`: write 시 만든 `K_p+K_v+Z`를 이미지 없이 주입
- `STORED-FULL FIDELITY`: 위 두 조건의 task/logit 차이

STORED-FULL 실패는 구현 오류로 자동 처리하지 않는다. CACHE-IDENTITY가 통과한 상태라면
“KV가 write 문맥과 위치에 결박됐다”는 연구 후보가 된다.

## 3. 저장 payload 원자

| 분류 | 원자 |
|---|---|
| 이미지 | `I` |
| 시각 text | `T_o`, `T_d`, `T_u` |
| episode text | `T_q`, `T_a`, `T_out`, `T_traj` |
| KV | `K_p`, `K_v`, `K_w` |
| 재조립 metadata | `Z` |

전체 조건 공간은 `vlm_diagnosis/core/memory_conditions.py`가 관리한다. 본실험은 전수조합을
돌리지 않고 아래 canonical manifest에서 시작한다.

## 4. canonical 10조건

1. `NO-MEM`
2. `IMAGE`
3. `T_visual`
4. `T_episode`
5. `IMAGE+T_visual`
6. `K_v`
7. `K_p+K_v`
8. `K_p+K_v+T_visual`
9. `IMAGE+K_v` — 중복 주입 sanity
10. `K_p+K_v+K_w` — smuggling-risk 진단

`IMAGE+T_episode`, `KV+T_episode`, full hybrid는 `T_episode` 주효과가 있을 때만 확장한다.

## 5. 한 번에 하나씩 바꿀 축

| 실험 | 고정하는 것 | 바꾸는 것 | 분리되는 원인 |
|---|---|---|---|
| M1-A | sequence, offset | IMAGE vs STORED-FULL | full payload fidelity |
| M1-B | offset, payload | write 주변 문맥 | context conditioning |
| M1-C | payload, 주변 문맥 | offset | mRoPE/position portability |
| M1-D | 위치 | `K_v` vs `K_p+K_v` | prefix 경계 누락 |
| M1-E | full block | single vs independent/joint 2·4 block | block composition |
| M1-F | canonical read 문맥 | canonical 10 payload | 표현 주효과 |
| M1-G | 효과가 난 조합 | image/text/KV 순서와 길이 control | 순서·길이 artifact |

## 6. 사용자가 수정·결정할 부분

| 결정 ID | 결정 | 왜 필요한가 |
|---|---|---|
| M1-01 | `K_p`가 끝나는 정확한 token index | visual-only와 prefix 포함 KV를 구분 |
| M1-02 | write/read token 순서 | q_w가 K_v에 조건화되는지 결정 |
| M1-03 | offset sweep | 위치 결박의 범위를 결정 |
| M1-04 | 2차 interaction 승격 기준 | 수천 registry 조건의 무분별한 실행 방지 |
| G02 | T_visual 생성 사양 | TEXT baseline 강도 결정 |

권장 시작:

```yaml
write_semantics: generic image write; q_w/action KV excluded from independent visual memory
offsets: [0, 128, 512, 2048]
block_counts: [1, 2, 4]
```

## 7. 현재 실행 가능 범위

condition registry 생성과 검증만 가능하다.

```bash
python -m vlm_diagnosis.core.memory_conditions \
  --scope core \
  --output /tmp/memory_conditions.jsonl
```

이 명령은 model experiment를 실행하지 않는다.

## 8. 목표 runner

```bash
python -m vlm_diagnosis.exps.m1_storage_reuse \
  --config experiments/configs/m1.yaml \
  --manifest experiments/manifests/m1_canonical.jsonl
```

구현해야 할 것:

- visual/prefix KV extraction과 serialization
- stored cache resume
- offset·mRoPE metadata 재주입
- canonical payload prefill/injection 조합
- task metric과 layer별 최초 divergence tracing

## 9. 결과 해석

| 결과 | 해석 | 다음 단계 |
|---|---|---|
| A 실패 | full KV가 이미지 동작을 재현하지 못함 | 누락 prefix/layer 확인, M2 보류 |
| A 성공, B 실패 | KV가 write 문맥에 조건화 | context portability 문제 후보 |
| B 성공, C만 악화 | 위치 이식 문제 | RoPE 축 원인 확인 후 M7 후보 |
| `K_v` 실패, `K_p+K_v` 성공 | 기억 단위가 visual block보다 큼 | 최소 prefix 경계 측정 |
| single 성공, independent concat 실패 | block 합성 문제 | M5 후보 |
| T_visual≈IMAGE | latent KV의 고유 가치가 약함 | M6에서 비용 비교 우선 |
| T_episode가 T0에서만 강함 | answer carryover | 미래 정보 보존 주장 금지 |

## 10. 완료 조건

M2에서 사용할 canonical KV 조건 하나가 다음을 만족해야 한다.

- IMAGE 대비 task metric 차이를 보고할 수 있음
- position/context가 고정되어 있음
- finite
- `K_w` smuggling 여부가 명시됨
- 결과가 `STORED-FULL FIDELITY` 성공인지 실패인지 판정됨
