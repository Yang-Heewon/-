# Invalid legacy D4 mini run

이 디렉터리의 raw JSONL/log는 다음 이유로 분석에서 제외한다.

1. FULL은 2D attention mask와 implicit position path를 사용했다.
2. keep-set은 4D attention mask와 explicit mRoPE position IDs를 사용했다.
3. 따라서 모든 `delta logp`에 attention/position 경로 차이가 섞였다.
4. shard0은 layer-27 fp32 patch 이후에도 `NaN logp: S0`로 실패했다.
5. 성공한 shard도 task metric과 T0–T4 label이 없다.

raw artifact는 실패 provenance로만 보존하며 M2-A/M3 결과에 포함하지 않는다.

