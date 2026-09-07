# Stage 0C — hidden ↔ K 보완성 (experiments/manifests/gqa_discovery.jsonl, 이미지 20, 시각 token)

| hidden 묶음 | K 대상 | 쌍별 코사인 Spearman | kNN@10 Jaccard | k-center 20% Jaccard | NovelCoverage |S_H−S_K|/|S_H| | |S_K−S_H|/|S_K| |
|---|---|---|---|---|---|---|
| early | K_late | 0.565 | 0.449 | 0.180 | 0.697 | 0.697 |
| early | K_all | 0.616 | 0.510 | 0.185 | 0.691 | 0.691 |
| mid | K_late | 0.624 | 0.535 | 0.176 | 0.704 | 0.704 |
| mid | K_all | 0.641 | 0.544 | 0.180 | 0.698 | 0.698 |
| late | K_late | 0.654 | 0.604 | 0.193 | 0.679 | 0.679 |
| late | K_all | 0.612 | 0.518 | 0.175 | 0.705 | 0.705 |
| early | hidden late (참고) | 0.793 | 0.529 | 0.179 | — | — |

통과 기준: ρ < 0.8 이고 k-center Jaccard < 0.6 (그리고 0A 에서 hidden 의 구조 ≥ K).