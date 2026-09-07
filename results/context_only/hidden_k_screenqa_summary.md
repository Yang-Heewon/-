# Stage 0C — hidden ↔ K 보완성 (experiments/manifests/screenqa_discovery.jsonl, 이미지 20, 시각 token)

| hidden 묶음 | K 대상 | 쌍별 코사인 Spearman | kNN@10 Jaccard | k-center 20% Jaccard | NovelCoverage |S_H−S_K|/|S_H| | |S_K−S_H|/|S_K| |
|---|---|---|---|---|---|---|
| early | K_late | 0.448 | 0.291 | 0.135 | 0.762 | 0.762 |
| early | K_all | 0.504 | 0.365 | 0.152 | 0.737 | 0.737 |
| mid | K_late | 0.590 | 0.408 | 0.152 | 0.736 | 0.736 |
| mid | K_all | 0.586 | 0.423 | 0.157 | 0.729 | 0.729 |
| late | K_late | 0.669 | 0.538 | 0.181 | 0.695 | 0.695 |
| late | K_all | 0.566 | 0.407 | 0.156 | 0.730 | 0.730 |
| early | hidden late (참고) | 0.625 | 0.351 | 0.142 | — | — |

통과 기준: ρ < 0.8 이고 k-center Jaccard < 0.6 (그리고 0A 에서 hidden 의 구조 ≥ K).