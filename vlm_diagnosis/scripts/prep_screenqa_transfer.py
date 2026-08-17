"""ScreenQA 전이 실험용 manifest 생성 — 근거 좌표로 쌍 라벨을 자동 계산한다.

DocVQA는 근거 위치 정보가 없어 201쌍을 사람이 라벨해야 했다. ScreenQA는 답마다
UI element bounds가 있으므로 두 질문의 근거가 같은 곳인지 자동으로 계산할 수 있고,
나아가 **근거 이동 거리를 연속값으로** 기록할 수 있다 (임계값에 의존하지 않는 축).

질문별 근거 bbox = 주석자별 ui_elements 합집합 → 좌표별 median (T4 생성기와 동일 규칙).

쌍 관계:
  evidence_iou        두 근거 bbox의 IoU
  center_distance     중심 간 거리 / 화면 대각선 (0~1)
  auto_label          T2 = 같은 근거(IoU>=0.5 또는 거리<0.03) / T3 = 다른 근거
                      partial = 그 사이 (해석 보류, 별도 보고)

역할 배정(문서 실험과 동일): q0=write 에피소드, q1..q3=소스(평가), q4+=held-out.

  python -m vlm_diagnosis.scripts.prep_screenqa_transfer --screens 150
"""
import argparse
import json
import math
import os
import random
import statistics
from collections import defaultdict, Counter
from datetime import datetime, timezone

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
PROBE = os.path.join(ROOT, "data", "screenqa_probe", "official")
OUT_DIR = os.path.join(ROOT, "experiments", "manifests")
ANNOT_REVISION = "1dfdbccaf56948821b5fa8ffe5d186fe4751e46d"

IOU_SAME = 0.5          # 이 이상이면 같은 element
DIST_SAME = 0.03        # 중심 거리(대각선 대비)가 이 미만이면 같은 줄/블록으로 취급
DIST_DIFF = 0.10        # 이 이상이면 확실히 다른 블록
MIN_Q = 5               # 화면당 최소 사용가능 질문 수 (소스3 + held-out2 확보용)


def union_bbox(ui_elements):
    xs = [b for e in ui_elements for b in (e["bounds"][0], e["bounds"][2])]
    ys = [b for e in ui_elements for b in (e["bounds"][1], e["bounds"][3])]
    return [min(xs), min(ys), max(xs), max(ys)]


def median_bbox(boxes):
    return [round(statistics.median(b[k] for b in boxes)) for k in range(4)]


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    ar = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ar if ar > 0 else 0.0


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def relation(a, b, W, H):
    o = iou(a, b)
    (ax, ay), (bx, by) = center(a), center(b)
    d = math.hypot((ax - bx) / W, (ay - by) / H) / math.hypot(1, 1)
    if o >= IOU_SAME or d < DIST_SAME:
        lab = "T2"
    elif d >= DIST_DIFF:
        lab = "T3"
    else:
        lab = "partial"
    return round(o, 4), round(d, 4), lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screens", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", default="screenqa_transfer",
                    help="출력 파일 접두 (discovery는 예: screenqa_discovery)")
    ap.add_argument("--exclude-manifest", default=None,
                    help="이 manifest에 있는 sample_id는 선택에서 제외 (파일럿과 분리)")
    ap.add_argument("--exclude-t4-pilot", action="store_true", default=True,
                    help="T4 파일럿에 쓴 화면은 제외 (실험 arm 분리)")
    a = ap.parse_args()

    bb = json.load(open(os.path.join(PROBE, "answers_and_bboxes", "validation.json")))
    short = {(r["image_id"], r["question"]): r["ground_truth"] for r in
             json.load(open(os.path.join(PROBE, "short_answers", "validation.json")))}

    excl = Counter()
    per_screen = defaultdict(list)
    for i, r in enumerate(bb):
        key = (r["image_id"], r["question"])
        answers = [x.strip() for x in short.get(key, [])
                   if x and x.strip() and "no answer" not in x.lower()]
        if not answers:
            excl["짧은답 없음"] += 1
            continue
        boxes = [union_bbox(g["ui_elements"]) for g in r["ground_truth"]
                 if g.get("ui_elements")]
        if not boxes:
            excl["bbox 없음"] += 1
            continue
        per_screen[r["image_id"]].append({
            "question_id": f"sqa_val_{i:05d}",
            "question": r["question"],
            "answers": sorted(set(answers), key=answers.index),
            "evidence_bbox": median_bbox(boxes),
            "n_annotator_boxes": len(boxes),
            "image_width": r["image_width"], "image_height": r["image_height"],
        })

    eligible = {k: v for k, v in per_screen.items() if len(v) >= MIN_Q}
    used_t4 = set()
    t4 = os.path.join(OUT_DIR, "t4_pilot.jsonl")
    if a.exclude_t4_pilot and os.path.exists(t4):
        used_t4 = {int(json.loads(l)["sample_id"]) for l in open(t4)}
        eligible = {k: v for k, v in eligible.items() if k not in used_t4}

    ids = sorted(eligible)
    if a.exclude_manifest:
        used = {json.loads(l)["sample_id"] for l in open(a.exclude_manifest)}
        before = len(ids)
        ids = [i for i in ids if str(i) not in used]
        print(f"[exclude] 파일럿 화면 {before - len(ids)}개 제외")
    random.Random(a.seed).shuffle(ids)
    picked = ids[:a.screens]
    print(f"사용가능 화면 {len(per_screen):,} → 질문 {MIN_Q}개 이상 {len(ids):,} "
          f"(T4 파일럿 {len(used_t4)}화면 제외) → 선택 {len(picked)}")
    print("질문 제외 사유:", dict(excl))

    # 이미지 확보 (T4 생성기의 다운로더 재사용)
    import importlib
    g = importlib.import_module("vlm_diagnosis.scripts.gen_t4_pilot")
    prov = g.ensure_images([{"image_id": s} for s in picked], ROOT, None)

    ROLE = {0: "episode", 1: "source", 2: "source", 3: "source"}
    screens_out, pairs_out = [], []
    rel_count = Counter()
    for sid in picked:
        qs = eligible[sid]
        rng = random.Random(f"{a.seed}:{sid}")
        rng.shuffle(qs)                       # 주석 순서 편향 제거
        for i, q in enumerate(qs):
            q["role"] = ROLE.get(i, "heldout")
        W, H = qs[0]["image_width"], qs[0]["image_height"]
        screens_out.append({
            "dataset": "ScreenQA", "dataset_revision": ANNOT_REVISION,
            "source_split": "validation", "split": "discovery",
            "sample_id": str(sid), "image": f"data/screenqa_pilot/{sid}.jpg",
            "image_width": W, "image_height": H,
            "image_mirror": g.IMAGES_REPO, "image_mirror_revision": g.IMAGES_REVISION,
            "image_sha256": prov.get(str(sid), {}).get("sha256"),
            "questions": qs, "selection_seed": a.seed,
        })
        for i in range(len(qs)):
            for j in range(len(qs)):
                if i == j:
                    continue
                o, d, lab = relation(qs[i]["evidence_bbox"], qs[j]["evidence_bbox"], W, H)
                rel_count[lab] += 1
                pairs_out.append({
                    "sample_id": str(sid),
                    "image": f"data/screenqa_pilot/{sid}.jpg",
                    "pair_id": f"{sid}_{qs[i]['question_id']}_{qs[j]['question_id']}",
                    "qA_id": qs[i]["question_id"], "qA": qs[i]["question"],
                    "qA_answers": qs[i]["answers"], "qA_role": qs[i]["role"],
                    "qA_bbox": qs[i]["evidence_bbox"],
                    "qB_id": qs[j]["question_id"], "qB": qs[j]["question"],
                    "qB_answers": qs[j]["answers"], "qB_role": qs[j]["role"],
                    "qB_bbox": qs[j]["evidence_bbox"],
                    "evidence_iou": o, "center_distance": d,
                    "auto_label": lab, "label_source": "auto_bbox",
                })

    man = os.path.join(OUT_DIR, f"{a.out_prefix}.jsonl")
    with open(man, "w") as f:
        for s in screens_out:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    pf = os.path.join(OUT_DIR, f"{a.out_prefix}_pairs.jsonl")
    with open(pf, "w") as f:
        for p in pairs_out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    meta = {
        "created": datetime.now(timezone.utc).isoformat(),
        "generator": "vlm_diagnosis.scripts.prep_screenqa_transfer",
        "why": "DocVQA는 근거 좌표가 없어 사람 라벨이 필요했지만 ScreenQA는 답마다 "
               "bbox가 있어 쌍 라벨을 자동 계산할 수 있다. 사람은 표본 검증만 한다.",
        "annotations": {"source": "official screen_qa GitHub JSON",
                        "revision": ANNOT_REVISION, "split": "validation"},
        "images": {"repo": g.IMAGES_REPO, "revision": g.IMAGES_REVISION,
                   "caveat": "community mirror — RICO 원본 대조는 confirmation 단계 전 권장"},
        "evidence_rule": "질문별 근거 bbox = 주석자별 ui_elements 합집합의 좌표별 median",
        "pair_rule": {"T2": f"IoU>={IOU_SAME} 또는 중심거리<{DIST_SAME}",
                      "T3": f"중심거리>={DIST_DIFF}",
                      "partial": "그 사이 — 해석 보류, 별도 보고",
                      "continuous": "evidence_iou / center_distance를 기록해 "
                                    "임계값 없는 연속 분석도 가능"},
        "roles": "shuffle 후 q0=episode, q1-3=source(평가), q4+=heldout",
        "stats": {"screens": len(screens_out),
                  "questions": sum(len(s["questions"]) for s in screens_out),
                  "pairs": len(pairs_out), "pair_labels": dict(rel_count),
                  "question_exclusions": dict(excl),
                  "eligible_screens": len(ids), "min_questions": MIN_Q},
        "human_task": "자동 라벨의 표본 검증만 (전수 라벨 불필요)",
    }
    json.dump(meta, open(man.replace(".jsonl", ".meta.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"화면 {len(screens_out)} · 질문 {meta['stats']['questions']} · "
          f"쌍 {len(pairs_out)} → {dict(rel_count)}")
    print(f"[saved] {man}\n[saved] {pf}")


if __name__ == "__main__":
    main()
