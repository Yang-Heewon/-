"""GQA(자연 이미지) 전이 실험용 manifest 생성 — 장면그래프 좌표로 쌍 라벨을 자동 계산한다.

DocVQA(문서) · ScreenQA(모바일 UI)에 이은 세 번째 도메인. GQA는 이미지당 질문이 많고
(val balanced 평균 12.9개), 각 질문의 `annotations.answer`가 장면그래프(scene graph)
객체 id를 가리키므로 답의 근거 bbox를 자동으로 얻을 수 있다. 즉 ScreenQA처럼 두 질문의
근거가 같은 곳인지(T2) 다른 곳인지(T3)를 사람 라벨 없이 계산할 수 있다.

근거 규칙: 질문별 근거 bbox = annotations.answer가 가리키는 장면그래프 객체(들)의
bbox 합집합(union), 이미지 경계로 클리핑. (val balanced에서는 answer 객체가 항상 1개다.)
answer 주석이 없는 질문(yes/no·전역 질문 대부분)은 제외하고 사유별로 집계한다.

쌍 관계(ScreenQA와 동일 임계값 — 도메인 간 비교 가능):
  evidence_iou        두 근거 bbox의 IoU
  center_distance     중심 간 거리 / 이미지 대각선 (0~1)
  auto_label          T2 = 같은 근거(IoU>=0.5 또는 거리<0.03) / T3 = 거리>=0.10
                      partial = 그 사이 (해석 보류, 별도 보고)

역할 배정(ScreenQA와 동일): shuffle 후 q0=write 에피소드, q1..q3=소스(평가), q4+=held-out.
이미지당 질문은 8개로 캡(비용 제한): 에피소드1 + 소스3 + held-out 최대 4.

이미지: 공식 전체 zip(~20GB)은 받지 않고, 선택된 150장만 HF 커뮤니티 미러
lmms-lab/GQA(val_balanced_images parquet)에서 row-group 단위로 받는다. 받은 이미지는
디코딩 + 장면그래프 width/height 일치 검증, sha256 기록.

  python -m vlm_diagnosis.scripts.prep_gqa_transfer --images 150
"""
import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict, Counter
from datetime import datetime, timezone

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
PROBE = os.path.join(ROOT, "data", "gqa_probe")
IMG_DIR_REL = os.path.join("data", "gqa_pilot")
OUT_DIR = os.path.join(ROOT, "experiments", "manifests")

# 공식 주석 출처 (Stanford, CC BY 4.0 — download 페이지 배지 확인 2026-08-16)
ANNOT_SOURCE = "official GQA v1.2 (downloads.cs.stanford.edu/nlp/data/gqa)"
ANNOT_FILES = {
    "questions": "questions1.2.zip::val_balanced_questions.json (Last-Modified 2019-03-26)",
    "scene_graphs": "sceneGraphs.zip::val_sceneGraphs.json (Last-Modified 2019-02-03)",
}
# 이미지 미러 (HF, lmms-lab/GQA -> lmms-lab-encoder/GQA 리다이렉트, 커밋 고정)
IMAGES_REPO = "lmms-lab/GQA"
IMAGES_REVISION = "a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8"
IMAGES_CONFIG = "val_balanced_images"
IMAGES_SHARDS = [f"val-{i:05d}-of-00003.parquet" for i in range(3)]

IOU_SAME = 0.5          # 이 이상이면 같은 근거 (ScreenQA와 동일)
DIST_SAME = 0.03        # 중심 거리(대각선 대비)가 이 미만이면 같은 근거
DIST_DIFF = 0.10        # 이 이상이면 확실히 다른 근거
MIN_Q = 5               # 이미지당 최소 사용가능 질문 수 (소스3 + held-out 확보용)
MAX_Q = 8               # 이미지당 질문 캡 (비용 제한)


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


def collect_questions(questions, graphs, excl, clip_count):
    """질문별 근거 bbox 계산. 반환: imageId -> [question record]."""
    per_img = defaultdict(list)
    for qid, r in sorted(questions.items()):
        g = graphs.get(r["imageId"])
        if g is None:
            excl["장면그래프 없음"] += 1
            continue
        ans_objs = r.get("annotations", {}).get("answer", {})
        if not ans_objs:
            excl["답 객체 주석 없음 (yes/no·전역 질문 포함)"] += 1
            continue
        W, H = g["width"], g["height"]
        boxes = []
        bad = None
        for oid in ans_objs.values():
            o = g["objects"].get(oid)
            if o is None:
                bad = "주석 객체가 장면그래프에 없음"
                break
            if o["w"] <= 0 or o["h"] <= 0:
                bad = "퇴화 bbox (w<=0 또는 h<=0)"
                break
            boxes.append([o["x"], o["y"], o["x"] + o["w"], o["y"] + o["h"]])
        if bad:
            excl[bad] += 1
            continue
        bb = [min(b[0] for b in boxes), min(b[1] for b in boxes),
              max(b[2] for b in boxes), max(b[3] for b in boxes)]
        clipped = [max(0, bb[0]), max(0, bb[1]), min(W, bb[2]), min(H, bb[3])]
        if clipped != bb:
            clip_count["클리핑된 bbox"] += 1
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            excl["클리핑 후 퇴화 bbox"] += 1
            continue
        per_img[r["imageId"]].append({
            "question_id": f"gqa_val_{qid}",
            "question": r["question"],
            "answers": [r["answer"]],          # EM 채점용 짧은 답만
            "full_answer": r["fullAnswer"],    # 문장형 답은 참고용으로 별도 보존
            "evidence_bbox": clipped,          # [left, top, right, bottom] 픽셀
            "evidence_object_ids": list(ans_objs.values()),
            "n_evidence_objects": len(ans_objs),
            "structural_type": r["types"]["structural"],
            "image_width": W, "image_height": H,
        })
    return per_img


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_images(image_ids, graphs):
    """선택된 이미지가 없으면 고정 커밋의 HF 미러 parquet에서 row-group 단위로 받는다.
    받은/기존 이미지 모두 디코딩 + 장면그래프 크기 일치 검증. 반환: provenance dict."""
    from PIL import Image
    img_dir = os.path.join(ROOT, IMG_DIR_REL)
    os.makedirs(img_dir, exist_ok=True)
    prov, missing = {}, []
    for sid in image_ids:
        path = os.path.join(img_dir, f"{sid}.jpg")
        if os.path.exists(path):
            prov[sid] = {"sha256": sha256_file(path)}
        else:
            missing.append(sid)
    if missing:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        base = f"datasets/{IMAGES_REPO}@{IMAGES_REVISION}/{IMAGES_CONFIG}"
        wanted = set(missing)
        print(f"[images] {len(wanted)}장을 미러에서 받는 중...", flush=True)
        for shard in IMAGES_SHARDS:
            if not wanted:
                break
            with fs.open(f"{base}/{shard}", "rb") as f:
                pf = pq.ParquetFile(f)
                for rg in range(pf.num_row_groups):
                    if not wanted:
                        break
                    ids = pf.read_row_group(rg, columns=["id"]).column("id").to_pylist()
                    hits = [(i, sid) for i, sid in enumerate(ids) if sid in wanted]
                    if not hits:
                        continue
                    tbl = pf.read_row_group(rg, columns=["id", "image"])
                    for i, sid in hits:
                        img = tbl.column("image")[i].as_py()
                        data = img["bytes"] if isinstance(img, dict) else img
                        path = os.path.join(img_dir, f"{sid}.jpg")
                        with open(path, "wb") as out:
                            out.write(data)
                        prov[sid] = {"sha256": hashlib.sha256(data).hexdigest(),
                                     "shard": shard, "row_group": rg}
                        wanted.discard(sid)
                        print(f"[images] {sid}.jpg ({shard} rg{rg}, {len(data)}B)",
                              flush=True)
        if wanted:
            raise RuntimeError(f"미러에 없는 이미지: {sorted(wanted)}")
    for sid in image_ids:  # 전수 검증: 디코딩 + 장면그래프 크기 일치
        path = os.path.join(img_dir, f"{sid}.jpg")
        with Image.open(path) as im:
            im.load()
            g = graphs[sid]
            if im.size != (g["width"], g["height"]):
                raise RuntimeError(
                    f"이미지 {sid}: 실제 크기 {im.size} != 장면그래프 "
                    f"({g['width']}, {g['height']})")
        prov[sid]["decoded_and_size_verified"] = True
    return prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    questions = json.load(open(os.path.join(PROBE, "val_balanced_questions.json")))
    graphs = json.load(open(os.path.join(PROBE, "val_sceneGraphs.json")))

    excl, clip_count = Counter(), Counter()
    per_img = collect_questions(questions, graphs, excl, clip_count)

    eligible = {k: v for k, v in per_img.items() if len(v) >= MIN_Q}
    ids = sorted(eligible)
    random.Random(a.seed).shuffle(ids)
    picked = ids[:a.images]
    print(f"근거 bbox 확보 질문 {sum(len(v) for v in per_img.values()):,} / "
          f"전체 {len(questions):,} · 사용가능 이미지(≥{MIN_Q}문항) {len(ids):,} "
          f"→ 선택 {len(picked)}")
    print("질문 제외 사유:", dict(excl))

    prov = ensure_images(picked, graphs)

    ROLE = {0: "episode", 1: "source", 2: "source", 3: "source"}
    images_out, pairs_out = [], []
    rel_count = Counter()
    for sid in picked:
        qs = list(eligible[sid])
        rng = random.Random(f"{a.seed}:{sid}")
        rng.shuffle(qs)                       # 주석 순서 편향 제거
        qs = qs[:MAX_Q]                       # 비용 캡
        for i, q in enumerate(qs):
            q["role"] = ROLE.get(i, "heldout")
        W, H = qs[0]["image_width"], qs[0]["image_height"]
        images_out.append({
            "dataset": "GQA", "dataset_revision": "v1.2",
            "source_split": "val_balanced", "split": "discovery",
            "sample_id": str(sid), "image": f"data/gqa_pilot/{sid}.jpg",
            "image_width": W, "image_height": H,
            "image_mirror": IMAGES_REPO, "image_mirror_revision": IMAGES_REVISION,
            "image_sha256": prov.get(sid, {}).get("sha256"),
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
                    "image": f"data/gqa_pilot/{sid}.jpg",
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

    man = os.path.join(OUT_DIR, "gqa_transfer.jsonl")
    with open(man, "w") as f:
        for s in images_out:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    pf = os.path.join(OUT_DIR, "gqa_transfer_pairs.jsonl")
    with open(pf, "w") as f:
        for p in pairs_out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    meta = {
        "created": datetime.now(timezone.utc).isoformat(),
        "generator": "vlm_diagnosis.scripts.prep_gqa_transfer",
        "why": "세 번째 도메인(자연 이미지). GQA는 질문의 annotations.answer가 "
               "장면그래프 객체를 가리켜 근거 bbox를 자동으로 얻을 수 있고, "
               "ScreenQA와 같은 규칙으로 쌍 라벨을 자동 계산한다.",
        "annotations": {"source": ANNOT_SOURCE, "files": ANNOT_FILES,
                        "split": "val_balanced",
                        "license": "CC BY 4.0 (gqadataset.org download 페이지 배지, "
                                   "2026-08-16 확인)"},
        "images": {"repo": IMAGES_REPO, "config": IMAGES_CONFIG,
                   "revision": IMAGES_REVISION,
                   "caveat": "커뮤니티 미러(카드 라이선스 표기는 mit로 부정확) — "
                             "confirmation 단계 전 공식 images.zip 표본 대조 권장"},
        "evidence_rule": "질문별 근거 bbox = annotations.answer가 가리키는 장면그래프 "
                         "객체(들)의 union bbox, 이미지 경계로 클리핑. "
                         "answers는 짧은 답([answer])만 — fullAnswer는 문장이라 "
                         "EM 채점에 부적합, full_answer 필드로 별도 보존.",
        "pair_rule": {"T2": f"IoU>={IOU_SAME} 또는 중심거리<{DIST_SAME}",
                      "T3": f"중심거리>={DIST_DIFF}",
                      "partial": "그 사이 — 해석 보류, 별도 보고",
                      "continuous": "evidence_iou / center_distance를 기록해 "
                                    "임계값 없는 연속 분석도 가능"},
        "roles": f"shuffle 후 q0=episode, q1-3=source(평가), q4+=heldout "
                 f"(이미지당 {MAX_Q}문항 캡)",
        "stats": {"images": len(images_out),
                  "questions": sum(len(s["questions"]) for s in images_out),
                  "pairs": len(pairs_out), "pair_labels": dict(rel_count),
                  "question_exclusions": dict(excl),
                  "bbox_clipped": dict(clip_count),
                  "eligible_questions": sum(len(v) for v in per_img.values()),
                  "eligible_images": len(ids), "min_questions": MIN_Q,
                  "max_questions": MAX_Q},
        "human_task": "자동 라벨의 표본 검증만 (전수 라벨 불필요)",
    }
    json.dump(meta, open(man.replace(".jsonl", ".meta.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"이미지 {len(images_out)} · 질문 {meta['stats']['questions']} · "
          f"쌍 {len(pairs_out)} → {dict(rel_count)}")
    print(f"[saved] {man}\n[saved] {pf}")


if __name__ == "__main__":
    main()
