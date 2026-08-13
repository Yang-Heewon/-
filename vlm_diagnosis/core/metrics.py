"""공통 task metric — M0 sanity와 이후 단계가 공유.

- normalized EM: 대소문자/공백/구두점 무시 완전 일치
- click-in-bbox: 생성 문자열에서 첫 좌표쌍을 파싱해 target bbox 포함 여부
"""
import re
import string

_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_text(s):
    return " ".join(s.lower().translate(_PUNCT).split())


def exact_match(pred, acceptable):
    p = normalize_text(pred)
    return float(any(p == normalize_text(a) for a in acceptable))


_COORD = re.compile(r"\(?\s*(\d+(?:\.\d+)?)\s*[,;]\s*(\d+(?:\.\d+)?)\s*\)?")


def parse_first_coordinates(pred):
    m = _COORD.search(pred)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def click_in_bbox(pred, bbox):
    """bbox = [x0, y0, x1, y1]. 좌표 파싱 실패는 0점."""
    xy = parse_first_coordinates(pred)
    if xy is None:
        return 0.0
    x, y = xy
    x0, y0, x1, y1 = bbox
    return float(x0 <= x <= x1 and y0 <= y <= y1)


def score_sample(pred, task_type, acceptable_answers=None, target_bbox=None):
    if task_type == "grounding":
        return click_in_bbox(pred, target_bbox)
    return exact_match(pred, acceptable_answers or [])
