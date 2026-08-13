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


def contains_match(pred, acceptable):
    """정규화 후 gold가 예측 문장 안에 포함되는가 — smoke 전용 관대 지표.
    본실험(M2-A)은 공식 ANLS + 짧은-답 prompt로 대체해야 한다."""
    p = normalize_text(pred)
    return float(any(normalize_text(a) in p for a in acceptable if a.strip()))


_COORD = re.compile(r"\(?\s*(\d+(?:\.\d+)?)\s*[,;]\s*(\d+(?:\.\d+)?)\s*\)?")


def parse_first_coordinates(pred):
    """좌표 추출. 점 (x,y) 하나면 그대로, 박스 (x1,y1),(x2,y2) 두 쌍이면 중심점.

    Qwen2.5-VL은 위치 응답을 bounding box 형식으로 내는 경향이 있어(학습 관례)
    박스 응답도 유효한 위치 지목으로 인정한다. 쉼표 없는 '896x896' 같은
    해상도 표기는 쌍으로 잡히지 않는다.
    """
    pairs = _COORD.findall(pred)
    if not pairs:
        return None
    if len(pairs) >= 2:
        (x1, y1), (x2, y2) = pairs[0], pairs[1]
        return ((float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2)
    return float(pairs[0][0]), float(pairs[0][1])


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
