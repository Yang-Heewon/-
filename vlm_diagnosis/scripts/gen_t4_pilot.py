"""Build the T4 pilot manifest v2 (ScreenQA content + auto-generated location questions).

This is the corrected (v2) generator. It supersedes the v1 session-scratchpad script
(select_t4.py) and fixes every issue from the 2026-08-15 formal audit:

  A. Nine v1 location questions failed visual audit (full-row/spinner bboxes, UI-state
     answers, icon-only text, on-screen duplicate text). Their source questions are
     hard-blacklisted (AUDIT_EXCLUSIONS below) and replacements are regenerated.
  B. Systematic fixes:
     1. Selection among valid candidates is seeded-RANDOM (rng.choice among the
        best-scoring options), not first-valid.
     2. New filters: (a) source bbox wider than 60% of screen width rejected
        (full-row/field bboxes); (b) UI-state answers {on,off,yes,no,true,false} and
        single-character answers rejected; (c) duplicate-text exclusion is
        unconditional — the same normalized text at 2+ distinct on-screen locations
        (center delta > 2% of W or H, anywhere on screen, even same half) skips the
        candidate; (d) midline +/-10% (half) and +/-5% grid-boundary bands kept.
     3. Stratified balance: half template targets an even top/bottom split
        (N//2 vs N-N//2); grid3x3 fills underrepresented classes first. If balance
        is unreachable with the base 25 screens, the pool expands (same eligibility
        rules, >=3 content questions) up to MAX_SCREENS=35.
     4. Position rule described as ACTUALLY implemented: position is judged from the
        MEDIAN OVER ANNOTATOR BBOX CENTERS (median of per-annotator center-x and
        center-y). The recorded source_bbox is the element-wise median of annotator
        bboxes and is NOT what position is judged from.

Inputs (relative to --root):
  data/screenqa_probe/official/answers_and_bboxes/validation.json
  data/screenqa_probe/official/short_answers/validation.json

Images: any selected screen missing from data/screenqa_pilot/ is downloaded from the
HF community mirror bevaya/RICO-ScreenQA at pinned revision IMAGES_REVISION via
row-group-targeted parquet reads (only the needed row groups' image column chunks are
fetched). Downloaded images are verified against annotation width/height.

Outputs (relative to --root):
  experiments/manifests/t4_pilot.jsonl        one row per screen (content + location
                                              questions; content questions carry
                                              type_draft per the frozen T4 rule)
  experiments/manifests/t4_pilot.meta.json    rules, stats, achieved balance,
                                              majority baselines, audit exclusions,
                                              statistical plan, image provenance
  experiments/manifests/t4_pairs_draft.jsonl  explicit content->location pair manifest
  experiments/manifests/t4_review.xlsx        human review sheet + persisted
                                              sanity_check sheet (10 seeded recomputes)

CLI:
  python vlm_diagnosis/scripts/gen_t4_pilot.py            # defaults: --root repo, --seed 42
  python vlm_diagnosis/scripts/gen_t4_pilot.py --root /root/research/heewon/VLM --seed 42

Requires: pillow, openpyxl, pyarrow, huggingface_hub (network only if images missing).
"""
import argparse
import hashlib
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------- constants
SEED_DEFAULT = 42
N_BASE = 25           # base screen count (v1 size)
MAX_SCREENS = 35      # balance-driven expansion cap (audit B3)
NO_ANSWER = '<no answer>'
STATE_WORDS = {'on', 'off', 'yes', 'no', 'true', 'false'}   # audit B2(b)
WIDE_BBOX_FRAC = 0.60                                       # audit B2(a)
DUP_DISTINCT_FRAC = 0.02   # centers differing by >2% of W or H = distinct location
MIDLINE_BAND = (0.40, 0.60)   # half template: exclude center-y in 40-60% of H
GRID_BOUNDARY_BAND = 0.05     # grid template: exclude center within 5% of 1/3, 2/3
STATUS_BAR_FRAC = 0.03
MIN_CONTENT_Q = 3

ANNOTATIONS_REVISION = '1dfdbccaf56948821b5fa8ffe5d186fe4751e46d'
IMAGES_REPO = 'bevaya/RICO-ScreenQA'
IMAGES_REVISION = '44c8508ea528b4d4e035cc06e28d7baf8db09ec6'
IMAGES_SHARDS = ['validation-00000-of-00003.parquet',
                 'validation-00001-of-00003.parquet',
                 'validation-00002-of-00003.parquet']

# Source questions of the 9 location questions that failed the 2026-08-15 visual
# audit. Blacklisted as location-question sources (several failure modes — icon-only
# text, duplicates outside the annotated elements — are not machine-detectable from
# the annotations alone).
AUDIT_EXCLUSIONS = {
    'sqa_val_06679': "v1 _loc2 '09:00': full-row bbox spanning the screen -> wrong grid cell",
    'sqa_val_01472': "v1 _loc1 '18': spinner/full-element bbox, not the text extent",
    'sqa_val_05301': "v1 _loc2 '18': spinner/full-element bbox, not the text extent",
    'sqa_val_02356': "v1 _loc2 'Brush': element is an icon; text not visibly rendered",
    'sqa_val_03384': "v1 _loc2 'on': UI toggle state, not visible text; multiple toggles on",
    'sqa_val_05761': "v1 _loc1 'on': UI toggle state, not visible text; multiple toggles on",
    'sqa_val_04404': "v1 _loc1 'Facebook': text appears at top AND bottom of screen (duplicate outside annotated elements)",
    'sqa_val_08120': "v1 _loc1 'Applesauce': text duplicated on screen (duplicate outside annotated elements)",
}
BAD_V1_LOC_IDS = {
    'sqa_val_06679_loc2', 'sqa_val_01472_loc1', 'sqa_val_05301_loc2',
    'sqa_val_02356_loc2', 'sqa_val_03384_loc2', 'sqa_val_05761_loc1',
    'sqa_val_04404_loc1', 'sqa_val_08120_loc1', 'sqa_val_08120_loc2',
}
# note: sqa_val_08120_loc2 ('40 MINS') was not itself flagged, but blacklisting is per
# source question; only the 9 flagged ids are recorded as audit failures in meta.
AUDIT_FAILED_LOC_IDS = sorted(BAD_V1_LOC_IDS - {'sqa_val_08120_loc2'})

# Exclusions from the v2 build's own visual sanity check (2026-08-15): failure modes
# not detectable from the annotations alone, found by inspecting the 10-question
# seeded sample with bboxes drawn on the images.
V2_VISUAL_EXCLUSIONS = {
    'sqa_val_03982': "opus 시각 검증(2026-08-15): 'Announce'가 화면에 4회 렌더링되어 "
                     "상·하반부에 모두 존재 → half 질문의 답이 모호. annotation에는 "
                     "1개 element만 있어 자동 중복 필터가 놓친 사례",
    'sqa_val_05364': "v2 candidate 'Sat, 11 Feb': same timestamp text rendered on "
                     "two news items (nytimes + washingtonpost rows); duplicate "
                     "outside annotated elements -> ambiguous location",
    'sqa_val_03018': "v2 candidate 'SAM App': text repeats in the headline 'SAM App "
                     "for Anxiety - Free Download' (duplicate outside annotated "
                     "elements; unconditional duplicate rule applies even though "
                     "both occurrences are in the top half)",
    'sqa_val_06821': "v2 candidate '1975': bbox sits on a wallpaper region with no "
                     "visibly rendered text (annotated answer not legible on screen)",
    'sqa_val_06233': "v2 candidate 'Facebook': header 'Log in With Facebook' (top) "
                     "AND footer 'This doesn't let the app post to Facebook.' "
                     "(bottom) — duplicate outside annotated elements, half answer "
                     "ambiguous",
    'sqa_val_06328': "v2 candidate 'Weather': word rendered many times on the "
                     "iTunes preview page (screen title, app names, category) — "
                     "duplicates outside annotated elements",
    'sqa_val_03457': "v2 candidate 'Greg Laurie': text is both the app-bar title "
                     "(top) and the player artist line (bottom) — duplicate "
                     "outside annotated elements, half answer ambiguous",
    'sqa_val_07727': "v2 candidate '1 mg': bbox sits on an empty region of the "
                     "medication list — dose text not visibly rendered",
    'sqa_val_02540': "v2 candidate 'Return': 'Return date' field label elsewhere "
                     "on screen repeats the word (duplicate outside annotated "
                     "elements, grid answer ambiguous)",
    'sqa_val_05020': "v2 candidate 'Grace Chan': profile name (middle-left) AND "
                     "post author line (bottom-left) — duplicate outside annotated "
                     "elements, grid answer ambiguous",
    'sqa_val_01041': "v2 candidate 'LISTEN': tab (top) AND 'Listen Later' item "
                     "(bottom) — whole-word duplicate outside annotated elements",
    'sqa_val_01410': "v2 candidate '30 °F': main temperature AND 'Low: 30 °F' line "
                     "— duplicate outside annotated elements",
    'sqa_val_04347': "v2 candidate '$0.00': invoice shows Total $0.00 AND Balance "
                     "Due $0.00 — duplicate (same cell, but unconditional rule)",
    'sqa_val_04720': "v2 candidate '13:00': '20 May at 13:00' AND 'Yesterday at "
                     "13:00' — duplicate outside annotated elements",
    'sqa_val_04778': "v2 candidate 'Today': tab (top) AND ad text 'Talk to your "
                     "doctor today' (bottom) — whole-word duplicate",
    'sqa_val_05441': "v2 candidate '$6.00': '$6.00 Value' rendered on three deal "
                     "cards — duplicates outside annotated elements",
    'sqa_val_06957': "v2 candidate 'Restaurant.com': title bar, install banner "
                     "(twice) and 'Specials by Restaurant.com' — duplicates in "
                     "both halves",
    'sqa_val_07958': "v2 candidate '2 hours ago': CNN item AND New York Times item "
                     "— duplicate outside annotated elements, grid ambiguous",
    'sqa_val_06467': "v2 candidate 'settings': target is the gear ICON in the "
                     "toolbar — annotated text not visibly rendered",
    'sqa_val_07953': "v2 candidate 'New York Times': byline on two news items "
                     "(top-left and middle) — duplicate outside annotated elements",
    'sqa_val_08138': "v2 candidate 'Wed, Feb 15': date rendered on four article "
                     "cards in both halves — duplicates outside annotated elements",
    'sqa_val_08438': "v2 candidate 'OsmAnd (online tiles)': row title AND 'Tile "
                     "data:' subtitle repeat the exact text — duplicate outside "
                     "annotated elements",
}

# Login-flow brand names: 'Log in with <brand>' headers pair with '... post to
# <brand>' footer disclaimers on the same dialog, so the brand text reliably
# appears in both halves. Failed visual audit three times (v1 sqa_val_04404, v2
# sqa_val_06233 and sqa_val_02822) -> systematic filter.
LOGIN_BRANDS = {'facebook', 'google', 'twitter', 'instagram'}
BLACKLIST = {**AUDIT_EXCLUSIONS, **V2_VISUAL_EXCLUSIONS}

# Screen-level exclusions found during the v2 build (image-vs-annotation checks).
SCREEN_EXCLUSIONS = {
    54145: 'mirror image is landscape 1920x1080 but the official annotation says '
           '1080x1920 — dimension mismatch would corrupt position labels',
}

# Star-rating answers ('5 stars', '4.5 stars', ...) are rendered as star WIDGETS,
# not visible text, and rating widgets typically repeat down list screens — the same
# failure mode as the v1 'Brush' icon case (systematic filter added after the v2
# visual check caught '5 stars').
import re
RATING_RE = re.compile(r'^\d+([.,]\d+)?\s*(stars?|★+)$')

GRID_NAMES = [['top-left', 'top-center', 'top-right'],
              ['middle-left', 'center', 'middle-right'],
              ['bottom-left', 'bottom-center', 'bottom-right']]
GRID_LABELS = [n for row in GRID_NAMES for n in row]
HALF_LABELS = ['top half', 'bottom half']

POSITIONAL_ANSWERS = set(
    HALF_LABELS + GRID_LABELS +
    ['top', 'bottom', 'left', 'right', 'center', 'middle', 'upper', 'lower',
     'left half', 'right half', 'upper half', 'lower half',
     'top left', 'top center', 'top right', 'middle left', 'middle right',
     'bottom left', 'bottom center', 'bottom right',
     'at the top', 'at the bottom', 'on the left', 'on the right'])


def norm(t):
    return ' '.join(t.lower().split())


def content_type_draft(question, answers):
    """Frozen T4 rule (docs/M3-01, 2026-08-14): answer IS a position -> layout;
    'how many' -> count; UI-state answers are excluded from location generation
    anyway; everything else -> OCR/semantic."""
    if any(norm(a) in POSITIONAL_ANSWERS for a in answers):
        return 'layout'
    if 'how many' in question.lower():
        return 'count'
    return 'OCR/semantic'


# ---------------------------------------------------------------- candidate rules
def loc_candidate(q, all_elems, excl):
    """Validate one question as a location-question source. Returns (cand, reason).

    Position is judged from the median over annotator bbox CENTERS (median of the
    per-annotator center-x values and center-y values). The recorded source_bbox is
    the element-wise median of annotator bboxes (display/reference only).
    """
    W, H = q['W'], q['H']
    if q['qid'] in BLACKLIST:
        return None, 'audit_blacklist'
    elems = []
    for g in q['gt']:
        ui = g.get('ui_elements') or []
        if len(ui) == 0:
            return None, 'no_bbox'
        if len(ui) > 1:
            return None, 'multi_element_answer'
        elems.append(ui[0])
    if not elems:
        return None, 'no_bbox'
    texts = {norm(e['text']) for e in elems}
    if len(texts) != 1:
        return None, 'annotator_text_mismatch'
    text = elems[0]['text'].strip()
    ntext = norm(text)
    if len(ntext) < 2 or sum(c.isalnum() for c in ntext) < 2:
        # <2 alphanumeric chars: single-char answers and punctuation-padded
        # single glyphs like '-2', which tend to be tiny/illegible on screen
        return None, 'text_too_short_or_single_char'
    if ntext in STATE_WORDS:
        return None, 'ui_state_answer'
    if ntext in LOGIN_BRANDS:
        return None, 'login_brand_answer'
    if RATING_RE.match(ntext):
        return None, 'rating_widget_answer'
    if ntext.isdigit() and len(ntext) <= 2:
        # bare 1-2 digit integers repeat pervasively in UIs (counts, scores,
        # table cells) and duplicates are rarely all annotated — v1 '18' and the
        # v2 visual check's '14' (score table) both failed this way
        return None, 'short_bare_integer'
    bbox = [round(statistics.median(e['bounds'][k] for e in elems)) for k in range(4)]
    if (bbox[2] - bbox[0]) > WIDE_BBOX_FRAC * W:
        return None, 'wide_bbox'
    cxs = [(e['bounds'][0] + e['bounds'][2]) / 2 for e in elems]
    cys = [(e['bounds'][1] + e['bounds'][3]) / 2 for e in elems]
    cx, cy = statistics.median(cxs), statistics.median(cys)
    if cy < STATUS_BAR_FRAC * H:
        return None, 'status_bar'
    # unconditional duplicate-text exclusion (audit B2(c)): same normalized text at
    # 2+ distinct locations anywhere on screen -> skip (threshold only separates
    # annotator jitter on the SAME element from genuinely distinct elements)
    centers = [c for t, c in all_elems if t == ntext]
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            (x1, y1), (x2, y2) = centers[i], centers[j]
            if abs(x1 - x2) > DUP_DISTINCT_FRAC * W or abs(y1 - y2) > DUP_DISTINCT_FRAC * H:
                return None, 'duplicate_text_on_screen'
    # substring ambiguity (v2 visual check round 2, 'Girl' vs 'Panda Girl'): if the
    # candidate text also occurs as a whole-word substring inside a DIFFERENT
    # annotated element's text elsewhere on screen, the quoted text matches multiple
    # visible strings -> ambiguous
    pat = re.compile(r'\b' + re.escape(ntext) + r'\b')
    for t, (ox, oy) in all_elems:
        if t != ntext and pat.search(t) and (
                abs(ox - cx) > DUP_DISTINCT_FRAC * W or
                abs(oy - cy) > DUP_DISTINCT_FRAC * H):
            return None, 'substring_of_other_text'
    cand = {'text': text, 'bbox': bbox, 'cx': cx, 'cy': cy}
    fy, fx = cy / H, cx / W
    # half template
    if MIDLINE_BAND[0] <= fy <= MIDLINE_BAND[1]:
        excl['half_midline_band'] += 1
    elif len({y / H < 0.5 for y in cys}) > 1:
        excl['half_annotator_disagree'] += 1
    else:
        cand['half'] = 'top half' if fy < 0.5 else 'bottom half'
    # grid3x3 template
    def cell(f):
        return 0 if f < 1 / 3 else (1 if f < 2 / 3 else 2)
    if any(abs(f - b) <= GRID_BOUNDARY_BAND for f in (fx, fy) for b in (1 / 3, 2 / 3)):
        excl['grid_boundary_band'] += 1
    elif len({(cell(x / W), cell(y / H)) for x, y in zip(cxs, cys)}) > 1:
        excl['grid_annotator_disagree'] += 1
    else:
        cand['grid'] = GRID_NAMES[cell(fy)][cell(fx)]
    if 'half' not in cand and 'grid' not in cand:
        return None, 'no_valid_template'
    return cand, None


# ---------------------------------------------------------------- balancing
def assign(screens, seed):
    """Greedy stratified assignment of one (half, grid) question pair per screen.

    Screens are processed most-constrained-first (fewest distinct grid labels).
    For each screen the pair whose grid label (then half label) currently has the
    LOWEST count is chosen — fills underrepresented classes first. Ties are broken
    by seeded rng.choice (audit B1: seeded-random, never first-valid).
    """
    rng = random.Random(seed)
    ch, cg = Counter(), Counter()
    order = sorted(
        screens,
        key=lambda e: (len({e['cands'][g][1]['grid'] for _, g in e['options']}),
                       len(e['options']), e['image_id']))
    chosen = {}
    for e in order:
        best, best_key = [], None
        for h, g in e['options']:
            key = (cg[e['cands'][g][1]['grid']], ch[e['cands'][h][1]['half']])
            if best_key is None or key < best_key:
                best_key, best = key, [(h, g)]
            elif key == best_key:
                best.append((h, g))
        h, g = rng.choice(sorted(best))
        chosen[e['image_id']] = (h, g)
        ch[e['cands'][h][1]['half']] += 1
        cg[e['cands'][g][1]['grid']] += 1
    return chosen, ch, cg


def balanced_selection(eligible, seed):
    """Base N_BASE screens (seeded shuffle, 1080x1920 preferred) + balance-driven
    expansion up to MAX_SCREENS. Returns (selected, chosen, ch, cg, expansion_log)."""
    rng = random.Random(seed)
    hi = sorted([e for e in eligible if (e['W'], e['H']) == (1080, 1920)],
                key=lambda e: e['image_id'])
    lo = sorted([e for e in eligible if (e['W'], e['H']) != (1080, 1920)],
                key=lambda e: e['image_id'])
    rng.shuffle(hi)
    rng.shuffle(lo)
    shuffled = hi + lo
    selected, rest = list(shuffled[:N_BASE]), list(shuffled[N_BASE:])
    log = []
    while True:
        chosen, ch, cg = assign(selected, seed + 1)
        half_diff = abs(ch['top half'] - ch['bottom half'])
        min_grid = min(cg[l] for l in GRID_LABELS)
        if (min_grid >= 2 and half_diff <= 1) or len(selected) >= MAX_SCREENS or not rest:
            break
        pick, why = None, None
        for lab in sorted(GRID_LABELS, key=lambda l: cg[l]):
            if cg[lab] >= 2:
                break
            for e in rest:
                if any(e['cands'][g][1]['grid'] == lab for _, g in e['options']):
                    pick, why = e, f'grid:{lab} count={cg[lab]}'
                    break
            if pick:
                break
        if pick is None and half_diff > 1:
            need = 'top half' if ch['top half'] < ch['bottom half'] else 'bottom half'
            for e in rest:
                if any(e['cands'][h][1]['half'] == need for h, _ in e['options']):
                    pick, why = e, f'half:{need}'
                    break
        if pick is None:
            break  # no remaining screen can supply a needed class
        selected.append(pick)
        rest.remove(pick)
        log.append({'added_image_id': pick['image_id'], 'reason': why})
    return selected, chosen, ch, cg, log


# ---------------------------------------------------------------- images
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def ensure_images(selected, root, old_meta):
    """Ensure every selected screen's image exists locally; download missing ones
    from the pinned HF mirror revision. Returns provenance dict."""
    img_dir = os.path.join(root, 'data', 'screenqa_pilot')
    os.makedirs(img_dir, exist_ok=True)
    old_prov = (old_meta or {}).get('image_provenance', {})
    prov, missing = {}, []
    for e in selected:
        sid = str(e['image_id'])
        path = os.path.join(img_dir, f'{sid}.jpg')
        if os.path.exists(path):
            sha = sha256_file(path)
            p = dict(old_prov.get(sid) or {'file_name': f'images/rico/{sid}.jpg'})
            p['sha256'] = sha
            if old_prov.get(sid, {}).get('sha256') not in (None, sha):
                p['note'] = 'local file sha256 differs from v1 meta record'
            prov[sid] = p
        else:
            missing.append(sid)
    need_scan = bool(missing) or any('shard' not in prov[s] for s in prov)
    if not need_scan:
        return prov
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    base = f'datasets/{IMAGES_REPO}@{IMAGES_REVISION}/data'
    wanted = {f'images/rico/{sid}.jpg': sid
              for sid in (set(missing) | {s for s in prov if 'shard' not in prov[s]})}
    print(f'[images] scanning shards for {len(wanted)} screens '
          f'({len(missing)} to download)...', flush=True)
    for shard in IMAGES_SHARDS:
        if not wanted:
            break
        with fs.open(f'{base}/{shard}', 'rb') as f:
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                if not wanted:
                    break
                names = pf.read_row_group(rg, columns=['file_name'])
                names = names.column('file_name').to_pylist()
                first = {}  # one row per QA pair -> same image repeats; keep first
                for i, n in enumerate(names):
                    if n in wanted and n not in first:
                        first[n] = i
                if not first:
                    continue
                hits = [(i, n) for n, i in first.items()]
                dl = [(i, n) for i, n in hits if wanted[n] in missing]
                tbl = pf.read_row_group(rg, columns=['file_name', 'image']) if dl else None
                for i, n in hits:
                    sid = wanted.pop(n)
                    entry = prov.get(sid, {'file_name': n})
                    entry.update({'file_name': n, 'shard': shard, 'row_group': rg})
                    if sid in missing:
                        img = tbl.column('image')[i].as_py()
                        data = img['bytes'] if isinstance(img, dict) else img
                        path = os.path.join(img_dir, f'{sid}.jpg')
                        with open(path, 'wb') as out:
                            out.write(data)
                        entry['sha256'] = sha256_file(path)
                        print(f'[images] downloaded {sid}.jpg '
                              f'({shard} rg{rg}, {len(data)} bytes)', flush=True)
                    prov[sid] = entry
    if wanted:
        raise RuntimeError(f'images not found in mirror: {sorted(wanted.values())}')
    return prov


def verify_images(selected, root):
    from PIL import Image
    for e in selected:
        path = os.path.join(root, 'data', 'screenqa_pilot', f'{e["image_id"]}.jpg')
        with Image.open(path) as im:
            im.load()
            if im.size != (e['W'], e['H']):
                raise RuntimeError(
                    f'image {e["image_id"]}: file {im.size} != annotation '
                    f'({e["W"]}, {e["H"]})')


# ---------------------------------------------------------------- main build
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    ap.add_argument('--root', default=default_root)
    ap.add_argument('--seed', type=int, default=SEED_DEFAULT)
    args = ap.parse_args()
    root, seed = args.root, args.seed

    full = json.load(open(f'{root}/data/screenqa_probe/official/answers_and_bboxes/validation.json'))
    short = json.load(open(f'{root}/data/screenqa_probe/official/short_answers/validation.json'))
    assert len(full) == len(short)
    for a, b in zip(full, short):
        assert a['image_id'] == b['image_id'] and a['question'] == b['question']

    screens = defaultdict(list)
    for idx, (f, s) in enumerate(zip(full, short)):
        screens[f['image_id']].append({
            'qid': f'sqa_val_{idx:05d}',
            'question': f['question'],
            'short_answers': [g for g in s['ground_truth']
                              if g.strip() and g.strip() != NO_ANSWER],
            'gt': f['ground_truth'],
            'W': f['image_width'], 'H': f['image_height'],
        })

    excl, screen_excl = Counter(), Counter()
    eligible = []
    for image_id, qs in sorted(screens.items()):
        if image_id in SCREEN_EXCLUSIONS:
            screen_excl['screen_blacklist'] += 1
            continue
        W, H = qs[0]['W'], qs[0]['H']
        content_qs = [q for q in qs if q['short_answers']]
        if len(content_qs) < MIN_CONTENT_Q:
            screen_excl['lt3_content_questions'] += 1
            continue
        all_elems = []
        for q in qs:
            for g in q['gt']:
                for e in (g.get('ui_elements') or []):
                    all_elems.append((norm(e['text']),
                                      ((e['bounds'][0] + e['bounds'][2]) / 2,
                                       (e['bounds'][1] + e['bounds'][3]) / 2)))
        cands = {}
        for q in qs:
            c, reason = loc_candidate(q, all_elems, excl)
            if c is None:
                excl[reason] += 1
            else:
                cands[q['qid']] = (q, c)
        halfq = sorted(q for q, (_, c) in cands.items() if 'half' in c)
        gridq = sorted(q for q, (_, c) in cands.items() if 'grid' in c)
        options = [(h, g) for h in halfq for g in gridq if h != g]
        if not options:
            screen_excl['no_valid_half_grid_pair'] += 1
            continue
        eligible.append({'image_id': image_id, 'W': W, 'H': H,
                         'content_qs': content_qs, 'cands': cands,
                         'options': options})

    print(f'screens scanned: {len(screens)}, eligible: {len(eligible)} '
          f'(1080x1920: {sum(1 for e in eligible if (e["W"], e["H"]) == (1080, 1920))})')

    selected, chosen, ch, cg, expansion_log = balanced_selection(eligible, seed)
    print(f'selected screens: {len(selected)} (expanded by {len(expansion_log)})')
    print('half:', dict(ch), ' grid:', dict(cg))

    # ---- build per-screen records
    meta_path = f'{root}/experiments/manifests/t4_pilot.meta.json'
    old_meta = json.load(open(meta_path)) if os.path.exists(meta_path) else None
    prov = ensure_images(selected, root, old_meta)
    verify_images(selected, root)

    rows, all_loc, all_content = [], [], []
    for e in sorted(selected, key=lambda e: e['image_id']):
        hqid, gqid = chosen[e['image_id']]
        loc_recs = []
        for qid, tmpl in ((hqid, 'half'), (gqid, 'grid3x3')):
            q, c = e['cands'][qid]
            if tmpl == 'half':
                question = (f"Is the '{c['text']}' shown in the top half or the "
                            f"bottom half of the screen?")
                ans = c['half']
            else:
                question = (f"In which region of the 3x3 grid of the screen is "
                            f"'{c['text']}' shown: top-left, top-center, top-right, "
                            f"middle-left, center, middle-right, bottom-left, "
                            f"bottom-center, or bottom-right?")
                ans = c['grid']
            loc_recs.append({
                'question_id': f'{qid}_loc{1 if tmpl == "half" else 2}',
                'question': question, 'answers': [ans],
                'source_question_id': qid, 'source_answer_text': c['text'],
                'source_bbox': c['bbox'], 'template': tmpl})
        content_recs = []
        for q in e['content_qs']:
            answers = list(dict.fromkeys(q['short_answers']))
            content_recs.append({
                'question_id': q['qid'], 'question': q['question'],
                'answers': answers,
                'type_draft': content_type_draft(q['question'], answers)})
        row = {
            'dataset': 'ScreenQA', 'dataset_revision': ANNOTATIONS_REVISION,
            'source_split': 'validation', 'split': 'pilot',
            'sample_id': str(e['image_id']),
            'image': f'data/screenqa_pilot/{e["image_id"]}.jpg',
            'image_width': e['W'], 'image_height': e['H'],
            'image_mirror': IMAGES_REPO, 'image_mirror_revision': IMAGES_REVISION,
            'content_questions': content_recs,
            'location_questions': loc_recs,
            'selection_seed': seed,
        }
        rows.append(row)
        all_loc.extend([(row, r) for r in loc_recs])
        all_content.extend([(row, r) for r in content_recs])

    final_loc_ids = {r['question_id'] for _, r in all_loc}
    assert final_loc_ids.isdisjoint(BAD_V1_LOC_IDS), 'audit-failed question regenerated!'

    with open(f'{root}/experiments/manifests/t4_pilot.jsonl', 'w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    # ---- pair manifest (content -> location only; see meta for rationale)
    pairs, n_pending = [], 0
    for row in rows:
        for cq in row['content_questions']:
            for lq in row['location_questions']:
                same_source = (lq['source_question_id'] == cq['question_id'])
                if same_source:
                    n_pending += 1
                    draft_label = ''
                    rationale = ('same evidence element: content answer text IS the '
                                 'location target; T2-vs-T4 precedence unresolved '
                                 '(docs/M3-01 pending note) — excluded from default '
                                 'T4 draft')
                elif cq['type_draft'] in ('OCR/semantic', 'count'):
                    draft_label = 'T4'
                    rationale = (f"cross-source {cq['type_draft']} -> layout "
                                 f"(type crossing, frozen T4 rule)")
                else:
                    draft_label = 'uncertain'
                    rationale = ("content question's own type_draft is layout — "
                                 "no clear type crossing; needs human review")
                pairs.append({
                    'dataset': 'ScreenQA', 'dataset_revision': ANNOTATIONS_REVISION,
                    'source_split': 'validation', 'split': 'pilot',
                    'sample_id': row['sample_id'], 'image': row['image'],
                    'pair_id': f"{row['sample_id']}_{cq['question_id']}_{lq['question_id']}",
                    'direction': 'content_to_location',
                    'qA_id': cq['question_id'], 'qA': cq['question'],
                    'qA_answers': ' | '.join(cq['answers']),
                    'qA_type_draft': cq['type_draft'],
                    'qB_id': lq['question_id'], 'qB': lq['question'],
                    'qB_answers': lq['answers'][0], 'qB_type_draft': 'layout',
                    'qB_template': lq['template'],
                    'same_source': same_source,
                    'pending_precedence_decision': same_source,
                    'draft_label': draft_label, 'draft_rationale': rationale,
                    'uncertain': True,
                    'label_A': '', 'notes_A': '', 'label_B': '', 'notes_B': '',
                    'adjudicated_label': '', 'adjudication_note': '',
                    'selection_seed': seed,
                })
    with open(f'{root}/experiments/manifests/t4_pairs_draft.jsonl', 'w') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    # ---- review workbook (review sheet + persisted sanity_check sheet)
    import openpyxl
    from PIL import Image
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 't4_location_review'
    ws.append(['sample_id', 'question_id', 'image', 'template', 'question',
               'generated_answer', 'source_answer_text', 'source_bbox',
               'valid_position_correct', 'valid_unambiguous', 'reviewer_notes'])
    for row, r in all_loc:
        ws.append([row['sample_id'], r['question_id'], row['image'], r['template'],
                   r['question'], r['answers'][0], r['source_answer_text'],
                   json.dumps(r['source_bbox']), None, None, None])

    ws2 = wb.create_sheet('sanity_check')
    ws2.append(['question_id', 'sample_id', 'template', 'claimed', 'recomputed',
                'center_fx', 'center_fy', 'match', 'method'])
    method = ('independent recompute: image W/H read from the actual image file '
              '(PIL); center = median over annotator bbox centers from the raw '
              'official annotation JSON; label from center fractions (no band '
              f'filters); seed={seed + 2}')
    srng = random.Random(seed + 2)
    sample = srng.sample(sorted(all_loc, key=lambda t: t[1]['question_id']),
                         min(10, len(all_loc)))
    n_match = 0
    for row, r in sample:
        idx = int(r['source_question_id'].split('_')[-1])
        gt = full[idx]['ground_truth']
        cxs = sorted((g['ui_elements'][0]['bounds'][0] + g['ui_elements'][0]['bounds'][2]) / 2
                     for g in gt)
        cys = sorted((g['ui_elements'][0]['bounds'][1] + g['ui_elements'][0]['bounds'][3]) / 2
                     for g in gt)
        with Image.open(os.path.join(root, row['image'])) as im:
            W, H = im.size
        fx = statistics.median(cxs) / W
        fy = statistics.median(cys) / H
        if r['template'] == 'half':
            recomputed = 'top half' if fy < 0.5 else 'bottom half'
        else:
            def cell(f):
                return 0 if f < 1 / 3 else (1 if f < 2 / 3 else 2)
            recomputed = GRID_NAMES[cell(fy)][cell(fx)]
        match = (recomputed == r['answers'][0])
        n_match += match
        ws2.append([r['question_id'], row['sample_id'], r['template'],
                    r['answers'][0], recomputed, round(fx, 4), round(fy, 4),
                    'yes' if match else 'NO', method])
    wb.save(f'{root}/experiments/manifests/t4_review.xlsx')
    print(f'sanity_check: {n_match}/{len(sample)} recomputed labels match')

    # ---- meta
    half_total = sum(ch.values())
    grid_total = sum(cg.values())
    meta = {
        'created': '2026-08-15',
        'version': 2,
        'generator': 'vlm_diagnosis/scripts/gen_t4_pilot.py (v2, audit-corrected; '
                     'v1 scratchpad script superseded)',
        'selection_seed': seed,
        'source': {
            'annotations': 'github.com/google-research-datasets/screen_qa '
                           'answers_and_bboxes/validation.json + '
                           'short_answers/validation.json (1:1 aligned, verified)',
            'annotations_revision': f'{ANNOTATIONS_REVISION} (repo HEAD at v1 build; '
                                    'local copy data/screenqa_probe/)',
            'images': f'HF community mirror {IMAGES_REPO}, validation parquet shards',
            'images_revision': IMAGES_REVISION,
            'residual_risk': 'Images come from a community mirror, not the official '
                             'RICO release (see docs/screenqa-access-smoke.md §6). '
                             'All selected images decode and match official '
                             'annotation image_width/image_height. Duplicate-text '
                             'detection can only see ANNOTATED evidence elements, '
                             'not the whole screen — duplicates outside annotations '
                             '(the v1 Facebook/Applesauce failures) are only caught '
                             'by human review; the v1 offenders are hard-blacklisted.',
        },
        'question_id_convention': 'sqa_val_<idx> where idx = 0-based line index in '
                                  'official validation.json; _loc1 = half template, '
                                  '_loc2 = grid3x3 template',
        'selection_rule': f'>=3 content questions with usable short answers AND >=1 '
                          f'valid (half, grid3x3) location pair from two distinct '
                          f'source questions; prefer 1080x1920; seeded shuffle, base '
                          f'{N_BASE} screens, balance-driven expansion up to '
                          f'{MAX_SCREENS} (stop when every grid class has >=2 items '
                          f'and half split differs by <=1, or no remaining screen '
                          f'can supply a needed class)',
        'location_generation_rules': {
            'position_rule': 'ACTUAL implementation: position is judged from the '
                             'median over annotator bbox CENTERS (median of '
                             'per-annotator center-x, median of center-y). The '
                             'recorded source_bbox is the element-wise median of '
                             'annotator bboxes and is NOT used for position '
                             'judgment.',
            'half_template': 'top/bottom by median center-y vs H/2; EXCLUDE '
                             'center-y in 40-60% of H (±10% midline band); EXCLUDE '
                             'annotator half disagreement',
            'grid3x3_template': 'cell by median center vs thirds; EXCLUDE center '
                                'within ±5% of any cell boundary; EXCLUDE annotator '
                                'cell disagreement',
            'common_exclusions': 'answer with 0 or >1 ui_elements per annotator; '
                                 'annotator text mismatch; normalized text <2 chars '
                                 'or <2 alphanumeric chars (single-char and '
                                 'punctuation-padded single-glyph answers); '
                                 'UI-state answers '
                                 '{on,off,yes,no,true,false}; star-rating answers '
                                 "(regex '^\\d+([.,]\\d+)?\\s*(stars?|★+)$' — "
                                 'rendered as widgets, not text); bare 1-2 digit '
                                 'integers (pervasive-duplicate risk in counts/'
                                 'scores/table cells); login-flow brand names '
                                 '{facebook,google,twitter,instagram} (rendered in '
                                 'both header and footer disclaimer of login '
                                 'dialogs); source bbox wider '
                                 'than 60% of screen width (full-row/field bboxes); '
                                 'same normalized text at 2+ distinct on-screen '
                                 'locations (center delta >2% of W or H, '
                                 'unconditional — even same half); candidate text '
                                 'occurring as a whole-word substring of a '
                                 'different annotated element elsewhere on screen '
                                 '(quoted text would match multiple visible '
                                 'strings); status bar '
                                 '(center-y < 3% of H); audit blacklist (below)',
            'candidate_selection': 'seeded-random among balance-optimal candidates '
                                   '(greedy fills underrepresented grid classes '
                                   'first, then half; ties broken by seeded '
                                   'rng.choice) — v1 first-valid bias removed',
        },
        'audit_exclusions': {
            'date': '2026-08-15',
            'failed_v1_location_questions': AUDIT_FAILED_LOC_IDS,
            'blacklisted_source_questions': AUDIT_EXCLUSIONS,
            'note': 'blacklist applies to the SOURCE question, so both templates '
                    'from a flagged source are excluded (sqa_val_08120_loc2 was not '
                    'itself flagged but its source is blacklisted)',
            'v2_visual_check_exclusions': V2_VISUAL_EXCLUSIONS,
            'screen_exclusions': {str(k): v for k, v in SCREEN_EXCLUSIONS.items()},
            'v2_visual_check_note': 'the v2 build itself was visually audited in '
                                    'rounds: each build\'s 10-question seeded '
                                    'sanity sample was rendered with bboxes drawn '
                                    'on the images and inspected, and the build '
                                    'was regenerated after each fix until a full '
                                    'sample passed. Failures found and fixed: a '
                                    'star-rating widget answer (systematic '
                                    'rating-widget filter added), a timestamp '
                                    'duplicated outside the annotated elements '
                                    '(source question blacklisted), a candidate '
                                    'text contained in another element\'s text '
                                    '(systematic whole-word substring filter '
                                    'added), an answer with a single legible '
                                    'glyph (systematic >=2-alphanumeric filter '
                                    'added), a bare 2-digit number duplicated in '
                                    'a score table (systematic short-bare-integer '
                                    'filter added), an answer with no visibly '
                                    'rendered text at the bbox (blacklisted), and '
                                    'further duplicate-outside-annotations cases '
                                    '(blacklisted).',
        },
        'pairs_manifest': {
            'file': 'experiments/manifests/t4_pairs_draft.jsonl',
            'direction': 'content_to_location ONLY. location->content is excluded '
                         'because every generated location question embeds the '
                         'source answer text verbatim — presenting it first leaks '
                         'the content answer (answer leakage).',
            'pending_precedence_decision': 'pairs where the location question\'s '
                                           'source element IS the content '
                                           'question\'s evidence (same source '
                                           'question) are marked '
                                           'pending_precedence_decision=true and '
                                           'carry no draft label (T2-vs-T4 '
                                           'precedence unresolved, docs/M3-01)',
            'draft_rule': 'cross-source pairs: draft_label=T4 when content '
                          'type_draft is OCR/semantic or count, else uncertain; '
                          'ALL rows uncertain=true for human review',
        },
        'statistical_plan': {
            'unit_of_analysis': 'screen (each screen contributes 1 half + 1 grid '
                                'question; questions within a screen are not '
                                'independent)',
            'resampling': 'screen-cluster bootstrap (resample screens with '
                          'replacement, keep all questions of a resampled screen)',
            'metrics': 'per-template balanced accuracy and macro-F1 (class '
                       'imbalance is bounded but not zero; majority baselines '
                       'recorded below)',
        },
        'stats': {
            'screens_scanned': len(screens),
            'screens_eligible': len(eligible),
            'eligible_1080x1920': sum(1 for e in eligible
                                      if (e['W'], e['H']) == (1080, 1920)),
            'screens_selected': len(selected),
            'expansion_beyond_base': expansion_log,
            'content_questions': len(all_content),
            'location_questions': len(all_loc),
            'content_type_draft_distribution': dict(Counter(
                r['type_draft'] for _, r in all_content)),
            'pairs_total': len(pairs),
            'pairs_pending_precedence': n_pending,
            'pairs_draft_T4': sum(1 for p in pairs if p['draft_label'] == 'T4'),
            'pairs_draft_uncertain': sum(1 for p in pairs
                                         if p['draft_label'] == 'uncertain'),
            'achieved_balance': {
                'half': {'distribution': dict(ch),
                         'majority_baseline': round(max(ch.values()) / half_total, 4)},
                'grid3x3': {'distribution': {l: cg.get(l, 0) for l in GRID_LABELS},
                            'majority_baseline': round(max(cg.values()) / grid_total, 4)},
            },
            'screen_exclusions': dict(screen_excl),
            'question_level_location_exclusions': dict(excl),
            'sanity_check': {'sampled': len(sample), 'matched': n_match,
                             'sheet': 't4_review.xlsx#sanity_check'},
        },
        'image_provenance': {k: prov[k] for k in sorted(prov, key=int)},
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)

    print(f'wrote {len(rows)} screens, {len(all_loc)} location questions, '
          f'{len(all_content)} content questions, {len(pairs)} pairs '
          f'({n_pending} pending precedence)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
