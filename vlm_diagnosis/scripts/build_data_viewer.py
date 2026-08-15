"""파일럿 데이터 열람용 HTML 생성기 — 이미지와 질문을 나란히 보며 검수하기 위한 뷰어.

DocVQA 32문서(T0–T3·T1 쌍의 원천)와 ScreenQA 26화면(T4 파일럿)을 한 페이지에 담는다.
이미지는 data URI로 내장하므로 파일 하나만 열면 되고, 총량은 --budget-mb로 제한한다.

  python -m vlm_diagnosis.scripts.build_data_viewer --out /tmp/viewer.html
"""
import argparse
import base64
import html
import io
import json
import os
from collections import defaultdict

from PIL import Image

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
M = os.path.join(ROOT, "experiments", "manifests")


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path)]


def encode(path, width, quality):
    im = Image.open(os.path.join(ROOT, path)).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode(), buf.tell()


def encode_all(items, width, quality, budget_bytes, label):
    """예산을 넘으면 품질·폭을 단계적으로 낮춰 재시도."""
    for w, q in ((width, quality), (int(width * 0.85), quality - 6),
                 (int(width * 0.7), quality - 12)):
        out, total = {}, 0
        for path in items:
            b64, n = encode(path, w, q)
            out[path] = b64
            total += n
        if total <= budget_bytes:
            print(f"  {label}: {len(items)}장, {w}px q{q} → {total/1e6:.1f}MB")
            return out
    print(f"  {label}: {len(items)}장, 최소 설정에서도 {total/1e6:.1f}MB (그대로 사용)")
    return out


CSS = """
:root{
  --ink:#12161c; --ink-soft:#39424e; --muted:#5c6672; --line:#dde3e9;
  --paper:#f5f7f9; --surface:#ffffff; --surface-2:#eef2f6;
  --accent:#1c6b76; --accent-soft:#e3f0f1;
  --pass:#2f7a4d; --fail:#b23c3c; --wait:#8a5d00;
  --shadow:0 1px 2px rgba(18,22,28,.06),0 8px 24px rgba(18,22,28,.06);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ink:#e8edf2; --ink-soft:#b9c3ce; --muted:#8b97a5; --line:#2a323c;
    --paper:#0f1318; --surface:#171c23; --surface-2:#1e242c;
    --accent:#5fb6c2; --accent-soft:#14323a;
    --pass:#6fbf8b; --fail:#e08585; --wait:#d3a44a;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"]{
  --ink:#e8edf2; --ink-soft:#b9c3ce; --muted:#8b97a5; --line:#2a323c;
  --paper:#0f1318; --surface:#171c23; --surface-2:#1e242c;
  --accent:#5fb6c2; --accent-soft:#14323a;
  --pass:#6fbf8b; --fail:#e08585; --wait:#d3a44a;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:-apple-system,"Apple SD Gothic Neo","Pretendard","Malgun Gothic",
    "Noto Sans KR",system-ui,sans-serif;
  font-size:15px; line-height:1.6; -webkit-font-smoothing:antialiased;
}
.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}
header{
  position:sticky; top:0; z-index:20; background:var(--surface);
  border-bottom:1px solid var(--line); box-shadow:var(--shadow);
}
.hwrap{max-width:1240px; margin:0 auto; padding:16px 20px;
  display:flex; flex-direction:column; gap:12px}
h1{margin:0; font-size:19px; letter-spacing:-.01em; font-weight:650}
h1 span{color:var(--muted); font-weight:400; font-size:14px; margin-left:8px}
.stats{display:flex; flex-wrap:wrap; gap:8px 20px; font-size:13px; color:var(--muted)}
.stats b{color:var(--ink); font-weight:600}
.controls{display:flex; flex-wrap:wrap; gap:8px; align-items:center}
input[type=search]{
  flex:1 1 240px; min-width:200px; padding:8px 12px; font:inherit; font-size:14px;
  color:var(--ink); background:var(--surface-2); border:1px solid var(--line);
  border-radius:7px;
}
input[type=search]:focus-visible{outline:2px solid var(--accent); outline-offset:1px}
.chip{
  padding:6px 12px; font-size:13px; font-weight:550; cursor:pointer; color:var(--ink-soft);
  background:var(--surface-2); border:1px solid var(--line); border-radius:999px;
}
.chip[aria-pressed="true"]{background:var(--accent); border-color:var(--accent); color:#fff}
.chip:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
main{max-width:1240px; margin:0 auto; padding:24px 20px 64px;
  display:flex; flex-direction:column; gap:20px}
.sec-label{
  font-size:12px; font-weight:650; letter-spacing:.09em; text-transform:uppercase;
  color:var(--muted); padding-top:8px;
}
.card{
  background:var(--surface); border:1px solid var(--line); border-radius:12px;
  box-shadow:var(--shadow); overflow:hidden; display:grid;
  grid-template-columns:minmax(0,320px) minmax(0,1fr);
}
@media (max-width:820px){.card{grid-template-columns:1fr}}
.shot{padding:16px; background:var(--surface-2); border-right:1px solid var(--line);
  display:flex; flex-direction:column; gap:10px}
@media (max-width:820px){.shot{border-right:none; border-bottom:1px solid var(--line)}}
.shot img{width:100%; height:auto; border-radius:6px; border:1px solid var(--line);
  cursor:zoom-in; background:#fff; display:block}
.shot .meta{font-size:12px; color:var(--muted); word-break:break-all}
.body{padding:16px 18px; display:flex; flex-direction:column; gap:14px; min-width:0}
.idline{display:flex; flex-wrap:wrap; align-items:baseline; gap:10px}
.idline .sid{font-size:17px; font-weight:650}
.qgroup{display:flex; flex-direction:column; gap:8px}
.q{display:flex; flex-direction:column; gap:3px; padding:9px 11px; border-radius:8px;
  background:var(--surface-2); border:1px solid var(--line)}
.q.held{border-left:3px solid var(--accent)}
.qtop{display:flex; flex-wrap:wrap; gap:8px; align-items:center; font-size:12px}
.qtext{font-size:14px}
.ans{font-size:13px; color:var(--ink-soft)}
.ans b{font-weight:600; color:var(--ink)}
.tag{font-size:11px; font-weight:650; letter-spacing:.03em; padding:2px 7px;
  border-radius:5px; background:var(--accent-soft); color:var(--accent);
  border:1px solid transparent; white-space:nowrap}
.tag.role{background:transparent; border-color:var(--line); color:var(--muted)}
.tag.pass{background:transparent; border-color:var(--pass); color:var(--pass)}
.tag.fail{background:transparent; border-color:var(--fail); color:var(--fail)}
.tag.wait{background:transparent; border-color:var(--wait); color:var(--wait)}
.para{margin:4px 0 0 14px; padding-left:12px; border-left:2px dashed var(--line);
  display:flex; flex-direction:column; gap:4px}
.para div{font-size:13px; color:var(--ink-soft)}
.note{font-size:12px; color:var(--muted)}
dialog{border:none; padding:0; background:transparent; max-width:96vw; max-height:96vh}
dialog::backdrop{background:rgba(8,11,15,.82)}
dialog img{max-width:96vw; max-height:88vh; border-radius:8px; display:block}
dialog form{display:flex; justify-content:flex-end; padding-top:8px}
dialog button{font:inherit; font-size:13px; padding:6px 14px; border-radius:7px;
  border:1px solid var(--line); background:var(--surface); color:var(--ink); cursor:pointer}
footer{max-width:1240px; margin:0 auto; padding:0 20px 48px; font-size:12px;
  color:var(--muted); display:flex; flex-direction:column; gap:4px}
.empty{padding:40px; text-align:center; color:var(--muted)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

JS = """
const q=document.getElementById('q'), chips=[...document.querySelectorAll('.chip')],
      cards=[...document.querySelectorAll('.card')], empty=document.getElementById('empty'),
      dlg=document.getElementById('zoom'), dimg=document.getElementById('zimg');
function apply(){
  const t=q.value.trim().toLowerCase();
  const on=chips.filter(c=>c.getAttribute('aria-pressed')==='true').map(c=>c.dataset.filter);
  let n=0;
  for(const c of cards){
    const okText=!t||c.dataset.search.includes(t);
    const okChip=on.every(f=>c.dataset.flags.split(' ').includes(f));
    const show=okText&&okChip; c.hidden=!show;
    const lab=c.previousElementSibling;
    if(show)n++;
  }
  for(const l of document.querySelectorAll('.sec-label')){
    let sib=l.nextElementSibling, any=false;
    while(sib&&sib.classList.contains('card')){ if(!sib.hidden)any=true; sib=sib.nextElementSibling; }
    l.hidden=!any;
  }
  empty.hidden=n>0;
}
q.addEventListener('input',apply);
chips.forEach(c=>c.addEventListener('click',()=>{
  c.setAttribute('aria-pressed',c.getAttribute('aria-pressed')==='true'?'false':'true');apply();}));
document.querySelectorAll('.shot img').forEach(im=>im.addEventListener('click',()=>{
  dimg.src=im.src; dimg.alt=im.alt; dlg.showModal();}));
"""


def tag(text, cls=""):
    return f'<span class="tag {cls}">{html.escape(text)}</span>'


def build():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/pilot_viewer.html")
    ap.add_argument("--budget-mb", type=float, default=9.0)
    a = ap.parse_args()

    docs = load_jsonl(os.path.join(M, "m2a_diagnostic.jsonl"))
    t1 = load_jsonl(os.path.join(M, "t1_paraphrases.jsonl"))
    pairs = load_jsonl(os.path.join(M, "m3_pairs_draft.jsonl"))
    screens = load_jsonl(os.path.join(M, "t4_pilot.jsonl"))
    audit = {r["question_id"]: r for r in load_jsonl(os.path.join(M, "t4_visual_audit.jsonl"))}

    para = defaultdict(list)
    for p in t1:
        para[p["source_question_id"]].append(p["paraphrase"])
    pair_n = defaultdict(int)
    for p in pairs:
        pair_n[str(p["sample_id"])] += 1

    print("이미지 인코딩:")
    budget = a.budget_mb * 1e6
    # 문서는 잔글씨 판독이 필요해 크게, 화면은 배치 확인이 목적이라 작게
    doc_imgs = encode_all([d["image"] for d in docs], 1500, 72, budget * 0.72, "DocVQA")
    scr_imgs = encode_all([s["image"] for s in screens], 560, 72, budget * 0.28, "ScreenQA")

    ROLE = {0: "q0 · write 에피소드", 1: "q1 · 소스", 2: "q2 · 소스", 3: "q3 · 소스",
            4: "q4 · held-out", 5: "q5 · held-out"}
    out = []

    # ---------- DocVQA ----------
    out.append('<p class="sec-label">DocVQA · 문서 32건 — T0–T3 쌍과 T1 패러프레이즈의 원천</p>')
    for d in docs:
        sid = str(d["sample_id"])
        qs = d["questions"]
        rows = []
        for i, qq in enumerate(qs):
            role = ROLE.get(i, f"q{i} · 미사용")
            held = " held" if i in (4, 5) else ""
            ps = para.get(qq["question_id"], [])
            block = [f'<div class="q{held}"><div class="qtop">{tag(role, "role")}'
                     f'<span class="mono note">{html.escape(qq["question_id"])}</span></div>'
                     f'<div class="qtext">{html.escape(qq["question"])}</div>'
                     f'<div class="ans">답 · <b>{html.escape(" / ".join(qq["answers"]))}</b></div>']
            if ps:
                block.append('<div class="para">'
                             + f'{tag("T1 패러프레이즈 · 검증 대기", "wait")}'
                             + "".join(f"<div>{html.escape(x)}</div>" for x in ps)
                             + "</div>")
            block.append("</div>")
            rows.append("".join(block))
        search = " ".join([sid] + [x["question"] for x in qs]).lower()
        flags = "docvqa" + (" heldout" if len(qs) > 4 else "")
        out.append(
            f'<article class="card" data-search="{html.escape(search, quote=True)}" '
            f'data-flags="{flags}">'
            f'<div class="shot"><img src="data:image/jpeg;base64,{doc_imgs[d["image"]]}" '
            f'alt="문서 {html.escape(sid)}"><div class="meta mono">{html.escape(d["image"])}</div></div>'
            f'<div class="body"><div class="idline"><span class="sid mono">{html.escape(sid)}</span>'
            f'<span class="note">질문 {len(qs)}개 · 쌍 {pair_n.get(sid, 0)}개 · '
            f'DocVQA {html.escape(d["dataset_revision"][:7])}</span></div>'
            f'<div class="qgroup">{"".join(rows)}</div></div></article>')

    # ---------- ScreenQA ----------
    out.append('<p class="sec-label">ScreenQA · 화면 26건 — T4(내용↔위치) 파일럿, 위치 질문은 전수 시각 검증 완료</p>')
    for s in screens:
        sid = str(s["sample_id"])
        rows = []
        for cq in s["content_questions"]:
            t = cq.get("type_draft", "")
            rows.append(
                f'<div class="q"><div class="qtop">{tag("내용", "role")}'
                + (tag(t) if t else "")
                + f'<span class="mono note">{html.escape(cq["question_id"])}</span></div>'
                f'<div class="qtext">{html.escape(cq["question"])}</div>'
                f'<div class="ans">답 · <b>{html.escape(" / ".join(cq["answers"]))}</b></div></div>')
        for lq in s["location_questions"]:
            v = audit.get(lq["question_id"], {})
            verdict = v.get("visual_verdict", "")
            vt = (tag(f'시각검증 {"통과" if verdict == "pass" else verdict}',
                      "pass" if verdict == "pass" else "fail") if verdict else "")
            who = v.get("verified_by", "")
            note = v.get("note", "")
            rows.append(
                f'<div class="q"><div class="qtop">{tag("위치 · 자동생성")}'
                f'{tag(lq["template"], "role")}{vt}'
                f'<span class="mono note">{html.escape(lq["question_id"])}</span></div>'
                f'<div class="qtext">{html.escape(lq["question"])}</div>'
                f'<div class="ans">답 · <b>{html.escape(lq["answers"][0])}</b>'
                f'<span class="note mono"> · 근거 “{html.escape(str(lq.get("source_answer_text", "")))}” '
                f'{html.escape(str(lq.get("source_bbox", "")))}</span></div>'
                + (f'<div class="note">{html.escape(who)} · {html.escape(note)}</div>' if note else "")
                + "</div>")
        search = " ".join([sid] + [x["question"] for x in s["content_questions"]]
                          + [x["question"] for x in s["location_questions"]]).lower()
        flags = "screenqa" + (" borderline" if any(
            "borderline" in audit.get(l["question_id"], {}).get("note", "")
            for l in s["location_questions"]) else "")
        out.append(
            f'<article class="card" data-search="{html.escape(search, quote=True)}" '
            f'data-flags="{flags}">'
            f'<div class="shot"><img src="data:image/jpeg;base64,{scr_imgs[s["image"]]}" '
            f'alt="화면 {html.escape(sid)}"><div class="meta mono">{html.escape(s["image"])}<br>'
            f'{s["image_width"]}×{s["image_height"]}</div></div>'
            f'<div class="body"><div class="idline"><span class="sid mono">{html.escape(sid)}</span>'
            f'<span class="note">내용 {len(s["content_questions"])} · '
            f'위치 {len(s["location_questions"])} · 쌍 {pair_n.get(sid, 0)}개</span></div>'
            f'<div class="qgroup">{"".join(rows)}</div></div></article>')

    n_para = len(t1)
    n_loc = sum(len(s["location_questions"]) for s in screens)
    n_cq = sum(len(s["content_questions"]) for s in screens)
    n_dq = sum(len(d["questions"]) for d in docs)
    passed = sum(1 for v in audit.values() if v["visual_verdict"] == "pass")

    doc = f"""<title>시각 기억 파일럿 데이터</title>
<style>{CSS}</style>
<header><div class="hwrap">
<h1>시각 기억 파일럿 데이터<span>검수 전 원본 확인용</span></h1>
<div class="stats">
<span>DocVQA <b>{len(docs)}</b>문서 · 질문 <b>{n_dq}</b></span>
<span>ScreenQA <b>{len(screens)}</b>화면 · 내용 <b>{n_cq}</b> / 위치 <b>{n_loc}</b></span>
<span>T1 패러프레이즈 <b>{n_para}</b> (검증 대기)</span>
<span>위치 질문 시각검증 <b>{passed}/{len(audit)}</b> 통과</span>
</div>
<div class="controls">
<input id="q" type="search" placeholder="ID·질문 내용으로 검색 (예: 4806, brand, top half)">
<button class="chip" data-filter="docvqa" aria-pressed="false">DocVQA만</button>
<button class="chip" data-filter="screenqa" aria-pressed="false">ScreenQA만</button>
<button class="chip" data-filter="heldout" aria-pressed="false">held-out 질문 있는 문서</button>
<button class="chip" data-filter="borderline" aria-pressed="false">경계선 표시된 화면</button>
</div>
</div></header>
<main>{"".join(out)}<p class="empty" id="empty" hidden>조건에 맞는 항목이 없습니다.</p></main>
<footer>
<span>이미지는 열람용으로 축소·재압축했습니다. 정확한 판정은 원본 경로의 파일로 확인하세요.</span>
<span>DocVQA(lmms-lab) · ScreenQA(CC BY 4.0) 주석 + RICO 스크린샷 — 각 데이터의 이용 조건을 따릅니다.</span>
</footer>
<dialog id="zoom"><img id="zimg" alt=""><form method="dialog"><button>닫기</button></form></dialog>
<script>{JS}</script>"""

    with open(a.out, "w") as f:
        f.write(doc)
    print(f"[saved] {a.out} ({os.path.getsize(a.out)/1e6:.1f}MB)")


if __name__ == "__main__":
    build()
