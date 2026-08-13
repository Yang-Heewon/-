"""D4 write-time 신호 스코어러 — 시각 토큰별 점수 (클수록 보존 우선), 시각 순서 기준.

S0 random          : 통제 하한
S1 query attention : SnapKV식 — 질문 i의 attention (query-aware, 전이 붕괴 예상)
S3 encoder saliency: 인코더 출력 임베딩 노름 (query-agnostic, zero-cost)
S4 pixel variance  : 패치 픽셀 분산 (무학습 대조군 — S3가 못 이기면 기여 증발)
S5 KVzip-VLM       : 멀티모달 재구성 질의(화면 서술)의 attention (방법 1차 후보, PLAN §1.4)
"""
import torch

from .attnstat import QKCapture, recv_column_mass
from .spans import token_spans

RECON_PROMPT = ("Describe this document image in as much detail as possible, "
                "including all visible text, numbers, and layout.")


def vlm_inputs(processor, img, user_text, device):
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": user_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ins = processor(text=[text], images=[img], return_tensors="pt").to(device)
    ins["pixel_values"] = ins["pixel_values"].to(torch.float16)
    return ins


def _forward_kwargs(ins, input_ids=None):
    ids = ins["input_ids"] if input_ids is None else input_ids
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                pixel_values=ins["pixel_values"], image_grid_thw=ins["image_grid_thw"],
                use_cache=False)


@torch.no_grad()
def _recv_visual(model, ins, input_ids, row_start, row_end, vis_pos):
    with QKCapture() as cap:
        model(**_forward_kwargs(ins, input_ids))
    recv = recv_column_mass(cap.qk, row_start, row_end)
    return recv[vis_pos].cpu()


def score_s0(n_vis, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n_vis, generator=g)


def spatial_uniform_keep(grid_thw, merge_size: int, keep_count: int) -> set[int]:
    """Deterministic farthest-point coverage on the merged visual-token grid.

    This is a query-agnostic control, not a saliency method.  Coordinates are
    normalized per temporal/spatial axis so portrait and landscape images get
    comparable coverage.
    """
    if merge_size <= 0:
        raise ValueError("merge_size must be positive")
    t, h, w = [int(x) for x in torch.as_tensor(grid_thw)[0]]
    if h % merge_size or w % merge_size:
        raise ValueError("grid height/width must be divisible by merge_size")
    rows, cols = h // merge_size, w // merge_size
    n_visual = t * rows * cols
    if not 1 <= keep_count <= n_visual:
        raise ValueError(f"keep_count must be in [1, {n_visual}], got {keep_count}")
    if keep_count == n_visual:
        return set(range(n_visual))

    tt, yy, xx = torch.meshgrid(
        torch.arange(t, dtype=torch.float32),
        torch.arange(rows, dtype=torch.float32),
        torch.arange(cols, dtype=torch.float32),
        indexing="ij",
    )
    coords = torch.stack((tt, yy, xx), dim=-1).reshape(-1, 3)
    scale = torch.tensor([max(t - 1, 1), max(rows - 1, 1), max(cols - 1, 1)])
    coords = coords / scale
    center = torch.tensor([0.5 if t > 1 else 0.0, 0.5, 0.5])
    first = int(((coords - center) ** 2).sum(dim=-1).argmin())
    selected = [first]
    min_distance = ((coords - coords[first]) ** 2).sum(dim=-1)
    min_distance[first] = -1
    for _ in range(1, keep_count):
        next_index = int(min_distance.argmax())
        selected.append(next_index)
        distance = ((coords - coords[next_index]) ** 2).sum(dim=-1)
        min_distance = torch.minimum(min_distance, distance)
        min_distance[selected] = -1
    return set(selected)


@torch.no_grad()
def score_s1(model, processor, img, question, device):
    """질문 토큰 행(시각 끝 이후 전부 = 질문+어시스턴트 헤더)의 열 질량."""
    ins = vlm_inputs(processor, img, question, device)
    sp = token_spans(ins["input_ids"], model.config)
    return _recv_visual(model, ins, ins["input_ids"],
                        sp["vis_end"] + 1, sp["L"], sp["visual"])


@torch.no_grad()
def score_s3(model, processor, img, device):
    """인코더 출력(머저 이후, LLM 정렬 차원) 임베딩의 L2 노름."""
    ins = vlm_inputs(processor, img, "x", device)
    visual = model.model.visual if hasattr(model.model, "visual") else model.visual
    emb = visual(ins["pixel_values"], grid_thw=ins["image_grid_thw"])
    return emb.float().norm(dim=-1).cpu()


def score_s4(processor, img, grid_thw):
    """머지드 토큰(28×28px 영역)별 그레이스케일 픽셀 분산 — 전처리 리사이즈 재현."""
    t, h, w = [int(x) for x in grid_thw[0]]
    m = processor.image_processor.merge_size
    img_r = img.convert("L").resize((w * 14, h * 14))
    arr = torch.tensor(list(img_r.getdata()), dtype=torch.float32).view(h * 14, w * 14)
    blocks = arr.view(h // m, m * 14, w // m, m * 14).permute(0, 2, 1, 3)
    return blocks.reshape(h // m, w // m, -1).var(dim=-1).flatten()


@torch.no_grad()
def score_s5(model, processor, img, device, max_new_tokens=96):
    """KVzip-VLM: 재구성 질의로 생성한 서술 토큰 행들의 열 질량."""
    ins = vlm_inputs(processor, img, RECON_PROMPT, device)
    gen = model.generate(**{k: v for k, v in ins.items()},
                         max_new_tokens=max_new_tokens, do_sample=False)
    full_ids = gen  # (1, P+G) — 프롬프트 + 생성 서술
    sp = token_spans(full_ids, model.config)
    P = ins["input_ids"].shape[1]
    return _recv_visual(model, ins, full_ids, P, full_ids.shape[1], sp["visual"])


def topk_keep(scores, budget):
    """상위 budget 비율의 시각 순서 인덱스 집합."""
    n = scores.shape[0]
    k = max(1, int(round(n * budget)))
    return set(torch.topk(scores, k).indices.tolist())
