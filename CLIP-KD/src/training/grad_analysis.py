import re
import torch
import torch.nn.functional as F
import torch.distributed as dist

from typing import Optional, Dict, List, Tuple

try:
    import wandb
except ImportError:
    wandb = None

def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

# ---------- (utils) block grouping ----------
def _parse_block(name: str):
    # visual blocks
    m = re.search(r"visual\.transformer\.resblocks\.(\d+)\.", name)
    if m:
        return "visual", int(m.group(1))

    # text blocks (open_clip often uses transformer.resblocks.* or text.transformer.resblocks.*)
    m = re.search(r"^(?:text\.)?transformer\.resblocks\.(\d+)\.", name)
    if m:
        return "text", int(m.group(1))

    return None, None

def _build_block_param_groups(model) -> Tuple[Dict[str, List[int]], List[torch.nn.Parameter], List[str]]:
    """
    returns:
      groups: {"visual/block_0": [param_idx,...], "text/block_0": [...] , ...}
      params: list of trainable params aligned to indices
      names:  param names aligned to indices
    """
    params = []
    names = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            names.append(n)
            params.append(p)

    groups: Dict[str, List[int]] = {}
    for idx, n in enumerate(names):
        t, b = _parse_block(n)
        if t is None:
            continue
        key = f"{t}/block_{b}"
        groups.setdefault(key, []).append(idx)
    
    return groups, params, names


def _cat_flat_by_indices(vec_list: List[torch.Tensor], idxs: List[int], max_elems: Optional[int]):
    parts = []
    for i in idxs:
        g = vec_list[i]
        if g is None:
            continue
        parts.append(g.reshape(-1))
    if not parts:
        return None
    v = torch.cat(parts)
    if max_elems is not None and v.numel() > max_elems:
        v = v[:max_elems]
    return v

# -------------------------
# (A) Total grad logging (after backward, after unscale for AMP)
# -------------------------
@torch.no_grad()
def _grad_l2_norm_from_pgrad(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        n = torch.linalg.vector_norm(g).item()
        total += n * n
    return total ** 0.5


@torch.no_grad()
def _weight_l2_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        w = p.detach()
        n = torch.linalg.vector_norm(w).item()
        total += n * n
    return total ** 0.5


def tb_log_total_grad(
    writer,
    model,
    step: int,
    every: int = 50,
    tag: str = "train/grad",
    enable: bool = True, 
):
    if (not enable) or writer is None or (step % every != 0) or (not is_main_process()):
        return

    gnorm = _grad_l2_norm_from_pgrad(model)
    wnorm = _weight_l2_norm(model)
    writer.add_scalar(f"{tag}/grad_norm_total", gnorm, step)
    writer.add_scalar(f"{tag}/grad_to_weight", gnorm / (wnorm + 1e-12), step)
    writer.flush()



# -------------------------
# (B) Loss-wise norm + pairwise cosine (BLOCKWISE)
# -------------------------

def tb_log_losswise_norm_and_cos(
    writer,
    model,
    loss_dict: dict,     # {"task": task_loss, "ckd": ckd_loss, "icl": icl_loss, "fd": fd_loss, ...}
    step: int,
    every: int = 1000,
    tag: str = "train/losswise",
    max_elems: int = 500_000,
    enable: bool = True,           # blockwise 로깅 on/off
    enable_text: bool = True,      # text tower까지 찍을지
    mode: str = "blockwise",       # "blockwise" | "global" | "both"
):
    """
    BLOCKWISE로 찍음:
      - loss별 grad norm: {tag}/grad_norm_from_{loss}/{tower}/block_i
      - pairwise cosine: {tag}/cos_{a}__{b}/{tower}/block_i
      - conflict_ratio_negcos: {tag}/conflict_ratio_negcos/{tower}/block_i
      - conflict_ratio_cancel: {tag}/conflict_ratio_cancel/{tower}/block_i   (상쇄 기반)
    """
    if (not enable) or writer is None or (step % every != 0) or (not is_main_process()):
        return

    # 활성 loss만
    active = {k: v for k, v in loss_dict.items() if (v is not None and torch.is_tensor(v))}
    if len(active) < 1:
        return

    groups, params, _ = _build_block_param_groups(model)
    if not enable_text:
        groups = {k: v for k, v in groups.items() if k.startswith("visual/")}

    if len(groups) == 0:
        return

    # loss별 grads (params 길이와 동일한 list로 맞춤)
    # autograd.grad는 전체 params를 대상으로 한 번씩 호출
    grad_per_loss: Dict[str, List[Optional[torch.Tensor]]] = {}

    # retain_graph=True: 다음 loss grad도 뽑아야 하니까 (train loop에서 backward도 할 거면 안전)
    for li, (lname, L) in enumerate(active.items()):
        grads = torch.autograd.grad(
            L,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        grad_per_loss[lname] = [None if g is None else g.detach() for g in grads]

    loss_names = list(grad_per_loss.keys())
    if len(loss_names) == 0:
        return

    # group별로 norm/cos/conflict를 찍는다
    for gname, idxs in groups.items():
        # (1) loss별 gvec
        gvec = {}
        gnorm = {}
        for lname in loss_names:
            v = _cat_flat_by_indices(grad_per_loss[lname], idxs, max_elems=max_elems)
            if v is None or v.numel() == 0:
                continue
            gvec[lname] = v
            gnorm[lname] = torch.linalg.vector_norm(v)

            writer.add_scalar(f"{tag}/grad_norm_from_{lname}/{gname}", gnorm[lname].item(), step)

        names = list(gvec.keys())
        if len(names) < 2:
            continue

        # (2) pairwise cosine + negcos ratio
        neg = 0
        tot = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                cos = F.cosine_similarity(gvec[a], gvec[b], dim=0).item()
                writer.add_scalar(f"{tag}/cos_{a}__{b}/{gname}", cos, step)
                tot += 1
                if cos < 0:
                    neg += 1
        writer.add_scalar(f"{tag}/conflict_ratio_negcos/{gname}", neg / max(tot, 1), step)

        # (3) cancellation 기반 conflict (내가 전에 말한 상쇄 지표)
        # 1 - ||sum|| / sum||.||
        gsum = torch.zeros_like(next(iter(gvec.values())))
        denom = 0.0
        for lname in names:
            gsum += gvec[lname]
            denom += gnorm[lname].item()
        cancel = 1.0 - (torch.linalg.vector_norm(gsum).item() / (denom + 1e-12))
        writer.add_scalar(f"{tag}/conflict_ratio_cancel/{gname}", cancel, step)

    writer.flush()

# -------------------------
# (B)-1 Loss-wise norm + pairwise cosine (before backward; read-only via autograd.grad)
# -------------------------
def _flatten_grads(grads, max_elems: Optional[int]):
    parts = [g.reshape(-1) for g in grads if g is not None]
    if not parts:
        return None
    v = torch.cat(parts)
    if max_elems is not None and v.numel() > max_elems:
        # 코드 최소화용: 앞부분만 사용 (trend 보기엔 충분)
        v = v[:max_elems]
    return v


def tb_log_losswise_norm_and_cos(
    writer,
    model,
    loss_dict: dict,     # {"fd": loss_fd or None, "crd": loss_crd or None, ...}
    step: int,
    every: int = 1000,
    tag: str = "train/losswise",
    max_elems: int = 500_000,
):
    """
    - loss별 grad norm (||g_i||) 기록
    - loss pairwise cosine (cos(g_i, g_j)) 기록
    - conflict_ratio: cos < 0 인 pair 비율
    """
    if writer is None or (step % every != 0) or (not is_main_process()):
        return

    # 활성 loss만 추림 (None / 비텐서 / grad 불가 등 방어)
    active = {}
    for name, loss in loss_dict.items():
        if loss is None:
            continue
        if not torch.is_tensor(loss):
            continue
        active[name] = loss

    if len(active) < 1:
        return

    params = [p for p in model.parameters() if p.requires_grad]

    gvec = {}
    names = []

    for name, loss in active.items():
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,   # 뒤에 total_loss backward를 해야 하니까
            allow_unused=True,
        )
        v = _flatten_grads(grads, max_elems=max_elems)
        if v is None or v.numel() == 0:
            continue

        v = v.detach()
        gvec[name] = v
        names.append(name)

        gnorm = torch.linalg.vector_norm(v).item()
        writer.add_scalar(f"{tag}/grad_norm_from_{name}", gnorm, step)

    if len(names) < 2:
        writer.flush()
        return

    conflict_cnt = 0
    pair_cnt = 0

    # 모든 pair (loss가 3개 이상이어도 자동으로 다 찍힘)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            cos = F.cosine_similarity(gvec[a], gvec[b], dim=0).item()
            writer.add_scalar(f"{tag}/cos_{a}__{b}", cos, step)
            pair_cnt += 1
            if cos < 0:
                conflict_cnt += 1

    writer.add_scalar(f"{tag}/conflict_ratio", conflict_cnt / max(pair_cnt, 1), step)
    writer.flush()


# -------------------------
# (C) 개별 method loss
# -------------------------
def log_losses(writer, args, step, total_loss, loss_dict, log_prefix="train/loss"):
    # 중복 방지: master만
    if not (is_main_process()):
        return

    # TensorBoard
    if writer is not None:
        # writer.add_scalar(f"{log_prefix}/total", float(total_loss.detach().item()), step)
        for k, v in loss_dict.items():
            if v is None:
                continue
            writer.add_scalar(f"{log_prefix}/{k}", float(v.detach().item()), step)
        writer.flush()

    # W&B (원하면)
    if args.wandb:
        assert wandb is not None, "Please install wandb."
        log_data = {f"{log_prefix}/total": float(total_loss.detach().item())}
        for k, v in loss_dict.items():
            if v is None:
                continue
            log_data[f"{log_prefix}/{k}"] = float(v.detach().item())
        wandb.log(log_data, step=step)

