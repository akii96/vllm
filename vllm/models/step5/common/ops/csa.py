# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-5 CSA helpers shared by the paged kernels.

Region selection depends on rounding to the OCP E4M3 grid, so it is done
explicitly, also on gfx942 whose native FP8 is E4M3FNUZ.
"""

import torch

from vllm.triton_utils import tl, triton

_OCP_E4M3 = torch.float8_e4m3fn

# A row wider than this falls back to the sort-based reference: the per-program
# radix select keeps the whole row in registers.
_MAX_FUSED_ROW = 16384


def round_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round to the OCP E4M3 grid, saturating like Triton's ``float8e4nv`` cast."""
    x = x.to(torch.bfloat16).clamp(-448, 448)
    return x.to(_OCP_E4M3).to(torch.bfloat16)


def indexer_norm(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-centred RMSNorm on ``index_q``, LayerNorm on ``index_k``, in FP32."""
    dtype = index_q.dtype
    q = index_q.float()
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + eps)
    q = q * (1.0 + q_norm_weight.float())
    k = torch.nn.functional.layer_norm(
        index_k.float(),
        (index_k.shape[-1],),
        k_norm_weight.float(),
        k_norm_bias.float(),
        eps,
    )
    return q.to(dtype), k.to(dtype)


def select_regions(
    scores: torch.Tensor,
    positions: torch.Tensor,
    region_size: int,
    topk: int,
) -> torch.Tensor:
    """Top-``topk`` earlier regions plus the query's own, as ascending int32 ids
    ``[T, G, topk + 1]`` padded with ``-1``. Ties pick the lower id."""
    tokens, groups, regions = scores.shape
    device = scores.device
    own = torch.div(positions.long(), region_size, rounding_mode="floor")
    visible = torch.arange(regions, device=device)[None, :] < own[:, None]
    masked = scores.masked_fill(~visible[:, None, :], float("-inf"))

    keep = min(topk, regions)
    picked = torch.sort(masked, dim=-1, descending=True, stable=True).indices
    picked = picked[..., :keep]
    count = torch.clamp(own, max=topk).view(tokens, 1, 1)
    slot = torch.arange(keep, device=device).view(1, 1, keep)
    # Invisible picks sort last (sentinel = regions) and are then dropped.
    picked = torch.where(slot < count, picked, regions)
    picked = torch.sort(picked, dim=-1).values

    out = torch.full((tokens, groups, topk + 1), -1, device=device, dtype=torch.int64)
    out[..., :keep] = picked.masked_fill(picked == regions, -1)
    out.scatter_(
        -1,
        count.expand(tokens, groups, 1),
        own.view(tokens, 1, 1).expand(tokens, groups, 1),
    )
    return out.to(torch.int32)


@triton.jit
def _select_regions_kernel(
    scores_ptr,
    positions_ptr,
    out_ptr,
    stride_s_token,
    stride_s_group,
    stride_o_token,
    stride_o_group,
    num_regions,
    REGION_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    TOPK_P1_POW2: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Per (token, group) top-``min(TOPK, own)`` visible regions via a fused
    radix select, plus the query's own region, ascending with ``-1`` padding.

    Equivalent to ``select_regions``: ties pick the lower region id. Descending
    float order is an ascending order on the monotone int32 key
    ``bits >= 0 ? bits ^ 0x7FFFFFFF : bits``, with -0.0 tied to +0.0.
    """
    token = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1)
    own = tl.load(positions_ptr + token) // REGION_SIZE
    rid = tl.arange(0, BLOCK_R)
    live = (rid < num_regions) & (rid < own)
    scores = tl.load(
        scores_ptr + token * stride_s_token + group * stride_s_group + rid,
        mask=live,
        other=float("-inf"),
    )
    bits = scores.to(tl.int32, bitcast=True)
    key = tl.where(bits >= 0, bits ^ 2147483647, bits)
    key = tl.where(bits == -2147483648, 2147483647, key)

    keep = tl.minimum(TOPK, num_regions)
    remaining = tl.minimum(keep, own)
    count = tl.minimum(tl.maximum(own, 0), TOPK)
    candidate = live
    selected = tl.zeros((BLOCK_R,), dtype=tl.int1)
    bins = tl.arange(0, 256)
    for shift in tl.static_range(24, -8, -8):
        digit = (key >> shift) & 0xFF
        hist = tl.histogram(tl.where(candidate, digit, 0), 256)
        hist -= tl.where(bins == 0, BLOCK_R - tl.sum(candidate.to(tl.int32), 0), 0)
        prefix = tl.cumsum(hist, axis=0)
        pivot = tl.min(tl.where(prefix >= remaining, bins, 256))
        selected |= candidate & (digit < pivot)
        remaining -= tl.sum(tl.where(bins < pivot, hist, 0))
        candidate &= digit == pivot
        if shift == 0:
            rank = tl.cumsum(candidate.to(tl.int32), axis=0)
            selected |= candidate & (rank <= remaining)

    slot = tl.cumsum(selected.to(tl.int32), axis=0) - selected.to(tl.int32)
    tl.store(
        out_ptr + token * stride_o_token + group * stride_o_group + slot,
        rid,
        mask=selected,
    )
    # The reference scatters the own region at clamp(own, max=topk); slots the
    # selection did not fill keep the pre-filled -1.
    lanes = tl.arange(0, TOPK_P1_POW2)
    values = tl.where(lanes == count, own, -1)
    tl.store(
        out_ptr + token * stride_o_token + group * stride_o_group + lanes,
        values,
        mask=(lanes >= count) & (lanes < TOPK + 1),
    )


def select_regions_fused(
    scores: torch.Tensor,
    positions: torch.Tensor,
    region_size: int,
    topk: int,
) -> torch.Tensor:
    """Fused top-k region selection; one kernel replaces the two sorts of
    ``select_regions``. Same outputs, including tie-breaking on the lower id."""
    tokens, groups, regions = scores.shape
    out = torch.full(
        (tokens, groups, topk + 1), -1, device=scores.device, dtype=torch.int32
    )
    if tokens == 0 or groups == 0 or regions == 0:
        return out
    block_r = triton.next_power_of_2(regions)
    if block_r > _MAX_FUSED_ROW:
        return select_regions(scores, positions, region_size, topk)
    _select_regions_kernel[(tokens, groups)](
        scores,
        positions,
        out,
        scores.stride(0),
        scores.stride(1),
        out.stride(0),
        out.stride(1),
        regions,
        REGION_SIZE=region_size,
        TOPK=topk,
        TOPK_P1_POW2=triton.next_power_of_2(topk + 1),
        BLOCK_R=block_r,
        num_warps=4 if block_r <= 1024 else 8,
    )
    return out
