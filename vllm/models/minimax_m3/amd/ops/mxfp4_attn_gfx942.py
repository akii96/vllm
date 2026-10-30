# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 MXFP4 K/V cache on gfx942: the page writer and the sparse attend.

gfx942 (MI300) has neither the fp4 converts nor the MX-scaled MFMA gfx950 uses,
so the pages are written with integer/fp32 ALU and read by decoding each e2m1
nibble straight into the bf16 bit pattern of ``value * 2^(e8m0 - 127)`` in
registers. That decode is exact: every e2m1 value times a power of two is a
bf16 number (bf16 has fp32's exponent range), so the bf16 MFMAs see precisely
the dequantized cache and the query stays bf16 (no e4m3 rounding of q).

The page is the one ``mxfp4_pages`` describes and gfx950 shares: per (block,
head slot) ``[128 tokens x 64 B e2m1][128 tokens x 4 B e8m0]``, low nibble the
even element, K heads in slots ``[0, H)`` and V heads in ``[H, 2H)``.

To avoid interleaving nibbles in registers, both dots run on the even and the
odd elements separately: ``q.k = q[0::2].k[0::2] + q[1::2].k[1::2]`` and the
two halves of ``p @ v`` are stored to the even and odd output columns. One
program serves one (query row, KV head) and its whole GQA group; the top-k
blocks are split over programs (and merged by a small reduce) when there are
too few rows to fill the GPU, which is the decode case.
"""

from functools import cache

import torch

from vllm.models.minimax_m3.amd.ops.mxfp4_pages import (
    DATA_BYTES,
    HEAD_DIM,
    PAGE_BYTES,
    PAGE_TOKENS,
)
from vllm.triton_utils import tl, triton

SCALE_GROUP = 32


@triton.jit
def _e2m1_rne(a):
    """Round-to-nearest-even e2m1 magnitude code (0..7) of ``a >= 0``,
    saturating at 6.0. The thresholds are the midpoints of the e2m1 grid;
    a tie goes to the code with the even mantissa."""
    code = (a > 0.25).to(tl.int32)
    code += (a >= 0.75).to(tl.int32)
    code += (a > 1.25).to(tl.int32)
    code += (a >= 1.75).to(tl.int32)
    code += (a > 2.5).to(tl.int32)
    code += (a >= 3.5).to(tl.int32)
    code += (a > 5.0).to(tl.int32)
    return code


@triton.jit
def _store_kv_mxfp4_kernel(
    k_ptr,
    v_ptr,
    cache_ptr,
    slot_ptr,
    stride_k_tok,
    stride_k_head,
    stride_v_tok,
    stride_v_head,
    stride_cache_block,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    PAGE: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    side = tl.program_id(1)  # 0..2H-1: K heads then V heads
    slot = tl.load(slot_ptr + tok).to(tl.int64)
    if slot < 0:
        return
    head = side % H
    is_v = side >= H
    if is_v:
        src = v_ptr + tok * stride_v_tok + head * stride_v_head
    else:
        src = k_ptr + tok * stride_k_tok + head * stride_k_head
    # Round to the activation dtype first (it already is), then fp32.
    x = tl.load(src + tl.arange(0, D)).to(tl.float32)
    x = tl.reshape(x, (D // 32, 32))
    amax = tl.max(tl.abs(x), axis=1)
    # e8m0 = ceil(log2(amax / 6)) + 127, from the exponent bits; all-zero
    # groups get the smallest normal scale. Same as AITER's MXFP4 writers.
    ratio = tl.maximum(amax, 6.0 * (2.0**-126)) * (1.0 / 6.0)
    e8m0 = (ratio.to(tl.uint32, bitcast=True) + 0x7FFFFF) >> 23
    inv = ((254 - e8m0) << 23).to(tl.float32, bitcast=True)
    y = x * inv[:, None]
    code = _e2m1_rne(tl.abs(y)) | tl.where(y < 0, 8, 0)
    code = tl.reshape(code, (D // 2, 2))
    lo, hi = tl.split(code)
    packed = (lo | (hi << 4)).to(tl.uint8)

    block = slot // BLOCK_TOKENS
    t = slot % BLOCK_TOKENS
    page = cache_ptr + block * stride_cache_block + side * PAGE
    tl.store(page + t * (D // 2) + tl.arange(0, D // 2), packed)
    tl.store(
        page + BLOCK_TOKENS * (D // 2) + t * (D // 32) + tl.arange(0, D // 32),
        e8m0.to(tl.uint8),
    )


def store_kv_mxfp4_gfx942(
    k: torch.Tensor,  # [tokens, H, 128] (any token / head strides)
    v: torch.Tensor,  # [tokens, H, 128]
    kv_cache: torch.Tensor,  # uint8 [blocks, 2H, 128, 68]
    slot_mapping: torch.Tensor,  # [tokens], negative = skip
) -> None:
    """Quantize K and V rows to MXFP4 and write them into their page slots."""
    num_tokens, num_heads, dim = k.shape
    if dim != HEAD_DIM or v.shape != k.shape:
        raise ValueError(f"MXFP4 K/V rows must be [tokens, H, {HEAD_DIM}]")
    if k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("MXFP4 K/V rows need a contiguous head dim")
    if kv_cache.dtype != torch.uint8 or kv_cache[0].numel() != 2 * num_heads * PAGE_BYTES:
        raise ValueError(
            f"MXFP4 K/V cache must be uint8 [blocks, {2 * num_heads}, 128, 68], got "
            f"{kv_cache.dtype} {tuple(kv_cache.shape)}"
        )
    if num_tokens == 0:
        return
    _store_kv_mxfp4_kernel[(num_tokens, 2 * num_heads)](
        k,
        v,
        kv_cache,
        slot_mapping,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        kv_cache.stride(0),
        H=num_heads,
        D=HEAD_DIM,
        BLOCK_TOKENS=PAGE_TOKENS,
        PAGE=PAGE_BYTES,
        num_warps=1,
    )


@triton.jit
def _e2m1_to_bf16(n, sc):
    """bf16 of ``e2m1(n) * 2^(e - 127)`` for int16 codes ``n`` and per-row
    ``sc = (e - 1) << 7``: the code's bits under the bf16 exponent plus the
    scale's exponent, the 0.5 code one exponent lower, 0 kept 0. Exact, since
    bf16 has fp32's exponent range."""
    m = n & 7
    bits = (m << 6) + sc
    bits = tl.where(m == 1, sc, bits)
    bits = tl.where(m == 0, 0, bits)
    bits = bits | ((n & 8) << 12)
    return bits.to(tl.bfloat16, bitcast=True)


@triton.jit
def _dequant_half(raw, s32, col, SHIFT: tl.constexpr):
    """bf16 ``[tokens, 64]`` of the even (SHIFT 0) or odd (SHIFT 4) elements of
    a page's rows: byte ``col`` holds elements ``2 col`` and ``2 col + 1``, of
    32-element chunk ``col // 16`` whose e8m0 is byte ``col // 16`` of the
    token's int32 scale word."""
    e = (s32[:, None] >> ((col[None, :] // 16) * 8)) & 255
    sc = ((e - 1) << 7).to(tl.int16)
    n = ((raw >> SHIFT) & 15).to(tl.int16)
    return _e2m1_to_bf16(n, sc)


@triton.jit
def _dequant_blocks_kernel(
    cache_ptr,
    bt_ptr,
    seq_lens_ptr,
    tmp_ptr,
    map_ptr,
    stride_cache_block,
    stride_bt,
    width,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    PAGE: tl.constexpr,
):
    """Block ``j`` of prefill request ``i``, side ``s`` (K heads then V heads),
    as exact bf16 rows ``[even 64 | odd 64]`` into slot ``i * width + j`` of the
    scratch, and ``map[block] = slot``."""
    i = tl.program_id(0)
    j = tl.program_id(1)
    side = tl.program_id(2)
    seq_len = tl.load(seq_lens_ptr + i)
    if j * BLOCK_TOKENS >= seq_len:
        return
    blk = tl.load(bt_ptr + i.to(tl.int64) * stride_bt + j)
    slot = i * width + j
    if side == 0:
        tl.store(map_ptr + blk, slot)
    HALF: tl.constexpr = D // 2
    tok = tl.arange(0, BLOCK_TOKENS)
    col = tl.arange(0, HALF)
    page = cache_ptr + blk.to(tl.int64) * stride_cache_block + side * PAGE
    raw = tl.load(page + tok[:, None] * HALF + col[None, :])
    s32 = tl.load((page + BLOCK_TOKENS * HALF + tok * 4).to(tl.pointer_type(tl.int32)))
    dst = (
        tmp_ptr
        + (slot.to(tl.int64) * (2 * H) + side) * (BLOCK_TOKENS * D)
        + tok[:, None] * D
        + col[None, :]
    )
    tl.store(dst, _dequant_half(raw, s32, col, 0))
    tl.store(dst + HALF, _dequant_half(raw, s32, col, 4))

@triton.jit
def _page_ptrs(cache_ptr, entries, slot, b, page_div, stride_cache_block, h, tok, col,
               H: tl.constexpr, HALF: tl.constexpr, BLOCK_TOKENS: tl.constexpr,
               PAGE: tl.constexpr):
    """K / V byte pointers ``[tokens, HALF]`` and int32 scale-word pointers
    ``[tokens]`` of the block top-k entry ``b`` of this row names."""
    entry = tl.sum(tl.where(slot == b, entries, 0), axis=0)
    block = cache_ptr + (entry // page_div).to(tl.int64) * stride_cache_block
    k_page = block + h * PAGE
    v_page = block + (H + h) * PAGE
    kp = k_page + tok[:, None] * HALF + col[None, :]
    vp = v_page + tok[:, None] * HALF + col[None, :]
    ksp = (k_page + BLOCK_TOKENS * HALF + tok * 4).to(tl.pointer_type(tl.int32))
    vsp = (v_page + BLOCK_TOKENS * HALF + tok * 4).to(tl.pointer_type(tl.int32))
    return kp, vp, ksp, vsp


@triton.jit
def _sparse_attn_mxfp4_kernel(
    q_ptr,
    cache_ptr,
    bt_ptr,
    ctx_ptr,
    out_ptr,
    part_acc_ptr,
    part_ml_ptr,
    stride_q_row,
    stride_q_head,
    stride_o_row,
    stride_o_head,
    stride_cache_block,
    stride_bt_row,
    bt_col_stride,
    page_div,
    qk_scale,  # softmax scale * log2(e)
    H: tl.constexpr,
    GROUP: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    D: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    PAGE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    MAX_BLOCKS_POW2: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    tmp_ptr=None,
    map_ptr=None,
    BF16_SRC: tl.constexpr = False,
):
    pid = tl.program_id(0)  # row * H + kv head
    split = tl.program_id(1)
    row = pid // H
    h = pid % H
    per_split: tl.constexpr = (MAX_BLOCKS + NUM_SPLITS - 1) // NUM_SPLITS
    bt_row = bt_ptr + pid.to(tl.int64) * stride_bt_row
    ctx = tl.load(ctx_ptr + pid)
    nblk = tl.cdiv(ctx, BLOCK_TOKENS)
    b0 = split * per_split
    b1 = tl.minimum(b0 + per_split, nblk)

    HALF: tl.constexpr = D // 2
    g = tl.arange(0, GROUP_PAD)
    gmask = g < GROUP
    tok = tl.arange(0, BLOCK_TOKENS)
    col = tl.arange(0, HALF)
    q_row = (
        q_ptr
        + row.to(tl.int64) * stride_q_row
        + (h * GROUP + g)[:, None] * stride_q_head
        + 2 * col[None, :]
    )
    qm = gmask[:, None]
    q_even = tl.load(q_row, mask=qm, other=0.0).to(tl.bfloat16)
    q_odd = tl.load(q_row + 1, mask=qm, other=0.0).to(tl.bfloat16)

    m_i = tl.full((GROUP_PAD,), -1.0e30, tl.float32)
    l_i = tl.zeros((GROUP_PAD,), tl.float32)
    # The output accumulates transposed, [HALF, GROUP]: the PV dot then has the
    # head dim as M, so its warps split the work instead of each repeating the
    # 16 query heads (and the V decode).
    acc_e = tl.zeros((HALF, GROUP_PAD), tl.float32)
    acc_o = tl.zeros((HALF, GROUP_PAD), tl.float32)
    # The split's table entries up front, then block b + 1's bytes are loaded
    # into registers while block b computes.
    slot = tl.arange(0, MAX_BLOCKS_POW2)
    entries = tl.load(bt_row + slot * bt_col_stride, mask=slot < MAX_BLOCKS, other=0)
    kp, vp, ksp, vsp = _page_ptrs(cache_ptr, entries, slot, b0, page_div, stride_cache_block,
                                  h, tok, col, H, HALF, BLOCK_TOKENS, PAGE)
    live = (b0 < b1) & (not BF16_SRC)
    k_raw = tl.load(kp, mask=live, other=0)
    k_s = tl.load(ksp, mask=live, other=0)
    v_raw = tl.load(vp, mask=live, other=0)
    v_s = tl.load(vsp, mask=live, other=0)
    if BF16_SRC:
        # Prefill: the blocks were dequantized once into the bf16 scratch.
        SIDE: tl.constexpr = BLOCK_TOKENS * D
        for b in range(b0, b1):
            entry = tl.sum(tl.where(slot == b, entries, 0), axis=0)
            sl = tl.load(map_ptr + entry // page_div).to(tl.int64)
            kb = tmp_ptr + (sl * (2 * H) + h) * SIDE + tok[:, None] * D + col[None, :]
            vb = kb + H * SIDE
            s = tl.dot(q_even, tl.trans(tl.load(kb)))
            s = tl.dot(q_odd, tl.trans(tl.load(kb + HALF)), s)
            valid = b * BLOCK_TOKENS + tok < ctx
            s = tl.where(valid[None, :], s * qk_scale, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            pt = tl.trans(p.to(tl.bfloat16))
            acc_e = tl.dot(tl.trans(tl.load(vb)), pt, acc_e * alpha[None, :])
            acc_o = tl.dot(tl.trans(tl.load(vb + HALF)), pt, acc_o * alpha[None, :])
            m_i = m_new
        b1 = b0  # skip the MXFP4 loop below
    for b in range(b0, b1):
        nxt = b + 1 < b1
        kp, vp, ksp, vsp = _page_ptrs(cache_ptr, entries, slot, b + 1, page_div,
                                      stride_cache_block, h, tok, col, H, HALF,
                                      BLOCK_TOKENS, PAGE)
        nk_raw = tl.load(kp, mask=nxt, other=0)
        nk_s = tl.load(ksp, mask=nxt, other=0)
        nv_raw = tl.load(vp, mask=nxt, other=0)
        nv_s = tl.load(vsp, mask=nxt, other=0)

        k_e = _dequant_half(k_raw, k_s, col, 0)
        k_o = _dequant_half(k_raw, k_s, col, 4)
        s = tl.dot(q_even, tl.trans(k_e))
        s = tl.dot(q_odd, tl.trans(k_o), s)
        valid = b * BLOCK_TOKENS + tok < ctx
        s = tl.where(valid[None, :], s * qk_scale, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        pb = p.to(tl.bfloat16)
        v_e = _dequant_half(v_raw, v_s, col, 0)
        v_o = _dequant_half(v_raw, v_s, col, 4)
        pt = tl.trans(pb)
        acc_e = tl.dot(tl.trans(v_e), pt, acc_e * alpha[None, :])
        acc_o = tl.dot(tl.trans(v_o), pt, acc_o * alpha[None, :])
        m_i = m_new
        k_raw, k_s, v_raw, v_s = nk_raw, nk_s, nv_raw, nv_s

    if NUM_SPLITS == 1:
        inv_l = (1.0 / tl.where(l_i > 0, l_i, 1.0))[None, :]
        o = out_ptr + row.to(tl.int64) * stride_o_row + (h * GROUP + g)[None, :] * stride_o_head
        om = gmask[None, :]
    else:
        inv_l = tl.full((1, GROUP_PAD), 1.0, tl.float32)
        part = (pid.to(tl.int64) * NUM_SPLITS + split) * GROUP_PAD
        o = part_acc_ptr + (part + g)[None, :] * D
        om = tl.full((1, GROUP_PAD), 1, tl.int1)
        tl.store(part_ml_ptr + (part + g) * 2, m_i)
        tl.store(part_ml_ptr + (part + g) * 2 + 1, l_i)
    ty = o.dtype.element_ty
    tl.store(o + 2 * col[:, None], (acc_e * inv_l).to(ty), mask=om)
    tl.store(o + 2 * col[:, None] + 1, (acc_o * inv_l).to(ty), mask=om)

@triton.jit
def _sparse_attn_reduce_kernel(
    part_acc_ptr,
    part_ml_ptr,
    out_ptr,
    stride_o_row,
    stride_o_head,
    H: tl.constexpr,
    GROUP: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // H
    h = pid % H
    g = tl.arange(0, GROUP_PAD)
    d = tl.arange(0, D)
    base = pid.to(tl.int64) * NUM_SPLITS * GROUP_PAD
    m = tl.full((GROUP_PAD,), -1.0e30, tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        m = tl.maximum(m, tl.load(part_ml_ptr + (base + s * GROUP_PAD + g) * 2))
    l = tl.zeros((GROUP_PAD,), tl.float32)
    acc = tl.zeros((GROUP_PAD, D), tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        off = base + s * GROUP_PAD + g
        w = tl.exp2(tl.load(part_ml_ptr + off * 2) - m)
        l += w * tl.load(part_ml_ptr + off * 2 + 1)
        acc += w[:, None] * tl.load(part_acc_ptr + off[:, None] * D + d[None, :])
    acc = acc / tl.where(l > 0, l, 1.0)[:, None]
    o = out_ptr + row.to(tl.int64) * stride_o_row + (h * GROUP + g)[:, None] * stride_o_head
    tl.store(o + d[None, :], acc.to(out_ptr.dtype.element_ty), mask=(g < GROUP)[:, None])


@cache
def _num_cus(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _num_splits(num_programs: int, max_blocks: int, device: torch.device) -> int:
    """Split the top-k blocks across programs until the GPU is about full."""
    cus = _num_cus(device.index if device.index is not None else 0)
    splits = 1
    while splits < max_blocks and num_programs * splits * 2 <= 2 * cus:
        splits *= 2
    return splits


def mxfp4_sparse_attend_gfx942(
    q: torch.Tensor,  # [rows, Hq, 128] bf16
    kv_cache: torch.Tensor,  # uint8 [blocks, 2H, 128, 68]
    sparse_bt: torch.Tensor,  # int32 [rows * H, >= max_blocks * bt_col_stride]
    sparse_ctx: torch.Tensor,  # int32 [rows * H] tokens attended per row
    out: torch.Tensor,  # [rows, Hq, 128]
    scale: float,
    *,
    num_kv_heads: int,
    max_blocks: int,
    bt_col_stride: int = 1,
    page_div: int = 1,
    num_splits: int | None = None,
    prefill_block_table: torch.Tensor | None = None,
    prefill_seq_lens: torch.Tensor | None = None,
    prefill_max_seq_len: int = 0,
) -> torch.Tensor:
    """Attend each (row, KV head) to its selected MXFP4 pages.

    With ``prefill_block_table`` / ``prefill_seq_lens`` (the block table and
    KV lengths of the requests the rows belong to), every block of those
    requests is first dequantized once into a bf16 scratch and the attend reads
    that instead of re-decoding a block for every query row; the result is the
    same (the dequantization is exact). This is the prefill path, where many
    rows share blocks.

    Row ``r * H + h`` of ``sparse_bt`` lists the pages for query row ``r`` and
    KV head ``h``, compacted with the partial (causal tail) block last, and
    ``sparse_ctx`` its token count. Entry ``j`` is read at column
    ``j * bt_col_stride`` and names cache block ``entry // page_div``, so the
    page-16 table AITER's top-k emits (``8`` and ``8 * H``) is read in place.
    """
    rows, num_q_heads, dim = q.shape
    H = num_kv_heads
    if dim != HEAD_DIM or num_q_heads % H:
        raise ValueError(f"q must be [rows, H * group, {HEAD_DIM}]")
    if q.stride(-1) != 1 or out.stride(-1) != 1:
        raise ValueError("q and out need a contiguous head dim")
    if kv_cache.dtype != torch.uint8 or kv_cache[0].numel() != 2 * H * PAGE_BYTES:
        raise ValueError("MXFP4 K/V cache must be uint8 [blocks, 2H, 128, 68]")
    if rows == 0:
        return out
    group = num_q_heads // H
    group_pad = max(16, triton.next_power_of_2(group))
    programs = rows * H
    extra = {}
    if prefill_block_table is not None:
        assert prefill_seq_lens is not None and prefill_max_seq_len > 0
        reqs = prefill_block_table.shape[0]
        width = triton.cdiv(prefill_max_seq_len, PAGE_TOKENS)
        tmp = torch.empty(
            (reqs * width, 2 * H, PAGE_TOKENS, HEAD_DIM),
            dtype=torch.bfloat16,
            device=q.device,
        )
        block_map = torch.empty(kv_cache.shape[0], dtype=torch.int32, device=q.device)
        _dequant_blocks_kernel[(reqs, width, 2 * H)](
            kv_cache,
            prefill_block_table,
            prefill_seq_lens,
            tmp,
            block_map,
            kv_cache.stride(0),
            prefill_block_table.stride(0),
            width,
            H=H,
            D=HEAD_DIM,
            BLOCK_TOKENS=PAGE_TOKENS,
            PAGE=PAGE_BYTES,
            num_warps=4,
        )
        extra = dict(tmp_ptr=tmp, map_ptr=block_map, BF16_SRC=True)
    if num_splits is None:
        num_splits = _num_splits(programs, max_blocks, q.device)
    num_splits = max(1, min(num_splits, max_blocks))
    if num_splits > 1:
        part_acc = torch.empty(
            (programs, num_splits, group_pad, HEAD_DIM), dtype=torch.float32, device=q.device
        )
        part_ml = torch.empty(
            (programs, num_splits, group_pad, 2), dtype=torch.float32, device=q.device
        )
    else:
        part_acc = part_ml = out
    _sparse_attn_mxfp4_kernel[(programs, num_splits)](
        q,
        kv_cache,
        sparse_bt,
        sparse_ctx,
        out,
        part_acc,
        part_ml,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        kv_cache.stride(0),
        sparse_bt.stride(0),
        bt_col_stride,
        page_div,
        scale * 1.4426950408889634,
        H=H,
        GROUP=group,
        GROUP_PAD=group_pad,
        D=HEAD_DIM,
        BLOCK_TOKENS=PAGE_TOKENS,
        PAGE=PAGE_BYTES,
        MAX_BLOCKS=max_blocks,
        NUM_SPLITS=num_splits,
        MAX_BLOCKS_POW2=triton.next_power_of_2(max_blocks),
        num_warps=4,
        num_stages=1,
        **extra,
    )
    if num_splits > 1:
        _sparse_attn_reduce_kernel[(programs,)](
            part_acc,
            part_ml,
            out,
            out.stride(0),
            out.stride(1),
            H=H,
            GROUP=group,
            GROUP_PAD=group_pad,
            D=HEAD_DIM,
            NUM_SPLITS=num_splits,
            num_warps=4,
        )
    return out


@triton.jit
def _dequant_page16_kernel(
    cache_ptr,
    bt_ptr,
    seq_lens_ptr,
    k_out_ptr,
    v_out_ptr,
    map_ptr,
    stride_cache_block,
    stride_bt,
    width,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    PAGE: tl.constexpr,
):
    """Block ``j`` of request ``i``, side ``s`` (K heads then V heads), to exact
    bf16 in AITER's shuffled page-16 layout at scratch slot ``i * width + j``
    (K ``[page, H, D/8, 16, 8]``, V ``[page, H, 2, D, 8]``, 8 pages a slot),
    and ``map[block] = slot``."""
    i = tl.program_id(0)
    j = tl.program_id(1)
    side = tl.program_id(2)
    seq_len = tl.load(seq_lens_ptr + i)
    if j * BLOCK_TOKENS >= seq_len:
        return
    blk = tl.load(bt_ptr + i.to(tl.int64) * stride_bt + j)
    slot = (i * width + j).to(tl.int64)
    if side == 0:
        tl.store(map_ptr + blk, slot.to(tl.int32))
    HALF: tl.constexpr = D // 2
    tok = tl.arange(0, BLOCK_TOKENS)
    col = tl.arange(0, HALF)
    page = cache_ptr + blk.to(tl.int64) * stride_cache_block + side * PAGE
    raw = tl.load(page + tok[:, None] * HALF + col[None, :])
    s32 = tl.load((page + BLOCK_TOKENS * HALF + tok * 4).to(tl.pointer_type(tl.int32)))
    h = side % H
    pg = slot * (BLOCK_TOKENS // 16) + tok // 16  # page-16 id within the scratch
    t16 = tok % 16
    for par in tl.static_range(2):
        d = 2 * col + par
        val = _dequant_half(raw, s32, col, 4 * par)
        if side < H:
            off = (
                ((pg[:, None] * H + h) * (D // 8) + d[None, :] // 8) * 128
                + t16[:, None] * 8
                + d[None, :] % 8
            )
            tl.store(k_out_ptr + off, val)
        else:
            off = (
                ((pg[:, None] * H + h) * 2 + t16[:, None] // 8) * (D * 8)
                + d[None, :] * 8
                + t16[:, None] % 8
            )
            tl.store(v_out_ptr + off, val)


@triton.jit
def _remap_table_kernel(bt_ptr, out_ptr, map_ptr, n, PER_BLOCK: tl.constexpr, BLK: tl.constexpr):
    """Page-16 entries ``(block * 8 + j) * H + h`` onto the scratch slots."""
    off = tl.program_id(0) * BLK + tl.arange(0, BLK)
    m = off < n
    e = tl.load(bt_ptr + off, mask=m, other=0)
    slot = tl.load(map_ptr + e // PER_BLOCK, mask=m, other=0)
    tl.store(out_ptr + off, slot * PER_BLOCK + e % PER_BLOCK, mask=m)


def mxfp4_sparse_attend_prefill_gfx942(
    q: torch.Tensor,  # [rows, Hq, 128] bf16
    kv_cache: torch.Tensor,  # uint8 [blocks, 2H, 128, 68]
    sparse_bt: torch.Tensor,  # int32 [rows * H, topk * 8] page-16 table
    sparse_ctx: torch.Tensor,  # int32 [rows * H]
    out: torch.Tensor,  # [rows, Hq, 128], contiguous
    scale: float,
    *,
    num_kv_heads: int,
    block_table: torch.Tensor,  # [reqs, max_blocks] the rows' requests
    seq_lens: torch.Tensor,  # [reqs]
    max_seq_len: int,
) -> torch.Tensor:
    """Prefill attend: every block of the rows' requests is dequantized once
    (exactly) to bf16 in AITER's page-16 layout and the page-16 table is
    remapped onto it, then AITER's sparse paged attention runs in bf16 -- the
    same kernel the fp8 K/V cache path uses."""
    from vllm.models.minimax_m3.amd.ops.sparse_pa import _run_gluon_decode

    H = num_kv_heads
    reqs = block_table.shape[0]
    width = triton.cdiv(max_seq_len, PAGE_TOKENS)
    pages = reqs * width * (PAGE_TOKENS // 16)
    k_s = torch.empty((pages, H, HEAD_DIM // 8, 16, 8), dtype=torch.bfloat16, device=q.device)
    v_s = torch.empty((pages, H, 2, HEAD_DIM, 8), dtype=torch.bfloat16, device=q.device)
    # Zeroed: table padding past a row's context may name any block.
    block_map = torch.zeros(kv_cache.shape[0], dtype=torch.int32, device=q.device)
    _dequant_page16_kernel[(reqs, width, 2 * H)](
        kv_cache,
        block_table,
        seq_lens,
        k_s,
        v_s,
        block_map,
        kv_cache.stride(0),
        block_table.stride(0),
        width,
        H=H,
        D=HEAD_DIM,
        BLOCK_TOKENS=PAGE_TOKENS,
        PAGE=PAGE_BYTES,
        num_warps=4,
    )
    table = torch.empty_like(sparse_bt)
    n = sparse_bt.numel()
    _remap_table_kernel[(triton.cdiv(n, 1024),)](
        sparse_bt, table, block_map, n, PER_BLOCK=(PAGE_TOKENS // 16) * H, BLK=1024
    )
    _run_gluon_decode(
        q.contiguous(), k_s, v_s, table, sparse_ctx, H, scale, out, None, None
    )
    return out

__all__ = [
    "DATA_BYTES",
    "SCALE_GROUP",
    "mxfp4_sparse_attend_gfx942",
    "mxfp4_sparse_attend_prefill_gfx942",
    "store_kv_mxfp4_gfx942",
]
