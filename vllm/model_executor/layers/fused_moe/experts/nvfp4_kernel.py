# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 MoE GEMM kernel for gfx942 using tl.dot_scaled with fp8 scales.

W layout: [E, K//2, N] packed e2m1 uint8 (K-major)
W scale:  [E, N, K//16] fp8_e4m3fn (NVFP4 per-16 scales)
X scale:  [M, K//16] fp8_e4m3fn (all 1.0 for non-microscaled X)
"""

import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.common.pid_preprocessing import pid_grid
from aiter.ops.triton._triton_kernels.moe.activations import _swiglu
from aiter.ops.triton._triton_kernels.moe.quant_moe import (
    _compute_static_fp8_quant,
)
from aiter.ops.triton.moe.moe_op_gemm_a8w4 import should_upcast_indices
from aiter.ops.triton.moe.reduce import reduce_grouped
from aiter.ops.triton.utils.device_info import get_num_xcds


@triton.jit
def _xcd_swizzle(pid, domain_size, XCD_SWIZZLE: tl.constexpr):
    pids_per_group = domain_size // XCD_SWIZZLE
    extra_pid_groups = domain_size % XCD_SWIZZLE
    group = pid % XCD_SWIZZLE
    local_pid = pid // XCD_SWIZZLE
    return group * pids_per_group + min(group, extra_pid_groups) + local_pid


@triton.jit
def _moe_gemm_nvfp4_dot_scaled(
    Y,
    stride_y_k,
    stride_y_m,
    stride_y_n,
    X,
    stride_x_m,
    stride_x_k,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    WScale,
    stride_ws_e,
    stride_ws_n,
    stride_ws_k,
    GlobalScale,
    X_static_scale,
    Quant_static_scale,
    B,
    stride_b_e,
    Gammas,
    N,
    K,
    GatherIndx,
    ExptHist,
    ExptOffs,
    ExptOffsSum,
    ExptData,
    grid_m,
    grid_n,
    APPLY_SWIGLU: tl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: tl.constexpr,
    SWIGLU_ADD_RESIDUAL: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    XCD_SWIZZLE: tl.constexpr,
    EVEN_K: tl.constexpr,
    MASK_K_LIMIT: tl.constexpr,
    SPLIT_K: tl.constexpr,
    W_CACHE_MODIFIER: tl.constexpr,
    UPCAST_INDICES: tl.constexpr = False,
):
    """NVFP4 MoE GEMM using tl.dot_scaled with fp8_e4m3fn scales."""
    NVFP4_SCALE_FACTOR: tl.constexpr = 16
    NVFP4_SCALE_BLOCK_K: tl.constexpr = BLOCK_K // NVFP4_SCALE_FACTOR
    PACKED_BLOCK_K: tl.constexpr = BLOCK_K // 2

    tl.static_assert(BLOCK_K % NVFP4_SCALE_FACTOR == 0)

    OUT_BLOCK_N: tl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    tl.assume(stride_y_k >= 0)
    tl.assume(stride_y_m >= 0)
    tl.assume(stride_y_n >= 0)
    tl.assume(stride_x_m >= 0)
    tl.assume(stride_x_k >= 0)
    tl.assume(stride_w_e >= 0)
    tl.assume(stride_w_k >= 0)
    tl.assume(stride_w_n >= 0)
    tl.assume(stride_ws_e >= 0)
    tl.assume(stride_ws_n >= 0)
    tl.assume(stride_ws_k >= 0)
    if B is not None:
        tl.assume(stride_b_e >= 0)
    tl.assume(grid_m >= 0)
    tl.assume(grid_n >= 0)

    pid = tl.program_id(0)
    if ExptOffsSum is not None and XCD_SWIZZLE > 1:
        padding_m = grid_m - tl.load(ExptOffsSum)
    else:
        padding_m: tl.constexpr = 0

    index_type: tl.constexpr = tl.int64 if UPCAST_INDICES else tl.int32

    unpadded_m = grid_m - padding_m
    tl.assume(unpadded_m >= 0)
    total_actual_tiles = unpadded_m * grid_n * SPLIT_K
    if padding_m > 0 and pid >= total_actual_tiles:
        return

    pid_emnk = pid
    if XCD_SWIZZLE != 1:
        pid_emnk = _xcd_swizzle(pid_emnk, total_actual_tiles, XCD_SWIZZLE)
    pid_mnk = pid_emnk % (unpadded_m * grid_n * SPLIT_K)
    pid_k = pid_mnk % SPLIT_K
    pid_mn = pid_mnk // SPLIT_K
    pid_m, pid_n = pid_grid(pid_mn, unpadded_m, grid_n, GROUP_M)

    if SPLIT_K > 1:
        Y += pid_k.to(index_type) * stride_y_k

    expt_data = tl.load(ExptData + pid_m)
    if expt_data == -1:
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M = tl.load(ExptHist + expt_id)
    start_m = tl.load(ExptOffs + expt_id)
    expt_id, block_id = expt_id.to(index_type), block_id.to(index_type)
    start_m = start_m.to(index_type)
    pid_n, pid_k = pid_n.to(index_type), pid_k.to(index_type)

    offs_x_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_x_m = tl.max_contiguous(tl.multiple_of(offs_x_m % M, BLOCK_M), BLOCK_M)
    if GatherIndx is None:
        X += start_m * stride_x_m
    else:
        GatherIndx += start_m
        offs_x_m = tl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT
    offs_x_k = BLOCK_K * pid_k + tl.arange(0, BLOCK_K)
    # X may be float8_e4m3fnuz (FNUZ) on gfx942. dot_scaled expects OCP
    # float8_e4m3fn. We load as uint8 so dot_scaled does the bitcast.
    # FNUZ and OCP have identical bit patterns for non-zero values.
    XPtrs = (
        X
        + offs_x_m.to(index_type)[:, None] * stride_x_m
        + offs_x_k.to(index_type)[None, :] * stride_x_k
    )

    offs_w_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_w_n = tl.max_contiguous(tl.multiple_of(offs_w_n % N, BLOCK_N), BLOCK_N)
    offs_w_k = PACKED_BLOCK_K * pid_k + tl.arange(0, PACKED_BLOCK_K)
    W += expt_id * stride_w_e
    WPtrs = W + (
        offs_w_k.to(index_type)[:, None] * stride_w_k
        + offs_w_n.to(index_type)[None, :] * stride_w_n
    )

    offs_ws_k = NVFP4_SCALE_BLOCK_K * pid_k + tl.arange(0, NVFP4_SCALE_BLOCK_K)
    WScale += expt_id * stride_ws_e
    WScalePtrs = WScale + (
        offs_w_n.to(index_type)[:, None] * stride_ws_n
        + offs_ws_k.to(index_type)[None, :] * stride_ws_k
    )

    num_k_iter = tl.cdiv(K, BLOCK_K * SPLIT_K)
    if not EVEN_K:
        num_k_iter -= 1

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _k in range(num_k_iter):
        x = tl.load(XPtrs)
        w = tl.load(WPtrs, cache_modifier=W_CACHE_MODIFIER)
        ws = tl.load(WScalePtrs, cache_modifier=W_CACHE_MODIFIER)

        # Identity X scales: OCP fp8 1.0 = 0x38
        xs = tl.full((BLOCK_M, NVFP4_SCALE_BLOCK_K), 0x38, dtype=tl.uint8)
        xs = tl.cast(xs, tl.float8e4nv, bitcast=True)

        acc = tl.dot_scaled(x, xs, "e4m3", w, ws, "e2m1", acc=acc, fast_math=True)

        XPtrs += (BLOCK_K * SPLIT_K) * stride_x_k
        WPtrs += (PACKED_BLOCK_K * SPLIT_K) * stride_w_k
        WScalePtrs += (NVFP4_SCALE_BLOCK_K * SPLIT_K) * stride_ws_k

    if not EVEN_K:
        mask_x_k = offs_x_k < MASK_K_LIMIT
        mask_w_k = offs_w_k < (MASK_K_LIMIT // 2)
        mask_ws_k = offs_ws_k * NVFP4_SCALE_FACTOR < MASK_K_LIMIT

        x = tl.load(XPtrs, mask=mask_x_k[None, :], other=0.0)
        w = tl.load(
            WPtrs, mask=mask_w_k[:, None], other=0, cache_modifier=W_CACHE_MODIFIER
        )
        ws = tl.load(
            WScalePtrs,
            mask=mask_ws_k[None, :],
            other=0.0,
            cache_modifier=W_CACHE_MODIFIER,
        )

        xs = tl.full((BLOCK_M, NVFP4_SCALE_BLOCK_K), 0x38, dtype=tl.uint8)
        xs = tl.cast(xs, tl.float8e4nv, bitcast=True)

        acc = tl.dot_scaled(x, xs, "e4m3", w, ws, "e2m1", acc=acc, fast_math=True)

    global_scale = tl.load(GlobalScale + expt_id).to(tl.float32)
    acc *= global_scale

    if X_static_scale is not None:
        acc *= tl.load(X_static_scale)

    offs_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_y_n = BLOCK_N * pid_n + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_y_n < N
    if B is not None:
        BPtrs = B + expt_id * stride_b_e + offs_y_n
        if pid_k == 0:
            bias = tl.load(BPtrs, mask=mask_n, other=0, cache_modifier=W_CACHE_MODIFIER)
        else:
            bias = tl.full([BLOCK_N], 0, dtype=tl.float32)
        acc = acc + bias[None, :]

    if APPLY_SWIGLU and SPLIT_K == 1:
        out = _swiglu(acc, alpha, limit, ADD_RESIDUAL=SWIGLU_ADD_RESIDUAL)
        tl.static_assert(out.shape[1] == OUT_BLOCK_N)
        offs_y_n = OUT_BLOCK_N * pid_n + tl.arange(0, OUT_BLOCK_N)
        mask_n = offs_y_n < yN
    else:
        tl.static_assert(ACTIVATION_REDUCTION_N == 1)
        out = acc

    if Gammas is not None:
        gammas = tl.load(Gammas + start_m + offs_m, mask=mask_m, other=0.0)
        out *= gammas[:, None]

    if Quant_static_scale is not None:
        out = _compute_static_fp8_quant(out, tl.load(Quant_static_scale))

    Y += start_m * stride_y_m
    offs_y_m = offs_m
    YPtrs = (
        Y
        + offs_y_m.to(index_type)[:, None] * stride_y_m
        + offs_y_n.to(index_type)[None, :] * stride_y_n
    )
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(YPtrs, out, mask=mask)


def moe_gemm_a8w4_nvfp4(
    x,
    w,
    w_scale,
    w_global_scale,
    x_static_scale=None,
    quant_static_scale=None,
    bias=None,
    routing_data=None,
    gather_indx=None,
    scatter_indx=None,
    gammas=None,
    out_dtype=torch.bfloat16,
    apply_swiglu=False,
    alpha=1.0,
    limit=1.0,
    swiglu_add_residual=True,
):
    """MoE GEMM with NVFP4 weights using tl.dot_scaled with fp8 scales."""
    assert w.dtype == torch.uint8
    assert w.ndim == 3
    assert w_global_scale.dtype == torch.float32

    num_tokens = x.shape[-2]
    M = num_tokens if gather_indx is None else gather_indx.shape[0]
    E, K_packed, N = w.shape
    K = K_packed * 2

    if w_scale.dtype == torch.uint8:
        w_scale = w_scale.view(torch.float8_e4m3fn)
    block_m = routing_data.block_m
    block_k = 128
    num_stages = 2
    group_m = 4
    xcd_swizzle = get_num_xcds()
    w_cache_modifier = ".cg" if block_m <= 32 else None
    split_k = 1
    block_n = 128
    num_warps = 4

    if apply_swiglu and split_k > 1:
        apply_swiglu_matmul = False
        reduction_n_matmul = 1
    elif apply_swiglu:
        apply_swiglu_matmul = True
        reduction_n_matmul = 2
    else:
        apply_swiglu_matmul = False
        reduction_n_matmul = 1
    reduction_n_reduction = 2 if (apply_swiglu and split_k > 1) else 1

    matmul_shape = (split_k, M, N // reduction_n_matmul)
    y = torch.empty(matmul_shape, device=x.device, dtype=out_dtype)

    stride_bias = None if bias is None else bias.stride(0)
    expt_data = routing_data.expt_data
    expt_hist = None if expt_data is None else expt_data.hist
    expt_hist_sum = None if expt_data is None else expt_data.token_offs_pad[-1]
    expt_token_offs_raw = None if expt_data is None else expt_data.token_offs_raw
    expt_block_pid_map = None if expt_data is None else expt_data.block_pid_map

    grid_m = routing_data.n_blocks(M, block_m)
    grid_n = triton.cdiv(N, block_n)
    grid = grid_m * grid_n * split_k

    _moe_gemm_nvfp4_dot_scaled[(grid,)](
        y,
        y.stride(0),
        y.stride(1),
        y.stride(2),
        x.view(torch.uint8),
        x.stride(0),
        x.stride(1),
        w,
        w.stride(0),
        w.stride(1),
        w.stride(2),
        w_scale,
        w_scale.stride(0),
        w_scale.stride(1),
        w_scale.stride(2),
        w_global_scale,
        x_static_scale,
        quant_static_scale,
        bias,
        stride_bias,
        gammas,
        N,
        K,
        gather_indx,
        expt_hist,
        expt_token_offs_raw,
        expt_hist_sum,
        expt_block_pid_map,
        grid_m,
        grid_n,
        apply_swiglu_matmul,
        alpha,
        limit,
        reduction_n_matmul,
        swiglu_add_residual,
        routing_data.n_expts_act,
        block_m,
        block_n,
        block_k,
        group_m,
        XCD_SWIZZLE=xcd_swizzle,
        EVEN_K=K % block_k == 0,
        MASK_K_LIMIT=K % block_k,
        SPLIT_K=split_k,
        W_CACHE_MODIFIER=w_cache_modifier,
        num_warps=num_warps,
        num_stages=num_stages,
        UPCAST_INDICES=should_upcast_indices(x.view(torch.uint8), w, w_scale, y),
    )

    group_indx = (
        None
        if scatter_indx is None
        else scatter_indx.view(-1, routing_data.n_expts_act)
    )
    if scatter_indx is not None or split_k > 1:
        y_rows = (
            scatter_indx.shape[0] // routing_data.n_expts_act
            if scatter_indx is not None
            else M
        )
        y_final = torch.empty(
            (y_rows, N // reduction_n_matmul // reduction_n_reduction),
            device=x.device,
            dtype=out_dtype,
        )
    else:
        y_final = None

    y_final = reduce_grouped(
        y,
        group_indx,
        y_final,
        apply_swiglu and split_k > 1,
        alpha,
        limit,
        reduction_n_reduction,
        out_dtype=out_dtype,
        swiglu_add_residual=swiglu_add_residual,
    )
    return y_final
