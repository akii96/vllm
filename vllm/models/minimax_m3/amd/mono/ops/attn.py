# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
"""MiniMax-M3 attention MonoKernel: one persistent launch per sparse layer and
TP rank, from the layer's input (the previous layer's reduced output and the
residual) to this rank's unreduced o_proj output, which the FFN launch
(``ffn.py``) all-reduces. FP8 per-channel weights, fp8 KV cache with a fixed
scale, bf16 index cache: vLLM's Triton decode path on CDNA3+.

Stages, each a task table over the grid (a task waits only on earlier ones)::

    norm    token k: x + residual -> residual out, Gemma norm, per-token FP8
    qkv     16 rows of q | k | v | index_q | index_k: FP8 GEMV, every token
    head    (token, head slot) a half wave: Gemma head norm, partial NeoX RoPE;
            q / index_q kept, K | V -> fp8 cache, index_k -> index cache
    score   (request, 8 index blocks): bf16 MFMA of each row's index_q with
            the blocks' keys, block max; a row's newest block forced first
    attend  (token, split s): selects the token's 16 best blocks (ties to the
            lower block: split s ranks a 16th of them and publishes its 16
            best, every split ranks the splits' candidates), then
            the s-th selected block: 16 heads, K / V dequantized to bf16,
            bf16 MFMA, online softmax in base 2 -> partial O and log-sum-exp
    oproj   16 o_proj rows: merge every token's partials, per-token FP8 of the
            attention output, FP8 GEMV -> bf16 partial out
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64

from vllm.models.common.mono.device import (
    CM_DEV,
    ROW16,
    WAVE,
    Mailbox,
    bf16_round,
    bf16_word,
    bf_hi,
    bf_lo,
    butterfly,
    fp8_pack4,
    fp8x4_bf16,
    hw_exp2,
    hw_log2,
    rsqrt_rn,
    load,
    load_f32,
    load_i32,
    memrealtime,
    mfma_bf16,
    mfma_fp8,
    publish,
    store,
    traced,
    uniform,
)
from vllm.models.minimax_m3.amd.mono.ops.head import (
    ELEMS,
    HEAD_DIM,
    ROTARY,
    norm_rope,
    unpack4,
)

THREADS = 512
WAVES = THREADS // 64
HALF_WAVES = THREADS // 32
MAX_TOKENS = 16
H = 6144
NQ = 16  # q heads a rank
SLOTS = NQ + 4  # head slots: q 0..15, k, v, index_q, index_k
K_SLOT, V_SLOT, IQ_SLOT, IK_SLOT = NQ, NQ + 1, NQ + 2, NQ + 3
QKV_ROWS = SLOTS * HEAD_DIM
O_K = NQ * HEAD_DIM
BLK = 128  # sparse block == KV page
TOPK = 16
SPLITS = TOPK  # attend tasks a token: one selected block each
SCORE_BLOCKS = 8  # index blocks a score task: one a wave
O_ROWS = 16
FP8_MAX = 224.0  # vLLM's dynamic range on fnuz (the fp8 index cache)
# qkv_proj / o_proj input quant: AITER's per-token quant (QuantFP8 on ROCm), the
# format's max; the KV cache writer saturates there too
LIN_MAX = 240.0
KV_MAX = 240.0
LOG2E = 1.4426950408889634
KV_ROW = 2 * HEAD_DIM  # fp8 bytes a token: K then V
NEG = -3.0e38
MAX_BLOCKS = 2048  # index blocks a request at most (256K context)
QKV_KC = H // 64 // WAVES  # 16 B weight loads a lane, a row group
O_KC = O_K // 64 // WAVES
KK = HEAD_DIM // 16  # 16-dim chunks of a head (bf16 MFMA k)
STAMPS = 12
KEY_NONE = -(2**31)  # a candidate pad's key: below every block's


def _layout(items):
    out, off = {}, 0
    for name, nbytes in items:
        out[name] = off
        off += (nbytes + 255) // 256 * 256
    return out, off


_P = 8  # a (value, tag) pair
SCRATCH, SCRATCH_BYTES = _layout(
    [
        ("xrdy", MAX_TOKENS * _P),
        ("xq", MAX_TOKENS * H),
        ("qkv", MAX_TOKENS * QKV_ROWS * _P),
        ("hrdy", MAX_TOKENS * SLOTS * _P),
        ("q", MAX_TOKENS * O_K * 2),
        ("iq", MAX_TOKENS * HEAD_DIM * 2),
        ("score", MAX_TOKENS * MAX_BLOCKS * _P),
        ("cand", MAX_TOKENS * SPLITS * TOPK * 2 * _P),  # (key, block) a split's best
        ("po", MAX_TOKENS * SPLITS * O_K * _P),
        ("plse", MAX_TOKENS * SPLITS * NQ * _P),
        ("ao", MAX_TOKENS * O_K),
        ("aordy", MAX_TOKENS * _P),
    ]
)


def gemv_k(wave, g4, c, kc_per_wave):
    """A lane's k offset of its c-th 16 B of a row: the 4 lanes of a row read
    128 contiguous bytes; A and B use the same k order."""
    return (wave * (kc_per_wave // 2) + c // 2) * 128 + g4 * 32 + (c % 2) * 16


def build_attn(
    tokens: int, grid: int, key: str, q_len: int, sm_scale: float, eps: float,
    max_blocks: int, timeline: bool = False, fp8_index: bool = False,
):  # fmt: skip
    """``@flyc.jit`` launcher of the attention launch for a step of ``tokens``
    rows, ``q_len`` consecutive rows a request, on ``grid`` CTAs; a request has
    at most ``max_blocks`` index blocks. ``fp8_index``: the index cache holds the
    device's fp8 (unit scale, as vLLM's fp8 indexer), scored with fp8 MFMA."""
    M, G = tokens, grid
    R = M // q_len
    assert 1 <= M <= MAX_TOKENS and M % q_len == 0 and max_blocks <= MAX_BLOCKS
    QKV_TASKS = QKV_ROWS // 16
    HEAD_TASKS = -(-M * SLOTS // HALF_WAVES)
    CHUNKS = -(-max_blocks // SCORE_BLOCKS)
    SCORE_TASKS = CHUNKS * R  # chunk-major: early chunks of every request first
    ATT_TASKS = M * SPLITS
    O_TASKS = H // O_ROWS
    counts = (("norm", M), ("qkv", QKV_TASKS), ("head", HEAD_TASKS),
              ("score", SCORE_TASKS), ("attend", ATT_TASKS),
              ("merge", M))  # fmt: skip
    BASE, off = {}, 0
    for stage, n in counts:
        BASE[stage] = off % G
        off += n
    N = dict(counts) | {"oproj": O_TASKS}
    BASE["oproj"] = (BASE["merge"] + M) % G
    ITERS = {s: -(-N[s] // G) for s in N}
    SCALE_LOG2 = sm_scale * LOG2E
    # a split ranks a SPLITS-th of a token's blocks (pads up to CAP), then the
    # splits' TOPK best each
    CAP = max(2 * TOPK, 1 << (-(-max_blocks // SPLITS) - 1).bit_length())
    CANDS = SPLITS * TOPK
    PAD = MAX_BLOCKS  # pad pairs' block ids: above every block's
    assert CAP <= THREADS and CANDS <= THREADS

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 256, 16]
        keys: fx.Array[fx.Int32, 2 * THREADS, 16]  # (key, block) pairs
        vt: fx.Array[fx.Int32, HEAD_DIM * BLK // 2, 16]  # V^T of a block: bf16 pairs
        pt: fx.Array[fx.Int32, NQ * BLK // 2, 16]  # P of a block: bf16 pairs
        hm: fx.Array[fx.Float32, 4 * NQ, 16]  # running max, running sum, alpha
        sm: fx.Array[fx.Float32, WAVES, 16]
        xsc: fx.Array[fx.Float32, MAX_TOKENS, 16]
        cnt: fx.Array[fx.Int32, WAVES, 16]

    name = f"minimax_m3_mono_attn_s{M}_q{q_len}_g{G}_b{max_blocks}_{'f8' if fp8_index else 'bf'}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def attn_kernel(
        x: Int64, residual: Int64, res_out: Int64, partial: Int64,
        g_in: Int64, w_qkv: Int64, s_qkv: Int64, g_q: Int64, g_k: Int64,
        g_iq: Int64, g_ik: Int64, cos_sin: Int64, w_o: Int64, s_o: Int64,
        kv_cache: Int64, kv_stride: Int64, index_cache: Int64, ix_stride: Int64,
        k_scale: Int64, v_scale: Int64, positions: Int64, slots: Int64,
        block_table: Int64, bt_stride: Int32, seq_lens: Int64, scratch: Int64,
        ctr: Int64, add_residual: Int32, tl: Int64,
    ):  # fmt: skip
        _ = key
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        g4 = lane // 16
        l16 = lane % 16
        l32 = tid % 32
        hw_ = tid // 32
        lds = fx.SharedAllocator().allocate(Smem).peek()
        red, keys, vt, pt = lds.red.ptr, lds.keys.ptr, lds.vt.ptr, lds.pt.ptr
        hm, sm, xsc, cnt = lds.hm.ptr, lds.sm.ptr, lds.xsc.ptr, lds.cnt.ptr
        tag = uniform(load_i32(ctr, bid)) + 1
        mb = Mailbox(tag)
        mb_put, mb_poll = mb.put, mb.poll

        def stamp(i):
            if const_expr(timeline):
                if tid == 0:
                    clk = memrealtime()
                    words = [fx.Int32(clk), fx.Int32(clk >> 32)]
                    store(fx.Vector.from_elements(words, fx.Int32), tl, (bid * STAMPS + i) * 2)

        def sreg(n):
            return scratch + fx.Int64(SCRATCH[n])

        def first(stage):
            return (bid + (G - BASE[stage])) % G

        def kv_len(k):
            """Token k's causal length: its request's length less its later rows."""
            return fx.max(load_i32(seq_lens, k // q_len) - (q_len - 1 - k % q_len), 0)

        @traced
        def cta_reduce(v, op, buf):
            wv = butterfly(v, WAVE, op)
            gpu.barrier()
            if lane == 0:
                fx.ptr_store(wv, buf + wave)
            gpu.barrier()
            t = fx.ptr_load(buf)
            for w in range_constexpr(1, WAVES):
                t = op(t, fx.ptr_load(buf + w))
            gpu.barrier()
            return t

        def add(a, b):
            return a + b

        def clamp(v, m):
            return fx.max(fx.min(v, fx.Float32(m)), fx.Float32(-m))

        def quant_row(vals, buf):
            """Per-token FP8 of a CTA-wide row (this thread's values): -> (words, scale)."""
            amax = fx.Float32(0.0)
            for v in vals:
                amax = fx.max(amax, fmath.absf(v))
            # AITER's per-token quant: scale = amax * f32(1 / 240) (a multiply, not
            # a division), no floor; x * (1 / scale), an IEEE reciprocal; an
            # all-zero row stays 0
            xs = cta_reduce(amax, fx.max, buf) * fx.Float32(1.0 / LIN_MAX)
            inv = (xs > 0.0).select(fx.Float32(1.0) / xs, fx.Float32(0.0))
            q = [clamp(v * inv, LIN_MAX) for v in vals]
            return [fp8_pack4(*q[4 * j : 4 * j + 4]) for j in range(len(vals) // 4)], xs

        # ------------------------------------------------------------ norm
        @traced
        def norm(k):
            w0 = tid * 6
            row = k * (H // 2) + w0
            raw = [[load(a, i, 4), load(a, i + 4, 2)] for a, i in ((x, row), (residual, row), (g_in, w0))]
            xv, rv, gv = [[v[0][j] for j in range(4)] + [v[1][j] for j in range(2)] for v in raw]
            addr = (add_residual != 0).select(fx.Float32(1.0), fx.Float32(0.0))
            s, gm = [], []  # s stays fp32: the norm reads it unrounded
            for i in range_constexpr(6):
                for f in (bf_lo, bf_hi):
                    s.append(f(xv[i]) + f(rv[i]) * addr)
                    gm.append(f(gv[i]) + 1.0)
            hw = [bf16_word(s[2 * i], s[2 * i + 1]) for i in range(6)]
            store(fx.Vector.from_elements(hw[:4], fx.Int32), res_out, row)
            store(fx.Vector.from_elements(hw[4:], fx.Int32), res_out, row + 4)
            ss = fx.Float32(0.0)
            for v in s:
                ss = fmath.fma(v, v, ss)
            rstd = rsqrt_rn(cta_reduce(ss, add, sm) / float(H) + eps)
            qw, xs = quant_row([bf16_round(s[i] * rstd * gm[i]) for i in range(12)], sm)
            dst = k * (H // 4) + tid * 3
            store(fx.Vector.from_elements(qw[:2], fx.Int32), sreg("xq"), dst, CM_DEV)
            store(qw[2], sreg("xq"), dst + 2, CM_DEV)
            publish(mb_put, sreg("xrdy"), k, [xs.bitcast(fx.Int32)], tid == 0)

        @traced
        def wait_scales(rdy):
            """Every token's activation scale, flagged at rdy -> xsc."""
            if wave == 0:
                got = mb_poll([(sreg(rdy), fx.min(lane, M - 1), 1)])
                if lane < M:
                    fx.ptr_store(got[0][0].bitcast(fx.Float32), xsc + lane)
            gpu.barrier()

        # ------------------------------------------------------------ qkv
        def w_load(w, rows, kdim, kc_per_wave, t, has):
            """This lane's FP8 weights of 16-row tile t (a no-op unless has)."""
            nbytes = has.select(fx.Int32(rows * kdim), fx.Int32(0))
            row = t * 16 + l16
            return [
                load(w, (row * kdim + gemv_k(wave, g4, c, kc_per_wave)) // 4, 4, nbytes=nbytes)
                for c in range(kc_per_wave)
            ]

        def qkv_load(t, has):
            return w_load(w_qkv, QKV_ROWS, H, QKV_KC, t, has)

        def gemv_tile(wts, b_addr, b_row, kc_per_wave):
            """A 16-row x 16-column FP8 GEMV tile; B column l16 is row b_row of
            the FP8 activations at b_addr. -> this lane's 4 accumulators."""
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for c in range_constexpr(kc_per_wave):
                b = load(b_addr, (b_row + gemv_k(wave, g4, c, kc_per_wave)) // 4, 4, cm=CM_DEV)
                a = wts[c].bitcast(fx.Int64)
                b = b.bitcast(fx.Int64)
                for h in range_constexpr(2):
                    acc = mfma_fp8(a[h], b[h], acc)
            for i in range_constexpr(4):
                fx.ptr_store(acc[i], red + ((wave * 16 + 4 * g4 + i) * 16 + l16))
            gpu.barrier()

        def tile_sum(r, col):
            tot = fx.Float32(0.0)
            for w in range_constexpr(WAVES):
                tot = tot + fx.ptr_load(red + ((w * 16 + r) * 16 + col))
            return tot

        @traced
        def qkv(t, wts):
            gemv_tile(wts, sreg("xq"), fx.min(l16, M - 1) * H, QKV_KC)
            if tid < 16 * M:
                r = tid % 16
                k = tid // 16
                v = bf16_round(tile_sum(r, k) * fx.ptr_load(xsc + k) * load_f32(s_qkv, t * 16 + r))
                mb_put(sreg("qkv"), k * QKV_ROWS + t * 16 + r, [v.bitcast(fx.Int32)])
            gpu.barrier()

        # ------------------------------------------------------------ head
        @traced
        def head(u):
            """(token, slot) u on this half wave: lane l32 holds dims 4 l32 .. + 4."""
            k = u // SLOTS
            slot = u % SLOTS
            d0 = l32 * ELEMS
            row = k * QKV_ROWS + slot * HEAD_DIM + d0
            got = mb_poll([(sreg("qkv"), row, 2), (sreg("qkv"), row + 2, 2)])
            xv = [got[j][i].bitcast(fx.Float32) for j in range(2) for i in range(2)]
            gptr = (slot < NQ).select(
                g_q, (slot == K_SLOT).select(g_k, (slot == IQ_SLOT).select(g_iq, g_ik))
            )
            gamma = unpack4(load(gptr, d0 // 2, 2))
            pos = load_i32(positions, 2 * k)
            half = ROTARY // 2
            ib = fx.min((d0 < half).select(d0, d0 - half), half - ELEMS)
            cs = unpack4(load(cos_sin, (pos * ROTARY + ib) // 2, 2))
            sn = unpack4(load(cos_sin, (pos * ROTARY + half + ib) // 2, 2))
            is_v = slot == V_SLOT
            y32 = norm_rope(xv, l32, gamma, cs, sn, eps)
            y = [bf16_round(is_v.select(xv[i], y32[i])) for i in range(ELEMS)]
            words = fx.Vector.from_elements([bf16_word(y[0], y[1]), bf16_word(y[2], y[3])], fx.Int32)
            sm_ = load_i32(slots, 2 * k)
            if slot < NQ:
                store(words, sreg("q"), (k * O_K + slot * HEAD_DIM + d0) // 2, CM_DEV)
            if const_expr(fp8_index):
                # vLLM's fp8 index path: unit scale, saturated; index_q goes
                # through bf16 first, index_k converts from fp32
                def sat8(vals):
                    return fp8_pack4(*[clamp(v, FP8_MAX) for v in vals])

                ix8 = (slot == IQ_SLOT).select(sat8(y), sat8(y32))
                if slot == IQ_SLOT:
                    store(ix8, sreg("iq"), (k * HEAD_DIM + d0) // 4, CM_DEV)
            else:
                if slot == IQ_SLOT:
                    store(words, sreg("iq"), (k * HEAD_DIM + d0) // 2, CM_DEV)
            if (sm_ >= 0) & ((slot == K_SLOT) | is_v):
                sc = is_v.select(load_f32(v_scale, 0), load_f32(k_scale, 0))
                inv = fx.Float32(1.0) / sc
                w8 = fp8_pack4(*[clamp(v * inv, KV_MAX) for v in y])
                page = kv_cache + fx.Int64(sm_ // BLK) * kv_stride
                store(w8, page, ((sm_ % BLK) * KV_ROW + is_v.select(HEAD_DIM, 0) + d0) // 4, CM_DEV)
            if (sm_ >= 0) & (slot == IK_SLOT):
                page = index_cache + fx.Int64(sm_ // BLK) * ix_stride
                if const_expr(fp8_index):
                    store(ix8, page, ((sm_ % BLK) * HEAD_DIM + d0) // 4, CM_DEV)
                else:
                    store(words, page, ((sm_ % BLK) * HEAD_DIM + d0) // 2, CM_DEV)

        @traced
        def head_task(t):
            u = t * HALF_WAVES + hw_
            if u < M * SLOTS:
                head(u)
            publish(mb_put, sreg("hrdy"), t, [fx.Int32(1)], tid == 0)

        @traced
        def wait_heads():
            """Every head task's stores (q, index_q, the caches) landed."""
            if wave == 0:
                mb_poll([(sreg("hrdy"), fx.min(lane, HEAD_TASKS - 1), 1)])
            gpu.barrier()

        # ------------------------------------------------------------ score
        @traced
        def score(t):
            """Request r, index blocks c * SCORE_BLOCKS ..: wave w scores blocks
            w, w + 8 against every row of the request (B column l16 = row)."""
            r = t % R
            c = t // R
            rlen = load_i32(seq_lens, r)
            nblk = (rlen + BLK - 1) // BLK
            qrow = fx.min(l16, q_len - 1)
            lim = rlen - (q_len - 1 - qrow)
            # q and the keys share one permuted k order (16 B a lane), so the
            # dot product does not see it
            qrow_ = (r * q_len + qrow) * HEAD_DIM
            if const_expr(fp8_index):
                qw = [load(sreg("iq"), (qrow_ + kk * 64 + g4 * 16) // 4, 4, cm=CM_DEV).bitcast(fx.Int64)
                      for kk in range(HEAD_DIM // 64)]  # fmt: skip
            else:
                qw = [load(sreg("iq"), (qrow_ + kk * 32 + g4 * 8) // 2, 4, cm=CM_DEV)
                      for kk in range(HEAD_DIM // 32)]  # fmt: skip
            row_bytes = HEAD_DIM * (1 if fp8_index else 2)
            for i in range_constexpr(SCORE_BLOCKS // WAVES):
                b = c * SCORE_BLOCKS + wave + i * WAVES
                if b < nblk:
                    page = index_cache + fx.Int64(load_i32(block_table, r * bt_stride + b)) * ix_stride
                    # the block's keys in flight at once, 16 B a load
                    kws = [
                        [load(page, ((sub * 16 + l16) * row_bytes + kk * 64 + g4 * 16) // 4, 4, cm=CM_DEV)
                         for kk in range(row_bytes // 64)]
                        for sub in range(BLK // 16)
                    ]  # fmt: skip
                    mx = fx.Float32(NEG)
                    for sub in range_constexpr(BLK // 16):
                        acc = fx.Vector.filled(4, 0.0, fx.Float32)
                        for kk in range_constexpr(len(qw)):
                            kv4 = kws[sub][kk]
                            if const_expr(fp8_index):
                                k8 = kv4.bitcast(fx.Int64)
                                for h in range_constexpr(2):
                                    acc = mfma_fp8(k8[h], qw[kk][h], acc)
                            else:
                                for h in range_constexpr(2):
                                    acc = mfma_bf16([kv4[2 * h], kv4[2 * h + 1]], [qw[kk][2 * h], qw[kk][2 * h + 1]], acc)
                        for e in range_constexpr(4):
                            kpos = b * BLK + sub * 16 + 4 * g4 + e
                            mx = fx.max(mx, (kpos < lim).select(acc[e], fx.Float32(NEG)))
                    mx = butterfly(mx, (16, 32), fx.max)
                    nb_q = (rlen - (q_len - 1 - fx.min(lane, q_len - 1)) + BLK - 1) // BLK
                    sc_out = (b < nb_q).select((b == nb_q - 1).select(fx.Float32(1e29), mx), fx.Float32(-1e30))
                    if lane < q_len:
                        mb_put(sreg("score"), (r * q_len + lane) * MAX_BLOCKS + b, [sc_out.bitcast(fx.Int32)])

        def max_len_blocks():
            """The longest request's index blocks (uniform)."""
            mx = load_i32(seq_lens, 0)
            for r in range_constexpr(1, R):
                mx = fx.max(mx, load_i32(seq_lens, r))
            return (mx + BLK - 1) // BLK

        @traced
        def score_loop(t0, n_live):
            for t in range(t0, n_live, G):
                score(t)

        # ------------------------------------------------------------ select
        def beats(ah, al, bh, bl):
            """(key, block) a orders before b: the higher key, ties to the lower block."""
            return (ah > bh) | ((ah == bh) & (al < bl))

        def rank_of(n):
            """Key tid // P of the n keys in keys (in block order) and its rank, the
            count of keys that beat it (ties to the lower index); P = THREADS // n
            threads share a key. -> (index, rank, reports: this thread speaks)."""
            P = THREADS // n
            c = tid // P
            part = tid % P
            kc = fx.ptr_load(keys + c)
            n_up = fx.Int32(0)
            for j0 in range_constexpr(0, n, P):
                j = part + j0
                up = beats(fx.ptr_load(keys + j), j, kc, c)
                n_up = n_up + up.select(fx.Int32(1), fx.Int32(0))
            n_up = butterfly(n_up, tuple(P >> i for i in range(1, P.bit_length())))
            return c, n_up, part == 0

        @traced
        def nth_cand(s):
            """The s-th best of the CANDS candidates in keys, SPLITS lists of TOPK
            sorted best first, by a SPLITS-way merge (every wave, each row of 16
            lanes, lane j heading list j): round r takes the best head and
            advances its list. -> its candidate index, wave-uniform."""
            j = lane % SPLITS

            def pair(h):
                x = 2 * (j * TOPK + fx.min(h, TOPK - 1))
                live = h < TOPK
                return (live.select(fx.ptr_load(keys + x), fx.Int32(KEY_NONE)),
                        live.select(fx.ptr_load(keys + (x + 1)), fx.Int32(2 * PAD)))  # fmt: skip

            h = fx.Int32(0)
            kh, bh = pair(h)
            idx = fx.Int32(-1)
            for r in range(TOPK):
                kn, bn = pair(h + 1)  # the next head, off the round's chain
                mk = butterfly(kh, ROW16, fx.max)
                mb = butterfly((kh == mk).select(bh, fx.Int32(2 * PAD)), ROW16, fx.min)
                win = (kh == mk) & (bh == mb)
                idx = (win & (r == s)).select(j * TOPK + h, idx)
                h = h + win.select(fx.Int32(1), fx.Int32(0))
                kh = win.select(kn, kh)
                bh = win.select(bn, bh)
            return uniform(butterfly(idx, ROW16, fx.max))

        @traced
        def select(k, s):
            """Token k's s-th best block (-1: none) and its page, vLLM's top-TOPK (the higher
            score, ties to the lower block): split s ranks its share, a SPLITS-th
            of the token's blocks, and publishes its TOPK best; the context's TOPK
            best are among the splits' candidates, which every split ranks."""
            nb = (kv_len(k) + BLK - 1) // BLK
            span = (nb + SPLITS - 1) // SPLITS
            b = s * span + tid
            live = (tid < span) & (b < nb)
            key = fx.Int32(KEY_NONE)
            if (nb > 0) & (tid < CAP):  # a CUDA-graph pad row (seq_len 0) has no scores
                bits = mb_poll([(sreg("score"), k * MAX_BLOCKS + fx.min(b, nb - 1), 1)])[0][0]
                key = (bits < 0).select(bits ^ fx.Int32(0x7FFFFFFF), bits)
            stamp(8)
            if tid < CAP:
                fx.ptr_store(live.select(key, fx.Int32(KEY_NONE)), keys + tid)
            gpu.barrier()
            c, rk, held = rank_of(CAP)
            if held & (rk < TOPK):
                # pads: below every block, distinct ids above every block's
                lc = (s * span + c < nb) & (c < span)
                bl = lc.select(s * span + c, fx.Int32(PAD) + s * CAP + c)
                mb_put(sreg("cand"), 2 * ((k * SPLITS + s) * TOPK + rk), [fx.ptr_load(keys + c), bl])
            gpu.barrier()
            stamp(9)
            if tid < CANDS:
                got = mb_poll([(sreg("cand"), 2 * (k * CANDS + tid), 2)])[0]
                fx.ptr_store(got[0], keys + 2 * tid)
                fx.ptr_store(got[1], keys + (2 * tid + 1))
                # its page, in flight during the merge
                pg = load_i32(block_table, (k // q_len) * bt_stride + fx.max(fx.min(got[1], nb - 1), 0))
                fx.ptr_store(pg, keys + (2 * CANDS + tid))
            gpu.barrier()
            stamp(10)
            c = nth_cand(s)
            b = fx.ptr_load(keys + (2 * c + 1))
            return (b < PAD).select(b, fx.Int32(-1)), fx.ptr_load(keys + (2 * CANDS + c))

        # ------------------------------------------------------------ attend
        def q_frags(k):
            """This lane's MFMA A operand of token k's q (rows = heads)."""
            return [load(sreg("q"), (k * O_K + l16 * HEAD_DIM + kk * 16 + g4 * 4) // 2, 2, cm=CM_DEV)
                    for kk in range(KK)]  # fmt: skip

        @traced
        def attend(t, b, pg, qa):
            """Token k, split s: its selected block b on page pg (``select``) for
            all 16 heads; qa: ``q_frags``.
            QK: wave w takes keys 16 w .. + 16 (MFMA A = q: rows = heads).
            P and V^T go through LDS; PV: wave w owns dims 16 w .. + 16.
            K and V are dequantized like vLLM's: fp8 -> bf16 (x scale -> bf16)."""
            k = t // SPLITS
            s = t % SPLITS
            klen = kv_len(k)
            ksc = load_f32(k_scale, 0)
            vsc = load_f32(v_scale, 0)
            if tid < NQ:
                fx.ptr_store(fx.Float32(NEG), hm + tid)
                fx.ptr_store(fx.Float32(0.0), hm + (NQ + tid))
            o = fx.Vector.filled(4, 0.0, fx.Float32)
            kr = wave * 16 + l16
            live = b >= 0
            page = kv_cache + fx.Int64(pg) * kv_stride
            # V^T -> LDS as bf16: vt[d][key]. Thread: one key's 16 dims, one
            # 16 B load; every load of the block in flight together
            v16 = fx.recast_iter(fx.Int16, vt)
            vkey = tid // (HEAD_DIM // 16)
            vd0 = (tid % (HEAD_DIM // 16)) * 16
            vraw = [load(page, ((vkey + 64 * p) * KV_ROW + HEAD_DIM + vd0) // 4, 4, cm=CM_DEV)
                    for p in range(BLK // 64)]  # fmt: skip
            # QK^T: A = q (16 heads x 16 dims), B = K (16 keys x 16 dims)
            kraw = [load_i32(page, (kr * KV_ROW + kk * 16 + g4 * 4) // 4, cm=CM_DEV) for kk in range(KK)]
            for p in range_constexpr(BLK // 64):
                for wd in range_constexpr(4):
                    vw = fp8x4_bf16(vraw[p][wd])
                    for e in range_constexpr(4):
                        v = bf16_round((bf_lo if e % 2 == 0 else bf_hi)(vw[e // 2]) * vsc)
                        d = vd0 + wd * 4 + e
                        bits = fx.Int32(fx.Uint32(v.bitcast(fx.Int32)) >> 16)
                        fx.ptr_store(fx.Int16(bits), v16 + (d * BLK + vkey + 64 * p))
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for kk in range_constexpr(KK):
                kw = fp8x4_bf16(kraw[kk])
                kw = [bf16_word(bf16_round(bf_lo(w) * ksc), bf16_round(bf_hi(w) * ksc)) for w in kw]
                acc = mfma_bf16([qa[kk][0], qa[kk][1]], kw, acc)
            ok = live & (fx.max(b, 0) * BLK + kr < klen)
            # acc[e]: head 4 g4 + e, key kr
            for e in range_constexpr(4):
                fx.ptr_store(ok.select(acc[e] * SCALE_LOG2, fx.Float32(NEG)), red + ((4 * g4 + e) * BLK + kr))
            gpu.barrier()
            # softmax: half wave hh owns head hh's 128 keys, 4 a lane
            hh = hw_
            v4 = [fx.ptr_load(red + (hh * BLK + l32 * 4 + i)) for i in range(4)]
            bm = butterfly(fx.max(fx.max(v4[0], v4[1]), fx.max(v4[2], v4[3])), (16, 8, 4, 2, 1), fx.max)
            m_old = fx.ptr_load(hm + hh)
            m_new = fx.max(m_old, bm)
            m_safe = (m_new > NEG).select(m_new, fx.Float32(0.0))
            p4 = [(v4[i] > NEG).select(hw_exp2(v4[i] - m_safe), fx.Float32(0.0)) for i in range(4)]
            p16 = [bf16_round(p) for p in p4]
            fx.ptr_store(bf16_word(p16[0], p16[1]), pt + (hh * (BLK // 2) + l32 * 2))
            fx.ptr_store(bf16_word(p16[2], p16[3]), pt + (hh * (BLK // 2) + l32 * 2 + 1))
            psum = butterfly(p4[0] + p4[1] + p4[2] + p4[3], (16, 8, 4, 2, 1))
            gpu.barrier()
            if l32 == 0:
                alpha = (m_old > NEG).select(hw_exp2(m_old - m_safe), fx.Float32(0.0))
                fx.ptr_store(m_new, hm + hh)
                fx.ptr_store(fx.ptr_load(hm + (NQ + hh)) * alpha + psum, hm + (NQ + hh))
                fx.ptr_store(alpha, hm + (2 * NQ + hh))
            gpu.barrier()
            # PV: A = P (16 heads x 16 keys), B = V^T (16 keys x 16 dims), wave w dims 16 w ..
            alpha = [fx.ptr_load(hm + (2 * NQ + 4 * g4 + e)) for e in range(4)]
            o = fx.Vector.from_elements([o[e] * alpha[e] for e in range(4)], fx.Float32)
            for kc in range_constexpr(BLK // 16):
                pa = [fx.ptr_load(pt + (l16 * (BLK // 2) + kc * 8 + g4 * 2 + i)) for i in range(2)]
                vb = [fx.ptr_load(vt + ((wave * 16 + l16) * (BLK // 2) + kc * 8 + g4 * 2 + i)) for i in range(2)]
                o = mfma_bf16(pa, vb, o)
            gpu.barrier()
            # o[e]: head 4 g4 + e, dim 16 wave + l16, unnormalized; lse in log2
            for e in range_constexpr(4):
                hh = 4 * g4 + e
                lsum = fx.ptr_load(hm + (NQ + hh))
                on = (lsum > 0.0).select(o[e] / lsum, fx.Float32(0.0))
                mb_put(sreg("po"), (k * SPLITS + s) * O_K + hh * HEAD_DIM + wave * 16 + l16, [on.bitcast(fx.Int32)])
            if tid < NQ:
                lsum = fx.ptr_load(hm + (NQ + tid))
                lse = (lsum > 0.0).select(fx.ptr_load(hm + tid) + hw_log2(lsum), fx.Float32(NEG))
                mb_put(sreg("plse"), (k * SPLITS + s) * NQ + tid, [lse.bitcast(fx.Int32)])

        # ------------------------------------------------------------ oproj
        def o_load(t, has):
            return w_load(w_o, H, O_K, O_KC, t, has)

        @traced
        def merge(k):
            """Token k's attention output: the splits' partials weighted by their
            lse (base 2), bf16, per-token FP8 -> ao (scratch), its scale flagged.
            Thread: 4 dims of one head."""
            hh = tid // 32
            d0 = (tid % 32) * 4
            got = mb_poll([(sreg("plse"), (k * SPLITS + s) * NQ + hh, 1) for s in range(SPLITS)])
            lse = [got[s][0].bitcast(fx.Float32) for s in range(SPLITS)]
            mx = lse[0]
            for s in range_constexpr(1, SPLITS):
                mx = fx.max(mx, lse[s])
            wts = [(lse[s] > NEG).select(hw_exp2(lse[s] - mx), fx.Float32(0.0)) for s in range(SPLITS)]
            po = mb_poll([(sreg("po"), (k * SPLITS + s) * O_K + hh * HEAD_DIM + d0 + 2 * h, 2)
                          for s in range(SPLITS) for h in range(2)])  # fmt: skip
            tot = wts[0]
            for s in range_constexpr(1, SPLITS):
                tot = tot + wts[s]
            vals = []
            for i in range_constexpr(4):
                acc = fx.Float32(0.0)
                for s in range_constexpr(SPLITS):
                    acc = acc + po[s * 2 + i // 2][i % 2].bitcast(fx.Float32) * wts[s]
                vals.append(bf16_round((tot > 0.0).select(acc / tot, fx.Float32(0.0))))
            qw, xs = quant_row(vals, sm)
            store(qw[0], sreg("ao"), k * (O_K // 4) + tid, CM_DEV)
            publish(mb_put, sreg("aordy"), k, [xs.bitcast(fx.Int32)], tid == 0)

        @traced
        def oproj(t, wts):
            wait_scales("aordy")
            gemv_tile(wts, sreg("ao"), fx.min(l16, M - 1) * O_K, O_KC)
            outs = fx.recast_iter(fx.Float32, keys)  # free once select is done
            if tid < O_ROWS * M:
                rr = tid % O_ROWS
                k = tid // O_ROWS
                v = tile_sum(rr, k) * fx.ptr_load(xsc + k) * load_f32(s_o, t * O_ROWS + rr)
                fx.ptr_store(v, outs + (k * O_ROWS + rr))
            gpu.barrier()
            if tid < O_ROWS * M // 2:
                k = tid // (O_ROWS // 2)
                pr = tid % (O_ROWS // 2)
                w = bf16_word(fx.ptr_load(outs + (k * O_ROWS + 2 * pr)), fx.ptr_load(outs + (k * O_ROWS + 2 * pr + 1)))
                store(w, partial, (k * H + t * O_ROWS) // 2 + pr)
            gpu.barrier()

        # ------------------------------------------------------------ schedule
        stamp(0)
        o_first = first("oproj")
        o_w = o_load(fx.min(o_first, O_TASKS - 1), o_first < O_TASKS)
        q0 = first("qkv")
        q_w = qkv_load(fx.min(q0, QKV_TASKS - 1), q0 < QKV_TASKS)
        if first("norm") < M:
            norm(first("norm"))
        stamp(1)
        if q0 < QKV_TASKS:
            wait_scales("xrdy")
        for it in range_constexpr(ITERS["qkv"]):
            t = q0 + it * G
            if const_expr(it + 1 < ITERS["qkv"]):
                nxt = qkv_load(fx.min(t + G, QKV_TASKS - 1), t + G < QKV_TASKS)
            if t < QKV_TASKS:
                qkv(t, q_w)
            if const_expr(it + 1 < ITERS["qkv"]):
                q_w = nxt
        if first("head") < HEAD_TASKS:
            head_task(first("head"))
        sc0 = first("score")
        if sc0 < SCORE_TASKS:
            wait_heads()
            stamp(3)
        # only chunks below the longest request's length hold blocks: a CTA's
        # task stride stays G, the loop stops at the live task count
        live_chunks = (max_len_blocks() + SCORE_BLOCKS - 1) // SCORE_BLOCKS
        score_loop(sc0, fx.min(live_chunks * R, fx.Int32(SCORE_TASKS)))
        if sc0 < SCORE_TASKS:
            stamp(2)
        at0 = first("attend")
        if at0 < ATT_TASKS:
            wait_heads()
            qa = q_frags(at0 // SPLITS)
            sel, pg = select(at0 // SPLITS, at0 % SPLITS)
            stamp(5)
            attend(at0, sel, pg, qa)
            stamp(6)
        if first("merge") < M:
            merge(first("merge"))
            stamp(7)
        for it in range_constexpr(ITERS["oproj"]):
            t = o_first + it * G
            if const_expr(it + 1 < ITERS["oproj"]):
                nxt_o = o_load(fx.min(t + G, O_TASKS - 1), t + G < O_TASKS)
            if t < O_TASKS:
                oproj(t, o_w)
            if const_expr(it + 1 < ITERS["oproj"]):
                o_w = nxt_o
        stamp(4)
        gpu.barrier()
        if tid == 0:
            store(tag, ctr, bid)

    @flyc.jit
    def launch(
        x: Int64, residual: Int64, res_out: Int64, partial: Int64,
        g_in: Int64, w_qkv: Int64, s_qkv: Int64, g_q: Int64, g_k: Int64,
        g_iq: Int64, g_ik: Int64, cos_sin: Int64, w_o: Int64, s_o: Int64,
        kv_cache: Int64, kv_stride: Int64, index_cache: Int64, ix_stride: Int64,
        k_scale: Int64, v_scale: Int64, positions: Int64, slots: Int64,
        block_table: Int64, bt_stride: Int32, seq_lens: Int64, scratch: Int64,
        ctr: Int64, add_residual: Int32, tl: Int64,
        stream: fx.Stream = fx.Stream(None),
    ):  # fmt: skip
        _ = key
        attn_kernel(
            x, residual, res_out, partial, g_in, w_qkv, s_qkv, g_q, g_k, g_iq,
            g_ik, cos_sin, w_o, s_o, kv_cache, kv_stride, index_cache, ix_stride,
            k_scale, v_scale, positions, slots, block_table, bt_stride, seq_lens,
            scratch, ctr, add_residual, tl,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch
