# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
"""MiniMax-M3 FFN MonoKernel for ROCm gfx942 (FlyDSL), FP8 per-channel weights.

One persistent launch per sparse MoE layer and TP rank, for 1..MAX_TOKENS rows:
from the unreduced o_proj partial to the all-reduced FFN output (RFC #60904).
The wide path follows the gfx950 wide-MoE kernel of ROCm/aiter#6047.

Stages are task tables over the grid (task t of a stage runs on CTA
``(base + t) % G``); a task waits only on earlier stages, all CTAs resident::

    push     o_proj partial -> every rank's all-reduce region (system scope)
    norm     token k: rank-ordered sum + residual -> h_mid, Gemma norm -> xn,
             per-token FP8 -> xq (routed) and xq_sh (shared expert)
    router   expert e: fp32 gate . xn for every token -> (sigmoid + bias, sigmoid)
    ug       8 gate + 8 up rows: FP8 GEMV -> SwiGLU-OAI -> bf16 mids
    quant    (token, unit): per-row FP8 of its 768 mids
    down     32 rows: FP8 GEMV, route weights, x routed_scale, + shared,
             rank-ordered all-reduce -> out

Narrow (M < WIDE_FROM): a ug task is one routed (token, slot) or the shared
expert for every token, B read from xq staged in LDS; a down task is one
(token, 32 rows) over its 5 units. Wide (M >= WIDE_FROM): a ug task is one
distinct routed expert (or the shared one) for every token at once, B (column =
token) held in VGPRs; a down task is 32 rows over every distinct expert, with
all mids staged in LDS.

Hand-offs are (value, tag) pairs; the tag is a per-CTA launch counter, equal on
every CTA and rank since every launch bumps it once.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from vllm.models.common.mono.device import (
    CM_DEV,
    CM_SYS,
    WAVE,
    Mailbox,
    ballot,
    bf16_round,
    bf16_trunc,
    bf16_word,
    bf16_word_trunc,
    bf_hi,
    bf_lo,
    butterfly,
    fp8_pack4,
    hw_exp2,
    hw_rcp,
    load,
    load_f32,
    load_i32,
    load_ptr64,
    memrealtime,
    mfma_fp8,
    popcount,
    publish,
    rsqrt_rn,
    store,
    traced,
    uniform,
    wave_max,
    wave_sum,
)

THREADS = 512
WAVES = THREADS // 64
MAX_TOKENS = 8
H = 6144
# expert intermediate size per rank (TP4)
I = 768  # noqa: E741
E = 128
TOP_K = 4
UNITS = TOP_K + 1  # routed slots, then the shared expert
FP8_MAX = 224.0  # vLLM's dynamic FP8 range on fnuz (routed experts' quant)
SH_FP8_MAX = 240.0  # the shared expert's linears quantize to the format's max
LOG2E = 1.4426950408889634
UG_ROWS = 8  # gate rows a task, and as many up rows: one 16-row MFMA tile
UG_TASKS_PER_UNIT = I // UG_ROWS  # 96
UG_KC = H // 64 // WAVES  # 64-k chunks a wave: 12
DN_ROWS = 32
DN_GROUPS = H // DN_ROWS  # 192
DN_KW = WAVES // 2  # waves splitting k; the two halves take a 16-row group each
DN_KL = I // 64 // DN_KW  # 16 B loads a lane, a unit: 3 (2 MFMA k-chunks each)
ROUTER_BASE = MAX_TOKENS
WIDE_FROM = 5  # from this many rows on, the wide path
W13_BYTES = 2 * I * H
W2_BYTES = H * I

# (value, tag) regions are laid out for MAX_TOKENS, so a tag word never sits
# where another width keeps data
_PAIR = 8
SCORE_COPIES = 8  # router score copies; CTA b polls copy b % 8 to spread polling


def _layout(items):
    out, off = {}, 0
    for name, nbytes in items:
        out[name] = off
        off += (nbytes + 255) // 256 * 256
    return out, off


SCRATCH, SCRATCH_BYTES = _layout(
    [
        ("xrdy", MAX_TOKENS * 2 * _PAIR),  # token k: pairs 2k, 2k + 1 = (xs, shared xs)
        ("scores", SCORE_COPIES * MAX_TOKENS * E * 2 * _PAIR),
        ("mids", MAX_TOKENS * UNITS * I * _PAIR),
        ("mq", MAX_TOKENS * UNITS * I),
        ("mqs", MAX_TOKENS * UNITS * _PAIR),
        ("xn", MAX_TOKENS * H * 2),
        ("xq", MAX_TOKENS * H),
        ("xq_sh", MAX_TOKENS * H),  # the shared expert's input quant
    ]
)
_REGION = 4 * MAX_TOKENS * (H // 2) * _PAIR  # [src rank][token][bf16 pair word]
PEER, PEER_BYTES = _layout([("ar", _REGION), ("ffn", _REGION)])
STAMPS = 8  # timeline build: device-clock stamps a CTA
_STREAM = fx.Stream(None)


def build_ffn(tokens: int, grid: int, tp: int, key: str, timeline: bool = False):
    """Return the ``@flyc.jit`` launcher for ``tokens`` rows on ``grid`` CTAs.

    With ``timeline``, CTA b writes clock stamps to ``tl[b * STAMPS + i]``.
    """
    M, G, W = tokens, grid, tp
    assert 1 <= M <= MAX_TOKENS and W == 4 and G >= ROUTER_BASE + E
    WIDE = M >= WIDE_FROM
    U_MAX = min(E, M * TOP_K)
    UG_TASKS = UG_TASKS_PER_UNIT * (1 + U_MAX)  # at most, when wide
    UG_BASE = ROUTER_BASE + E
    Q_TASKS = M * UNITS
    Q_BASE = (UG_BASE + UG_TASKS) % G
    assert Q_TASKS <= G
    DN_TASKS = M * DN_GROUPS
    DN_BASE = (Q_BASE + Q_TASKS) % G
    UG_ITERS = -(-UG_TASKS // G)
    if WIDE:  # a down task is DN_ROWS rows over every distinct expert
        DN_TASKS = DN_GROUPS
        assert DN_TASKS + Q_TASKS <= G
    DN_ITERS = -(-DN_TASKS // G)
    MID_ROWS = M * UNITS
    MID_ROW = I // 4 + 4  # words, padded 16 B
    assert MID_ROWS * DN_ROWS <= 2 * WAVES * 256 or not WIDE
    DN_LD = I // 64  # 16 B loads a lane, an expert's 16 rows: 12
    DN_EXP = -(-(U_MAX + 1) // DN_KW)  # experts a wave runs, at most
    PUSH_THREADS = M * H // 4  # 4 bf16 a thread

    # Narrow: xq rows in LDS are padded 16 B so the 16 B-a-lane B reads of 16
    # columns spread over banks. Wide: B stays in VGPRs, xq gets no LDS.
    XQ_ROW = H // 4 + 4  # words

    @fx.struct
    class Smem:
        xq: fx.Array[fx.Int32, 4 if WIDE else XQ_ROW * M, 16]
        mid: fx.Array[fx.Int32, MID_ROWS * MID_ROW if WIDE else 4, 16]
        uexp: fx.Array[fx.Int32, U_MAX + 2, 16]
        umask: fx.Array[fx.Int32, U_MAX + 1, 16]
        red: fx.Array[fx.Float32, 2 * WAVES * 256, 16]
        sig: fx.Array[fx.Float32, M * E, 16]
        rexp: fx.Array[fx.Int32, M * TOP_K, 16]
        rwt: fx.Array[fx.Float32, M * TOP_K, 16]
        msc: fx.Array[fx.Float32, max(8, MID_ROWS), 16]
        part: fx.Array[fx.Float32, DN_ROWS * (M if WIDE else 1), 16]
        sm: fx.Array[fx.Float32, 2 * WAVES, 16]
        xsc: fx.Array[fx.Float32, 2 * MAX_TOKENS, 16]  # x scales: routed, then shared

    @flyc.kernel(
        name=f"minimax_m3_mono_ffn_s{M}_g{G}", known_block_size=[THREADS, 1, 1]
    )
    def ffn_kernel(
        partial: Int64,
        residual: Int64,
        h_mid: Int64,
        out: Int64,
        g_post: Int64,
        gate: Int64,
        bias: Int64,
        w13: Int64,
        s13: Int64,
        w2: Int64,
        s2: Int64,
        sgu: Int64,
        sgu_s: Int64,
        sdn: Int64,
        sdn_s: Int64,
        scratch: Int64,
        peer_local: Int64,
        peers: Int64,
        ctr: Int64,
        rank: Int32,
        tl: Int64,
    ):
        _ = key

        def stamp(i):
            if const_expr(timeline):  # noqa: SIM102 (a build-time and a run-time test)
                if tid == 0:
                    clk = memrealtime()
                    words = [fx.Int32(clk), fx.Int32(clk >> 32)]
                    store(
                        fx.Vector.from_elements(words, fx.Int32),
                        tl,
                        (bid * STAMPS + i) * 2,
                    )

        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        g4 = lane // 16
        l16 = lane % 16
        lds = fx.SharedAllocator().allocate(Smem).peek()
        red, xql = lds.red.ptr, lds.xq.ptr
        sig, rexp, rwt = lds.sig.ptr, lds.rexp.ptr, lds.rwt.ptr
        msc, part, sm, xsc = lds.msc.ptr, lds.part.ptr, lds.sm.ptr, lds.xsc.ptr
        uexp, umask, midl = lds.uexp.ptr, lds.umask.ptr, lds.mid.ptr

        tag = uniform(load_i32(ctr, bid)) + 1
        mb = Mailbox(tag)
        mb_put, mb_poll = mb.put, mb.poll
        peer_base = [load_ptr64(peers, p) for p in range(W)]

        def sreg(name):
            return scratch + fx.Int64(SCRATCH[name])

        def own(name):
            return peer_local + fx.Int64(PEER[name])

        def first(base):
            return (bid + (G - base % G)) % G

        def sigmoid(x):
            return hw_rcp(1.0 + hw_exp2(-x * LOG2E))

        def quant(v, scale, fmax=FP8_MAX):
            """Clamp ``v / scale``; a true division, as vLLM's quant kernels."""
            return fx.max(fx.min(v / scale, fx.Float32(fmax)), fx.Float32(-fmax))

        def row_scale(amax):
            """Routed experts' per-row FP8 scale, as vLLM's scaled_fp8_quant:
            amax / 224, correctly rounded, floored."""
            return fx.max(amax / FP8_MAX, fx.Float32(1.0 / (FP8_MAX * 512.0)))

        def sh_scale(amax):
            """Shared expert's input quant, as AITER's per-token quant via QuantFP8:
            scale amax * f32(1 / 240), no floor, values x * (1 / scale).
            Returns (scale, its reciprocal or 0 for an all-zero row)."""
            xs = amax * fx.Float32(1.0 / SH_FP8_MAX)
            return xs, (xs > 0.0).select(fx.Float32(1.0) / xs, fx.Float32(0.0))

        def words6(a, b):
            return [a[i] for i in range(4)] + [b[i] for i in range(2)]

        @traced
        def block_reduce(v, op):
            """Reduce ``v`` with ``op`` over the CTA: per wave, then the 8 waves."""
            wv = butterfly(v, WAVE, op)
            gpu.barrier()
            if lane == 0:
                fx.ptr_store(wv, sm + wave)
            gpu.barrier()
            t = fx.ptr_load(sm)
            for w in range_constexpr(1, WAVES):
                t = op(t, fx.ptr_load(sm + w))
            return t

        @traced
        def wait_x():
            """Wait for every token's x scales -> xsc (wave 0, a token a lane).
            xn / xq are then read with CM_DEV, past stale L2 lines."""
            if wave == 0:
                got = mb_poll([(sreg("xrdy"), 2 * fx.min(lane, M - 1), 2)])
                if lane < M:
                    fx.ptr_store(got[0][0].bitcast(fx.Float32), xsc + lane)
                    fx.ptr_store(
                        got[0][1].bitcast(fx.Float32), xsc + (MAX_TOKENS + lane)
                    )
            gpu.barrier()

        @traced
        def stage_xq():
            """Narrow path: stage every token's xq in LDS once, so the B reads
            never queue behind the weight stream."""
            for k in range_constexpr(M):
                for p in range_constexpr(-(-(H // 16) // THREADS)):
                    w = tid + p * THREADS
                    if w < H // 16:
                        v = load(sreg("xq"), (k * H) // 4 + w * 4, 4, cm=CM_DEV)
                        for e in range_constexpr(4):
                            fx.ptr_store(v[e], xql + (k * XQ_ROW + w * 4 + e))
            gpu.barrier()

        # ------------------------------------------------------------ push
        @traced
        def push(g):
            k = g // (H // 4)
            j = g % (H // 4)
            wv = load(partial, k * (H // 2) + j * 2, width=2)
            for p in range_constexpr(W):
                mb_put(
                    peer_base[p] + fx.Int64(PEER["ar"]),
                    (rank * MAX_TOKENS + k) * (H // 2) + j * 2,
                    [wv[0], wv[1]],
                    CM_SYS,
                )

        # ------------------------------------------------------------ norm
        @traced
        def norm(k):
            w0 = tid * 6  # 12 elements a thread
            # residual and gamma loads are in flight while the peers' parts land
            res_w = words6(
                load(residual, k * (H // 2) + w0, 4),
                load(residual, k * (H // 2) + w0 + 4, 2),
            )
            g_w = words6(load(g_post, w0, 4), load(g_post, w0 + 4, 2))
            got = mb_poll(
                [
                    (own("ar"), (src * MAX_TOKENS + k) * (H // 2) + w0 + 2 * h, 2)
                    for src in range(W)
                    for h in range(3)
                ],
                CM_SYS,
            )
            stamp(6)
            # As vLLM's AITER fused all-reduce + Gemma norm on gfx94x: rank-ordered
            # fp32 sum, truncated to bf16, + residual in fp32; the new residual is
            # that truncated, the norm reads it unrounded and its output truncates.
            s, gm = [], []
            for i in range_constexpr(6):
                for f in (bf_lo, bf_hi):
                    a = fx.Float32(0.0)
                    for src in range_constexpr(W):
                        a = a + f(got[src * 3 + i // 2][i % 2])
                    s.append(bf16_trunc(a) + f(res_w[i]))
                    gm.append(f(g_w[i]) + 1.0)
            hw = [bf16_word_trunc(s[2 * i], s[2 * i + 1]) for i in range(6)]
            store(fx.Vector.from_elements(hw[:4], fx.Int32), h_mid, k * (H // 2) + w0)
            store(
                fx.Vector.from_elements(hw[4:], fx.Int32), h_mid, k * (H // 2) + w0 + 4
            )
            ss = fx.Float32(0.0)
            for v in s:
                ss = fmath.fma(v, v, ss)
            rstd = rsqrt_rn(block_reduce(ss, lambda a, b: a + b) / float(H) + 1e-6)
            xn = [bf16_trunc(s[i] * gm[i] * rstd) for i in range(12)]
            xw = [bf16_word(xn[2 * i], xn[2 * i + 1]) for i in range(6)]
            store(
                fx.Vector.from_elements(xw[:4], fx.Int32),
                sreg("xn"),
                k * (H // 2) + w0,
                CM_DEV,
            )
            store(
                fx.Vector.from_elements(xw[4:], fx.Int32),
                sreg("xn"),
                k * (H // 2) + w0 + 4,
                CM_DEV,
            )
            amax = fx.Float32(0.0)
            for v in xn:
                amax = fx.max(amax, fmath.absf(v))
            amax = block_reduce(amax, fx.max)
            xs = row_scale(amax)
            xs_sh, inv_sh = sh_scale(amax)
            for region, q in (
                ("xq", [quant(v, xs) for v in xn]),
                ("xq_sh", [quant(v * inv_sh, 1.0, SH_FP8_MAX) for v in xn]),
            ):
                qw = [fp8_pack4(*q[4 * i : 4 * i + 4]) for i in range(3)]
                store(
                    fx.Vector.from_elements(qw[:2], fx.Int32),
                    sreg(region),
                    k * (H // 4) + tid * 3,
                    CM_DEV,
                )
                store(qw[2], sreg(region), k * (H // 4) + tid * 3 + 2, CM_DEV)
            publish(
                mb_put,
                sreg("xrdy"),
                2 * k,
                [xs.bitcast(fx.Int32), xs_sh.bitcast(fx.Int32)],
                tid == 0,
            )
            stamp(7)

        # ------------------------------------------------------------ router
        @traced
        def router(e):
            c0 = wave * (H // WAVES) + lane * 12
            gw = [load(gate, e * H + c0 + 4 * i, 4, T.f32) for i in range(3)]
            gf = [gw[i // 4][i % 4] for i in range(12)]
            bias_e = load_f32(bias, e)
            wait_x()
            # every token's xn slice in flight at once, then the dot products
            xws = [
                words6(
                    load(sreg("xn"), k * (H // 2) + c0 // 2, 4, cm=CM_DEV),
                    load(sreg("xn"), k * (H // 2) + c0 // 2 + 4, 2, cm=CM_DEV),
                )
                for k in range(M)
            ]
            for k in range_constexpr(M):
                xw = xws[k]
                acc = fx.Float32(0.0)
                for i in range_constexpr(6):
                    acc = fmath.fma(gf[2 * i], bf_lo(xw[i]), acc)
                    acc = fmath.fma(gf[2 * i + 1], bf_hi(xw[i]), acc)
                acc = wave_sum(acc)
                if lane == 0:
                    fx.ptr_store(acc, red + (k * WAVES + wave))
            gpu.barrier()
            if tid < M:
                logit = fx.ptr_load(red + tid * WAVES)
                for w in range_constexpr(1, WAVES):
                    logit = logit + fx.ptr_load(red + (tid * WAVES + w))
                sc = sigmoid(logit)
                for c in range_constexpr(SCORE_COPIES):
                    mb_put(
                        sreg("scores"),
                        ((c * MAX_TOKENS + tid) * E + e) * 2,
                        [(sc + bias_e).bitcast(fx.Int32), sc.bitcast(fx.Int32)],
                    )
            gpu.barrier()

        # ------------------------------------------------------------ routing
        @traced
        def route():
            """Every token's top-4 by sigmoid + bias (ties to the lower expert) ->
            rexp, and the picks' sigmoids renormalized -> rwt."""
            cp = (bid % SCORE_COPIES) * MAX_TOKENS
            for k in range_constexpr(M):
                if wave == k % WAVES:
                    got = mb_poll(
                        [
                            (sreg("scores"), ((cp + k) * E + lane) * 2, 2),
                            (sreg("scores"), ((cp + k) * E + 64 + lane) * 2, 2),
                        ]
                    )
                    v0 = got[0][0].bitcast(fx.Float32)
                    v1 = got[1][0].bitcast(fx.Float32)
                    fx.ptr_store(got[0][1].bitcast(fx.Float32), sig + (k * E + lane))
                    fx.ptr_store(
                        got[1][1].bitcast(fx.Float32), sig + (k * E + 64 + lane)
                    )
                    pid = fx.Int32(0)
                    for p in range_constexpr(TOP_K):
                        m = wave_max(fx.max(v0, v1))
                        cand = fx.min(
                            (v0 == m).select(lane, fx.Int32(E)),
                            (v1 == m).select(lane + 64, fx.Int32(E)),
                        )
                        ex = butterfly(cand, WAVE, fx.min)
                        v0 = (lane == ex).select(fx.Float32(-3.0e38), v0)
                        v1 = (lane + 64 == ex).select(fx.Float32(-3.0e38), v1)
                        pid = (lane == p).select(ex, pid)
                    w = (lane < TOP_K).select(
                        fx.ptr_load(sig + (k * E + fx.min(pid, E - 1))), fx.Float32(0.0)
                    )
                    tot = butterfly(w, (1, 2))
                    if lane < TOP_K:
                        fx.ptr_store(pid, rexp + (k * TOP_K + lane))
                        fx.ptr_store(w / tot, rwt + (k * TOP_K + lane))
            gpu.barrier()
            if const_expr(WIDE):
                build_experts()

        @traced
        def build_experts():
            """Wide path: the step's nu distinct routed experts, ascending (so
            every CTA agrees) -> uexp[:nu], their token masks -> umask[:nu], and
            nu -> uexp[U_MAX]. Thread e < E owns expert e."""
            m = fx.Int32(0)
            if tid < E:
                for p in range_constexpr(M * TOP_K):
                    hit = fx.ptr_load(rexp + p) == tid
                    m = m | hit.select(fx.Int32(1 << (p // TOP_K)), fx.Int32(0))
            bal = ballot(m != 0)
            below = popcount(bal & ((fx.Int64(1) << fx.Int64(lane)) - 1))
            if tid == 0:
                fx.ptr_store(popcount(bal), umask + U_MAX)
            gpu.barrier()
            if tid < E:
                off = (wave == 0).select(fx.Int32(0), fx.ptr_load(umask + U_MAX))
                if m != 0:
                    fx.ptr_store(fx.Int32(tid), uexp + (off + below))
                    fx.ptr_store(m, umask + (off + below))
                if tid == E - 1:
                    fx.ptr_store(off + popcount(bal), uexp + U_MAX)
            gpu.barrier()

        # ------------------------------------------------------------ up / gate
        def ug_k(c):
            """This lane's k offset of its c-th 16 B of a row: a row's 4 lanes read
            one whole 128 B line a load. A and B share this k permutation, which
            the k-reduction does not see."""
            return (wave * (UG_KC // 2) + c // 2) * 128 + g4 * 32 + (c % 2) * 16

        def dn_k(c):
            """Narrow down: wave w takes k quarter w % DN_KW, a row's 4 lanes read
            192 contiguous bytes, 16 a load."""
            return (wave % DN_KW) * (I // DN_KW) + g4 * (16 * DN_KL) + c * 16

        def ug_task(t):
            """(shared, unit, token k, row group cg). Narrow: unit = k * TOP_K +
            slot; wide: unit is the distinct expert index and k is unused."""
            shared = t < UG_TASKS_PER_UNIT
            unit = fx.max(t - UG_TASKS_PER_UNIT, 0) // UG_TASKS_PER_UNIT
            return shared, unit, unit // TOP_K, t % UG_TASKS_PER_UNIT

        def ug_expert(shared, unit):
            tab = uexp if const_expr(WIDE) else rexp
            ex = uniform(fx.ptr_load(tab + fx.min(unit, U_MAX - 1)))
            return shared.select(fx.Int32(0), fx.max(fx.min(ex, E - 1), 0))

        def ug_count():
            """Number of up / gate tasks this step; needs the routing."""
            if const_expr(not WIDE):
                return fx.Int32(UG_TASKS)
            nu = fx.max(fx.min(uniform(fx.ptr_load(uexp + U_MAX)), U_MAX), 0)
            return (1 + nu) * UG_TASKS_PER_UNIT

        def slot_of(k, e):
            """Token k's slot routed to expert e, or TOP_K if none."""
            s = fx.Int32(TOP_K)
            for j in range_constexpr(TOP_K - 1, -1, -1):
                s = (fx.ptr_load(rexp + (k * TOP_K + j)) == e).select(fx.Int32(j), s)
            return s

        def ug_load(t, has):
            """Task t's weights and scales: tile rows 0..7 are gate rows, 8..15 the
            matching up rows; the waves split k. Nothing is read unless ``has``."""
            shared, unit, _, cg = ug_task(t)
            ex = ug_expert(shared, unit)
            base = shared.select(sgu, w13 + fx.Int64(ex) * W13_BYTES)
            row = (l16 < 8).select(cg * UG_ROWS + l16, I + cg * UG_ROWS + l16 - 8)
            nbytes = has.select(fx.Int32(W13_BYTES), fx.Int32(0))
            wts = [
                load(base, (row * H + ug_k(c)) // 4, 4, nbytes=nbytes)
                for c in range(UG_KC)
            ]
            s_base = shared.select(sgu_s, s13 + fx.Int64(ex) * (2 * I * 4))
            rr = cg * UG_ROWS + tid % UG_ROWS
            return wts + [load_f32(s_base, rr), load_f32(s_base, I + rr)]

        def ug_prep(t, first=False):
            """Task t's B operand (column l16 = a token): xq, or xq_sh for the
            shared expert. Shared tasks are t < 96 < G, so only a CTA's ``first``
            task can be one."""
            shared, _, k, _ = ug_task(t)
            if const_expr(WIDE):
                routed = xb_wide
            else:
                v4 = fx.Vector.make_type(4, fx.Int32)
                routed = [
                    fx.Vector(
                        fx.ptr_load(xql + (k * XQ_ROW + ug_k(c) // 4), result_type=v4)
                    )
                    for c in range(UG_KC)
                ]
            if const_expr(not first):
                return routed
            sh_n = shared.select(fx.Int32(M * H), fx.Int32(0))
            sh_b = [
                load(
                    sreg("xq_sh"),
                    (fx.min(l16, M - 1) * H + ug_k(c)) // 4,
                    4,
                    cm=CM_DEV,
                    nbytes=sh_n,
                )
                for c in range(UG_KC)
            ]
            return [shared.select(a, b) for a, b in zip(sh_b, routed, strict=True)]

        @traced
        def ug(t, wts, bs, buf=0):
            """Up / gate task t. Consecutive tasks alternate the ``red`` half
            (``buf``) so a task needs no trailing barrier."""
            shared, unit, k, cg = ug_task(t)
            red_ = red + buf * (WAVES * 256)
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for c in range_constexpr(UG_KC):
                b = bs[c].bitcast(fx.Int64)
                a = wts[c].bitcast(fx.Int64)
                for h in range_constexpr(2):
                    acc = mfma_fp8(a[h], b[h], acc)
            for i in range_constexpr(4):
                fx.ptr_store(acc[i], red_ + ((wave * 16 + 4 * g4 + i) * 16 + l16))
            gpu.barrier()
            rr = tid % UG_ROWS
            col = tid // UG_ROWS
            if const_expr(WIDE):
                # column col is token col, live if routed to this expert
                kk = fx.min(col, M - 1)
                ex = ug_expert(shared, unit)
                um = fx.ptr_load(umask + fx.min(unit, U_MAX - 1))
                live = (tid < UG_ROWS * M) & (shared | (((um >> kk) & 1) != 0))
                slot = shared.select(fx.Int32(TOP_K), slot_of(kk, ex))
            else:
                live = tid < shared.select(fx.Int32(UG_ROWS * M), fx.Int32(UG_ROWS))
                kk = shared.select(col, k)
                slot = shared.select(fx.Int32(TOP_K), unit % TOP_K)
            if live:
                gs = fx.Float32(0.0)
                us = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    gs = gs + fx.ptr_load(red_ + ((w * 16 + rr) * 16 + col))
                    us = us + fx.ptr_load(red_ + ((w * 16 + 8 + rr) * 16 + col))
                x_s = fx.ptr_load(
                    xsc + (kk + shared.select(fx.Int32(MAX_TOKENS), fx.Int32(0)))
                )
                gv = bf16_round(gs * x_s * wts[UG_KC])
                uv = bf16_round(us * x_s * wts[UG_KC + 1])
                gv = fx.min(gv, fx.Float32(7.0))
                uv = fx.max(fx.min(uv, fx.Float32(7.0)), fx.Float32(-7.0))
                # routed: vLLM's silu_and_mul_with_clamp rounds g * sigmoid(a g) to
                # bf16 before the product; the shared expert's swiglu_oai_split
                # does not
                gsig = gv * sigmoid(1.702 * gv)
                gsig = shared.select(gsig, bf16_round(gsig))
                y = bf16_round(gsig * (uv + 1.0))
                mb_put(
                    sreg("mids"),
                    (kk * UNITS + slot) * I + cg * UG_ROWS + rr,
                    [y.bitcast(fx.Int32)],
                )

        # ------------------------------------------------------------ down
        def dn_experts(k):
            return [
                fx.min(uniform(fx.ptr_load(rexp + (k * TOP_K + u))), E - 1)
                for u in range(TOP_K)
            ]

        def dn_load(t, has):
            k = fx.min(t // DN_GROUPS, M - 1)
            dg = t % DN_GROUPS
            exs = dn_experts(k)
            nbytes = has.select(fx.Int32(W2_BYTES), fx.Int32(0))
            row = dg * DN_ROWS + (wave // DN_KW) * 16 + l16
            wts = []
            for u in range_constexpr(UNITS):
                base = w2 + fx.Int64(exs[u]) * W2_BYTES if u < TOP_K else sdn
                for c in range_constexpr(DN_KL):
                    wts.append(load(base, (row * I + dn_k(c)) // 4, 4, nbytes=nbytes))
            return wts

        # ------------------------------------------------------------ mid quant
        @traced
        def mid_quant(q):
            """(token, unit) row q's mids -> FP8 in mq, its scale published to mqs;
            done once here rather than in every down task."""
            v4 = [fx.Float32(0.0) for _ in range(4)]
            if tid < I // 4:
                got = mb_poll(
                    [(sreg("mids"), q * I + 4 * tid + 2 * h, 2) for h in range(2)]
                )
                v4 = [got[h][e].bitcast(fx.Float32) for h in range(2) for e in range(2)]
            amax = fx.max(fx.max(fmath.absf(v4[0]), fmath.absf(v4[1])),
                          fx.max(fmath.absf(v4[2]), fmath.absf(v4[3])))  # fmt: skip
            amax = block_reduce(amax, fx.max)
            sh = q % UNITS == TOP_K
            ms_r = row_scale(amax)
            ms_s, inv_s = sh_scale(amax)
            ms = sh.select(ms_s, ms_r)
            if tid < I // 4:
                qv = [
                    sh.select(quant(v * inv_s, 1.0, SH_FP8_MAX), quant(v, ms_r))
                    for v in v4
                ]
                store(fp8_pack4(*qv), sreg("mq"), q * (I // 4) + tid, CM_DEV)
            publish(mb_put, sreg("mqs"), q, [ms.bitcast(fx.Int32)], tid == 0)

        @traced
        def dn_prep(t):
            """Narrow down task t's B operand: its token's 5 units' FP8 mids from
            scratch, after their scales land in msc."""
            k = fx.min(t // DN_GROUPS, M - 1)
            if wave == 0:
                got = mb_poll([(sreg("mqs"), k * UNITS + fx.min(lane, UNITS - 1), 1)])
                if lane < UNITS:
                    fx.ptr_store(got[0][0].bitcast(fx.Float32), msc + lane)
            gpu.barrier()
            return [
                load(sreg("mq"), ((k * UNITS + u) * I + dn_k(c)) // 4, 4, cm=CM_DEV)
                for u in range(UNITS)
                for c in range(DN_KL)
            ]

        @traced
        def down(t, wts, bs):
            """Narrow down task t = (token, 32 rows): waves 0..3 / 4..7 take rows
            0..15 / 16..31, each wave a k quarter; all 16 B columns are the token."""
            k = t // DN_GROUPS
            dg = t % DN_GROUPS
            exs = dn_experts(k)
            rg = wave // DN_KW
            for u in range_constexpr(UNITS):
                acc = fx.Vector.filled(4, 0.0, fx.Float32)
                for c in range_constexpr(DN_KL):
                    a = wts[u * DN_KL + c].bitcast(fx.Int64)
                    b = bs[u * DN_KL + c].bitcast(fx.Int64)
                    for h in range_constexpr(2):
                        acc = mfma_fp8(a[h], b[h], acc)
                if l16 == 0:
                    for j in range_constexpr(4):
                        fx.ptr_store(
                            acc[j],
                            red
                            + (((u * 2 + rg) * DN_KW + wave % DN_KW) * 16 + 4 * g4 + j),
                        )
            gpu.barrier()
            if tid < DN_ROWS:
                dn_finish(k, dg, exs, lambda u: tid // 16 + u * 2, tid)
            gpu.barrier()
            if tid < 16:
                ffn_reduce(k, dg, tid, 0)
            gpu.barrier()

        # ------------------------------------------------------------ wide down
        # Task dg = 32 rows over every distinct expert and the shared one; wave w
        # takes rows 16 * (w // 4).. and experts w % 4, w % 4 + 4, ..; B column
        # l16 is token l16's mid row. Per-(token, slot) terms are summed in slot
        # order, as vLLM does.
        WD_UNIT = DN_LD + 1  # loads an expert: its s2 rows, then 12 weight words
        WD_DEPTH = 3 * WD_UNIT  # loads in flight

        def wd_unit(i):
            """(distinct index, live, shared, expert) of this wave's i-th expert;
            index nu is the shared expert."""
            du = wave % DN_KW + DN_KW * i
            nu = fx.max(fx.min(uniform(fx.ptr_load(uexp + U_MAX)), U_MAX), 0)
            ex = fx.max(
                fx.min(uniform(fx.ptr_load(uexp + fx.min(du, U_MAX - 1))), E - 1), 0
            )
            return du, du <= nu, du == nu, ex

        def wd_load(dg, n, has):
            du, ok, sh, ex = wd_unit(n // WD_UNIT)
            ok = ok & has
            c = n % WD_UNIT
            r0 = dg * DN_ROWS + (wave // DN_KW) * 16
            if const_expr(c == 0):
                base = sh.select(sdn_s, s2 + fx.Int64(ex) * (H * 4))
                return load(
                    base,
                    r0 + 4 * g4,
                    4,
                    T.f32,
                    nbytes=ok.select(fx.Int32(H * 4), fx.Int32(0)),
                )
            base = sh.select(sdn, w2 + fx.Int64(ex) * W2_BYTES)
            return load(
                base,
                ((r0 + l16) * I + wd_k(c - 1)) // 4,
                4,
                nbytes=ok.select(fx.Int32(W2_BYTES), fx.Int32(0)),
            )

        def wd_k(c):
            """As ug_k: a row's 4 lanes read one whole 128 B line a load."""
            return (c // 2) * 128 + g4 * 32 + (c % 2) * 16

        def dn_load_wide(t, has):
            return [wd_load(t, n, has) for n in range(min(WD_DEPTH, DN_EXP * WD_UNIT))]

        @traced
        def stage_mids():
            """Wide path: every (token, unit)'s FP8 mid row -> LDS, scales -> msc."""
            if wave == 0:
                got = mb_poll([(sreg("mqs"), fx.min(lane, MID_ROWS - 1), 1)])
                if lane < MID_ROWS:
                    fx.ptr_store(got[0][0].bitcast(fx.Float32), msc + lane)
            gpu.barrier()
            nv = MID_ROWS * (I // 16)
            for p in range_constexpr(-(-nv // THREADS)):
                v = tid + p * THREADS
                if v < nv:
                    w = load(sreg("mq"), v * 4, 4, cm=CM_DEV)
                    fx.ptr_store(
                        w, midl + (v // (I // 16) * MID_ROW + v % (I // 16) * 4)
                    )
            gpu.barrier()

        @traced
        def down_wide(dg, wts, bs):
            inflight = list(wts)
            nxt = len(inflight)
            ok_all = fx.Int32(dg) < DN_TASKS
            stage_mids()
            rg = wave // DN_KW
            kk = fx.min(l16, M - 1)
            v4 = fx.Vector.make_type(4, fx.Int32)
            for i in range_constexpr(DN_EXP):
                ld = []
                for _ in range_constexpr(WD_UNIT):
                    ld.append(inflight.pop(0))
                    if const_expr(nxt < DN_EXP * WD_UNIT):
                        inflight.append(wd_load(dg, nxt, ok_all))
                        nxt += 1
                du, ok, sh, ex = wd_unit(i)
                um = fx.ptr_load(umask + fx.min(du, U_MAX - 1))
                slot = sh.select(fx.Int32(TOP_K), slot_of(kk, ex))
                mrow = kk * UNITS + slot
                acc = fx.Vector.filled(4, 0.0, fx.Float32)
                for c in range_constexpr(DN_LD):
                    b = fx.Vector(
                        fx.ptr_load(
                            midl + (mrow * MID_ROW + wd_k(c) // 4), result_type=v4
                        )
                    )
                    a = ld[c + 1].bitcast(fx.Int64)
                    b = b.bitcast(fx.Int64)
                    for h in range_constexpr(2):
                        acc = mfma_fp8(a[h], b[h], acc)
                ms = fx.ptr_load(msc + mrow)
                wgt = sh.select(
                    fx.Float32(1.0),
                    fx.ptr_load(rwt + (kk * TOP_K + fx.min(slot, TOP_K - 1))),
                )
                live = ok & (l16 < M) & (sh | (((um >> kk) & 1) != 0))
                if live:
                    for j in range_constexpr(4):
                        term = bf16_round(acc[j] * ms * ld[0][j] * wgt)
                        fx.ptr_store(
                            term, red + (mrow * DN_ROWS + rg * 16 + 4 * g4 + j)
                        )
            gpu.barrier()
            if tid < M * DN_ROWS:
                k = tid // DN_ROWS
                r = tid % DN_ROWS
                routed = fx.Float32(0.0)
                for u in range_constexpr(TOP_K):
                    routed = routed + fx.ptr_load(red + ((k * UNITS + u) * DN_ROWS + r))
                routed = bf16_round(bf16_round(routed) * 2.0)
                shared = fx.ptr_load(red + ((k * UNITS + TOP_K) * DN_ROWS + r))
                fx.ptr_store(bf16_round(routed + shared), part + tid)
            gpu.barrier()
            if tid < M * 16:
                ffn_reduce(tid // 16, dg, tid % 16, (tid // 16) * DN_ROWS)
            gpu.barrier()

        @traced
        def dn_finish(k, dg, exs, tile_of, r):
            """Row r of (token k, row group dg): each unit's wave-summed tile
            (``tile_of(unit)``) scaled and route-weighted, x routed_scale, + shared."""
            rr = r % 16
            row = dg * DN_ROWS + r
            d = []
            for u in range_constexpr(UNITS):
                s = fx.Float32(0.0)
                for w in range_constexpr(DN_KW):
                    s = s + fx.ptr_load(red + ((tile_of(u) * DN_KW + w) * 16 + rr))
                d.append(s * fx.ptr_load(msc + u))
            routed = fx.Float32(0.0)
            for u in range_constexpr(TOP_K):
                s2v = load_f32(s2 + fx.Int64(exs[u]) * (H * 4), row)
                routed = routed + bf16_round(
                    d[u] * s2v * fx.ptr_load(rwt + (k * TOP_K + u))
                )
            routed = bf16_round(bf16_round(routed) * 2.0)
            shared = bf16_round(d[TOP_K] * load_f32(sdn_s, row))
            fx.ptr_store(bf16_round(routed + shared), part + r)

        @traced
        def ffn_reduce(k, dg, i, p0):
            """All-reduce bf16 pair i of (token k, row group dg) from part[p0:]:
            summed in rank order and truncated to bf16, as AITER's all-reduce on
            gfx94x, -> out."""
            pair = (rank * MAX_TOKENS + k) * (H // 2) + dg * 16 + i
            word = bf16_word(
                fx.ptr_load(part + (p0 + 2 * i)), fx.ptr_load(part + (p0 + 2 * i + 1))
            )
            for p in range_constexpr(W):
                mb_put(peer_base[p] + fx.Int64(PEER["ffn"]), pair, [word], CM_SYS)
            parts = mb_poll(
                [
                    (own("ffn"), (src * MAX_TOKENS + k) * (H // 2) + dg * 16 + i, 1)
                    for src in range(W)
                ],
                CM_SYS,
            )
            lo = bf_lo(parts[0][0])
            hi = bf_hi(parts[0][0])
            for src in range_constexpr(1, W):
                lo = lo + bf_lo(parts[src][0])
                hi = hi + bf_hi(parts[src][0])
            store(bf16_word_trunc(lo, hi), out, k * (H // 2) + dg * 16 + i)

        # ------------------------------------------------------------ schedule
        stamp(0)
        if bid * THREADS + tid < PUSH_THREADS:
            push(bid * THREADS + tid)
        if bid < M:
            norm(bid)
        stamp(1)
        if (bid >= ROUTER_BASE) & (bid < ROUTER_BASE + E):
            router(bid - ROUTER_BASE)
        stamp(2)
        # This CTA's up / gate, mid quant and down jobs, software-pipelined: job
        # j + 1's weights are issued before job j computes.
        ug0 = first(UG_BASE)
        q0 = first(Q_BASE)
        dn0 = first(DN_BASE)

        # the first task's weights stream while the routing is on its way
        shared0 = ug0 < UG_TASKS_PER_UNIT
        early = ug_load(fx.min(ug0, UG_TASKS_PER_UNIT - 1), shared0)
        if (ug0 < UG_TASKS) | (dn0 < DN_TASKS):
            route()
        n_ug = fx.min(ug_count(), UG_TASKS)
        if ug0 < n_ug:
            wait_x()
            if const_expr(not WIDE):
                stage_xq()
        stamp(3)
        late = ug_load(fx.min(ug0, n_ug - 1), (ug0 < n_ug) & ~shared0)
        cur = [shared0.select(a, b) for a, b in zip(early, late, strict=True)]
        if const_expr(WIDE):
            # B is task-invariant (column l16 = token l16): load this lane's xq
            # k-slices once, behind the first task's weights
            xnb = (ug0 < n_ug).select(fx.Int32(M * H), fx.Int32(0))
            xb_wide = [
                load(
                    sreg("xq"),
                    (fx.min(l16, M - 1) * H + ug_k(c)) // 4,
                    4,
                    cm=CM_DEV,
                    nbytes=xnb,
                )
                for c in range(UG_KC)
            ]
        jobs = [(ug_load, ug_prep, ug, ug0 + it * G, n_ug) for it in range(UG_ITERS)]
        jobs += [(None, None, mid_quant, q0, Q_TASKS)]
        dn_job = (
            (dn_load_wide, None, down_wide)
            if const_expr(WIDE)
            else (dn_load, dn_prep, down)
        )
        jobs += [(*dn_job, dn0 + it * G, DN_TASKS) for it in range(DN_ITERS)]

        def issue(j):
            ld, _, _, t, n = jobs[j]
            return [] if ld is None else ld(fx.min(t, n - 1), t < n)

        # loads complete in issue order: a task's small B loads go out before the
        # next task's weights, so waiting on them does not wait on the prefetch
        for j in range_constexpr(len(jobs)):
            _, prep, run, t, n = jobs[j]
            if const_expr(prep is ug_prep):
                bs = ug_prep(fx.min(t, n - 1), first=j == 0)
            else:
                bs = [] if prep is None else prep(fx.min(t, n - 1))
            if const_expr(run is mid_quant):  # noqa: SIM102
                # its publish waits on vmcnt(0): run it before the next prefetch
                # goes out, or it waits on the whole prefetch
                if t < n:
                    mid_quant(t)
            if const_expr(j + 1 < len(jobs)):
                nxt = issue(j + 1)
            if const_expr(run is ug):  # noqa: SIM102
                if t < n:
                    ug(t, cur, bs, j % 2)
            elif const_expr(run is not mid_quant):  # noqa: SIM102
                if t < n:
                    run(t, cur, bs)
            if const_expr(j == UG_ITERS - 1):
                gpu.barrier()  # the last up / gate task's reduce reads done
                stamp(4)
            if const_expr(j + 1 < len(jobs)):
                cur = nxt
        stamp(5)
        gpu.barrier()
        if tid == 0:
            store(tag, ctr, bid)

    @flyc.jit
    def launch(
        partial: Int64, residual: Int64, h_mid: Int64, out: Int64, g_post: Int64,
        gate: Int64, bias: Int64, w13: Int64, s13: Int64, w2: Int64, s2: Int64,
        sgu: Int64, sgu_s: Int64, sdn: Int64, sdn_s: Int64, scratch: Int64,
        peer_local: Int64, peers: Int64, ctr: Int64, rank: Int32, tl: Int64,
        stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = key
        ffn_kernel(
            partial, residual, h_mid, out, g_post, gate, bias, w13, s13, w2, s2,
            sgu, sgu_s, sdn, sdn_s, scratch, peer_local, peers, ctr, rank, tl,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)  # fmt: skip

    return launch
