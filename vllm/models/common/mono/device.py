# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM), MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.: atom/mono/device/{ops,sync}.py
"""Arch-neutral (gfx942 / gfx950) FlyDSL device primitives of the MonoKernels.

Traced inside ``@flyc.kernel`` bodies. Lane exchanges use DPP within a row and
``shuffle_xor`` across rows, so nothing here needs gfx950-only instructions.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

# Cache policy bits: SC1 is device scope (past the per-XCD L2s), SC0 | SC1 system
# scope (peer GPUs over XGMI).
CM_DEV = 16
CM_SYS = 17
POLL_MAX = 12


def traced(fn):
    """Lower a helper's ``while`` / ``if`` over traced values to scf ops."""
    return ASTRewriter.transform(fn)


def rsrc(addr, nbytes=None):
    return bo.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)


def load(addr, index, width=4, dtype=None, cm=0, nbytes=None):
    """``width`` (> 1) elements of ``dtype`` (i32 by default) at element ``index``.
    ``nbytes``: the buffer's bound (0 makes the load a no-op returning zeros)."""
    return fx.Vector(
        bo.buffer_load(
            rsrc(addr, nbytes),
            index,
            vec_width=width,
            dtype=dtype or T.i32,
            cache_modifier=cm,
        )
    )


def load_i32(addr, index, cm=0):
    return fx.Int32(
        bo.buffer_load(rsrc(addr), index, vec_width=1, dtype=T.i32, cache_modifier=cm)
    )


def load_f32(addr, index, cm=0):
    return fx.Float32(
        bo.buffer_load(rsrc(addr), index, vec_width=1, dtype=T.f32, cache_modifier=cm)
    )


def store(value, addr, index, cm=0):
    """A scalar or vector at element ``index`` of ``addr``."""
    bo.buffer_store(value, rsrc(addr), index, cache_modifier=cm)


def uniform(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v).ir_value()))


def load_ptr64(table, i):
    """Entry ``i`` of an i64 table, wave-uniform."""
    w = load(table, i * 2, width=2)
    return (fx.Int64(uniform(w[1])) << 32) | fx.Int64(fx.Uint32(uniform(w[0])))


def memrealtime():
    """The 100 MHz device clock."""
    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


def _hw_f32(name, x):
    return fx.Float32(
        llvm.call_intrinsic(T.f32, name, [fx.Float32(x).ir_value()], [], [])
    )


def hw_rcp(x):
    return _hw_f32("llvm.amdgcn.rcp.f32", x)


def hw_exp2(x):
    return _hw_f32("llvm.amdgcn.exp2.f32", x)


def rsqrt_rn(x):
    """1 / sqrt(x), each step correctly rounded (Triton's ``1.0 / tl.sqrt``):
    the approximate v_rsq can flip a bf16 rounding downstream."""
    return fx.Float32(1.0) / fx.Float32(fmath.sqrt(fx.Float32(x)))


_DPP_XOR = {
    8: ((0x118, 0xC), (0x108, 0x3)),
    4: ((0x114, 0xA), (0x104, 0x5)),
    2: ((0x4E, 0xF),),
    1: ((0xB1, 0xF),),
}


def xshfl(v, off):
    """Lane ``lane ^ off``'s value: DPP within a 16-lane row, ``shuffle_xor``
    (ds_swizzle / ds_bpermute) across rows."""
    if off >= 16:
        return v.shuffle_xor(off, 64)
    is_f = isinstance(v, fx.Float32)
    x = v.bitcast(fx.Int32) if is_f else fx.Int32(v)
    y = x
    for ctrl, bank in _DPP_XOR[off]:
        y = fx.Int32(update_dpp_i32(y, x, ctrl, 0xF, bank, False))
    return y.bitcast(fx.Float32) if is_f else y


def ballot(pred):
    """The wave's lanes where ``pred`` holds, a bit a lane."""
    return fx.Int64(rocdl.ballot(T.i64, pred.ir_value()))


def popcount(x):
    return fx.Int32(fx.ctpop(x))


def _add(a, b):
    return a + b


def butterfly(v, offsets, op=_add):
    """Reduce over the xor ``offsets`` in order (the order is the numerics)."""
    for off in offsets:
        v = op(v, xshfl(v, off))
    return v


WAVE = (32, 16, 8, 4, 2, 1)


def wave_sum(v):
    return butterfly(v, WAVE)


def wave_max(v):
    return butterfly(v, WAVE, fx.max)


def bf16_round(a):
    return fx.Float32(fx.Float32(a).to(fx.BFloat16))


def bf16_word(a, b):
    """(bf16(a), bf16(b)) packed in an Int32."""
    return (
        fx.Vector.from_elements([a, b], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
    )


def bf16_trunc(a):
    """f32 -> bf16 by dropping the low half (round toward zero), back as f32:
    AITER's fp32 -> bf16 store on gfx94x."""
    return (fx.Float32(a).bitcast(fx.Int32) & fx.Int32(-65536)).bitcast(fx.Float32)


def bf16_word_trunc(a, b):
    """(trunc(a), trunc(b)) packed in an Int32."""
    lo = fx.Int32(fx.Uint32(fx.Float32(a).bitcast(fx.Int32)) >> 16)
    return (fx.Float32(b).bitcast(fx.Int32) & fx.Int32(-65536)) | lo


def bf_lo(w):
    return (w << 16).bitcast(fx.Float32)


def bf_hi(w):
    return (w & fx.Int32(-65536)).bitcast(fx.Float32)


def fp8_pack4(a, b, c, d):
    """Four in-range f32 -> one dword of four FP8 bytes (fnuz on gfx942), RNE."""
    lo = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, a, b, fx.Int32(0), False))
    return fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, c, d, lo, True))


def mfma_fp8(a, b, c):
    """16x16x32 FP8 MFMA, a / b one i64 (8 bytes) a lane."""
    return fx.Vector(
        rocdl.mfma_f32_16x16x32_fp8_fp8(T.vec(4, T.f32), [a, b, c, 0, 0, 0])
    )


@traced
def _spin(load_all, pending):
    v = load_all()
    while pending(v):
        rocdl.s_sleep(1)
        v = load_all()
    return v


class Mailbox:
    """Tagged (value, tag) pairs: a consumer polls until every tag is the
    launch's, so a hand-off is one round trip with no separate flag."""

    def __init__(self, tag):
        self.tag = tag

    def put(self, addr, pair, words, cm=CM_DEV):
        """Words (1 or 2, Int32) at pairs ``pair``.. of ``addr``."""
        vals = []
        for w in words:
            vals += [fx.Int32(w), self.tag]
        store(fx.Vector.from_elements(vals, fx.Int32), addr, pair * 2, cm)

    def poll(self, specs, cm=CM_DEV, batch=POLL_MAX):
        """``specs`` = [(addr, pair, npairs in {1, 2})] -> their value words."""
        if const_expr(len(specs) > batch):
            return self.poll(specs[:batch], cm, batch) + self.poll(
                specs[batch:], cm, batch
            )
        tag = self.tag

        def load_all():
            words = []
            for a, i, n in specs:
                w = load(a, fx.Int32(i) * 2, width=2 * n, cm=cm)
                words += [w[e] for e in range(2 * n)]
            return fx.Vector.from_elements(words, fx.Int32)

        nw = sum(2 * n for _, _, n in specs)

        def pending(v):
            bad = v[1] != tag
            for e in range_constexpr(3, nw, 2):
                bad = bad | (v[e] != tag)
            return bad

        v = _spin(load_all, pending)
        outs, e = [], 0
        for _, _, n in specs:
            outs.append([v[e + 2 * q] for q in range(n)])
            e += 2 * n
        return outs


@traced
def publish(put, addr, pair, words, who):
    """The CTA's plain stores drained, then thread ``who`` puts the flag pair
    (``put``: a ``Mailbox.put``)."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if who:
        put(addr, pair, words)
