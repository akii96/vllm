# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-5 CSA metadata vs a direct reference.

A region ``r`` of a request is summarised in the step that caches its last token:
``computed < (r + 1) * region_size <= seq_len``. Getting this wrong does not crash;
it leaves stale or missing summaries and silently picks worse regions.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.models.step5.common.csa_cache import (
    Step5CSAMetadataBuilder,
    _completed_regions,
)
from vllm.platforms import current_platform

REGION = 8


def _reference_regions(seq_lens: list[int], query_lens: list[int]) -> set:
    return {
        (req, r)
        for req, (seq, q) in enumerate(zip(seq_lens, query_lens))
        for r in range(seq // REGION)
        if seq - q < (r + 1) * REGION <= seq
    }


def _random_batch(rng: torch.Generator) -> BatchSpec:
    reqs = int(torch.randint(1, 7, (), generator=rng))
    query_lens, seq_lens = [], []
    for _ in range(reqs):
        kind = int(torch.randint(0, 3, (), generator=rng))
        q = (
            1 + int(torch.randint(0, 4, (), generator=rng))  # decode / MTP verify
            if kind == 0
            else int(torch.randint(1, 300, (), generator=rng))  # (chunked) prefill
        )
        computed = 0 if kind == 1 else int(torch.randint(0, 500, (), generator=rng))
        query_lens.append(q)
        seq_lens.append(computed + q)
    return BatchSpec(seq_lens=seq_lens, query_lens=query_lens)


@pytest.mark.cpu_test
def test_completed_regions_matches_reference():
    rng = torch.Generator().manual_seed(0)
    for _ in range(500):
        spec = _random_batch(rng)
        m = create_common_attn_metadata(spec, block_size=16, device=torch.device("cpu"))
        work = _completed_regions(m, REGION)
        live = work[work[:, 0] >= 0]
        got = {tuple(row) for row in live.tolist()}
        assert len(got) == live.shape[0], "duplicate work rows"
        assert got == _reference_regions(spec.seq_lens, spec.query_lens), spec
        assert (work[live.shape[0] :] == -1).all(), "padding must trail live rows"


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="token_to_req_indices is a GPU kernel"
)
def test_builder_positions_fallback():
    """Spec-decode metadata carries no positions; the builder must derive them."""
    spec = BatchSpec(seq_lens=[40, 9, 130], query_lens=[4, 9, 1])
    m = create_common_attn_metadata(spec, block_size=16, device=torch.device("cuda"))
    assert m.positions is None
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                sparse_config=SimpleNamespace(region_block_size=REGION)
            )
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64, max_num_seqs=8),
    )
    builder = Step5CSAMetadataBuilder(
        SimpleNamespace(tokens_per_state=1), [], config, torch.device("cuda")
    )
    md = builder.build(0, m)
    assert md.positions is not None and md.token_req is not None

    want_pos = [
        p for s, q in zip(spec.seq_lens, spec.query_lens) for p in range(s - q, s)
    ]
    want_req = [r for r, q in enumerate(spec.query_lens) for _ in range(q)]
    assert md.positions.tolist() == want_pos
    assert md.token_req.tolist() == want_req


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="needs CUDA graph capture"
)
def test_builder_persistent_buffers_survive_graph_replay():
    """A captured decode graph keeps reading the buffers it captured.

    The builder must rewrite its persistent per-step buffers in place, so a
    replayed graph observes the refreshed metadata of the current step
    instead of the capture-time values.
    """
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                sparse_config=SimpleNamespace(region_block_size=REGION)
            )
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64, max_num_seqs=8),
    )
    builder = Step5CSAMetadataBuilder(
        SimpleNamespace(tokens_per_state=1), [], config, torch.device("cuda")
    )

    def build(seq_lens: list[int]):
        m = create_common_attn_metadata(
            BatchSpec(seq_lens=seq_lens, query_lens=[1] * len(seq_lens)),
            block_size=16,
            device=torch.device("cuda"),
        )
        return builder.build(0, m)

    md1 = build([24, 8])
    pos_out = torch.zeros(2, dtype=torch.int32, device="cuda")
    regions_out = torch.zeros(2, 2, dtype=torch.int32, device="cuda")

    def consume():
        pos_out.copy_(md1.positions)
        regions_out.copy_(md1.completed_regions)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        consume()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        consume()

    md2 = build([40, 16])
    assert md2.positions.data_ptr() == md1.positions.data_ptr(), (
        "positions must live in a persistent builder buffer"
    )
    assert md2.completed_regions.data_ptr() == md1.completed_regions.data_ptr(), (
        "completed_regions must live in a persistent builder buffer"
    )
    graph.replay()
    assert pos_out.tolist() == md2.positions.tolist(), (
        "replayed graph must observe the step-2 positions"
    )
    live2 = md2.completed_regions[md2.completed_regions[:, 0] >= 0]
    live_out = regions_out[regions_out[:, 0] >= 0]
    assert live_out.tolist() == live2.tolist(), (
        "replayed graph must observe the step-2 completed regions"
    )
