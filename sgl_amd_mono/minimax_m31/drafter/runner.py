# SPDX-License-Identifier: Apache-2.0
"""Host side of the DSpark draft launch: one rank's layer-packed MXFP8 weights, the
per-layer pointer table, the scratch, the peer buffer and a step's launch."""

from __future__ import annotations

import torch

from sgl_amd_mono.minimax_m31.drafter import layout

# (shape a rank, the checkpoint projections stacked along N in order)
GEMV = {
    "qkv": ((2304, 6144), ("q_proj", "k_proj", "v_proj")),
    "o": ((6144, 2048), ("o_proj",)),
    "gu": ((6144, 6144), ("gate_proj", "up_proj")),
    "d": ((6144, 3072), ("down_proj",)),
}
# the TP-sharded dimension of each checkpoint projection: 0 rows (column-parallel),
# 1 the K columns (row-parallel)
SHARD_DIM = {
    "q_proj": 0,
    "k_proj": 0,
    "v_proj": 0,
    "o_proj": 1,
    "gate_proj": 0,
    "up_proj": 0,
    "down_proj": 1,
}
WAIT_S = 10.0


def shard(
    tensor: torch.Tensor, proj: str, kind: str, rank: int, tp: int
) -> torch.Tensor:
    """This rank's part of a checkpoint projection's weight or its e8m0 scales."""
    t = tensor.view(torch.uint8)
    dim = SHARD_DIM[proj]
    n = t.shape[dim] // tp
    return t.narrow(dim, rank * n, n)


class DraftRunner:
    """The draft launch of one TP4 rank. ``weights[layer][(proj, kind)]``: this rank's
    checkpoint shards (uint8). The K / V caches are bound at the first eager call."""

    def __init__(
        self, model, weights, rank, cpu_group, device, deadline_s=WAIT_S, timeline=False
    ):
        from sgl_amd_mono.common.peer import PeerBuffer
        from sgl_amd_mono.minimax_m31.drafter.layer import build_drafter
        from sgl_amd_mono.minimax_m31.runner import _writable_flydsl_cache

        _writable_flydsl_cache()
        dev = torch.device(device)
        self.rank, self.model = rank, model
        self.packed = {}
        for name, ((n, k), projs) in GEMV.items():
            w = torch.stack(
                [
                    torch.cat([weights[i][p, "weight"] for p in projs])
                    for i in range(layout.LAYERS)
                ]
            )
            s = torch.stack(
                [
                    torch.cat([weights[i][p, "weight_scale_inv"] for p in projs])
                    for i in range(layout.LAYERS)
                ]
            )
            assert tuple(w.shape[1:]) == (n, k) and tuple(s.shape[1:]) == (
                n,
                k // 32,
            ), (name, w.shape, s.shape)
            self.packed[name] = (w.to(dev).contiguous(), s.to(dev).contiguous())
        layers = model.layers
        self.norms = [
            [
                d.input_layernorm.weight,
                d.post_attention_layernorm.weight,
                d.self_attn.q_norm.weight,
                d.self_attn.k_norm.weight,
            ]
            for d in (L.decoder_layer for L in layers)
        ]
        for ws in self.norms:
            assert all(w.dtype == torch.bfloat16 and w.is_contiguous() for w in ws)
        self.final_norm = model.norm.weight
        cs = layers[0].self_attn.rotary_emb.cos_sin_cache
        assert cs.dim() == 2 and cs.shape[1] == layout.ROT, tuple(cs.shape)
        self.cos_sin = cs.to(device=dev, dtype=torch.bfloat16).contiguous()
        self.layer_ids = [L.self_attn.attn.layer_id for L in layers]
        nbytes = layout.SCRATCH_BYTES + (layout.TL_BYTES if timeline else 0)
        self.scratch = torch.zeros(nbytes, dtype=torch.uint8, device=dev)
        self.peers = PeerBuffer(layout.SYM_BYTES, cpu_group, rank, 4, dev)
        self.peers.bytes.zero_()
        self.pe = [int(a) for a in self.peers.addresses.tolist()]
        self.health = torch.zeros(4, dtype=torch.int32, device=dev)
        self.step = torch.zeros(1, dtype=torch.int32, device=dev)
        self.out = torch.empty(
            layout.MAX_ROWS, layout.HIDDEN, dtype=torch.bfloat16, device=dev
        )
        self.launch = build_drafter(wait_ticks=int(deadline_s * 1e8), timeline=timeline)
        self.tab = None
        self._caches = None

    def _bind(self, pool) -> None:
        """The pointer table of the pool's draft K / V caches (once; checked a call)."""
        caches = [pool.get_kv_buffer(i) for i in self.layer_ids]
        for k, v in caches:
            for t in (k, v):
                if not (
                    t.dtype == torch.bfloat16
                    and t.is_contiguous()
                    and t.dim() == 3
                    and tuple(t.shape[1:]) == (1, layout.HD)
                ):
                    raise RuntimeError(
                        f"M3.1 draft KV cache {t.dtype} {tuple(t.shape)}"
                    )
        tab = torch.zeros(layout.LAYERS, 16, dtype=torch.int64)
        for i, ((k, v), ws) in enumerate(zip(caches, self.norms)):
            for j, w in enumerate(ws):
                tab[i, j] = w.data_ptr()
            tab[i, layout.T_K_CACHE] = k.data_ptr()
            tab[i, layout.T_V_CACHE] = v.data_ptr()
            tab[i, layout.T_SLOTS] = k.shape[0]
        self.tab = tab.to(self.out.device)
        self._caches = [(k.data_ptr(), v.data_ptr()) for k, v in caches]

    def forward(self, x, positions, forward_batch, backend) -> torch.Tensor:
        """The 5 layers and the final norm of a draft step: (rows, 6144) bf16."""
        pool = backend.token_to_kv_pool
        if self._caches is None:
            self._bind(pool)
        elif self._caches != [
            (k.data_ptr(), v.data_ptr())
            for k, v in (pool.get_kv_buffer(i) for i in self.layer_ids)
        ]:
            raise RuntimeError(
                "M3.1 draft: the KV caches moved since the launch bound them"
            )
        r2t = backend.req_to_token_pool.req_to_token
        rpi, seq, slots = (
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.out_cache_loc,
        )
        rows = x.shape[0]
        assert rows in (layout.QPR, 2 * layout.QPR), rows
        assert (
            x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.shape[1] == layout.HIDDEN
        )
        assert r2t.dtype == torch.int32 and r2t.is_contiguous()
        for t in (positions, rpi, seq, slots):
            assert t.dtype == torch.int64, t.dtype
        pk = self.packed
        out = self.out[:rows]
        self.launch(
            x.data_ptr(), out.data_ptr(), self.tab.data_ptr(),
            pk["qkv"][0].data_ptr(), pk["qkv"][1].data_ptr(), pk["o"][0].data_ptr(), pk["o"][1].data_ptr(),
            pk["gu"][0].data_ptr(), pk["gu"][1].data_ptr(), pk["d"][0].data_ptr(), pk["d"][1].data_ptr(),
            self.final_norm.data_ptr(), self.cos_sin.data_ptr(), r2t.data_ptr(), rpi.data_ptr(),
            positions.data_ptr(), slots.data_ptr(), seq.data_ptr(), self.scratch.data_ptr(),
            int(self.peers.local), *self.pe, self.step.data_ptr(), self.health.data_ptr(),
            self.rank, r2t.shape[1], r2t.shape[0], rows,
            stream=torch.cuda.current_stream(),
        )  # fmt: skip
        self.step.add_(1)
        return out
