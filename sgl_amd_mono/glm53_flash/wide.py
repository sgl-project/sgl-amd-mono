# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash's wide verify steps (``gfx950.wide.layer``): a step of R >= 3 MTP
requests runs one persistent launch a KDA + MoE layer, from the attention's mHC
read through the router's top-8 (and, past a run's first layer, the previous
layer's MoE all-reduce and mHC write-back), SGLang's routed experts between
launches, and SGLang's own path for the dense layers and each run's exit (the MoE
all-reduce and ``hc_post``). A DSA + MoE layer joins the run as two seam launches
around SGLang's DSA attention: the previous MoE's exit and the attention input,
then the attention's all-reduce, the FFN input and its route.

The KDA stage is SGLang's ring verify (``enable_linear_replayssm_spec``): it reads
the committed conv and recurrent state, writes every draft token's conv window and
its raw k / v / g / beta to the ReplaySSM ring, and the commit after the step folds
the accepted ones.
"""

from __future__ import annotations

import torch

from sgl_amd_mono.glm53_flash import HF_CONFIG, LINEAR_ATTN_CONFIG

# the widths a narrow launch runs: wider steps are the wide launch's
NARROW = 16
TP = 4
SUPPORTED_L = (6,)
# requests a bucket launch runs; a step of fewer pads up to the next bucket
REQ_BUCKETS = (3, 4, 6, 8, 12, 16, 24, 32)
HC = HF_CONFIG["hc_mult"]
HIDDEN = HF_CONFIG["hidden_size"]
TOPK_SLOTS = HF_CONFIG["num_experts_per_tok"] + HF_CONFIG["n_shared_experts"]

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/ops/layernorm/mhc.py": ("hc_contract",),
    "srt/distributed/communication_op.py": ("tensor_model_parallel_all_reduce",),
    "srt/environ.py": (
        "envs",
        "SGLANG_SIMULATE_UNIFORM_EXPERTS",
        "SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS",
    ),
    "srt/layers/attention/linear/utils.py": (
        "select_verify_intermediate_state_indices",
    ),
    "srt/layers/layer_boundary/adapters/attention.py": (
        "AttentionInputs",
        "get_attn_tp_context",
    ),
    "srt/layers/layer_boundary/residual/batch.py": (
        "start",
        "stream_of",
        "set_written",
        "complete_output",
        "final_norm",
    ),
    "srt/layers/moe/topk.py": (
        "TopKOutputFormat",
        "build_precomputed_topk_output",
        "is_expert_remapping_enabled",
    ),
    "srt/layers/zero_copy_context.py": ("set_moe_output",),
    "srt/model_executor/forward_context.py": ("get_attn_backend",),
    "srt/models/glm5_next.py": ("Glm5NextMoE",),
    "srt/utils/common.py": ("BumpAllocator",),
}


def widths(L: int) -> tuple:
    """The rows a wide step takes: every request count up to the largest bucket."""
    return tuple(L * r for r in range(1, max(REQ_BUCKETS) + 1) if L * r > NARROW)


def _say(msg):
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"[glm53 wide] {msg}", flush=True)


def is_covered(layer) -> bool:
    """A KDA layer with the routed MoE: the wide launch's layer."""
    from sglang.srt.models.glm5_next import Glm5NextMoE

    return bool(layer.is_linear_attn) and isinstance(layer.mlp, Glm5NextMoE)


def is_dsa_covered(layer) -> bool:
    """A DSA layer with the routed MoE: its seams run on the wide launch."""
    from sglang.srt.models.glm5_next import Glm5NextMoE

    return (
        not layer.is_linear_attn
        and isinstance(layer.mlp, Glm5NextMoE)
        and moe_reason(layer.mlp) is None
    )


def moe_reason(moe) -> str | None:
    """Why SGLang's routed experts cannot take the launch's (ids, weights), or None:
    the launch routes as SGLang's own top-k does for this model (sigmoid + bias,
    ungrouped top-8, renormalized, x routed_scaling_factor) and appends the fused
    shared expert at id 288, weight 1, so only a STANDARD output whose
    post-processing is a no-op qualifies."""
    try:
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.topk import (
            TopKOutputFormat,
            is_expert_remapping_enabled,
        )

        cfg = moe.topk.topk_config
        checks = {
            "output_format": cfg.output_format in (None, TopKOutputFormat.STANDARD),
            "sigmoid": cfg.scoring_func == "sigmoid",
            "bias": cfg.correction_bias is not None,
            "ungrouped": (cfg.num_expert_group or 1) == 1
            and (cfg.topk_group or 1) == 1,
            "fused_shared": moe.num_fused_shared_experts == 1,
            "shared_weight": cfg.fused_shared_experts_scaling_factor in (None, 1.0),
            "no_eplb_remap": not is_expert_remapping_enabled(),
            "no_sim": not envs.SGLANG_SIMULATE_UNIFORM_EXPERTS.get()
            and not envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS.get(),
        }
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"
    bad = [k for k, v in checks.items() if not v]
    return ", ".join(bad) if bad else None


def _router(i, moe, copies):
    """The router in bf16 and its bias in f32, as the launch reads them."""
    gate = moe.gate
    w_r = gate.weight
    if w_r.dtype != torch.bfloat16:
        r16 = w_r.to(torch.bfloat16)
        if not torch.equal(r16.float(), w_r):
            raise ValueError("needs router weights exactly representable in bf16")
        w_r = copies["wide router bf16", i] = r16.contiguous()
    bias = gate.e_score_correction_bias
    if bias.dtype != torch.float32:
        bias = copies["wide router bias f32", i] = bias.float().contiguous()
    return w_r.contiguous(), bias.contiguous()


def _ffn_mhc(module):
    ffn = (
        module.hc_ffn_fn,
        module.hc_ffn_scale,
        module.hc_ffn_base,
        module.post_attention_layernorm.weight,
    )
    assert all(t.dtype == torch.float32 for t in ffn[:3])
    return ffn


class _Layer:
    """A covered layer's weights as the launch reads them: views of SGLang's
    parameters (``weights.kda_weights``), the router in bf16 and its bias in f32."""

    def __init__(self, i, module, copies):
        from sgl_amd_mono.glm53_flash.gfx950.weights import kda_weights

        self.i, self.module, self.moe = i, module, module.mlp
        a = kda_weights(module, i, TP, copies)
        attn = module.self_attn
        self.onorm_eps = float(getattr(attn.o_norm, "eps", HF_CONFIG["rms_norm_eps"]))
        # the launch's f_b | g_b: SGLang's [2, P, D] parameter, which a.w_fb / w_gb view
        w_fgb = attn.fused_fg_b_proj.weight
        assert a.w_fb.data_ptr() == w_fgb.data_ptr() and w_fgb.is_contiguous()
        f32 = torch.float32
        self.ptr_a = (
            a.hc_fn,
            a.hc_scale,
            a.hc_base,
            a.g_norm,
            a.w_in,
            a.conv_w,
            w_fgb,
            a.a_log,
            a.dt_bias,
            a.w_on,
            a.w_o,
        )
        self.ffn = _ffn_mhc(module)
        self.w_r, self.bias = _router(i, self.moe, copies)
        self.cache = None  # (conv, iw, state, rawv, rawk, g, beta) bound at first use


class _DsaLayer:
    """A DSA + MoE layer's seams: the attention and FFN mHC (views of SGLang's
    parameters) and the router."""

    def __init__(self, i, module, copies):
        self.i, self.module, self.moe = i, module, module.mlp
        attn = (
            module.hc_attn_fn,
            module.hc_attn_scale,
            module.hc_attn_base,
            module.input_layernorm.weight,
        )
        assert all(t.dtype == torch.float32 for t in attn[:3])
        assert attn[3].dtype == torch.bfloat16 and attn[3].shape == (HIDDEN,)
        self.attn, self.ffn = attn, _ffn_mhc(module)
        self.w_r, self.bias = _router(i, self.moe, copies)


class Wide:
    """One rank's wide launches: per bucket, the T1 and non-T1 builds and their
    buffers; ``backbone`` runs a step."""

    def __init__(self, model, rank, group, device, L, state_bf16, copies):
        assert L in SUPPORTED_L, L
        self.lm = model
        self.bb = bb = model.model
        self.rank, self.group, self.device, self.L = rank, group, device, L
        self.state_bf16 = state_bf16
        idx = [
            i for i in range(bb.start_layer, bb.end_layer) if is_covered(bb.layers[i])
        ]
        self.layers = {i: _Layer(i, bb.layers[i], copies) for i in idx}
        didx = [
            i
            for i in range(bb.start_layer, bb.end_layer)
            if is_dsa_covered(bb.layers[i])
        ]
        self.dsa = {i: _DsaLayer(i, bb.layers[i], copies) for i in didx}
        eps = {ly.onorm_eps for ly in self.layers.values()}
        assert len(eps) == 1, eps
        self.onorm_eps = eps.pop()
        self.lower_bound = float(LINEAR_ATTN_CONFIG["gate_lower_bound"])
        self.last = bb.end_layer - 1
        self.buckets = tuple(L * r for r in REQ_BUCKETS if L * r > NARROW)
        self.bufs, self.runs = {}, {}
        self.kda_ready = False
        # whether aiter wrote the launch's buffer, set by the first experts call
        self.moe_out = None
        runs, cur = [], []
        for i in range(bb.start_layer, bb.end_layer):
            if i in self.layers or i in self.dsa:
                cur.append(i)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        _say(
            f"{len(idx)} covered layers and {len(didx)} DSA layers' seams in {len(runs)} runs "
            f"({', '.join(f'{r[0]}..{r[-1]}' for r in runs)}), {L} tokens a request, "
            f"buckets {self.buckets}, state {'bf16' if state_bf16 else 'f32'}"
        )

    # ------------------------------------------------------------ per bucket
    # (kind, t1): a KDA layer's launch, a DSA layer's seam before ("a") and after
    # ("b") its attention
    KINDS = (("kda", False), ("kda", True), ("a", False), ("a", True), ("b", True))

    def _build(self, b, kind, t1):
        from sgl_amd_mono.glm53_flash.gfx950.wide import layer as lay

        body, stop = (
            (True, "route")
            if kind == "kda"
            else (False, "agate" if kind == "a" else "route")
        )
        return lay.build(
            b,
            self.L,
            TP,
            self.lower_bound,
            self.onorm_eps,
            self.state_bf16,
            True,
            stop,
            shared_w=1.0,
            t1=t1,
            moe_sort=stop == "route",
            body=body,
        )

    def _kinds(self):
        return self.KINDS if self.dsa else self.KINDS[:2]

    def _ensure(self, b):
        """Bucket b's launches, collective state and activations (the peer buffers'
        handshake is collective: every rank reaches it on the same step)."""
        if b in self.bufs:
            return self.bufs[b]
        from sgl_amd_mono.common.peer import PeerBuffer
        from sgl_amd_mono.glm53_flash.gfx950.config import MOE_SLOTS, N_EXPERTS
        from sgl_amd_mono.glm53_flash.gfx950.wide import kda, sort

        assert MOE_SLOTS == TOPK_SLOTS
        dev, bf, f32, i32 = self.device, torch.bfloat16, torch.float32, torch.int32

        def z(*s, dt=bf):
            return torch.zeros(*s, dtype=dt, device=dev)

        # one epoch, scratch and peer buffer a build: a build's counters and flags
        # move on once a launch of it, and the two builds' layouts differ
        var = {}
        for kind, t1 in self._kinds():
            launch = self._build(b, kind, t1)
            pl = launch.plan
            pb = PeerBuffer(pl["peer_bytes"], self.group, self.rank, TP, dev)
            pb.bytes.zero_()
            var[kind, t1] = dict(
                launch=launch,
                peers=pb,
                epoch=z(pl["epoch_words"], dt=i32),
                scratch=z(pl["scratch_words"], dt=i32),
            )

        def out():
            return dict(
                res_mid=z(b, HC, HIDDEN),
                x_f=z(b, HIDDEN),
                post_f=z(b, HC, dt=f32),
                comb_f=z(b, HC * HC, dt=f32),
                ids=z(b, MOE_SLOTS, dt=i32),
                wts=z(b, MOE_SLOTS, dt=f32),
                logits=z(b, N_EXPERTS, dt=f32),
                moe=z(b, HIDDEN),
                res_cur=z(b, HC, HIDDEN),
            )

        R = b // self.L
        # the launch's MoE sort (AITER's BM16 aux layout), at rest all pad
        sly = sort.layout(b)
        sbuf = torch.empty(sly["nbytes"], dtype=torch.uint8, device=dev)
        sort.init(sbuf, b)
        bufs = dict(
            var=var,
            r_in=z(b, HC, HIDDEN),
            out=[out(), out()],
            sbuf=sbuf,
            sly=sly,
            x_a=z(b, HIDDEN),
            post_a=z(b, HC, dt=f32),
            comb_a=z(b, HC * HC, dt=f32),
            proj=z(b, kda.NPROJ),
            core=z(b, kda.P),
            acc_o=z(b, HIDDEN),
            slots=torch.full((R,), -1, dtype=i32, device=dev),
            iw_idx=torch.full((R,), -1, dtype=i32, device=dev),
            ring_idx=torch.full((R,), -1, dtype=i32, device=dev),
            # a DSA layer's attention input, its mixes and the streams it reads
            dsa=dict(
                x=z(b, HIDDEN),
                post=z(b, HC, dt=f32),
                comb=z(b, HC * HC, dt=f32),
                res=z(b, HC, HIDDEN),
            ),
        )
        self.bufs[b] = bufs
        peers = ", ".join(
            f"{k[0]} t1={int(k[1])} {v['peers'].bytes.numel() / 1e6:.1f} MB"
            for k, v in var.items()
        )
        _say(f"bucket {b}: peers {peers}")
        return bufs

    # ------------------------------------------------------------ KDA state
    def _kda_step(self, fb, R, bufs):
        """The step's KDA rows (device work only: a graph replays it): request r's
        state slot, its verify window row and its ring row; pad requests -1."""
        from sglang.srt.layers.attention.linear.utils import (
            select_verify_intermediate_state_indices,
        )
        from sglang.srt.model_executor.forward_context import get_attn_backend

        kb = get_attn_backend().linear_attn_backend
        fm = kb.forward_metadata
        slots = fm.mamba_cache_indices[:R]
        vrows = select_verify_intermediate_state_indices(
            kb.verify_intermediate_state_indices,
            fb.req_pool_indices,
            lambda: slots >= 0,
            kb.req_to_token_pool.size,
        )[:R]
        ring = slots
        if kb.state_pool is not None and kb.state_pool.replayssm_request_ring:
            ring = fm.replayssm_ring_indices[:R]
        for k, v in (("slots", slots), ("iw_idx", vrows), ("ring_idx", ring)):
            bufs[k].fill_(-1)
            bufs[k][:R] = v.to(torch.int32)
        if not self.kda_ready:
            self.kda_ready = True
            for ly in self.layers.values():
                self._bind(ly, kb)

    def _bind(self, ly, kb):
        c = kb.state_layer_cache(ly.i)
        if getattr(c, "replayssm_rawv", None) is None:
            raise RuntimeError("the wide launch runs on the ReplaySSM ring")
        conv = c.conv[0].transpose(1, 2)  # [slots, 3P, 3]
        iw = c.intermediate_conv_window[0].transpose(2, 3)  # [rows, L, 3P, 3]
        state = c.temporal
        ring = (c.replayssm_rawv, c.replayssm_rawk, c.replayssm_g, c.replayssm_beta)
        P3 = ly.ptr_a[5].shape[0]
        want_state = torch.bfloat16 if self.state_bf16 else torch.float32
        ok = (
            conv.dtype == torch.bfloat16
            and tuple(conv.shape[1:]) == (P3, 3)
            and conv.stride(1) == 1
            and tuple(iw.shape[1:]) == (self.L, P3, 3)
            and iw.stride(2) == 1
            and state.dtype == want_state
            and state.stride()[1:]
            == (state.shape[2] * state.shape[3], state.shape[3], 1)
            and ring[0].shape[-2] >= self.L
            and all(t.stride(-1) == 1 for t in ring)
        )
        if not ok:
            raise RuntimeError(
                f"GLM-5.3-Flash wide: layer {ly.i}: KDA cache conv {conv.dtype} "
                f"{tuple(conv.shape)} {conv.stride()}, window {tuple(iw.shape)} "
                f"{iw.stride()}, state {state.dtype} {tuple(state.shape)}, ring "
                f"{[tuple(t.shape) for t in ring]}"
            )
        ly.cache = (conv, iw, state, *ring)

    # ------------------------------------------------------------ one launch
    def _launch(self, ly, b, bufs, t1, prev, cur, r_in, m):
        v = bufs["var"]["kda", t1]
        conv, iw, state, rawv, rawk, g, beta = ly.cache
        (
            hc_fn,
            hc_scale,
            hc_base,
            g_norm,
            w_in,
            conv_w,
            w_fgb,
            a_log,
            dt_bias,
            on_w,
            w_o,
        ) = ly.ptr_a
        sl, iwi, ri = bufs["slots"], bufs["iw_idx"], bufs["ring_idx"]
        ints = (
            conv.stride(0),
            conv.stride(2),
            iw.stride(0),
            iw.stride(1),
            iw.stride(3),
            rawv.stride(0),
            rawk.stride(0),
            g.stride(0),
            beta.stride(0),
            rawv.shape[-2],
        )
        ptrs = (
            r_in,
            hc_fn,
            hc_scale,
            hc_base,
            g_norm,
            bufs["x_a"],
            bufs["post_a"],
            bufs["comb_a"],
            w_in,
            bufs["proj"],
            conv_w,
            conv,
            sl,
            iw,
            iwi,
            w_fgb,
            a_log,
            dt_bias,
            state,
            sl,
            rawv,
            rawk,
            g,
            beta,
            ri,
            on_w,
            bufs["core"],
            w_o,
            bufs["acc_o"],
            *ly.ffn,
            cur["res_mid"],
            cur["x_f"],
            cur["post_f"],
            cur["comb_f"],
            ly.w_r,
            ly.bias,
            cur["ids"],
            cur["wts"],
            cur["logits"],
            prev["moe"],
            prev["post_f"],
            prev["comb_f"],
            cur["res_cur"],
            v["epoch"],
            v["scratch"],
        )
        v["launch"](
            *[t.data_ptr() for t in ptrs],
            v["peers"].addresses.data_ptr(),
            self.rank,
            *ints,
            bufs["sbuf"].data_ptr(),
            cur["moe"].data_ptr(),
            m,
            stream=torch.cuda.current_stream(),
        )

    def _seam(self, bufs, kind, t1, mhc4, out4, r_in, prev, ly, cur, m):
        """A DSA layer's seam launch: [t1: fold prev's partial (prev["moe"], post_f,
        comb_f) into r_in] mHC pre with ``mhc4`` (fn, scale, base, norm) -> out4 (x,
        post, comb, streams); kind "b" routes x with ly's router into cur's ids /
        weights, sorts and zeroes cur["moe"]."""
        v = bufs["var"][kind, t1]
        x, post, comb, res = out4
        p = lambda t: 0 if t is None else t.data_ptr()  # noqa: E731
        route = kind == "b"
        ptrs = (
            [p(r_in), *[p(t) for t in mhc4], p(x), p(post), p(comb)]
            + [0] * 21  # in_proj, KDA, o_proj: no body
            + [0] * 8  # FFN mHC, R_mid, x_f, post_f, comb_f: no body
            + (
                [p(ly.w_r), p(ly.bias), p(cur["ids"]), p(cur["wts"]), p(cur["logits"])]
                if route
                else [0] * 5
            )
            + [
                p(prev["moe"]) if t1 else 0,
                p(prev["post_f"]) if t1 else 0,
                p(prev["comb_f"]) if t1 else 0,
                p(res),
            ]
            + [p(v["epoch"]), p(v["scratch"])]
        )
        assert len(ptrs) == 48, len(ptrs)
        v["launch"](
            *ptrs,
            v["peers"].addresses.data_ptr(),
            self.rank,
            *([0] * 10),
            bufs["sbuf"].data_ptr() if route else 0,
            p(cur["moe"]) if route else 0,
            m,
            stream=torch.cuda.current_stream(),
        )

    def _attention(self, dl, positions, x, fb, zero_allocator):
        """SGLang's DSA attention on the seam's x: this rank's o_proj partial."""
        from sglang.srt.layers.layer_boundary.adapters.attention import (
            AttentionInputs,
            get_attn_tp_context,
        )

        attn = dl.module.self_attn
        ctx = get_attn_tp_context()
        ctx.set_attn_inputs(AttentionInputs(x, fb, attn.prepare_qkv_latent))
        try:
            o = attn(
                positions=positions,
                hidden_states=x,
                forward_batch=fb,
                zero_allocator=zero_allocator,
                input_on_attn_tp_slices=False,
                prev_topk_indices=None,
            )
        finally:
            ctx.clear_attn_inputs()
        if isinstance(o, tuple):
            o = o[0]
        assert o.dtype == torch.bfloat16 and o.shape == (x.shape[0], HIDDEN), (
            o.dtype,
            o.shape,
        )
        return o.contiguous()

    def _experts(self, ly, cur, rows, bufs):
        """SGLang's routed experts (the fused shared expert at slot 8) on the
        launch's FFN input and (ids, weights): this rank's partial -> cur["moe"], which
        aiter writes directly (published by ``set_moe_output``), from the launch's sort
        and zeroing (``register_presorted_moe``): only the two GEMMs run. An aiter
        runner that does not take it zeroes its own output and the partial is copied in."""
        from aiter.presorted_moe import register_presorted_moe
        from sglang.srt.layers.moe.topk import build_precomputed_topk_output
        from sglang.srt.layers.zero_copy_context import set_moe_output

        from sgl_amd_mono.glm53_flash.gfx950.config import N_EXPERTS
        from sgl_amd_mono.glm53_flash.gfx950.wide import sort

        moe = ly.moe
        out = cur["moe"][:rows]
        topk = build_precomputed_topk_output(
            cur["wts"][:rows], cur["ids"][:rows], moe.topk.topk_config, ly.i
        )
        register_presorted_moe(
            dict(
                sort.views(bufs["sbuf"], bufs["sly"], rows),
                topk_ids_ptr=cur["ids"].data_ptr(),
                topk_weights_ptr=cur["wts"].data_ptr(),
                M=rows,
                block_m=sort.BM,
                num_experts=N_EXPERTS + 1,
                topk=TOPK_SLOTS,
                moe_buf_zeroed=True,
                moe_buf_ptr=out.data_ptr(),
            )
        )
        with set_moe_output(out):
            y = moe.experts(cur["x_f"][:rows], topk)
        direct = y.data_ptr() == out.data_ptr()
        if self.moe_out is None:
            self.moe_out = direct
            _say(f"experts {'write the launch buffer' if direct else 'output copied'}")
        if not direct:
            out.copy_(y)

    # ------------------------------------------------------------ the step
    def warmup(self):
        """Build and run every bucket's launches once on dummy activations with no
        KDA rows (every index -1), in one order on every rank, before any step."""
        import time

        t0 = time.time()
        dev = self.device
        ly = next(iter(self.layers.values()))
        P3 = ly.ptr_a[5].shape[0]
        H, D = ly.ptr_a[7].shape[0], 128
        st = torch.bfloat16 if self.state_bf16 else torch.float32
        bf, f32 = torch.bfloat16, torch.float32
        dummy = (
            torch.zeros(1, 3, P3, dtype=bf, device=dev).transpose(1, 2),
            torch.zeros(1, self.L, 3, P3, dtype=bf, device=dev).transpose(2, 3),
            torch.zeros(1, H, D, D, dtype=st, device=dev),
            torch.zeros(1, H, self.L, D, dtype=bf, device=dev),
            torch.zeros(1, H, self.L, D, dtype=bf, device=dev),
            torch.zeros(1, H, self.L, D, dtype=f32, device=dev),
            torch.zeros(1, H, self.L, dtype=f32, device=dev),
        )
        saved = ly.cache
        ly.cache = dummy
        try:
            for b in self.buckets:
                bufs = self._ensure(b)
                o0, o1 = bufs["out"]
                self._launch(ly, b, bufs, False, o1, o0, bufs["r_in"], b)
                self._launch(ly, b, bufs, True, o0, o1, o0["res_mid"], b)
                if self.dsa:
                    dl, d = next(iter(self.dsa.values())), bufs["dsa"]
                    out4 = (d["x"], d["post"], d["comb"], d["res"])
                    self._seam(
                        bufs, "a", False, dl.attn, out4, bufs["r_in"], None, dl, None, b
                    )
                    self._seam(
                        bufs, "a", True, dl.attn, out4, o1["res_mid"], o1, dl, None, b
                    )
                    fin = dict(moe=d["x"], post_f=d["post"], comb_f=d["comb"])
                    out4 = (o0["x_f"], o0["post_f"], o0["comb_f"], o0["res_mid"])
                    self._seam(bufs, "b", True, dl.ffn, out4, d["res"], fin, dl, o0, b)
            torch.cuda.synchronize()
        finally:
            ly.cache = saved
        n = len(self._kinds()) * len(self.buckets)
        _say(f"warmup: {n} launches built and run in {time.time() - t0:.0f} s")

    def backbone(self, input_ids, positions, fb):
        """``Glm5NextModel.forward`` with each run of covered layers on the wide
        launch: the final norm's rows."""
        from sglang.kernels.ops.layernorm.mhc import hc_contract
        from sglang.srt.distributed import tensor_model_parallel_all_reduce
        from sglang.srt.layers.layer_boundary.residual.batch import (
            complete_output,
            final_norm,
            set_written,
            start,
            stream_of,
        )
        from sglang.srt.utils.common import BumpAllocator

        m = self.bb
        hidden = m.embed_tokens(input_ids)
        rows = hidden.shape[0]
        L = int(fb.spec_info.draft_token_num)
        assert L == self.L and rows % L == 0, (rows, L, self.L)
        R = rows // L
        b = next(x for x in self.buckets if x >= rows)
        bufs = self._ensure(b)
        if not getattr(self, "_said", False):
            self._said = True
            _say(f"first wide step: rows {rows} (bucket {b})")
        start(fb)
        dev = hidden.device
        zero_allocator = BumpAllocator(
            buffer_size=2 * (m.end_layer - m.start_layer),
            dtype=torch.float32,
            device=dev,
        )
        gz = (
            BumpAllocator(
                buffer_size=m.gemm_output_zero_allocator_size,
                dtype=torch.float32,
                device=dev,
            )
            if getattr(m, "gemm_output_zero_allocator_size", 0) > 0
            else None
        )
        self._kda_step(fb, R, bufs)
        topk_indices = None
        i, ping = m.start_layer, 0
        while i < m.end_layer:
            if i not in self.layers and i not in self.dsa:
                hidden, topk_indices = m.layers[i](
                    positions,
                    hidden,
                    fb,
                    zero_allocator,
                    gz,
                    prev_topk_indices=topk_indices,
                )
                i += 1
                continue
            # a run of covered layers, from the written streams (the first launch reads
            # SGLang's rows in place; the bucket's pad rows read 0)
            if stream_of(fb).pending is not None:
                raise RuntimeError(
                    f"wide run at layer {i}: a residual update is still pending"
                )
            r0 = hidden.view(rows, HC, HIDDEN)
            if not r0.is_contiguous():
                r0 = bufs["r_in"][:rows].copy_(r0)
            run = []
            while i in self.layers or i in self.dsa:
                run.append(i)
                i += 1
            d = bufs["dsa"]
            for n, li in enumerate(run):
                prev, cur = bufs["out"][ping], bufs["out"][1 - ping]
                r_in = r0 if n == 0 else prev["res_mid"]
                if li in self.layers:
                    ly = self.layers[li]
                    self._launch(ly, b, bufs, n > 0, prev, cur, r_in, rows)
                else:
                    # seam a, SGLang's attention, seam b (the attention's all-reduce)
                    ly = self.dsa[li]
                    self._seam(
                        bufs,
                        "a",
                        n > 0,
                        ly.attn,
                        (d["x"], d["post"], d["comb"], d["res"]),
                        r_in,
                        prev,
                        ly,
                        None,
                        rows,
                    )
                    o = self._attention(
                        ly, positions, d["x"][:rows], fb, zero_allocator
                    )
                    self._seam(
                        bufs,
                        "b",
                        True,
                        ly.ffn,
                        (cur["x_f"], cur["post_f"], cur["comb_f"], cur["res_mid"]),
                        d["res"] if n > 0 else r_in,
                        dict(moe=o, post_f=d["post"], comb_f=d["comb"]),
                        ly,
                        cur,
                        rows,
                    )
                self._experts(ly, cur, rows, bufs)
                ping = 1 - ping
            # leave: the last layer's MoE all-reduce and mHC write-back on SGLang
            last = self.layers.get(run[-1]) or self.dsa[run[-1]]
            o = bufs["out"][ping]
            x = tensor_model_parallel_all_reduce(o["moe"][:rows])
            hidden = last.module.hc_post(
                x,
                o["res_mid"][:rows].view(rows, -1),
                o["comb_f"][:rows],
                o["post_f"][:rows],
            )
            if run[-1] == self.last:
                hidden = hc_contract(hidden, HC)
            hidden = set_written(hidden, fb)
            topk_indices = None
        hidden = complete_output(hidden, fb)
        return final_norm(hidden, fb, m.norm)
