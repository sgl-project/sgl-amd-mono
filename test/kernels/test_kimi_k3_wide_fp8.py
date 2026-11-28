# SPDX-License-Identifier: Apache-2.0
"""GPU kernel checks for Kimi-K3 FP8 wide KDA layer, bf16 state, non-ReplaySSM, split K1,
and the narrow mla_pre on bf16 / Quark FP8 weights without an indexer."""

import os
import subprocess
import sys
import unittest

import torch


def _gfx950_gpus() -> int:
    if not torch.cuda.is_available():
        return 0
    if not (torch.cuda.get_device_properties(0).gcnArchName or "").startswith("gfx950"):
        return 0
    return torch.cuda.device_count()


def _quant_fp8_per_channel(w_bf16: torch.Tensor):
    wf = w_bf16.float()
    scale = (wf.abs().amax(dim=1) / 448.0).clamp(min=1e-12).contiguous()
    w_q = (
        (wf / scale[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
    )
    w_deq = (w_q.float() * scale[:, None]).to(torch.bfloat16).contiguous()
    return w_q, scale, w_deq


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(1, -1), b.float().reshape(1, -1)
    ).item()


@unittest.skipIf(_gfx950_gpus() < 1, "needs gfx950 GPU")
class TestKimiK3Fp8SingleGpu(unittest.TestCase):
    def test_preshuffle_matches_aiter(self):
        from aiter.ops.shuffle import shuffle_weight

        from sgl_amd_mono.kimi_k3.gfx950.stages import wfmt

        torch.manual_seed(0)
        raw = torch.randn(128, 7168, device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        got = wfmt.preshuffle_fp8(raw)
        ref = shuffle_weight(raw, layout=(16, 16))
        self.assertTrue(
            torch.equal(got.view(torch.uint8), ref.view(torch.uint8)),
            "preshuffle_fp8 differs from aiter shuffle_weight",
        )
        self.assertTrue(
            torch.equal(
                wfmt.unshuffle_fp8(ref).view(torch.uint8), raw.view(torch.uint8)
            )
        )

    def test_k1_split_fab_matches_merged_bit_for_bit(self):
        from sgl_amd_mono.kimi_k3.gfx950.attention import kda
        from sgl_amd_mono.kimi_k3.gfx950.stages import wfmt
        from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

        dev = torch.device("cuda:0")
        torch.manual_seed(42)
        rows = 8
        bf, f32 = torch.bfloat16, torch.float32
        w_in_bf = torch.randn(kda.NPROJ, HIDDEN, device=dev, dtype=bf) * 0.02
        w_q, scale, _ = _quant_fp8_per_channel(w_in_bf)
        w_sh = wfmt.preshuffle_fp8(w_q)
        w_qkvg = w_sh[: kda.FA_0].contiguous()
        w_fab = w_sh[kda.FA_0 :].contiguous()

        prefix = torch.randn(rows, HIDDEN, device=dev, dtype=bf) * 0.5
        blocks = torch.randn(rows, 3, HIDDEN, device=dev, dtype=bf) * 0.5
        ares_nw = torch.ones(HIDDEN, device=dev, dtype=bf)
        ares_qk = torch.randn(HIDDEN, device=dev, dtype=bf) * 0.1
        in_nw = torch.ones(HIDDEN, device=dev, dtype=bf)
        w_fb = torch.randn(kda.PROJ, kda.HD, device=dev, dtype=bf) * 0.02
        conv_w = torch.randn(3 * kda.PROJ, kda.CONV_W, device=dev, dtype=f32) * 0.1
        conv_st = (
            torch.randn(2, 3 * kda.PROJ, rows + kda.CONV_W - 1, device=dev, dtype=bf)
            * 0.2
        )
        a_log = torch.randn(kda.NH, device=dev, dtype=f32) * 0.1
        dt_bias = torch.randn(kda.PROJ, device=dev, dtype=f32) * 0.1
        on_w = torch.ones(kda.HD, device=dev, dtype=bf)

        for ring in (True, False):
            common = dict(
                tokens=rows,
                qlen=rows,
                nblocks=2,
                delta=False,
                state_len=rows + kda.CONV_W - 1,
                eps=1e-5,
                out_eps=1e-5,
                onorm_eps=1e-5,
                lower_bound=-5.0,
                write_idx=-1,
                ring=ring,
                init_committed=not ring,
                wfmt=kda.W_FP8_SHUFFLED,
                state_bf16=True,
            )
            k_merged = kda.KdaPreBuild(**common, split_fab=False)
            k_split = kda.KdaPreBuild(**common, split_fab=True)

            def run(key, wi, wfab):
                rstate = (
                    torch.randn(
                        1 if ring else rows,
                        kda.NH,
                        kda.HD,
                        kda.HD,
                        device=dev,
                        dtype=bf,
                    )
                    * 0.1
                )
                rstate0 = (
                    torch.randn(1, kda.NH, kda.HD, kda.HD, device=dev, dtype=bf) * 0.1
                    if not ring
                    else None
                )
                # fixed seeds for identical initial states
                torch.manual_seed(99)
                if ring:
                    rstate.normal_(0, 0.1)
                else:
                    rstate.zero_()
                    rstate0.normal_(0, 0.1)
                ring_bufs = (
                    tuple(
                        torch.zeros(1, kda.NH, rows, *tail, dtype=dt, device=dev)
                        for tail, dt in (
                            ((kda.HD,), bf),
                            ((kda.HD,), bf),
                            ((kda.HD,), f32),
                            ((), f32),
                        )
                    )
                    if ring
                    else None
                )
                st_idx = (
                    torch.zeros(1, rows, dtype=torch.int32, device=dev)
                    if ring
                    else torch.arange(rows, dtype=torch.int32, device=dev).view(1, rows)
                )
                core = torch.empty(rows, kda.PROJ, dtype=bf, device=dev)
                scratch = torch.zeros(
                    kda.scratch_bytes(key), dtype=torch.uint8, device=dev
                )
                epoch = torch.tensor([1], dtype=torch.int32, device=dev)
                kda.kda_pre(
                    key,
                    prefix=prefix,
                    delta=None,
                    blocks=blocks,
                    ares_nw=ares_nw,
                    ares_qk=ares_qk,
                    in_nw=in_nw,
                    w_in=wi,
                    w_in_scale=scale,
                    w_in_fab=wfab,
                    w_fb=w_fb,
                    conv_w=conv_w,
                    conv_state=conv_st.clone(),
                    a_log=a_log,
                    dt_bias=dt_bias,
                    on_w=on_w,
                    rstate=rstate,
                    st_idx=st_idx,
                    num_acc=torch.ones(1, dtype=torch.int32, device=dev),
                    ring=ring_bufs,
                    rstate0=rstate0,
                    init_slot=None
                    if ring
                    else torch.zeros(1, dtype=torch.int32, device=dev),
                    core_out=core,
                    scratch=scratch,
                    layer=0,
                    epoch=epoch,
                )
                torch.cuda.synchronize()
                return core, rstate, ring_bufs

            c_m, rs_m, rg_m = run(k_merged, w_sh, None)
            c_s, rs_s, rg_s = run(k_split, w_qkvg, w_fab)
            self.assertTrue(torch.equal(c_m, c_s), f"ring={ring} core mismatch")
            if ring:
                for a, b in zip(rg_m, rg_s):
                    self.assertTrue(torch.equal(a, b), "ring mismatch")
            else:
                self.assertTrue(torch.equal(rs_m, rs_s), "inter rstate mismatch")

    def test_mla_pre_fp8_no_indexer_matches_reference(self):
        import torch.nn.functional as F

        from sgl_amd_mono.kimi_k3.gfx950.attention import mla_pre as mp
        from sgl_amd_mono.kimi_k3.gfx950.stages import wfmt
        from sgl_amd_mono.kimi_k3.gfx950.stages.dims import HIDDEN

        torch.manual_seed(0)
        dev = torch.device("cuda:0")

        def rmsnorm(x, w, eps):
            xf = x.float()
            return (
                xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps) * w.float()
            ).to(torch.bfloat16)

        def quant_row(w_bf16):
            wf = w_bf16.float()
            sc = (wf.abs().amax(dim=1) / 448.0).clamp(min=1e-12).contiguous()
            wq = (wf / sc.unsqueeze(1)).clamp(-448, 448).to(torch.float8_e4m3fn)
            return wq.contiguous(), sc

        nblocks = 4
        cases = [
            (1, wfmt.BF16, False),
            (1, wfmt.FP8, True),
            (8, wfmt.FP8, True),
            (8, wfmt.FP8_SHUFFLED, True),
        ]
        for s_tok, fmt, use_qn in cases:
            ares_nw = (1.0 + 0.1 * torch.randn(HIDDEN, device=dev)).to(torch.bfloat16)
            ares_qk = (0.1 * torch.randn(HIDDEN, device=dev)).to(torch.bfloat16)
            in_nw = (1.0 + 0.1 * torch.randn(HIDDEN, device=dev)).to(torch.bfloat16)
            q_nw = (1.0 + 0.1 * torch.randn(mp.QL, device=dev)).to(torch.bfloat16)
            kv_nw = (1.0 + 0.1 * torch.randn(mp.KV, device=dev)).to(torch.bfloat16)
            w_qkv_bf = 0.05 * torch.randn(
                mp.LATW, HIDDEN, device=dev, dtype=torch.bfloat16
            )
            w_g_bf = 0.05 * torch.randn(
                mp.GATE, HIDDEN, device=dev, dtype=torch.bfloat16
            )
            w_qb_bf = 0.05 * torch.randn(mp.QB, mp.QL, device=dev, dtype=torch.bfloat16)
            w_kc = 0.05 * torch.randn(
                mp.NHQ, mp.QN, mp.KV, device=dev, dtype=torch.bfloat16
            )
            if fmt == wfmt.BF16:
                w_qkv, w_qkv_s, w_qkv_d = w_qkv_bf, None, w_qkv_bf
                w_g, w_g_s, w_g_d = w_g_bf, None, w_g_bf
                w_qb, w_qb_s, w_qb_d = w_qb_bf, None, w_qb_bf
            else:
                w_qkv_q, w_qkv_s = quant_row(w_qkv_bf)
                w_g_q, w_g_s = quant_row(w_g_bf)
                w_qb_q, w_qb_s = quant_row(w_qb_bf)
                w_qkv = (
                    wfmt.preshuffle_fp8(w_qkv_q)
                    if fmt == wfmt.FP8_SHUFFLED
                    else w_qkv_q
                )
                w_g = wfmt.preshuffle_fp8(w_g_q) if fmt == wfmt.FP8_SHUFFLED else w_g_q
                w_qb = (
                    wfmt.preshuffle_fp8(w_qb_q) if fmt == wfmt.FP8_SHUFFLED else w_qb_q
                )
                w_qkv_d = (w_qkv_q.float() * w_qkv_s.unsqueeze(1)).to(torch.bfloat16)
                w_g_d = (w_g_q.float() * w_g_s.unsqueeze(1)).to(torch.bfloat16)
                w_qb_d = (w_qb_q.float() * w_qb_s.unsqueeze(1)).to(torch.bfloat16)
            w_dict = {
                "ares_nw": ares_nw,
                "ares_qk": ares_qk,
                "in_nw": in_nw,
                "w_qkv": w_qkv,
                "w_qkv_s": w_qkv_s,
                "w_g": w_g,
                "w_g_s": w_g_s,
                "w_qb": w_qb,
                "w_qb_s": w_qb_s,
                "q_nw": q_nw,
                "kv_nw": kv_nw,
            }
            if use_qn:
                w_dict["w_kc"] = (
                    w_kc.transpose(1, 2).contiguous().view(mp.NHQ * mp.KV, mp.QN)
                )
            key = mp.MlaPreBuild(
                tokens=s_tok,
                nblocks=nblocks,
                qn=use_qn,
                has_indexer=False,
                qkv_fmt=fmt,
                qb_fmt=fmt,
                write_idx=nblocks,
            )
            scratch = torch.zeros(mp.scratch_bytes(key), dtype=torch.uint8, device=dev)
            epoch = torch.zeros(1, dtype=torch.int64, device=dev)
            out = {
                "q": torch.empty(s_tok, mp.QB, dtype=torch.bfloat16, device=dev),
                "latent": torch.empty(
                    s_tok, mp.KV + mp.ROPE, dtype=torch.bfloat16, device=dev
                ),
                "k_nope": torch.empty(s_tok, mp.KV, dtype=torch.bfloat16, device=dev),
                "q_lora": torch.empty(s_tok, mp.QL, dtype=torch.bfloat16, device=dev),
                "gate": torch.empty(s_tok, mp.GATE, dtype=torch.bfloat16, device=dev),
            }
            if use_qn:
                out["qn"] = torch.empty(
                    s_tok, mp.NHQ, mp.KV, dtype=torch.bfloat16, device=dev
                )
            for _ in range(2):
                epoch += 1
                prefix = torch.randn(s_tok, HIDDEN, dtype=torch.bfloat16, device=dev)
                blocks = torch.randn(
                    s_tok, nblocks + 2, HIDDEN, dtype=torch.bfloat16, device=dev
                )
                blocks_ref = blocks.clone()
                mp.mla_pre_launch(
                    key,
                    prefix=prefix,
                    blocks=blocks,
                    w=w_dict,
                    out=out,
                    scratch=scratch,
                    layer=3,
                    epoch=epoch,
                )
                torch.cuda.synchronize()
                srcs = [blocks_ref[:, idx, :] for idx in range(nblocks)] + [prefix]
                stack = torch.stack(srcs, dim=1)
                wgts = torch.softmax(
                    (rmsnorm(stack, ares_nw, key.eps).float() * ares_qk.float()).sum(
                        dim=-1
                    ),
                    dim=-1,
                )
                mixed = (
                    (wgts.unsqueeze(-1) * stack.float()).sum(dim=1).to(torch.bfloat16)
                )
                x_in = rmsnorm(mixed, in_nw, key.out_eps)
                lat = (x_in.float() @ w_qkv_d.float().T).to(torch.bfloat16)
                gate_ref = (x_in.float() @ w_g_d.float().T).to(torch.bfloat16)
                ql_n = rmsnorm(lat[:, : mp.QL], q_nw, key.q_eps)
                kn_ref = rmsnorm(lat[:, mp.QL : mp.QL + mp.KV], kv_nw, key.kv_eps)
                q_ref = (ql_n.float() @ w_qb_d.float().T).to(torch.bfloat16)
                checks = [
                    (out["q"], q_ref),
                    (out["q_lora"], ql_n),
                    (out["latent"], lat[:, mp.QL :]),
                    (out["k_nope"], kn_ref),
                    (out["gate"], gate_ref),
                ]
                if use_qn:
                    qn_ref = (
                        torch.bmm(
                            q_ref.view(s_tok, mp.NHQ, mp.QHD)[:, :, : mp.QN]
                            .transpose(0, 1)
                            .float(),
                            w_kc.float(),
                        )
                        .transpose(0, 1)
                        .to(torch.bfloat16)
                    )
                    checks.append((out["qn"], qn_ref))
                for got, want in checks:
                    cos = F.cosine_similarity(
                        got.float().reshape(-1), want.float().reshape(-1), dim=0
                    ).item()
                    self.assertGreater(cos, 0.9995, (s_tok, fmt, use_qn, cos))
                    # cosine misses a uniform rescale: the norms must agree too
                    rel = (
                        (got.float() - want.float()).norm() / want.float().norm()
                    ).item()
                    self.assertLess(rel, 0.05, (s_tok, fmt, use_qn, rel))
                self.assertTrue(torch.equal(blocks[:, nblocks, :], prefix))


def _tp8_worker():
    import torch.distributed as dist

    from sgl_amd_mono.common import guard
    from sgl_amd_mono.common.peer import PeerBuffer
    from sgl_amd_mono.kimi_k3.gfx950.attention import kda as k1
    from sgl_amd_mono.kimi_k3.gfx950.stages import wfmt
    from sgl_amd_mono.kimi_k31 import wide
    from sgl_amd_mono.kimi_k31.gfx950 import layer as lay
    from sgl_amd_mono.kimi_k31.gfx950 import pre_attn as pa
    from sgl_amd_mono.kimi_k31.gfx950.chain import Gemm

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == 8
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    dev = torch.device(f"cuda:{rank}")
    bf, f32 = torch.bfloat16, torch.float32
    H, SL, LAT, E, NF = 7168, 896, 3584, 896, 2096

    for L, R, ring, state_bf16 in [
        (5, 2, True, True),
        (6, 2, True, True),
        (5, 2, False, True),
        (5, 2, False, False),
    ]:
        m = L * R
        cfg = lay.LayerCfg(
            *(Gemm(*(int(v) for v in c.split(":"))) for c in wide.cfg_rows(m, L))
        )
        torch.manual_seed(100 + rank + L * 7)
        w_ip_bf = torch.randn(k1.NPROJ, H, device=dev, dtype=bf) * 0.02
        w_ip_q, w_ip_s, w_ip_deq = _quant_fp8_per_channel(w_ip_bf)
        w_ip_sh = wfmt.preshuffle_fp8(w_ip_q)
        w_ip_qkvg = w_ip_sh[: k1.FA_0].contiguous()
        w_ip_fab = w_ip_sh[k1.FA_0 :].contiguous()

        wo_bf = torch.randn(H, k1.PROJ, device=dev, dtype=bf) * 0.02
        wo_q, wo_s, wo_deq = _quant_fp8_per_channel(wo_bf)
        wo_sh = wfmt.preshuffle_fp8(wo_q)

        # shared inputs across bf16-deq reference and fp8-split launch
        lat_p = torch.randn(m, LAT, device=dev, dtype=bf) * 0.1
        sh_p = torch.randn(m, H, device=dev, dtype=bf) * 0.1
        w_up = torch.randn(SL, LAT, device=dev, dtype=bf) * 0.02
        prefix = torch.randn(m, H, device=dev, dtype=bf) * 0.2
        bank = torch.randn(m, 8, SL, device=dev, dtype=bf) * 0.2
        cw_a = torch.randn(H, device=dev, dtype=f32) * 0.1
        w_in = torch.ones(H, device=dev, dtype=bf)
        cw_m = torch.randn(H, device=dev, dtype=f32) * 0.1
        w_ln = torch.ones(H, device=dev, dtype=bf)
        wf = torch.randn(NF, H, device=dev, dtype=bf) * 0.02
        bias = torch.randn(E, device=dev, dtype=f32) * 0.1
        w_sd = torch.randn(H, 768, device=dev, dtype=bf) * 0.02
        w_fb = torch.randn(k1.PROJ, k1.HD, device=dev, dtype=bf) * 0.02
        conv_w = torch.randn(3 * k1.PROJ, k1.CONV_W, device=dev, dtype=f32) * 0.1
        conv_st = torch.randn(4, k1.CONV_W - 1, 3 * k1.PROJ, device=dev, dtype=bf) * 0.2
        a_log = torch.randn(k1.NH, device=dev, dtype=f32) * 0.1
        dt_bias = torch.randn(k1.PROJ, device=dev, dtype=f32) * 0.1
        on_w = torch.ones(k1.HD, device=dev, dtype=bf)

        sdt = bf if state_bf16 else f32
        rstate_init = torch.randn(4, k1.NH, k1.HD, k1.HD, device=dev, dtype=sdt) * 0.1

        def run_one(use_fp8):
            fn, pl = lay.kda_layer(
                m,
                cfg,
                L,
                NF,
                nvb_a=2,
                nvb_m=2,
                write_a=False,
                nbk=8,
                launches=1,
                eps=1e-5,
                eps_n=1e-5,
                capture=False,
                t1=True,
                restart=False,
                ip_fmt=wfmt.FP8_SHUFFLED if use_fp8 else wfmt.BF16,
                ip_split=use_fp8,
                o_fmt=wfmt.FP8_SHUFFLED if use_fp8 else wfmt.BF16,
                state_bf16=state_bf16,
                ring=ring,
            )
            half = pl.peer_offsets[1]
            pb = PeerBuffer(2 * half + guard.FLAG_BYTES, dist.group.WORLD, rank, 8, dev)
            pb.bytes.zero_()
            scr_pb = PeerBuffer(pl.scratch_bytes(), dist.group.WORLD, rank, 1, dev)
            scr_pb.bytes.zero_()
            epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
            ticks = guard.deadline_ticks(60.0)
            epoch[guard.GUARD_DEADLINE] = (
                (ticks & 0xFFFFFFFF) - (1 << 32)
                if (ticks & 0xFFFFFFFF) >= (1 << 31)
                else (ticks & 0xFFFFFFFF)
            )
            epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
            epoch[0] = 1

            mid = torch.zeros(m, SL, device=dev, dtype=bf)
            pout = torch.zeros(m, H, device=dev, dtype=bf)
            ip_buf = torch.zeros(m, k1.NPROJ, device=dev, dtype=bf)
            acc_u = torch.zeros(m, SL, device=dev, dtype=bf)
            acc_o = torch.zeros(m * H, device=dev, dtype=bf)
            core = torch.zeros(m, k1.PROJ, device=dev, dtype=bf)
            lat_out = torch.zeros(m, LAT, device=dev, dtype=bf)
            ids = torch.zeros(m, 16, device=dev, dtype=torch.int32)
            tw = torch.zeros(m, 16, device=dev, dtype=f32)
            logits = torch.zeros(m, E, device=dev, dtype=f32)
            sh_out = torch.zeros(m, H, device=dev, dtype=bf)
            cap = torch.zeros(m, H, device=dev, dtype=bf)
            cnt = torch.zeros(lay.counters(m, cfg, NF), device=dev, dtype=torch.int32)
            sort_b = torch.zeros(256, device=dev, dtype=torch.uint8)
            conv_win = torch.zeros(
                R, L, k1.CONV_W - 1, 3 * k1.PROJ, device=dev, dtype=bf
            )
            win_row = torch.arange(R, dtype=torch.int32, device=dev)
            num_acc = torch.ones(R, dtype=torch.int32, device=dev)
            if ring:
                rstate = rstate_init.clone()
                rstate0 = None
                init_slot = None
                st_idx = (
                    torch.arange(R, dtype=torch.int32, device=dev)
                    .view(R, 1)
                    .expand(R, L)
                    .contiguous()
                )
                ring_bufs = (
                    torch.zeros(4, k1.NH, 16, k1.HD, dtype=bf, device=dev),
                    torch.zeros(4, k1.NH, 16, k1.HD, dtype=bf, device=dev),
                    torch.zeros(4, k1.NH, 16, k1.HD, dtype=f32, device=dev),
                    torch.zeros(4, k1.NH, 16, dtype=f32, device=dev),
                )
            else:
                rstate = torch.zeros(R * L, k1.NH, k1.HD, k1.HD, dtype=sdt, device=dev)
                rstate0 = rstate_init.clone()
                init_slot = torch.arange(R, dtype=torch.int32, device=dev)
                st_idx = (
                    torch.arange(R, dtype=torch.int32, device=dev).view(R, 1) * L
                    + torch.arange(L, dtype=torch.int32, device=dev).view(1, L)
                ).contiguous()
                ring_bufs = None

            tab = pa.kda_table(
                dev,
                w_fb=w_fb,
                conv_w=conv_w,
                conv_state=conv_st.mT,
                a_log=a_log,
                dt_bias=dt_bias,
                on_w=on_w,
                rstate=rstate,
                rstate0=rstate0,
                init_slot=init_slot,
                st_idx=st_idx,
                num_acc=num_acc,
                core_out=core,
                conv_win=conv_win,
                win_row=win_row,
                ring=ring_bufs,
                w_ip_scale=w_ip_s if use_fp8 else None,
                w_ip_fab=w_ip_fab if use_fp8 else None,
                w_o_scale=wo_s if use_fp8 else None,
            )
            col0 = 2 * rank * SL
            fn(
                lat_p.data_ptr(),
                sh_p.data_ptr(),
                w_up.data_ptr(),
                prefix.data_ptr() + col0,
                mid.data_ptr(),
                pout.data_ptr() + col0,
                bank.data_ptr(),
                cw_a.data_ptr(),
                w_in.data_ptr(),
                (w_ip_qkvg if use_fp8 else w_ip_deq).data_ptr(),
                ip_buf.data_ptr(),
                acc_u.data_ptr(),
                tab.data_ptr(),
                core.data_ptr(),
                (wo_sh if use_fp8 else wo_deq).data_ptr(),
                cw_m.data_ptr(),
                w_ln.data_ptr(),
                wf.data_ptr(),
                acc_o.data_ptr(),
                bias.data_ptr(),
                lat_out.data_ptr(),
                ids.data_ptr(),
                tw.data_ptr(),
                w_sd.data_ptr(),
                sh_out.data_ptr(),
                cap.data_ptr(),
                H,
                logits.data_ptr(),
                cnt.data_ptr(),
                scr_pb.bytes.data_ptr(),
                pb.addresses.data_ptr(),
                rank,
                epoch.data_ptr(),
                0,
                prefix.shape[0] * H * 2 - col0,
                sort_b.data_ptr(),
                stream=torch.cuda.current_stream(),
            )
            torch.cuda.synchronize()
            dist.barrier()
            return (
                core,
                pout[:, rank * SL : (rank + 1) * SL],
                lat_out,
                logits,
                sh_out,
                rstate,
                ring_bufs,
                conv_win,
            )

        ref = run_one(False)
        got = run_one(True)
        names = ("core", "pout", "lat_out", "logits", "sh_out")
        for nm, a, b in zip(names, ref[:5], got[:5]):
            c = _cos(a, b)
            assert c > 0.9995, f"rank {rank} L={L} ring={ring} {nm} cos={c}"
        assert _cos(ref[7], got[7]) > 0.9995, "conv_win mismatch"
        if ring:
            for j, (a, b) in enumerate(zip(ref[6], got[6])):
                c = _cos(a, b)
                assert c > 0.9995, f"ring[{j}] cos={c}"
        else:
            c = _cos(ref[5], got[5])
            assert c > 0.9995, f"inter rstate cos={c}"
        if rank == 0:
            print(
                f"TP8 PASS L={L} R={R} ring={ring} state_bf16={state_bf16}: "
                + ", ".join(
                    f"{nm} cos={_cos(a, b):.6f}"
                    for nm, a, b in zip(names, ref[:5], got[:5])
                ),
                flush=True,
            )
    from sgl_amd_mono.kimi_k31 import presort
    from sgl_amd_mono.kimi_k31.gfx950 import dsa_post as dp
    from sgl_amd_mono.kimi_k31.gfx950.stages import moe_front as mf

    H, SL, KO, LAT, SI, E, NF = 7168, 896, 1536, 3584, 768, 896, 2096
    cols = slice(rank * SL, (rank + 1) * SL)
    col0 = 2 * cols.start
    for m, L in ((12, 6), (64, 8), (288, 6), (480, 8)):
        cfg = lay.LayerCfg(
            *(Gemm(*(int(v) for v in c.split(":"))) for c in wide.cfg_rows(m, L))
        )
        pl = dp.plan(m, cfg.oproj, cfg.front, cfg.sdown, NF, 3)[0]
        half = pl.peer_offsets[1]
        pb = PeerBuffer(2 * half + guard.FLAG_BYTES, dist.group.WORLD, rank, 8, dev)
        pb.bytes.zero_()
        scr_pb = PeerBuffer(pl.scratch_bytes(), dist.group.WORLD, rank, 1, dev)
        scr_pb.bytes.zero_()
        epoch = torch.zeros(guard.GUARD_WORDS, dtype=torch.int32, device=dev)
        ticks = guard.deadline_ticks(120.0)
        lo = ticks & 0xFFFFFFFF
        epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32
        torch.manual_seed(100 + rank)
        wo_q = torch.randint(0, 120, (H, KO), device=dev, dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        wo_s = (torch.rand(H, device=dev, dtype=f32) * 0.02 + 0.005).contiguous()
        wo_shuf = wfmt.preshuffle_fp8(wo_q)
        wo_bf = (wo_q.float() * wo_s[:, None]).to(bf)
        torch.manual_seed(200)
        cw_m = torch.randn(H, device=dev, dtype=f32) * 0.1
        w_ln = torch.ones(H, device=dev, dtype=bf)
        bias = torch.randn(E, device=dev, dtype=f32) * 0.05
        torch.manual_seed(300 + rank)
        wf = torch.randn(NF, H, device=dev, dtype=bf) * 0.02
        w_sd = torch.randn(H, SI, device=dev, dtype=bf) * 0.02
        pout_fp8, pout_bf = (
            torch.zeros(m, H, device=dev, dtype=bf),
            torch.zeros(m, H, device=dev, dtype=bf),
        )
        lat_fp8, lat_bf = (
            torch.zeros(m, LAT, device=dev, dtype=bf),
            torch.zeros(m, LAT, device=dev, dtype=bf),
        )
        sh_fp8, sh_bf = (
            torch.zeros(m, H, device=dev, dtype=bf),
            torch.zeros(m, H, device=dev, dtype=bf),
        )
        moe_out = torch.ones(m, LAT, device=dev, dtype=bf)
        ids = torch.zeros(m, 16, device=dev, dtype=torch.int32)
        w = torch.zeros(m, 16, device=dev, dtype=f32)
        logits = torch.zeros(m, E, device=dev, dtype=f32)
        acc_o = torch.zeros(m * H, device=dev, dtype=bf)
        cnt_fp8 = torch.zeros(
            dp.counters(m, cfg.oproj, cfg.front, NF), device=dev, dtype=torch.int32
        )
        cnt_bf = torch.zeros_like(cnt_fp8)
        sbuf = torch.zeros(
            mf.sort_layout(m, "aux16")["nbytes"], dtype=torch.uint8, device=dev
        )
        mf.sort_init(sbuf, m, "aux16")
        sviews = wide._sort_views(sbuf, m, "aux16")
        fn_fp8 = dp.dsa_post(
            m,
            cfg.oproj,
            cfg.front,
            cfg.sdown,
            NF,
            nvb_m=3,
            nbk=8,
            launches=1,
            eps=1e-5,
            half_bytes=half,
            sort="aux16",
            o_fmt=wfmt.FP8_SHUFFLED,
        )[0]
        fn_bf = dp.dsa_post(
            m,
            cfg.oproj,
            cfg.front,
            cfg.sdown,
            NF,
            nvb_m=3,
            nbk=8,
            launches=1,
            eps=1e-5,
            half_bytes=half,
            sort=None,
            o_fmt=wfmt.BF16,
        )[0]
        pfx_bytes = m * H * 2 - col0
        for rep in range(2):
            torch.manual_seed(500 + rep * 10 + rank)
            core = torch.randn(m, KO, device=dev, dtype=bf) * 0.2
            torch.manual_seed(600 + rep * 10)
            prefix = torch.randn(m, H, device=dev, dtype=bf) * 0.2
            bank = torch.randn(m, 8, SL, device=dev, dtype=bf) * 0.2
            moe_out.fill_(1.5)
            epoch[0] += 1
            fn_fp8(
                core.data_ptr(),
                wo_shuf.data_ptr(),
                wo_s.data_ptr(),
                prefix.data_ptr() + col0,
                pout_fp8.data_ptr() + col0,
                bank.data_ptr(),
                cw_m.data_ptr(),
                w_ln.data_ptr(),
                wf.data_ptr(),
                acc_o.data_ptr(),
                bias.data_ptr(),
                lat_fp8.data_ptr(),
                ids.data_ptr(),
                w.data_ptr(),
                w_sd.data_ptr(),
                sh_fp8.data_ptr(),
                moe_out.data_ptr(),
                logits.data_ptr(),
                cnt_fp8.data_ptr(),
                scr_pb.bytes.data_ptr(),
                pb.addresses.data_ptr(),
                rank,
                epoch.data_ptr(),
                0,
                pfx_bytes,
                sbuf.data_ptr(),
                stream=torch.cuda.current_stream(),
            )
            torch.cuda.synchronize()
            chk = presort.check_aux16(sviews, ids, w)
            assert (
                chk["num_valid"] == 0
                and chk["eids"] == 0
                and chk["rows"] == 0
                and chk["reverse"] == 0
            ), chk
            assert float(moe_out.abs().max()) == 0.0
        epoch[0] += 1
        fn_bf(
            core.data_ptr(),
            wo_bf.data_ptr(),
            0,
            prefix.data_ptr() + col0,
            pout_bf.data_ptr() + col0,
            bank.data_ptr(),
            cw_m.data_ptr(),
            w_ln.data_ptr(),
            wf.data_ptr(),
            acc_o.data_ptr(),
            bias.data_ptr(),
            lat_bf.data_ptr(),
            ids.data_ptr(),
            w.data_ptr(),
            w_sd.data_ptr(),
            sh_bf.data_ptr(),
            moe_out.data_ptr(),
            logits.data_ptr(),
            cnt_bf.data_ptr(),
            scr_pb.bytes.data_ptr(),
            pb.addresses.data_ptr(),
            rank,
            epoch.data_ptr(),
            0,
            pfx_bytes,
            sbuf.data_ptr(),
            stream=torch.cuda.current_stream(),
        )
        torch.cuda.synchronize()
        assert _cos(pout_fp8[:, cols], pout_bf[:, cols]) > 0.9999
        assert _cos(lat_fp8, lat_bf) > 0.9999
        assert _cos(sh_fp8, sh_bf) > 0.9999
        if rank == 0:
            print(f"TP8 PASS dsa_post FP8+aux16 m={m} L={L}", flush=True)
    dist.destroy_process_group()


@unittest.skipIf(_gfx950_gpus() < 8, "needs 8 gfx950 GPUs")
class TestKimiK3WideFp8Tp8(unittest.TestCase):
    def test_wide_kda_layer_fp8_bf16_state_and_non_replayssm(self):
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--master-addr",
            "127.0.0.1",
            "--master-port",
            "29531",
            "--nproc-per-node",
            "8",
            __file__,
            "--tp8-worker",
        ]
        env = {**os.environ, "GLOO_SOCKET_IFNAME": "lo"}
        res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        if res.returncode != 0:
            self.fail(
                f"TP8 worker failed:\nSTDOUT:\n{res.stdout[-3000:]}\nSTDERR:\n{res.stderr[-3000:]}"
            )
        self.assertIn("TP8 PASS L=5 R=2 ring=True state_bf16=True", res.stdout)
        self.assertIn("TP8 PASS L=6 R=2 ring=True state_bf16=True", res.stdout)
        self.assertIn("TP8 PASS L=5 R=2 ring=False state_bf16=True", res.stdout)
        self.assertIn("TP8 PASS L=5 R=2 ring=False state_bf16=False", res.stdout)
        for m, L in ((12, 6), (64, 8), (288, 6), (480, 8)):
            self.assertIn(f"TP8 PASS dsa_post FP8+aux16 m={m} L={L}", res.stdout)


if __name__ == "__main__":
    if "--tp8-worker" in sys.argv:
        _tp8_worker()
    else:
        unittest.main()
