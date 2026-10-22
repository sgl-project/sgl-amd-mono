# SPDX-License-Identifier: Apache-2.0
# Adapted from sgl-project/sglang#45353 at 6ce98069c3 (Apache-2.0), by salykova:
# python/sglang/kernels/ops/speculative/qwen4_exp_monokernel/weights.py
"""Qwen3.8-Flash-Next monokernel geometry (``amd/Qwen3.8-Flash-Next-Quark-MXFP4-
AttnFP8-PLEFP8``, SGLang's Qwen4-Exp) and the build key of one launch."""

import msgspec

# the text config's shapes, which the kernel builds in
HIDDEN = 2560
HC = 4
HC_DIM = HIDDEN * HC  # 10240: the hyper-connection residual streams
HC_LOWRANK = 320
EPS = 1e-6
# Gated DeltaNet
GDN_HK = 16
GDN_HV = 48
GDN_DK = 128
GDN_DV = 128
GDN_KEY_DIM = GDN_HK * GDN_DK  # 2048
GDN_VAL_DIM = GDN_HV * GDN_DV  # 6144
GDN_CONV_DIM = 2 * GDN_KEY_DIM + GDN_VAL_DIM  # 10240
GDN_QKVZ = 2 * GDN_KEY_DIM + 2 * GDN_VAL_DIM  # 16384
GDN_CONV_K = 4
# MoE: 512 routed MXFP4 experts, top-10, and the shared expert fused at index 512
N_EXPERTS = 512
TOPK = 10
MOE_INTER = 640
# full attention (QSA)
ATT_HQ = 24
ATT_HKV = 2
ATT_D = 256
ROTARY_DIM = 64

# a launch's rows: R requests x L verify tokens (row t: request t // L, step t % L)
MAX_R = 8
MAX_L = 4
MAX_T = 32


class ConvStrides(msgspec.Struct, frozen=True):
    """Element strides (draft step, channel, window, request row) of a GDN layer's
    intermediate conv-window view (SGLang's ``intermediate_conv_window[0]``)."""

    step: int = 30720
    channel: int = 3
    window: int = 1
    row: int = 4 * 30720


class MonoBuild(msgspec.Struct, frozen=True):
    """One launch: its layers and rows, and the cache strides it is compiled for;
    every field is a scalar of the JIT cache key (``plan.key_tuple``).

    ``plan``: the launch's layer kinds, comma-separated (``gdn`` | ``qsa``);
    ``nreq`` R requests of ``ntok`` L verify tokens; ``rtt_stride``: req_to_token
    row stride (int32 elements); ``pos_stride``: element stride between the 3
    MRoPE position axes (0: 1-D text positions); ``map_i64``: the request -> mamba
    slot map is int64; ``ple_first``: the launch starts with layer 1's PLE;
    ``final_mix``: it ends with the model-level mixer; ``prelude``: it embeds the
    tokens (launch 0); ``iconv``, ``issm_row`` (elements per request row of
    intermediate_ssm), ``ple_ist_row`` (bf16 elements per request row of the PLE
    intermediate short-conv state) and ``ictx_row`` (int64 elements per request
    row of the n-gram intermediate context): SGLang's verify scratch strides;
    ``first_src``: where layer 0's input comes from (``global``: the previous
    launch, ``mtp``: the MTP input fusion); ``ssm_bf16``: the GDN SSM states are
    bf16 (SGLANG_MAMBA_SSM_DTYPE=bfloat16; the recurrence stays f32, the states
    are rounded when stored, as SGLang does)."""

    plan: str
    nreq: int
    ntok: int
    rtt_stride: int = 0
    pos_stride: int = 1
    map_i64: bool = False
    ple_first: bool = False
    final_mix: bool = False
    prelude: bool = False
    eos_token_id: int = 0
    iconv: ConvStrides = msgspec.field(default_factory=ConvStrides)
    issm_row: int = 4 * GDN_HV * GDN_DK * GDN_DV
    ple_ist_row: int = 4 * HC_DIM * 9
    ictx_row: int = 8
    first_src: str = "global"
    ssm_bf16: bool = False
    # SGLANG_AMD_MONO_FAULT_INJECT: the wait this launch stalls at (0: none), once
    # fault_after steps have run
    fault_site: int = 0
    fault_after: int = 0

    def __post_init__(self) -> None:
        kinds = self.layers
        if not kinds or any(k not in ("gdn", "qsa") for k in kinds):
            raise ValueError(f"plan {self.plan!r}: layer kinds are gdn or qsa")
        if not (1 <= self.nreq <= MAX_R and 1 <= self.ntok <= MAX_L):
            raise ValueError(
                f"{self.nreq} requests x {self.ntok} tokens: the kernel takes 1..{MAX_R}"
                f" requests of 1..{MAX_L} tokens"
            )
        if self.nreq * self.ntok > MAX_T:
            raise ValueError(f"{self.nreq * self.ntok} rows, past {MAX_T}")
        if self.first_src not in ("global", "embed", "mtp"):
            raise ValueError(f"first_src {self.first_src!r}")

    @property
    def layers(self) -> tuple:
        return tuple(self.plan.split(",")) if self.plan else ()
