# Qwen3.8-Flash-Next TP1 and TP2 decode (wide kernels)

`--monokernel-backend amd` on MI355X (gfx950) runs each pure decode step of Qwen3.8-Flash-Next
(`amd/Qwen3.8-Flash-Next-Quark-MXFP4`: MXFP4 experts, BF16 attention and GDN projections) of 1 to 192 rows at TP2, or
1 to 64 rows at TP1, as one persistent FlyDSL launch per GDN layer and two per QSA layer around SGLang's QSA
attention. The TP size and speculative decoding route `qwen38_flash_next.spec.SPEC` here (`_route`): TP2, and TP1
without speculative decoding; TP1 with NEXTN runs the verify monokernel of `../README.md` on the FP8-projection
checkpoint.

The kernels (`wide/`) are the wide decode kernels of vllm-project/vllm#65188 by @samremes, adapted at 294644b81e
(Apache-2.0; each file names its origin), with the TP1 changes of vllm-project/vllm#67706 at d78712abdb: the GDN
projection in one K split with its 96 `[b | a]` rows in two 64-row blocks, the MoE's 640 intermediate rows (10 tiles a
wave, a larger f32 bridge panel, the last wave's scale pair clamped) and the QSA projection in one K split. Three
build flags add SGLang's setting, each off by default as upstream:
`conv_dim_first` (SGLang's conv state layout), `wait_s` (SGLang's wait deadline) and `stop` (a failed wait stops every
rank), with `fault_site` / `fault_after` for the fault-injection test.

## Launches

| layer | launches | phases |
| --- | --- | --- |
| GDN (36) | `gdn` | [stats] attention mix, GDN projection + conv, gated delta rule + gated norm, out_proj + TP exchange + combine, MLP mix, router, routing + MXFP4 quantization, experts, TP exchange + combine |
| QSA (12) | `qsa_front`, SGLang's attention, `qsa_back` | attention mix -> x, qkv_proj + index_qk_proj; SGLang's prepare, indexer and sparse attention fed those projections (`spec.qsa_core`); o_proj with the output gate + TP exchange + combine, the MLP half |

The embedding, the PLE layer, the final mixer, the logits and sampling stay SGLang's (`spec._Built.forward`). The
first launch of a step and the first after a PLE layer compute the state's norms first (`stats`). After each GDN
launch the forward calls SGLang's decode track-save (the radix cache's GDN state copies).

The kernels are built for each decode CUDA graph capture size in 1-192 at TP2, in 1-64 at TP1
(`Shard.MAX_TOKENS`); SGLang pads a step to the smallest captured size that holds it. The spec declares SGLang's
default decode capture sizes up to 192 (1, 2, 4, 8, 12, 16, 24, ..., 192) and the verify path's widths (4 to 32 in
steps of 4: 20 and 28 are added), and SGLang captures the declared widths up to its largest capture size, then its
own sizes above them. Every width builds four kernels (`gdn` with and without `stats`, `qsa_front`, `qsa_back`),
compiled at startup (each rank starts at its own share; FlyDSL's disk cache hands a rank the kernels another rank
built): at TP2 with `--cuda-graph-max-bs-decode 192`, 30 widths and 120 builds; at TP1 with
`--cuda-graph-max-bs-decode 64`, 14 widths and 56 builds. Steps above the TP's largest width, prefill and mixed
steps run SGLang's forward.

TP1 stops at 64 rows as vllm-project/vllm#67706 does: there the gain over vLLM's TP1 decode falls with the width, and
an earlier TP1 build of these kernels was slower than vLLM's above 64 rows. The spec holds one set of widths for both
TP sizes, so at TP1 SGLang also routes the declared widths above 64 that it captures (a `--cuda-graph-max-bs-decode`
above 64), or every declared width with CUDA graphs off, to `spec._Built.forward`, which runs SGLang's forward for
them. SGLang's startup lines then list those widths among the steps that run the monokernels (the binding's
`monokernel ... widths [...]` and its capture check's `runs decode steps of batch size [...]`); the build's own line,
`Qwen3.8-Flash-Next monokernel: TP1, N builds for widths [...] (the declared widths [...] run SGLang's forward)`,
names the widths TP1 builds and those it hands back. A TP1 server none of whose capture sizes is in 1-64 is refused
at build. The TP1 recipe (`test/e2e/qwen38_flash_next_decode_server.py`, `TP1_ARGS`) caps the running requests at 64.

## Supported

- TP2 on two, or TP1 on one, dedicated gfx950 GPUs (256 CUs), plain decode (no speculative decoding), CUDA graphs or
  eager;
- the checkpoint's shapes (`wide/layout.py`, `SHAPES`) and layouts: bf16 in_proj_qkvz and out_proj, which the build
  quantizes to MXFP4 copies in AITER's shuffled layout (`weights.mxfp4_copy`, read as W4A16: 144 tensors, 525.9 MiB
  a rank at TP2 and 1,051.9 MiB at TP1, in SGLang's startup line of weight copies); AITER's shuffled MXFP4 experts
  with the shared expert fused as expert 512, padded to a multiple of 128 intermediate rows;
- the GDN states as SGLang keeps them: a bf16 conv state `[slots, CONV_DIM, 3]` and an f32 recurrent state
  `[slots, NV, DV, DK]`, a slot dense, on the triton GDN decode backend; pad rows (slot <= 0) write none;
- the TP all-reduces send exact bf16 sums in rank order; SGLang's decode all-reduce is exact at these sizes too, on
  AITER's custom all-reduce, which takes it below 64 MiB at TP2 (a 192-row exchange is 0.94 MiB). At TP1 the exchange
  is a round trip through the rank's own mailbox (`common.peer.PeerBuffer` opens no peer at one rank).

## Refusals

Default-deny, each stage listing every reason (`spec.py`), the TP ranks agreeing on the build:

- the server (`SERVER_ARGS`, `config_disable_reasons`): speculative decoding, an SSM state other than f32 or a conv
  state other than bf16, `--enable-unified-memory` and `--enable-page-major-kv-layout` (their envelope slots), a GDN
  decode backend other than triton, other checkpoint shapes or quantization, GDN and attention projections the
  checkpoint's Quark config does not keep in bf16 (`projection_dtypes`: the FP8-projection checkpoint, which runs TP1
  with NEXTN), `SGLANG_AMD_MONO_ROCM_LLVM`, no FlyDSL;
  and what `SPEC.requires` and SGLang refuse for both paths (ReplaySSM, EP, PP, DP attention, ...);
- the loaded model (`model_disable_reasons`): every layer's module variants, weight shapes, dtypes and layouts, one
  HC norm epsilon, the expert padding; SGLang's linear-attention backend calls the forward makes;
- the GDN states at every bind (`gdn_state_reasons`): dtype, shape, a slot's rows dense, the conv state's slot pitch a
  multiple of 4 elements and its base 8-byte aligned (its taps are read 8 bytes at a time).

## Failure

Every in-kernel wait (grid barrier, TP exchange line, tagged hand-off) gives up at SGLang's deadline D
(`BuildContext.deadline_s`, half the watchdog timeout, at most 60 s), or a grace of D/8 or two later: a tagged hand-off,
or an exchange line while only the rank's own lines are missing, at D; a grid barrier at 1.125 D; an exchange line while
a peer's line is missing at 1.25 D (`wide.common.grid.deadlines`): past the deadline SGLang hands the plugin, but at
most 0.625 of the watchdog timeout. In stop mode the CTA raises its rank's flag in every other rank's peer buffer (the
first `PEER_HEAD` bytes, zeroed at allocation and written by nothing else) and faults the launch at its wait kind's
page; a rank whose wait finds a peer's flag up faults at 0x4000, without waiting for its own deadline. At TP1 there is
no peer: every line is the rank's own, and its own deadlines stop it. The runtime reports a queue's first fault (CLR's
"faulting addr", ROCr's "on address"), so a rank does not raise its own flag, and the staggered deadlines make the first
wait to give up on a GPU one of the kind furthest upstream:

| a GPU's fault at | its first wait to give up | meaning |
| --- | --- | --- |
| 0x3000 | a tagged hand-off between the launch's CTAs (`wide/stages/tagged.py`, such as split-K partials) | a CTA of this GPU never handed off its values |
| 0x2000 | a TP exchange line | a line never came: of this rank (a CTA of its own never wrote it), or of a peer |
| 0x1000 | a grid barrier | no hand-off or line of its own was missing: a CTA never arrived (not every CTA resident, or one waits on a peer's line) |
| 0x4000 | a peer's flag | another rank gave up first; its page names the stall |

`SGLANG_AMD_MONO_FAULT_INJECT=rank:site:after` stalls a wait of that rank's launches once the step epoch reaches
`after` (test only).

## Check mode

`SGLANG_AMD_MONO_CHECK=1` (eager steps; `--disable-cuda-graph`) runs SGLang's decoder layer next to each layer's
launches on the same streams and GDN states and logs, per step, each metric's worst value: the layer's streams and
own contribution, the GDN conv and recurrent state rows, the radix track copies, and for a QSA layer the block input
and the fed projections against SGLang's (`check.py`, bounds `spec.BOUNDS`). `SGLANG_AMD_MONO_CHECK_REF=mxfp4` gives
the reference's GDN projections the launches' MXFP4 operands (`spec.CHECK_MXFP4_BOUNDS`).

## Code

| file | contents |
| --- | --- |
| `spec.py` | widths, refusals, the forward (`_Built`) and the build |
| `launcher.py` | the per-rank launch state, build keys and launches (`Launcher`) |
| `weights.py` | each layer's launch operands and the private copies |
| `check.py` | check mode |
| `wide/` | the kernels: build key, ABI and scratch (`layer.py`), shapes (`layout.py`), health words (`abi.py`), device primitives and the grid barrier (`common/`), stages (`stages/`) |
