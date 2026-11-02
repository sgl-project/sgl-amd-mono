# SPDX-License-Identifier: Apache-2.0
"""Monokernels for SGLang on AMD Instinct GPUs (RFC sgl-project/sglang#43949).

Persistent kernels run most of a decoder layer, its TP collectives included, in one
or two launches. SGLang holds the interface (``sglang.kernels.plugin_kernels.amd``);
this package holds the HIP device runtime (``common``) and, per model, its specs and
their forwards (``<model>/spec.py``, ``<model>/draft.py``) and its kernels per ISA
(``<model>/<isa>/``). SGLang loads it as a general plugin (the ``sglang.srt.plugins``
entry point in ``pyproject.toml``), which registers each spec.
"""

import inspect

__version__ = "0.1.0.dev0"

# the backend name every spec registers under
BACKEND = "amd"
# each spec by (HF architecture, model type), None for any model type; its module
# imports only when a server runs this backend. DeepSeek-V4-Pro and V4.1 share an
# architecture in SGLang, so their specs name their model types
SPECS = {
    ("DeepseekV4ForCausalLM", "deepseek_v4"): "sgl_amd_mono.deepseek_v4.spec:SPEC",
    ("DeepseekV4ForCausalLM", "deepseek_v41"): "sgl_amd_mono.deepseek_v41.spec:SPEC",
    (
        "DeepseekV4ForCausalLMDSpark",
        "deepseek_v41",
    ): "sgl_amd_mono.deepseek_v41.draft:SPEC",
    ("Glm5NextForConditionalGeneration", None): "sgl_amd_mono.glm53_flash.spec:SPEC",
    ("GlmMoeDsaForCausalLM", None): "sgl_amd_mono.glm53.spec:SPEC",
    ("GlmMoeDsaForCausalLMNextN", None): "sgl_amd_mono.glm53.draft:SPEC",
    ("MiniMaxM3SparseForCausalLM", None): "sgl_amd_mono.minimax.spec:SPEC",
    (
        "MiniMaxM3SparseForConditionalGeneration",
        None,
    ): "sgl_amd_mono.minimax.spec:SPEC",
    # the MTP draft routes its own steps (qwen38_flash_next/draft.py)
    ("Qwen4ExpForCausalLMMTP", None): "sgl_amd_mono.qwen38_flash_next.draft:SPEC",
    (
        "Qwen4ExpForConditionalGeneration",
        None,
    ): "sgl_amd_mono.qwen38_flash_next.spec:SPEC",
    ("MuseSparkForConditionalGeneration", None): "sgl_amd_mono.muse_spark.spec:SPEC",
    ("DFlash2DraftModel", None): "sgl_amd_mono.muse_spark.draft:SPEC",
}

# The SGLang names this module reads (test/unit/test_contract.py pins them)
SGLANG_READS = {
    "kernels/plugin_kernels/amd/spec.py|srt/monokernels/spec.py": (
        "register",
        "register(model_type)",
    )
}


def register_specs() -> None:
    """The plugin entry point, which SGLang runs in every process."""
    try:
        from sglang.kernels.plugin_kernels.amd.spec import register
    except ModuleNotFoundError:  # SGLang before sgl-project/sglang#45389
        from sglang.srt.monokernels.spec import register

    # The pre-#45389 register() is keyword-only (backend, architecture, path) and
    # has no model_type parameter at all -- passing it unconditionally raises
    # TypeError on that older signature, which the module-path fallback above
    # does not by itself protect against.
    takes_model_type = "model_type" in inspect.signature(register).parameters

    seen_architectures = set()
    for (architecture, model_type), path in SPECS.items():
        if not takes_model_type:
            # The old registry is a flat architecture -> path map with no
            # model_type axis at all, so an architecture that needs model_type
            # to disambiguate (DeepSeek-V4 vs V4.1 share one architecture
            # string) can only keep one spec here -- the first one in SPECS
            # wins; the rest are simply unavailable under old SGLang, same as
            # they always were before this plugin grew model_type-scoped
            # specs.
            if architecture in seen_architectures:
                continue
            seen_architectures.add(architecture)
        kwargs = dict(backend=BACKEND, architecture=architecture, path=path)
        if takes_model_type:
            kwargs["model_type"] = model_type
        register(**kwargs)
