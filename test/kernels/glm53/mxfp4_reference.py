"""Independent PyTorch E2M1/E8M0 arithmetic for native GLM expert checks.

No device helper or native MFMA implementation is used by these goldens.
Values are rounded to BF16 before A4 quantization, in groups of 32, as in
FlyDSL #1256. Ties choose the E2M1 encoding with an even low bit.
"""

import math

import torch


def quantize(x):
    x = x.bfloat16().float()
    groups = x.reshape(*x.shape[:-1], -1, 32)
    amax = groups.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    # frexp gives the ceiling exponent except for exact powers of two.
    mantissa, power = torch.frexp(amax * (1.0 / 6.0))
    exponent = (power - (mantissa == 0.5).int() + 127).clamp(1, 254)
    scales = torch.ldexp(torch.ones_like(amax), exponent - 127)
    normalized = (groups / scales).clamp(-6.0, 6.0)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    # Search thresholds; at exact midpoint select the even encoding.
    thresholds = (levels[1:] + levels[:-1]) * 0.5
    absolute = normalized.abs()
    codes = torch.zeros_like(normalized, dtype=torch.uint8)
    for index, threshold in enumerate(thresholds):
        codes += (
            (absolute > threshold) | ((absolute == threshold) & (index % 2 == 1))
        ).byte()
    values = levels[codes.long()] * torch.where(torch.signbit(normalized), -1, 1)
    codes |= torch.signbit(normalized).byte() << 3
    return (
        (values * scales).reshape(x.shape),
        codes.reshape(x.shape),
        exponent.squeeze(-1).byte(),
    )


def unpack(packed, scales):
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
    values = levels[(codes & 7).long()] * torch.where((codes & 8) != 0, -1, 1)
    return values * torch.ldexp(
        torch.ones_like(scales, dtype=torch.float32), scales.int() - 127
    ).repeat_interleave(32, -1)


def expert_factory(original, rank=0):
    """Build the actual AITER shuffle used by SGL, retaining logical controls."""
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    def make(inter, hidden, generator, device):
        # TP expert shards differ; replicated indexer tensors use the caller's
        # untouched generator so CP inputs and index weights still agree.
        generator = torch.Generator(device=device).manual_seed(
            generator.initial_seed() + 997 * rank
        )
        result = []
        for name, rows, k in (("ug", 2 * inter, hidden), ("dn", hidden, inter)):
            q = torch.randint(
                0,
                256,
                (257, rows, k // 2),
                generator=generator,
                device=device,
                dtype=torch.uint8,
            )
            base = round(math.log2(k**-0.5 / 4)) + 127
            scale = torch.randint(
                base - 1,
                base + 2,
                (257, rows, k // 32),
                generator=generator,
                device=device,
                dtype=torch.uint8,
            )
            original[name] = (q, scale)
            result.extend(
                (
                    shuffle_weight(q, (16, 16)),
                    e8m0_shuffle(scale.flatten(0, 1)).reshape(scale.shape),
                )
            )
        return result

    return make
