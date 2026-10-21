"""Compare twelve-row owner reductions with distinct attention TP partials.

Run this harness under baseline and candidate PYTHONPATHs, saving a control
on the first run and checking it on the second. Expert TP shards already
differ in the independent A4W4 fixture; this also distinguishes O partials.
"""

import argparse
from types import SimpleNamespace

import check_mxfp4
import torch.distributed as dist


def main(options):
    original = check_mxfp4.make_weights

    def ranked_weights(*args, **kwargs):
        weights = original(*args, **kwargs)
        # Keep QKV/index/cache replicas equal while distinguishing O's shards.
        weights.s_o.mul_(1.0 + 0.25 * dist.get_rank())
        return weights

    check_mxfp4.make_weights = ranked_weights
    try:
        check_mxfp4.check(
            SimpleNamespace(
                rows=12,
                request_width=6,
                capacity=2048,
                seed=821,
                pad_rows=0,
                cp=False,
                external=True,
                ties=False,
                quant_edges=False,
                route_pattern="random",
                check_causality=True,
                profile_launches=False,
                bench=False,
                save_control=options.save_control,
                check_control=options.check_control,
            )
        )
    finally:
        check_mxfp4.make_weights = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-control")
    parser.add_argument("--check-control")
    options = parser.parse_args()
    dist.init_process_group("gloo")
    try:
        main(options)
    finally:
        dist.destroy_process_group()
