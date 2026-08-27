"""Baseline ET-only PEST++ IES calibration for mt_mesonet, reusing swim-rs Example 5.

Thin wrapper over Example 5's ``run_pest_sequence`` (its own ``calibrate.py`` hardcodes the
Example 5 TOML and results layout, so it cannot be pointed at this project directly). The
sequence itself — PestBuilder, spinup, prior-data-conflict dry run, pestpp-ies with the
Example 5 settings (noptmax=3), archive of the full trajectory per RUN_POLICY — is Example
5's code unchanged. Config comes from ``mt_mesonet.toml`` (`[calibration]`: ensemble ETf
target over six OpenET members, 200 realizations, 10 workers, spread weighting default).

This is the retrieval/water-balance **baseline** on nine non-irrigated grass/shrub sites:
conditioned on the ETf ensemble only, no soil-moisture observations. In-situ VWC stays
validation-only. SM-conditioning experiments are a separate, later leg.

``run_pest_sequence`` deletes and rebuilds ``{project_workspace}/pestrun`` (including
spinup) — by design; ``pestrun`` holds no source data. Results land in
``{project_workspace}/results/<tag>``.

Must run with the swim-rs venv (needs swimrs, pyemu, pestpp-ies on PATH):
    /home/dgketchum/code/swim-rs/.venv/bin/python examples/10_MT_Mesonet/calibrate.py --results-tag run0_baseline
"""

import argparse
import os
import sys
import time
from pathlib import Path

SWIMRS_EXAMPLES = Path("/home/dgketchum/code/swim-rs/examples")
E5 = SWIMRS_EXAMPLES / "5_Flux_Ensemble"

CONFIG = "/data/ssd2/nisar/swim/mt_mesonet/mt_mesonet.toml"


def _example5_calibrate():
    """Import Example 5's calibrate module (deferred: needs swimrs/pyemu)."""
    if str(E5) not in sys.path:
        sys.path.insert(0, str(E5))
    import calibrate as c5

    return c5


def _load_config(conf: str = CONFIG):
    from swimrs.swim.config import ProjectConfig

    cfg = ProjectConfig()
    cfg.read_config(conf, calibrate=True)
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Baseline IES calibration for mt_mesonet")
    p.add_argument("--config", default=CONFIG)
    p.add_argument(
        "--results-tag",
        default="scratch",
        help="Results subdirectory under {project_workspace}/results (Example 5 "
        "convention: untagged runs go to scratch, never an archived run dir)",
    )
    p.add_argument(
        "--keep-pestrun",
        action="store_true",
        help="Leave pest/master/workers dirs in place after archiving",
    )
    args = p.parse_args()

    cfg = _load_config(args.config)
    results = os.path.join(cfg.project_ws, "results", args.results_tag)

    print(f"Config: {args.config}")
    print(
        f"ETf target: {cfg.etf_target_model} ({len(cfg.etf_ensemble_members)} members)"
    )
    print(f"Realizations: {cfg.realizations}, workers: {cfg.workers}")
    print(f"Weighting mode: {cfg.etf_weighting_mode}")
    print(f"Results dir: {results}")

    c5 = _example5_calibrate()
    t0 = time.time()
    c5.run_pest_sequence(cfg, results, keep_pestrun=args.keep_pestrun)
    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f} s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
