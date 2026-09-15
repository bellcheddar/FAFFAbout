#!/usr/bin/env python
"""08_train.py: launch the LoRA fine-tune, with the checks that stop a wasted night.

Wraps `mlx_lm lora` rather than replacing it, and does four things the bare command will
not:

  1. runs scripts/preflight.sh and refuses to launch on a failure
  2. reconciles every key in config/train_config.yaml against the installed mlx-lm, since
     field names drift between versions and an unknown key is silently ignored
  3. starts the W&B run itself, because mlx-lm names the run after the basename of
     adapter_path and offers no override
  4. picks the next round number by globbing the adapters directory, so a deleted round is
     never reissued and the counter cannot drift from what is on disk

Resume note: a resume restarts the iteration counter AND the learning-rate schedule, so
pass the REMAINING budget as --iters and record the offset for reporting.

Usage
  .venv/bin/python scripts/08_train.py                       # next round, from config
  .venv/bin/python scripts/08_train.py --iters 1500 --resume adapters/...-round01
  .venv/bin/python scripts/08_train.py --dry-run             # checks only
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "train_config.yaml"
ADAPTERS = ROOT / "adapters"
RUN_PREFIX = "faffabout"
WANDB_PROJECT = "faffabout"


def model_slug(model: str) -> str:
    s = model.rstrip("/").split("/")[-1].lower()
    s = s.replace("meta-", "").replace("-instruct", "").replace("_", "-")
    return re.sub(r"[^a-z0-9.-]+", "-", s).strip("-")


def next_run_name(model: str) -> str:
    """Round number from the directory listing, never from a counter file."""
    stem = f"{RUN_PREFIX}-{model_slug(model)}-round"
    used = set()
    if ADAPTERS.exists():
        for d in ADAPTERS.glob(f"{stem}*"):
            m = re.fullmatch(rf"{re.escape(stem)}(\d+)", d.name)
            if m:
                used.add(int(m.group(1)))
    return f"{stem}{max(used, default=0) + 1:02d}"


def check_config_keys(cfg: dict) -> list[str]:
    """Report keys the installed mlx-lm will silently ignore."""
    try:
        from mlx_lm.lora import CONFIG_DEFAULTS  # type: ignore
        known = set(CONFIG_DEFAULTS)
    except Exception:
        try:
            from mlx_lm.tuner.trainer import TrainingArgs  # type: ignore
            known = set(getattr(TrainingArgs, "__dataclass_fields__", {}))
            known |= {"model", "train", "data", "fine_tune_type", "adapter_path",
                      "num_layers", "lora_parameters", "learning_rate", "lr_schedule",
                      "seed", "mask_prompt", "test", "resume_adapter_file", "wandb"}
        except Exception:
            return []
    return sorted(k for k in cfg if k not in known)


# Values worth sanity-checking against the library's own default, with how far off is
# tolerable. Round 01 attempt 1 ran at learning_rate 1.0e-4 against mlx-lm's default of
# 1e-5 and diverged: loss went 0.234 -> 4.831 -> 13.217 within twenty iterations of the
# warmup ending. Every existing guard passed, because they check that a KEY exists, never
# that a VALUE is sane.
VALUE_LIMITS = {"learning_rate": 5.0}


def check_config_values(cfg: dict) -> list[str]:
    """Flag values wildly adrift from what the installed mlx-lm defaults to."""
    try:
        from mlx_lm.lora import CONFIG_DEFAULTS  # type: ignore
    except Exception:  # noqa: BLE001
        return []
    out = []
    for key, factor in VALUE_LIMITS.items():
        got, default = cfg.get(key), CONFIG_DEFAULTS.get(key)
        if got is None or not isinstance(default, (int, float)) or not default:
            continue
        if got > default * factor:
            out.append(f"{key}={got:g} is {got/default:.0f}x mlx-lm's default of {default:g}")
    return out


def check_cli_flags(cmd: list[str]) -> list[str]:
    """Flags we construct, checked against the installed parser.

    check_config_keys only covers the YAML. `--wandb` was built here in Python, so it
    sailed past every check and the run died at launch: mlx-lm 0.31 wants report_to and
    project_name instead.
    """
    try:
        help_text = subprocess.run(cmd[:4] + ["--help"], capture_output=True,
                                   text=True, timeout=120).stdout
    except Exception:  # noqa: BLE001
        return []
    if not help_text:
        return []
    return [f for f in cmd if f.startswith("--") and f not in help_text]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=None, help="override; use the REMAINING budget on a resume")
    ap.add_argument("--resume", type=str, default=None, help="adapter directory to continue from")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true", help="only for a deliberate re-launch")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--allow-high-lr", action="store_true",
                    help="proceed despite a learning rate far above mlx-lm's default")
    args = ap.parse_args()

    cfg = yaml.safe_load(CFG.read_text())

    if not args.skip_preflight:
        print("running preflight ...")
        if subprocess.run(["bash", str(ROOT / "scripts" / "preflight.sh")]).returncode:
            sys.exit("preflight failed: fix the FAIL lines above, or pass --skip-preflight deliberately")

    unknown = check_config_keys(cfg)
    if unknown:
        print(f"\n!! keys the installed mlx-lm does not recognise and will IGNORE: {unknown}")
        print("   reconcile against the installed version before trusting the run")
        if not args.dry_run:
            sys.exit(1)

    risky = check_config_values(cfg)
    if risky and not args.allow_high_lr:
        print("\n!! " + "\n!! ".join(risky))
        print("   Round 01 attempt 1 diverged at learning_rate 1.0e-4 (10x the default):")
        print("   loss 0.234 -> 4.831 -> 13.217 within twenty iterations of warmup ending.")
        print("   Pass --allow-high-lr to proceed deliberately.")
        sys.exit(1)

    run_name = next_run_name(cfg["model"])
    adapter_path = ADAPTERS / run_name
    iters = args.iters if args.iters is not None else cfg["iters"]

    # steps_per_eval must be a whole multiple of steps_per_report, never equal to it
    spr, spe = cfg["steps_per_report"], cfg["steps_per_eval"]
    if spe % spr or spe == spr:
        sys.exit(f"steps_per_eval ({spe}) must be a whole multiple of steps_per_report ({spr}), and not equal")

    print(f"\nrun name    : {run_name}")
    print(f"adapter path: {adapter_path}")
    print(f"model       : {cfg['model']}")
    print(f"iters       : {iters}" + ("  (REMAINING budget on a resume)" if args.resume else ""))
    print(f"batch/layers: {cfg['batch_size']} / {cfg['num_layers']}   mask_prompt={cfg['mask_prompt']}")

    cmd = [str(ROOT / ".venv" / "bin" / "python"), "-m", "mlx_lm", "lora",
           "--config", str(CFG), "--train",
           "--adapter-path", str(adapter_path), "--iters", str(iters)]
    if args.resume:
        cmd += ["--resume-adapter-file", str(Path(args.resume) / "adapters.safetensors")]
    # W&B is configured in the YAML (report_to / project_name), not as a flag.

    unknown_flags = check_cli_flags(cmd)
    if unknown_flags:
        sys.exit(f"mlx-lm does not accept {unknown_flags}. Flag names drift between "
                 "versions; reconcile against `python -m mlx_lm lora --help`.")

    print("\n" + " ".join(cmd))
    if args.dry_run:
        print("\n(dry run: nothing launched)")
        return

    adapter_path.mkdir(parents=True, exist_ok=True)
    (adapter_path / "RUN_NOTES.md").write_text(
        f"# {run_name}\n\nmodel: {cfg['model']}\niters: {iters}\n"
        f"resumed from: {args.resume or '(none)'}\n\n"
        "Note: a resume restarts the iteration counter and the LR schedule, so `iters`\n"
        "above is the remaining budget, not the cumulative total.\n")
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
