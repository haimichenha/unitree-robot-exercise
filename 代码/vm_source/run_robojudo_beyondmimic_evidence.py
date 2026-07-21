"""Run a bounded RoboJuDo BeyondMimic simulation and save an off-screen frame.

Run from the RoboJuDo repository with its root on PYTHONPATH.  The regular
``scripts/run_pipeline.py`` loop is intentionally unbounded, so this evidence
helper runs a fixed number of simulation steps and writes a reproducible PNG.
"""

import argparse
from pathlib import Path

import cv2
import mujoco
import robojudo.pipeline

from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.rl_pipeline import RlPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="g1_beyondmimic")
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--switch-to", type=int)
    parser.add_argument("--switch-at", type=int, default=10)
    parser.add_argument(
        "--output",
        default="/home/hiamichenha/unitree_h1_learn/validation/robojudo_g1_beyondmimic_step90.png",
    )
    args = parser.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")

    cfg: RlPipelineCfg = ConfigManager(config_name=args.config).get_cfg()
    pipeline_type = cfg.pipeline_type
    pipeline_class: type[RlPipeline] = getattr(robojudo.pipeline, pipeline_type)
    pipeline = pipeline_class(cfg=cfg)

    for step in range(args.steps):
        if args.switch_to is not None and step == args.switch_at:
            if not hasattr(pipeline, "policy_manager"):
                raise RuntimeError("--switch-to requires a multi-policy pipeline")
            pipeline.policy_manager.switch_policy(args.switch_to)
            print(f"switch_requested={args.switch_to}@step{step}")
        pipeline.step()

    renderer = mujoco.Renderer(pipeline.env.model, height=480, width=640)
    try:
        renderer.update_scene(pipeline.env.data)
        rgb = renderer.render()
    finally:
        renderer.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Unable to write {output}")
    print(f"config={args.config}")
    print(f"steps_completed={args.steps}")
    if hasattr(pipeline, "policy_manager"):
        print(f"active_policy_id={pipeline.policy_manager.current_policy_id}")
    print(f"saved_frame={output}")


if __name__ == "__main__":
    main()
