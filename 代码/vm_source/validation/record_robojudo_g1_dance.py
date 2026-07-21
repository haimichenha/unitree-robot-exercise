"""Record a bounded off-screen RoboJuDo G1 dance rollout as an MP4.

The ``g1_beyondmimic_with_ctrl`` configuration uses the shipped
``dance1_subject2`` reference motion and the ``Dance_wose`` policy.  This
helper intentionally runs a finite number of MuJoCo steps so that the result
is reproducible and does not require the interactive viewer.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import mujoco
import robojudo.pipeline

from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.rl_pipeline import RlPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="g1_beyondmimic_with_ctrl")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--output",
        default="/home/hiamichenha/unitree_h1_learn/validation/robojudo_g1_dance_60s_30fps.mp4",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output FPS; simulation still advances at its native timestep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seconds <= 0 or args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise SystemExit("--seconds, --width, --height, and --fps must be positive")

    cfg: RlPipelineCfg = ConfigManager(config_name=args.config).get_cfg()
    pipeline_type = cfg.pipeline_type
    pipeline_class: type[RlPipeline] = getattr(robojudo.pipeline, pipeline_type)
    pipeline = pipeline_class(cfg=cfg)

    # Interactive RoboJuDo sessions start this controller when the user sends
    # the ``[MOTION_FADE_IN]`` key command.  An off-screen recorder has no
    # keyboard event loop, so enable the same controller state explicitly.
    try:
        motion_ctrl = pipeline.ctrl_manager.controllers.BeyondMimicCtrl.inst
    except AttributeError as exc:
        raise RuntimeError(
            "The selected configuration has no BeyondMimicCtrl to supply the dance motion"
        ) from exc
    motion_ctrl.playing = True

    # ``MujocoEnv.step`` normally redraws the interactive GLFW viewer.  Under
    # Xvfb that draw shares a software GL backend with the off-screen recorder
    # and can contaminate captured frames.  The recorder owns rendering below,
    # so suppress only the interactive redraw.
    viewer = getattr(pipeline.env, "viewer", None)
    if viewer is not None:
        viewer.render = lambda: None

    dt = float(pipeline.dt)
    if dt <= 0:
        raise RuntimeError(f"Invalid simulation timestep: {dt}")
    fps = int(args.fps)
    steps = int(math.ceil(args.seconds / dt))
    target_frames = int(round(args.seconds * fps))
    expected_seconds = target_frames / fps

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open MP4 writer: {output}")

    renderer = mujoco.Renderer(pipeline.env.model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.0
    camera.elevation = -10.0
    camera.azimuth = 180.0
    try:
        recorded_frames = 0
        for step in range(steps):
            pipeline.step()
            simulated_time = (step + 1) * dt
            while (
                recorded_frames < target_frames
                and simulated_time + 1e-9 >= (recorded_frames + 1) / fps
            ):
                # Follow the simulated base.  This matches the interactive
                # viewer's framing and keeps the dancing G1 centered while
                # the root translates across the scene.
                camera.lookat[:] = pipeline.env.data.qpos[:3]
                renderer.update_scene(pipeline.env.data, camera=camera)
                rgb = renderer.render()
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                recorded_frames += 1
                if recorded_frames % fps == 0:
                    print(
                        f"recorded_seconds={recorded_frames / fps:.1f}/{expected_seconds:.1f}",
                        flush=True,
                    )
    finally:
        renderer.close()
        writer.release()

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"MP4 was not written: {output}")
    if recorded_frames != target_frames:
        raise RuntimeError(f"Expected {target_frames} frames, wrote {recorded_frames}")
    print(f"config={args.config}")
    print(f"policy={getattr(cfg.policy, 'policy_name', 'unknown')}")
    print(f"motion=assets/motions/g1/beyondmimic/dance1_subject2.npz")
    print("motion_playback=enabled")
    print("interactive_viewer_redraw=disabled_for_capture")
    print(f"dt={dt}")
    print(f"fps={fps}")
    print(f"simulation_steps={steps}")
    print(f"frames={recorded_frames}")
    print(f"duration_seconds={expected_seconds:.3f}")
    print(f"saved_video={output}")


if __name__ == "__main__":
    main()
