"""Record an auditable RoboJuDo locomotion-to-dance policy-switch demo.

The helper uses the repository's ``g1_locomimic_beyondmimic`` configuration.
It injects release events into its existing ``KeyboardCtrl`` event queue, so
the tested path is the same keyboard-trigger mapping used by the interactive
pipeline: ``[`` -> ``[POLICY_MIMIC]`` and ``]`` -> ``[POLICY_LOCO]``.

The recording is off-screen for repeatability.  An English on-frame status
overlay identifies the exact key mapping, active policy, and transition phase.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np
import robojudo.pipeline

from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="g1_locomimic_beyondmimic")
    parser.add_argument("--seconds", type=float, default=55.0)
    parser.add_argument("--walk-before", type=float, default=15.0)
    parser.add_argument("--dance-seconds", type=float, default=25.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--output",
        default="/home/hiamichenha/unitree_h1_learn/validation/robojudo_g1_walk_dance_switch_55s_20fps.mp4",
    )
    return parser.parse_args()


def put_keyboard_release(keyboard_ctrl, key: str) -> None:
    """Queue the same release event that the interactive keyboard controller consumes."""
    keyboard_ctrl.event_queue.put({"type": "keyboard", "name": key, "pressed": False})


def overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = frame.copy()
    row_h = 25
    cv2.rectangle(canvas, (0, 0), (640, 18 + row_h * len(lines)), (10, 10, 10), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (12, 24 + row_h * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return canvas


def main() -> None:
    args = parse_args()
    if (
        args.seconds <= 0
        or args.walk_before <= 0
        or args.dance_seconds <= 0
        or args.fps <= 0
        or args.width <= 0
        or args.height <= 0
    ):
        raise SystemExit("All durations, FPS, width, and height must be positive")
    switch_to_dance_at = args.walk_before
    switch_to_walk_at = args.walk_before + args.dance_seconds
    if switch_to_walk_at >= args.seconds:
        raise SystemExit("--walk-before + --dance-seconds must be smaller than --seconds")

    cfg: RlPipelineCfg = ConfigManager(config_name=args.config).get_cfg()
    pipeline_type = cfg.pipeline_type
    pipeline_class = getattr(robojudo.pipeline, pipeline_type)
    pipeline = pipeline_class(cfg=cfg)

    try:
        keyboard_ctrl = pipeline.ctrl_manager.controllers.KeyboardCtrl.inst
        loco_policy = pipeline.policy_manager.policy_by_id(0).policy
    except AttributeError as exc:
        raise RuntimeError("Expected KeyboardCtrl and locomotion policy were not available") from exc

    # Prevent the Xvfb interactive viewer from contaminating the separate
    # off-screen renderer.  Pipeline state/control execution remains intact.
    if viewer := getattr(pipeline.env, "viewer", None):
        viewer.render = lambda: None

    dt = float(pipeline.dt)
    if dt <= 0:
        raise RuntimeError(f"Invalid simulation timestep: {dt}")
    total_steps = int(math.ceil(args.seconds / dt))
    target_frames = int(round(args.seconds * args.fps))
    expected_duration = target_frames / args.fps

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open MP4 writer: {output}")

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.0
    camera.elevation = -10.0
    camera.azimuth = 180.0
    renderer = mujoco.Renderer(pipeline.env.model, height=args.height, width=args.width)

    dance_key_sent = False
    walk_key_sent = False
    recorded_frames = 0
    event_log: list[str] = []
    status_counts: dict[str, int] = {}
    root_trace: list[str] = []
    last_trace_second = -1
    try:
        for step in range(total_steps):
            simulated_time = step * dt
            if not dance_key_sent and simulated_time >= switch_to_dance_at:
                put_keyboard_release(keyboard_ctrl, "[")
                dance_key_sent = True
                event_log.append(f"t={simulated_time:.2f}: key=[ -> [POLICY_MIMIC]")
            if not walk_key_sent and simulated_time >= switch_to_walk_at:
                put_keyboard_release(keyboard_ctrl, "]")
                walk_key_sent = True
                event_log.append(f"t={simulated_time:.2f}: key=] -> [POLICY_LOCO]")

            # The AMO locomotion policy receives a forward velocity command
            # while it is active; the command is intentionally not supplied
            # during the BeyondMimic dance segment.
            if pipeline.policy_manager.current_policy_id == 0:
                loco_policy.cmd[2] = 0.55
            pipeline.step()

            current_time = (step + 1) * dt
            current_id = int(pipeline.policy_manager.current_policy_id)
            interp = pipeline.policy_manager.interp_state.name
            root_xy = pipeline.env.data.qpos[:2].copy()
            trace_second = int(current_time)
            if trace_second % 5 == 0 and trace_second != last_trace_second:
                root_trace.append(f"t={current_time:.1f}:xy=({root_xy[0]:.3f},{root_xy[1]:.3f})")
                last_trace_second = trace_second
            if current_id == 0 and not dance_key_sent:
                phase = "WALK: locomotion policy"
                key_hint = "Keyboard ] -> [POLICY_LOCO]"
            elif current_id == 0 and dance_key_sent and not walk_key_sent:
                phase = "TRANSITION: walk -> dance"
                key_hint = "Keyboard [ -> [POLICY_MIMIC]"
            elif current_id != 0 and not walk_key_sent:
                phase = "DANCE: BeyondMimic Dance_wose"
                key_hint = "Keyboard [ triggered mimic policy"
            elif current_id == 0 and interp == "IDLE":
                phase = "WALK: locomotion resumed"
                key_hint = "Keyboard ] triggered loco policy"
            else:
                phase = "TRANSITION/RETURN: dance -> walk"
                key_hint = "Keyboard ] -> [POLICY_LOCO]"
            status_counts[f"policy={current_id};interp={interp}"] = (
                status_counts.get(f"policy={current_id};interp={interp}", 0) + 1
            )

            while recorded_frames < target_frames and current_time + 1e-9 >= (recorded_frames + 1) / args.fps:
                camera.lookat[:] = pipeline.env.data.qpos[:3]
                renderer.update_scene(pipeline.env.data, camera=camera)
                rgb = renderer.render()
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                frame = overlay(
                    frame,
                    [
                        f"t={current_time:05.2f}s | {phase}",
                        key_hint,
                        f"active_policy={current_id} | interpolation={interp}",
                        f"root_xy=({root_xy[0]:+.2f}, {root_xy[1]:+.2f}) | forward_cmd=0.55",
                    ],
                )
                writer.write(frame)
                recorded_frames += 1
    finally:
        renderer.close()
        writer.release()

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"MP4 was not written: {output}")
    if recorded_frames != target_frames:
        raise RuntimeError(f"Expected {target_frames} frames, wrote {recorded_frames}")
    if not (dance_key_sent and walk_key_sent):
        raise RuntimeError("Not all planned keyboard-trigger events were sent")

    print(f"config={args.config}")
    print("keyboard_mapping=[ -> [POLICY_MIMIC]; ] -> [POLICY_LOCO]")
    print("mimic_policy=Dance_wose")
    print("locomotion_command=forward_vel_x=0.55")
    print(f"dt={dt}")
    print(f"fps={args.fps}")
    print(f"simulation_steps={total_steps}")
    print(f"frames={recorded_frames}")
    print(f"duration_seconds={expected_duration:.3f}")
    for item in event_log:
        print(f"keyboard_event={item}")
    print("root_xy_trace=" + ";".join(root_trace))
    print("status_counts=" + ";".join(f"{key}:{value}" for key, value in sorted(status_counts.items())))
    print(f"saved_video={output}")


if __name__ == "__main__":
    main()
