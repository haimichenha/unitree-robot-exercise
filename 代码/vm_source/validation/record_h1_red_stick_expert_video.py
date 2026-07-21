"""Record a complete, labelled H1 expert-policy red-stick transfer video.

This is an expert-trajectory demonstration for data-collection evidence.  It
is intentionally separate from ACT inference videos and writes the label into
each frame so the two kinds of evidence cannot be confused.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# This script is intentionally runnable by absolute path from the validation
# directory, so explicitly expose the project root before local imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from DataCollecter.h1_policy import H1Policy
from Mujoco_env.envs.h1_ik import make_sim_env


R2D = 180 / 3.1415926
DEFAULT_PEG_POSE = np.array([0.40, 0.10, 1.03, 0.0, 0.0, 0.0], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="validation/h1_red_stick_expert_demo_30s.mp4",
        help="MP4 output path, relative to the project root by default.",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--force-exit",
        action="store_true",
        help="Opt in to os._exit(0) after the MP4 is closed for the VM native teardown issue.",
    )
    args = parser.parse_args()
    if args.fps < 1:
        raise SystemExit("--fps must be positive")
    return args


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    env = make_sim_env(freq=50)
    policy = H1Policy(task_name="Transmitting the red stick")
    env.reset(DEFAULT_PEG_POSE)

    writer: cv2.VideoWriter | None = None
    try:
        for step in range(600):
            end_effector_pose = policy.predict(DEFAULT_PEG_POSE, step)
            joint_action = env.ik_func(end_effector_pose)
            env.step_all_simple(joint_action)

            images = env._get_image_obs()["images"]
            top = cv2.resize(images["top"], (320, 240), interpolation=cv2.INTER_LINEAR)
            angle = cv2.resize(images["angle"], (320, 240), interpolation=cv2.INTER_LINEAR)
            frame = np.hstack((top, angle))
            cv2.putText(
                frame,
                "Expert red-stick transfer demonstration",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"step {step + 1}/600",
                (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Unable to open MP4 writer: {output}")
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()

    print("task=Transmitting the red stick")
    print("video_kind=expert_policy_demonstration")
    print("rollout_steps=600")
    print(f"saved_video={output}")
    if args.force_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
