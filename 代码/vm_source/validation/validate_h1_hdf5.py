"""Validate H1 ACT HDF5 episodes without modifying the dataset."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np

EXPECTED_DIM = 32
REAL_DIM = 17
CAMERAS = ("top", "angle")
IMAGE_SHAPE_SUFFIX = (240, 320, 3)
EP_RE = re.compile(r"^episode_(\d+)\.hdf5$")


def inspect_episode(path: Path) -> dict:
    result = {"file": path.name, "valid": True, "errors": []}
    with h5py.File(path, "r") as root:
        required = ("/observations/qpos", "/action", "/timestamp", "/task") + tuple(
            f"/observations/images/{camera}" for camera in CAMERAS
        )
        missing = [key for key in required if key not in root]
        if missing:
            result["valid"] = False
            result["errors"].append({"missing": missing})
            return result
        qpos = root["/observations/qpos"][()]
        action = root["/action"][()]
        timestamp = root["/timestamp"][()]
        time_steps = qpos.shape[0] if qpos.ndim == 2 else 0
        result.update(
            qpos_shape=list(qpos.shape),
            action_shape=list(action.shape),
            qpos_dtype=str(qpos.dtype),
            action_dtype=str(action.dtype),
            timestamp_shape=list(timestamp.shape),
            image_shapes={camera: list(root[f"/observations/images/{camera}"].shape) for camera in CAMERAS},
            image_dtypes={camera: str(root[f"/observations/images/{camera}"].dtype) for camera in CAMERAS},
        )
        shape_ok = qpos.shape == (time_steps, EXPECTED_DIM) and action.shape == (time_steps, EXPECTED_DIM)
        timestamp_ok = timestamp.shape == (time_steps,)
        image_ok = all(root[f"/observations/images/{camera}"].shape == (time_steps, *IMAGE_SHAPE_SUFFIX) for camera in CAMERAS)
        image_dtype_ok = all(root[f"/observations/images/{camera}"].dtype == np.uint8 for camera in CAMERAS)
        numerical_ok = bool(np.isfinite(qpos).all() and np.isfinite(action).all() and np.isfinite(timestamp).all())
        tail_zero = bool(np.abs(qpos[:, REAL_DIM:]).max(initial=0) == 0 and np.abs(action[:, REAL_DIM:]).max(initial=0) == 0)
        timestamp_monotonic = bool(np.all(np.diff(timestamp) >= 0))
        result.update(
            frames=int(time_steps),
            schema_shape_ok=shape_ok,
            timestamp_shape_ok=timestamp_ok,
            image_shape_ok=image_ok,
            image_dtype_ok=image_dtype_ok,
            finite=numerical_ok,
            tail_17_to_31_all_zero=tail_zero,
            timestamp_monotonic=timestamp_monotonic,
            qpos_real17_min_deg=float(qpos[:, :REAL_DIM].min(initial=0)),
            qpos_real17_max_deg=float(qpos[:, :REAL_DIM].max(initial=0)),
            action_real17_min_deg=float(action[:, :REAL_DIM].min(initial=0)),
            action_real17_max_deg=float(action[:, :REAL_DIM].max(initial=0)),
        )
        if not all((shape_ok, timestamp_ok, image_ok, image_dtype_ok, numerical_ok, tail_zero, timestamp_monotonic)):
            result["valid"] = False
            result["errors"].append("schema, finite-value, tail-padding, or timestamp validation failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset folder does not exist: {dataset_dir}")
    files = sorted((p for p in dataset_dir.iterdir() if p.is_file() and EP_RE.match(p.name)), key=lambda p: int(EP_RE.match(p.name).group(1)))
    all_items = sorted(p.name for p in dataset_dir.iterdir())
    ids = [int(EP_RE.match(p.name).group(1)) for p in files]
    expected_ids = list(range(len(files)))
    report = {
        "dataset_dir": str(dataset_dir),
        "episode_files": [p.name for p in files],
        "non_episode_items": [name for name in all_items if name not in {p.name for p in files}],
        "consecutive_ids_from_zero": ids == expected_ids,
        "episodes": [inspect_episode(p) for p in files],
    }
    report["valid"] = bool(files) and report["consecutive_ids_from_zero"] and not report["non_episode_items"] and all(item["valid"] for item in report["episodes"])
    report["summary"] = {
        "episode_count": len(files),
        "total_frames": int(sum(item.get("frames", 0) for item in report["episodes"])),
        "failure_count": int(sum(not item["valid"] for item in report["episodes"])),
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
