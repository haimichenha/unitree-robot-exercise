"""Print the HDF5 evidence required by the Unitree H1 practical assignment.

This utility is read-only: it never changes an episode file.  Run it from the
repository root after recording data, for example:

    python3 DataCollecter/analyze_hdf5.py DataCollecter/dataset/episode_0.hdf5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def dataset_paths(root: h5py.File) -> list[str]:
    """Return every dataset path, including the camera-image datasets."""

    paths: list[str] = []

    def collect(name: str, value: h5py.Dataset | h5py.Group) -> None:
        if isinstance(value, h5py.Dataset):
            paths.append(f"/{name}")

    root.visititems(collect)
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read one H1 episode and print its keys, image shape, and first five actions."
    )
    parser.add_argument("episode", type=Path, help="Path to episode_N.hdf5")
    parser.add_argument("--camera", default="top", help="Camera name below /observations/images (default: top)")
    args = parser.parse_args()

    episode = args.episode.expanduser().resolve()
    if not episode.is_file():
        raise SystemExit(f"HDF5 episode not found: {episode}")

    image_path = f"/observations/images/{args.camera}"
    with h5py.File(episode, "r") as root:
        required = ("/action", image_path)
        missing = [path for path in required if path not in root]
        if missing:
            available = ", ".join(dataset_paths(root))
            raise SystemExit(f"Missing dataset(s): {missing}. Available datasets: {available}")

        action = root["/action"]
        image = root[image_path]
        print(f"file: {episode}")
        print("keys:")
        for path in dataset_paths(root):
            print(f"  {path}")
        print(f"{image_path} shape: {image.shape}; dtype: {image.dtype}")
        print(f"/action shape: {action.shape}; dtype: {action.dtype}")
        print("/action first 5 rows:")
        print(np.asarray(action[:5]))


if __name__ == "__main__":
    main()
