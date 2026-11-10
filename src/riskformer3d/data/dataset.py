
"""
RiskFormer-3D V0.2
FlightBench temporal imitation-learning dataset.

Expected directory structure:

data_root/
    rollout_001/
        metadata.csv
        0.npy
        0.txt
        1.npy
        1.txt
        ...

metadata.csv columns:
    step,episode_id,goal_x,goal_y,goal_z

A metadata.csv with verified episode boundaries is required.
Never infer episode boundaries simply from file numbering.

Outputs:
    depth:    [T, 1, 64, 96]
    state:    [T, 13]
    goal_rel: [3]
    action:   [4] -- raw teacher CTBR action
"""

from pathlib import Path
import csv
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class FlightBenchSequenceDataset(Dataset):
    """Load time-aligned FlightBench imitation-learning samples."""

    def __init__(
        self,
        root,
        history=4,
        image_size=(64, 96),
    ):
        self.root = Path(root)
        self.history = int(history)
        self.image_size = tuple(image_size)
        self.samples = []

        if self.history < 1:
            raise ValueError("history must be at least 1")

        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root}"
            )

        rollout_dirs = sorted(
            p for p in self.root.iterdir()
            if p.is_dir()
        )

        if not rollout_dirs:
            raise ValueError("No rollout directories found")

        for folder in rollout_dirs:
            self._index_rollout(folder)

        if not self.samples:
            raise ValueError(
                "No valid temporal windows found. "
                "Check metadata.csv, episode lengths, and history."
            )

    def _index_rollout(self, folder):
        """Build windows without crossing episode boundaries."""

        metadata_path = folder / "metadata.csv"

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing {metadata_path}. "
                "Verified episode metadata is required."
            )

        with metadata_path.open(
            "r", newline="", encoding="utf-8"
        ) as f:
            reader = csv.DictReader(f)
            required = {
                "step", "episode_id",
                "goal_x", "goal_y", "goal_z"
            }

            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"{metadata_path} must contain {required}"
                )

            rows = list(reader)

        rows.sort(key=lambda row: int(row["step"]))

        previous_episode = None
        previous_step = None
        sequence = []

        for row in rows:
            step = int(row["step"])
            episode = row["episode_id"].strip()

            if not episode:
                raise ValueError("Empty episode_id")

            goal = np.array(
                [
                    float(row["goal_x"]),
                    float(row["goal_y"]),
                    float(row["goal_z"]),
                ],
                dtype=np.float32,
            )

            if not np.isfinite(goal).all():
                raise ValueError("Invalid goal coordinates")

            depth_path = folder / f"{step}.npy"
            state_path = folder / f"{step}.txt"

            if not depth_path.is_file():
                raise FileNotFoundError(depth_path)

            if not state_path.is_file():
                raise FileNotFoundError(state_path)

            # A new episode, or a break in step numbering,
            # invalidates the previous temporal history.
            new_segment = (
                previous_episode != episode
                or previous_step is None
                or step != previous_step + 1
            )

            if new_segment:
                sequence = []

                # Skip the first saved frame of each segment:
                # FlightBench may store an uninitialized or
                # previous-episode depth image at this point.
            else:
                sequence.append(step)

            if len(sequence) >= self.history:
                window = tuple(sequence[-self.history:])

                self.samples.append(
                    (folder, window, goal.copy())
                )

            previous_episode = episode
            previous_step = step

    def __len__(self):
        return len(self.samples)

    def _read_depth(self, path):
        """Read a 240x320 depth image in meters."""

        image = np.load(
            path,
            allow_pickle=False,
            mmap_mode="r",
        )

        if image.shape != (240, 320):
            raise ValueError(
                f"{path}: expected (240,320), "
                f"got {image.shape}"
            )

        # Copy data out of the read-only memory map.
        image = np.array(image, dtype=np.float32, copy=True)

        if not np.isfinite(image).all():
            raise ValueError(f"Non-finite depth: {path}")

        if not np.any(image > 0):
            raise ValueError(
                f"All-zero depth image: {path}. "
                "Check sensor validity and episode alignment."
            )

        tensor = torch.from_numpy(image)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        # Nearest-neighbor resizing preserves invalid-depth
        # values instead of mixing them with valid distances.
        tensor = F.interpolate(
            tensor,
            size=self.image_size,
            mode="nearest",
        )

        return tensor.squeeze(0)  # [1,H,W]

    def _read_state_action(self, path):
        """Parse concatenated state history and teacher action."""

        data = np.loadtxt(
            path,
            dtype=np.float32,
            ndmin=1,
        ).reshape(-1)

        # Official logger saves:
        # [13 * state_len values] + [4 teacher actions].
        state_size = len(data) - 4

        if (
            state_size < 13
            or state_size % 13 != 0
        ):
            raise ValueError(
                f"{path}: invalid state/action length "
                f"{len(data)}"
            )

        if not np.isfinite(data).all():
            raise ValueError(f"Non-finite data: {path}")

        # FlightBench concatenates history newest first.
        latest_state = data[:13].copy()
        teacher_action = data[-4:].copy()

        return (
            torch.from_numpy(latest_state),
            torch.from_numpy(teacher_action),
        )

    def __getitem__(self, index):
        folder, steps, goal = self.samples[index]

        depths = []
        states = []
        action = None

        for step in steps:
            depth = self._read_depth(
                folder / f"{step}.npy"
            )

            state, teacher_action = (
                self._read_state_action(
                    folder / f"{step}.txt"
                )
            )

            depths.append(depth)
            states.append(state)

            # Only the teacher action at the latest
            # time step is used as the prediction target.
            action = teacher_action

        depth_history = torch.stack(depths, dim=0)
        state_history = torch.stack(states, dim=0)

        goal_world = torch.from_numpy(goal.copy())

        # Relative target in world coordinates.
        goal_rel = goal_world - state_history[-1, :3]

        return {
            "depth": depth_history,
            "state": state_history,
            "goal_rel": goal_rel,
            "action": action,
        }


def smoke_test():
    """Test loader using synthetic data only."""

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "test_rollout"
        folder.mkdir()

        metadata_path = folder / "metadata.csv"

        with metadata_path.open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)

            writer.writerow([
                "step", "episode_id",
                "goal_x", "goal_y", "goal_z"
            ])

            for step in range(12):
                # Two episodes: steps 0-5 and 6-11.
                episode_id = step // 6

                writer.writerow([
                    step, episode_id, 14.0, 14.0, 2.0
                ])

                # The first image in each episode is
                # deliberately invalid, like the original
                # DAgger logger's first saved image.
                depth_value = (
                    0.0
                    if step in (0, 6)
                    else 1.0 + 0.05 * step
                )

                depth = np.full(
                    (240, 320),
                    depth_value,
                    dtype=np.float32,
                )

                np.save(folder / f"{step}.npy", depth)

                state = np.zeros(13, dtype=np.float32)
                state[0] = float(step)
                state[2] = 2.0
                state[3] = 1.0  # quaternion w

                raw_action = np.array(
                    [0.0, 0.1, -0.1, 0.0],
                    dtype=np.float32,
                )

                np.savetxt(
                    folder / f"{step}.txt",
                    np.concatenate([state, raw_action]),
                )

        dataset = FlightBenchSequenceDataset(
            root=tmp,
            history=3,
        )

        # Each episode has 6 frames. The first is dropped.
        # Remaining 5 frames give 3 valid windows of length 3.
        assert len(dataset) == 6

        sample = dataset[0]

        assert sample["depth"].shape == (3, 1, 64, 96)
        assert sample["state"].shape == (3, 13)
        assert sample["goal_rel"].shape == (3,)
        assert sample["action"].shape == (4,)

        assert torch.isfinite(sample["depth"]).all()
        assert torch.isfinite(sample["state"]).all()

        # First window must not include step 0.
        assert sample["state"][0, 0].item() == 1.0

        # Check the second episode starts at step 7.
        second_episode_sample = dataset[3]
        assert (
            second_episode_sample["state"][0, 0].item()
            == 7.0
        )

        print("Samples:", len(dataset))
        print("Depth:", tuple(sample["depth"].shape))
        print("State:", tuple(sample["state"].shape))
        print("Goal:", tuple(sample["goal_rel"].shape))
        print("Action:", tuple(sample["action"].shape))
        print("Episode boundary test: PASSED")
        print("Dataset smoke test: PASSED")


if __name__ == "__main__":
    smoke_test()
