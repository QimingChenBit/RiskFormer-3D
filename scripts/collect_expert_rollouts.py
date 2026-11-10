
"""
RiskFormer-3D V0.5
FlightBench expert trajectory collection adapter.

Purpose:
- Collect synchronized state/depth/teacher-action records.
- Avoid storing depth images from terminal auto-reset steps.
- Generate metadata compatible with V0.4 audit.
- Verify the entire pipeline using a CPU-only mock environment.

Real collection requires a working FlightBench environment
and a trained teacher policy. The mock teacher is NOT an expert.
"""

import argparse
import csv
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

METADATA_FIELDS = [
    "step",
    "episode_id",
    "goal_x",
    "goal_y",
    "goal_z",
    "obs_step",
    "depth_step",
    "teacher_step",
    "depth_valid",
    "done",
]


class ExpertRecorder:
    """Save one episode of aligned training records."""

    def __init__(self, folder, episode_id, goal_xyz):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)

        self.episode_id = int(episode_id)
        self.goal = np.asarray(goal_xyz, dtype=np.float32)

        if self.goal.shape != (3,):
            raise ValueError("goal_xyz must contain 3 coordinates")

        if not np.isfinite(self.goal).all():
            raise ValueError("Invalid goal coordinates")

        self.step = 0
        self.finished = False

        self.metadata_path = self.folder / "metadata.csv"

        with self.metadata_path.open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=METADATA_FIELDS
            )
            writer.writeheader()

    def append(self, obs, depth, teacher_action, done):
        """
        Save the pre-action observation and expert action.

        done describes the result of executing this action.
        Terminal post-step observations are not saved.
        """

        if self.finished:
            raise RuntimeError("Cannot append after episode end")

        imu = np.asarray(
            obs["imu"], dtype=np.float32
        )

        if imu.ndim != 3 or imu.shape[:2] != (1, 1):
            raise ValueError(
                f"Expected imu [1,1,13*K], got {imu.shape}"
            )

        state = imu[0, 0].copy()

        if len(state) < 13 or len(state) % 13 != 0:
            raise ValueError("Invalid state history length")

        image = np.asarray(depth, dtype=np.float32)

        if image.shape != (240, 320):
            raise ValueError(
                f"Expected depth [240,320], got {image.shape}"
            )

        if not np.isfinite(image).all():
            raise ValueError("Non-finite depth image")

        if not np.any(image > 0):
            raise ValueError(
                "All-zero depth: current V0.2 loader "
                "cannot use this frame"
            )

        action = np.asarray(
            teacher_action, dtype=np.float32
        ).reshape(-1)

        if action.shape != (4,):
            raise ValueError("Teacher action must be 4D")

        if not np.isfinite(state).all():
            raise ValueError("Invalid state values")

        if not np.isfinite(action).all():
            raise ValueError("Invalid teacher action")

        step = self.step

        # Depth image corresponding to the current state.
        np.save(
            self.folder / f"{step}.npy",
            image
        )

        # Official-like format:
        # [13*K state values] + [4 raw teacher actions]
        np.savetxt(
            self.folder / f"{step}.txt",
            np.concatenate([state, action])
        )

        # These are recorder-local logical frame indices.
        # They are not independent sensor timestamps.
        row = {
            "step": step,
            "episode_id": self.episode_id,
            "goal_x": float(self.goal[0]),
            "goal_y": float(self.goal[1]),
            "goal_z": float(self.goal[2]),
            "obs_step": step,
            "depth_step": step,
            "teacher_step": step,
            "depth_valid": 1,
            "done": int(bool(done)),
        }

        with self.metadata_path.open(
            "a", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=METADATA_FIELDS
            )
            writer.writerow(row)

        self.step += 1

        if done:
            self.finished = True


def get_teacher_action(teacher, obs):
    """Obtain raw CTBR action for the current observation."""

    action = np.asarray(
        teacher(obs), dtype=np.float32
    ).reshape(-1)

    if action.shape != (4,):
        raise ValueError("Teacher must return 4 values")

    if not np.isfinite(action).all():
        raise ValueError("Teacher returned invalid action")

    return action


def collect_episode(
    env,
    teacher,
    folder,
    episode_id,
    goal_xyz,
    max_steps=1000,
):
    """
    Collect a single episode.

    Expected FlightBench-like API:
        env.reset(random=True) -> obs

        env.depthStep(action[1,1,4])
            -> obs_next, reward, done, info, depth

    Requires one environment (n_rollout_threads = 1).
    """

    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    obs = env.reset(random=True)

    # FlightBench reset() provides state but not a fresh
    # depth image. The first action is a warm-up step.
    warm_action = get_teacher_action(teacher, obs)

    obs, _, done, _, depth = env.depthStep(
        warm_action.reshape(1, 1, 4)
    )

    if bool(np.asarray(done).reshape(-1)[0]):
        raise RuntimeError(
            "Episode ended during depth warm-up"
        )

    recorder = ExpertRecorder(
        folder=folder,
        episode_id=episode_id,
        goal_xyz=goal_xyz,
    )

    # depth and obs now refer to the first nonterminal
    # post-step frame, according to the FlightBench API.
    current_depth = np.asarray(depth)[0].copy()

    for _ in range(max_steps):
        # Teacher labels the CURRENT observation,
        # not the observation produced by its action.
        action = get_teacher_action(teacher, obs)

        next_obs, _, done, _, next_depth = env.depthStep(
            action.reshape(1, 1, 4)
        )

        terminated = bool(
            np.asarray(done).reshape(-1)[0]
        )

        # Save the state/depth BEFORE this transition,
        # together with the action selected at that state.
        recorder.append(
            obs=obs,
            depth=current_depth,
            teacher_action=action,
            done=terminated,
        )

        if terminated:
            # FlightBench may return a reset observation
            # together with a terminal pre-reset image.
            # Never combine or store that mismatched pair.
            break

        obs = next_obs
        current_depth = np.asarray(next_depth)[0].copy()

    return recorder.step


class MockFlightBench:
    """
    CPU-only test double.

    Simulates the relevant FlightBench reset/depthStep
    behavior, including auto-reset on terminal steps.
    """

    def __init__(self, terminal_step=8):
        self.terminal_step = terminal_step
        self.t = 0

    def _obs(self):
        state = np.zeros(13, dtype=np.float32)
        state[0] = float(self.t)
        state[2] = 2.0
        state[3] = 1.0  # quaternion w

        return {
            "imu": state.reshape(1, 1, 13)
        }

    def reset(self, random=True):
        self.t = 0
        return self._obs()

    def depthStep(self, action):
        action = np.asarray(action)

        if action.shape != (1, 1, 4):
            raise ValueError("Wrong action shape")

        self.t += 1

        # Encoded depth value lets us test alignment.
        value = 1.0 + 0.1 * self.t

        image = np.full(
            (1, 240, 320),
            value,
            dtype=np.float32,
        )

        terminated = self.t >= self.terminal_step

        if terminated:
            # Simulate FlightBench auto-reset behavior:
            # returned state resets, image is still terminal.
            self.t = 0

        return (
            self._obs(),
            np.zeros((1, 1, 1), dtype=np.float32),
            np.array([[terminated]], dtype=bool),
            [{}],
            image,
        )


def mock_teacher(obs):
    """Synthetic action rule; NOT a real navigation expert."""

    x = float(obs["imu"][0, 0, 0])

    return np.array(
        [0.0, 0.02 * x, -0.01 * x, 0.0],
        dtype=np.float32,
    )


def self_test():
    """
    CPU integration test:
    collection -> audit -> dataset -> BC training.
    """

    with tempfile.TemporaryDirectory() as tmp:
        data_root = Path(tmp) / "expert_data"
        data_root.mkdir()

        # Collect two independent mock episodes.
        for episode_id in range(2):
            env = MockFlightBench(terminal_step=8)

            folder = (
                data_root /
                f"rollout_{episode_id:04d}"
            )

            count = collect_episode(
                env=env,
                teacher=mock_teacher,
                folder=folder,
                episode_id=episode_id,
                goal_xyz=(14.0, 14.0, 2.0),
                max_steps=20,
            )

            assert count == 7, count

        print("Mock collection: PASSED")

        # Verify actual state/depth/label correspondence.
        folder = data_root / "rollout_0000"

        for step in range(7):
            depth = np.load(folder / f"{step}.npy")
            data = np.loadtxt(
                folder / f"{step}.txt"
            ).reshape(-1)

            current_x = float(data[0])

            expected_x = step + 1
            expected_depth = 1.0 + 0.1 * expected_x

            assert np.isclose(current_x, expected_x)
            assert np.isclose(
                float(depth[0, 0]), expected_depth
            )

            expected_action = np.array([
                0.0,
                0.02 * current_x,
                -0.01 * current_x,
                0.0,
            ])

            assert np.allclose(
                data[-4:], expected_action
            )

        print("State/depth/action alignment: PASSED")

        # Run existing V0.4 audit.
        from audit_imitation_data import audit_dataset

        errors, frames, windows = audit_dataset(
            data_root, history=3
        )

        assert not errors, errors
        assert frames == 14, frames
        assert windows == 8, windows

        print("V0.4 integrity audit: PASSED")

        # Run existing V0.2 dataset.
        sys.path.insert(0, str(ROOT / "src"))

        from riskformer3d.data.dataset import (
            FlightBenchSequenceDataset,
        )

        dataset = FlightBenchSequenceDataset(
            data_root,
            history=3,
        )

        assert len(dataset) == 8

        sample = dataset[0]

        assert sample["depth"].shape == (3, 1, 64, 96)
        assert sample["state"].shape == (3, 13)
        assert sample["goal_rel"].shape == (3,)
        assert sample["action"].shape == (4,)

        print("V0.2 sequence loading: PASSED")

        # Use existing V0.3 training pipeline for one epoch.
        from train_imitation import train

        args = SimpleNamespace(
            history=3,
            batch_size=2,
            epochs=1,
            lr=1e-3,
            save_to=None,
        )

        train(args, train_root=data_root)

        print("V0.5 collector smoke test: PASSED")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run CPU-only simulated collection and tests"
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
    else:
        parser.error(
            "Real FlightBench integration is not configured. "
            "Use --self-test."
        )


if __name__ == "__main__":
    main()

