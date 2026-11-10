
"""
RiskFormer-3D V0.4
Expert trajectory data integrity audit.

This checks:
1. Required metadata
2. Observation/depth/teacher step alignment
3. Episode and reset boundaries
4. Depth and action data validity
5. Available temporal training windows

Important:
Metadata must be recorded by the real collector.
It cannot be reconstructed reliably from file names alone.
"""

import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = {
    "step",
    "episode_id",
    "goal_x", "goal_y", "goal_z",
    "obs_step",
    "depth_step",
    "teacher_step",
    "depth_valid",
    "done",
}


def audit_dataset(root, history=4):
    root = Path(root)
    errors = []
    frames_checked = 0
    windows = 0

    if history < 1:
        return ["history must be >= 1"], 0, 0

    if not root.is_dir():
        return [f"Directory not found: {root}"], 0, 0

    folders = sorted(p for p in root.iterdir() if p.is_dir())

    if not folders:
        return ["No rollout directories found"], 0, 0

    for folder in folders:
        meta_path = folder / "metadata.csv"

        if not meta_path.is_file():
            errors.append(f"{folder.name}: missing metadata.csv")
            continue

        with meta_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            missing = REQUIRED_COLUMNS - set(
                reader.fieldnames or []
            )

            if missing:
                errors.append(
                    f"{folder.name}: missing columns {sorted(missing)}"
                )
                continue

            rows = list(reader)

        previous_step = None
        previous_episode = None
        previous_done = False

        episode_lengths = {}
        seen_episodes = set()

        for row in rows:
            try:
                step = int(row["step"])
                episode = row["episode_id"].strip()

                if not episode:
                    raise ValueError("empty episode_id")

                obs_step = int(row["obs_step"])
                depth_step = int(row["depth_step"])
                teacher_step = int(row["teacher_step"])

                depth_valid = int(row["depth_valid"])
                done = int(row["done"])

                if depth_valid not in (0, 1):
                    raise ValueError("depth_valid must be 0 or 1")

                if done not in (0, 1):
                    raise ValueError("done must be 0 or 1")

                goal = np.array([
                    float(row["goal_x"]),
                    float(row["goal_y"]),
                    float(row["goal_z"])
                ])

                if not np.isfinite(goal).all():
                    raise ValueError("non-finite goal")

                # Check step numbering
                if previous_step is not None:
                    if step != previous_step + 1:
                        errors.append(
                            f"{folder.name}: discontinuous step {step}"
                        )

                new_episode = episode != previous_episode

                if new_episode:
                    if previous_episode is not None:
                        if not previous_done:
                            errors.append(
                                f"{folder.name}: episode changed "
                                f"without previous done at step {step}"
                            )

                    if episode in seen_episodes:
                        errors.append(
                            f"{folder.name}: episode ID reused: {episode}"
                        )

                    seen_episodes.add(episode)
                    episode_lengths[episode] = 0

                elif previous_done:
                    errors.append(
                        f"{folder.name}: step {step} continues "
                        f"an episode after done"
                    )

                episode_lengths[episode] += 1

                # Observation and teacher actions must refer
                # to the current state step.
                if obs_step != step:
                    errors.append(
                        f"{folder.name}: observation mismatch "
                        f"at step {step}"
                    )

                if teacher_step != step:
                    errors.append(
                        f"{folder.name}: teacher mismatch "
                        f"at step {step}"
                    )

                # Only the first frame of an episode may be
                # marked invalid in the current V0.3 loader.
                if depth_valid == 0:
                    if not new_episode:
                        errors.append(
                            f"{folder.name}: invalid depth "
                            f"inside episode at step {step}"
                        )
                elif depth_step != step:
                    errors.append(
                        f"{folder.name}: depth mismatch at "
                        f"step {step}; depth_step={depth_step}"
                    )

                # Check actual image file
                depth_path = folder / f"{step}.npy"

                depth = np.load(
                    depth_path,
                    allow_pickle=False,
                    mmap_mode="r"
                )

                if depth.shape != (240, 320):
                    raise ValueError(
                        f"wrong depth shape: {depth.shape}"
                    )

                if not np.isfinite(depth).all():
                    raise ValueError("non-finite depth values")

                if depth_valid and not np.any(depth > 0):
                    raise ValueError("valid frame is all-zero")

                # Check state and teacher action
                state_path = folder / f"{step}.txt"

                data = np.loadtxt(
                    state_path,
                    dtype=np.float32
                ).reshape(-1)

                if len(data) < 17:
                    raise ValueError("state/action vector too short")

                if (len(data) - 4) % 13 != 0:
                    raise ValueError(
                        f"invalid state/action length: {len(data)}"
                    )

                if not np.isfinite(data).all():
                    raise ValueError(
                        "non-finite state or teacher action"
                    )

                frames_checked += 1
                previous_step = step
                previous_episode = episode
                previous_done = bool(done)

            except Exception as exc:
                errors.append(
                    f"{folder.name}, row step="
                    f"{row.get('step', '?')}: {exc}"
                )

        # V0.3 dataset discards the first frame of every
        # episode. Count possible remaining history windows.
        for length in episode_lengths.values():
            windows += max(0, length - history)

    return errors, frames_checked, windows


def make_synthetic_rollout(root):
    """Create two synthetic episodes for audit tests."""

    folder = Path(root) / "rollout_001"
    folder.mkdir(parents=True)

    columns = [
        "step", "episode_id",
        "goal_x", "goal_y", "goal_z",
        "obs_step", "depth_step",
        "teacher_step", "depth_valid", "done"
    ]

    with (folder / "metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        for step in range(12):
            episode = step // 6
            first_frame = step in (0, 6)
            done = step in (5, 11)

            writer.writerow({
                "step": step,
                "episode_id": episode,
                "goal_x": 14,
                "goal_y": 14,
                "goal_z": 2,
                "obs_step": step,
                "depth_step": -1 if first_frame else step,
                "teacher_step": step,
                "depth_valid": 0 if first_frame else 1,
                "done": int(done),
            })

            value = 0.0 if first_frame else 1.0 + step * 0.05

            np.save(
                folder / f"{step}.npy",
                np.full(
                    (240, 320),
                    value,
                    dtype=np.float32
                )
            )

            state = np.zeros(13, dtype=np.float32)
            state[0] = float(step)
            state[2] = 2.0
            state[3] = 1.0

            teacher = np.array(
                [0.0, 0.1, -0.1, 0.0],
                dtype=np.float32
            )

            np.savetxt(
                folder / f"{step}.txt",
                np.concatenate([state, teacher])
            )


def self_test():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        good = base / "good"
        bad = base / "bad"

        make_synthetic_rollout(good)

        errors, frames, windows = audit_dataset(
            good, history=3
        )

        assert not errors, errors
        assert frames == 12
        assert windows == 6

        # Also test compatibility with V0.3 dataset.
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "src"))

        from riskformer3d.data.dataset import (
            FlightBenchSequenceDataset
        )

        dataset = FlightBenchSequenceDataset(
            good, history=3
        )

        assert len(dataset) == windows

        print("Valid rollout test: PASSED")
        print("Temporal window test: PASSED")

        # Introduce a deliberate sensor timing error.
        shutil.copytree(good, bad)

        metadata = bad / "rollout_001" / "metadata.csv"

        with metadata.open(
            "r", encoding="utf-8", newline=""
        ) as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            rows = list(reader)

        rows[3]["depth_step"] = "2"

        with metadata.open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        errors, _, _ = audit_dataset(
            bad, history=3
        )

        assert any(
            "depth mismatch" in error
            for error in errors
        )

        print("Misalignment detection: PASSED")
        print("V0.4 data audit smoke test: PASSED")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=str)
    parser.add_argument("--history", type=int, default=4)

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.root:
        parser.error("Provide --root or --self-test")

    errors, frames, windows = audit_dataset(
        args.root, args.history
    )

    print("Frames checked:", frames)
    print("Potential training windows:", windows)
    print("Integrity issues:", len(errors))

    for error in errors[:30]:
        print("ERROR:", error)

    if errors:
        raise SystemExit(1)

    print("Data audit: PASSED")


if __name__ == "__main__":
    main()
