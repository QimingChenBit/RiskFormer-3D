
"""
RiskFormer-3D V0.3
Offline Behavior Cloning (BC) trainer.

Two modes:
1. --smoke-test: verify the training pipeline with synthetic data.
2. --train-root: train using verified expert trajectory data.

This script does NOT run FlightBench simulation.
"""

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 找到项目根目录，使脚本能够导入 src 中的模块
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from riskformer3d.data.dataset import FlightBenchSequenceDataset
from riskformer3d.models.transformer import RiskFormer


def make_test_rollout(root, name, shift=0.0):
    """
    创建一条模拟轨迹。
    仅用于验证训练代码，不代表真实专家数据。
    """
    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)

    with (folder / "metadata.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "episode_id",
            "goal_x", "goal_y", "goal_z"
        ])

        for step in range(9):
            writer.writerow([step, 0, 14.0, 14.0, 2.0])

            # 第一帧模拟官方采集器中的未初始化深度图
            depth_value = (
                0.0 if step == 0 else 1.0 + 0.05 * step + shift
            )
            depth = np.full(
                (240, 320), depth_value, dtype=np.float32
            )
            np.save(folder / f"{step}.npy", depth)

            # 13维状态：位置、四元数、速度、角速度
            state = np.zeros(13, dtype=np.float32)
            state[0] = 0.1 * step + shift
            state[2] = 2.0
            state[3] = 1.0  # 单位四元数

            # 人工标签：只用于测试监督训练
            teacher_action = np.array(
                [0.0, 0.08, -0.05, 0.02],
                dtype=np.float32
            )

            np.savetxt(
                folder / f"{step}.txt",
                np.concatenate([state, teacher_action])
            )


def evaluate(model, loader):
    """验证阶段：不更新模型参数。"""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            prediction = model(
                batch["depth"],
                batch["state"],
                batch["goal_rel"]
            )
            target = batch["action"]

            loss = F.mse_loss(prediction, target)

            total_loss += loss.item() * target.shape[0]
            total_samples += target.shape[0]

    return total_loss / total_samples


def train(args, train_root, val_root=None):
    """训练四维 CTBR 动作预测网络。"""

    # CPU-only，限制线程数
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)

    # 1. 加载训练数据
    train_dataset = FlightBenchSequenceDataset(
        root=train_root,
        history=args.history
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    # 2. 验证数据必须来自独立目录
    val_loader = None
    if val_root is not None:
        val_dataset = FlightBenchSequenceDataset(
            root=val_root,
            history=args.history
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )

    # 3. 初始化已经通过测试的 RiskFormer
    model = RiskFormer(
        d_model=64,
        nhead=4,
        temporal_layers=2,
        max_history=max(8, args.history)
    ).to("cpu")

    # 4. AdamW 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    print("Device: cpu")
    print("Training samples:", len(train_dataset))
    print(
        "Model parameters:",
        sum(p.numel() for p in model.parameters())
    )

    # 5. 开始训练
    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            depth = batch["depth"]       # [B,T,1,64,96]
            state = batch["state"]       # [B,T,13]
            goal = batch["goal_rel"]     # [B,3]
            target = batch["action"]     # [B,4]

            # 前向传播：得到原始CTBR动作
            prediction = model(depth, state, goal)

            assert prediction.shape == target.shape

            # Behavior Cloning监督损失
            loss = F.mse_loss(prediction, target)

            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite training loss")

            # 反向传播并更新参数
            optimizer.zero_grad()
            loss.backward()

            # 裁剪梯度，避免梯度过大
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item() * target.shape[0]
            total_samples += target.shape[0]

        train_loss = total_loss / total_samples

        message = (
            f"Epoch {epoch}/{args.epochs} "
            f"| train_mse={train_loss:.6f}"
        )

        if val_loader is not None:
            val_loss = evaluate(model, val_loader)
            message += f" | val_mse={val_loss:.6f}"

        print(message)

    # 6. 只在明确指定路径时保存检查点
    if args.save_to:
        save_path = Path(args.save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "history": args.history,
                "model_config": {
                    "d_model": 64,
                    "nhead": 4,
                    "temporal_layers": 2,
                    "max_history": max(8, args.history)
                }
            },
            save_path
        )
        print("Checkpoint saved:", save_path)

    print("Training pipeline: PASSED")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--train-root", type=str)
    parser.add_argument("--val-root", type=str)
    parser.add_argument("--save-to", type=str)

    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()

    if args.history < 1 or args.batch_size < 1 or args.epochs < 1:
        parser.error("history, batch-size and epochs must be positive")

    if args.smoke_test:
        print("MODE: SYNTHETIC SMOKE TEST")
        print("These are NOT real expert trajectories.")

        with tempfile.TemporaryDirectory() as tmp:
            train_root = Path(tmp) / "train"
            val_root = Path(tmp) / "val"

            make_test_rollout(train_root, "rollout_001")
            make_test_rollout(val_root, "rollout_002", shift=0.2)

            # 测试模式不写入正式模型检查点
            args.save_to = None
            train(args, train_root, val_root)

    else:
        if args.train_root is None:
            parser.error(
                "Specify --train-root or use --smoke-test"
            )

        train(args, args.train_root, args.val_root)


if __name__ == "__main__":
    main()
