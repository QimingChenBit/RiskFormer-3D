
"""
RiskFormer-3D V0.1
Lightweight Spatio-Temporal Transformer for UAV Navigation.

Input:
    depth:    [B, T, 1, 64, 96]  (meters)
    state:    [B, T, 13]
    goal_rel: [B, 3]             (relative goal position)

Output:
    raw_action: [B, 4]

The FlightBench environment applies tanh(raw_action).
"""

import torch
import torch.nn as nn


class DepthEncoder(nn.Module):
    """CNN + Spatial Transformer for depth images."""

    def __init__(self, d_model=64, nhead=4):
        super().__init__()

        # CNN将深度图转换为紧凑的空间特征
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),

            nn.Conv2d(64, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),

            # 固定成4×6个空间Token，降低CPU计算量
            nn.AdaptiveAvgPool2d((4, 6))
        )

        self.num_tokens = 24

        # 可学习的空间位置编码
        self.spatial_pos = nn.Parameter(
            torch.zeros(1, self.num_tokens, d_model)
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            batch_first=True
        )

        self.spatial_transformer = nn.TransformerEncoder(
            layer,
            num_layers=1
        )

        nn.init.normal_(self.spatial_pos, std=0.02)

    def forward(self, depth):
        # depth: [B*T, 1, 64, 96]

        x = self.cnn(depth)
        # [B*T, d_model, 4, 6]

        x = x.flatten(2).transpose(1, 2)
        # [B*T, 24, d_model]

        x = x + self.spatial_pos

        x = self.spatial_transformer(x)
        # [B*T, 24, d_model]

        # 对空间Token进行平均池化
        return x.mean(dim=1)
        # [B*T, d_model]


class RiskFormer(nn.Module):
    """Spatio-temporal navigation policy."""

    def __init__(
        self,
        d_model=64,
        nhead=4,
        temporal_layers=2,
        max_history=8
    ):
        super().__init__()

        self.max_history = max_history

        # 1. 视觉编码
        self.depth_encoder = DepthEncoder(
            d_model=d_model,
            nhead=nhead
        )

        # 2. 运动状态编码
        self.state_encoder = nn.Sequential(
            nn.Linear(13, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # 3. 相对目标编码
        self.goal_encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # 4. 多模态特征融合
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # 5. 时间位置编码
        self.temporal_pos = nn.Parameter(
            torch.zeros(1, max_history, d_model)
        )

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            batch_first=True
        )

        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer,
            num_layers=temporal_layers
        )

        # 6. CTBR动作输出头
        self.action_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4)
        )

        nn.init.normal_(self.temporal_pos, std=0.02)

    def forward(self, depth, state, goal_rel):
        """
        depth: [B,T,1,64,96], depth in meters
        state: [B,T,13]
        goal_rel: [B,3]
        """

        if depth.ndim != 5:
            raise ValueError("depth must be [B,T,1,64,96]")

        B, T, C, H, W = depth.shape

        if (C, H, W) != (1, 64, 96):
            raise ValueError("Expected depth shape [B,T,1,64,96]")

        if state.shape != (B, T, 13):
            raise ValueError("Expected state shape [B,T,13]")

        if goal_rel.shape != (B, 3):
            raise ValueError("Expected goal_rel shape [B,3]")

        if not (1 <= T <= self.max_history):
            raise ValueError("Invalid history length")

        # 把FlightBench中无效的0深度暂按远距离处理
        # 这是第一版预处理假设，后续需要结合渲染结果验证
        depth = depth.float()
        depth = torch.nan_to_num(
            depth, nan=4.1, posinf=4.1, neginf=4.1
        )

        depth = torch.where(
            depth > 0,
            depth,
            torch.full_like(depth, 4.1)
        )

        depth = depth.clamp(0.0, 4.1) / 4.1

        # 空间编码
        visual = self.depth_encoder(
            depth.reshape(B * T, 1, 64, 96)
        ).reshape(B, T, -1)

        # 状态编码
        state_features = self.state_encoder(state.float())

        # 目标编码，复制到每个时间步
        goal_features = self.goal_encoder(goal_rel.float())
        goal_features = goal_features.unsqueeze(1).expand(-1, T, -1)

        # 特征融合
        fused = self.fusion(
            torch.cat(
                [visual, state_features, goal_features],
                dim=-1
            )
        )

        # 时间编码
        fused = fused + self.temporal_pos[:, :T, :]

        temporal_features = self.temporal_transformer(fused)

        # 取最新时间步对应的特征
        latest = temporal_features[:, -1, :]

        # 输出原始4维CTBR动作
        # 不在此处tanh，FlightBench会进行tanh转换
        return self.action_head(latest)


if __name__ == "__main__":

    # 仅使用CPU
    device = torch.device("cpu")
    torch.set_num_threads(4)

    print("Device:", device)

    model = RiskFormer(
        d_model=64,
        nhead=4,
        temporal_layers=2,
        max_history=8
    ).to(device)

    # 构造测试输入：2个样本，每个样本4帧
    B = 2
    T = 4

    depth = torch.rand(B, T, 1, 64, 96) * 4.1

    # 使用静止姿态作为基本测试状态
    state = torch.zeros(B, T, 13)
    state[..., 3] = 1.0  # 单位四元数w=1

    goal_rel = torch.tensor([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0]
    ])

    # 前向传播
    raw_action = model(depth, state, goal_rel)

    # 对应环境内部的动作归一化
    normalized_action = torch.tanh(raw_action)

    print("Depth shape:", tuple(depth.shape))
    print("State shape:", tuple(state.shape))
    print("Goal shape:", tuple(goal_rel.shape))
    print("Raw action shape:", tuple(raw_action.shape))
    print("Normalized actions:")
    print(normalized_action.detach())

    assert raw_action.shape == (B, 4)
    assert torch.isfinite(raw_action).all()

    # 验证是否支持反向传播
    # 测试目标仅用于检查计算图，不是真实专家标签
    target = torch.zeros_like(raw_action)
    loss = nn.functional.mse_loss(raw_action, target)

    model.zero_grad()
    loss.backward()

    has_grad = any(
        p.grad is not None
        and torch.isfinite(p.grad).all().item()
        for p in model.parameters()
    )

    assert has_grad, "Backward pass failed"

    params = sum(p.numel() for p in model.parameters())

    print("Parameters:", f"{params:,}")
    print("Test loss:", float(loss.item()))
    print("Forward / backward test: PASSED")
