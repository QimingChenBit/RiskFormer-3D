
"""
RiskFormer-3D：三维障碍物模块

实现：
1. Sphere：球形障碍物
2. Box：长方体障碍物
3. Cylinder：圆柱形障碍物
4. MovingSphere：动态球形障碍物
5. check_collision：碰撞检测
"""

import numpy as np
from dataclasses import dataclass


# ==========================================
# 1. 球形障碍物
# ==========================================

@dataclass
class Sphere:
    center: np.ndarray
    radius: float

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)

        if self.center.shape != (3,) or self.radius <= 0:
            raise ValueError("Invalid sphere parameters")

    def signed_distance(self, point):
        point = np.asarray(point, dtype=float)

        return float(
            np.linalg.norm(point - self.center)
            - self.radius
        )


# ==========================================
# 2. 长方体障碍物
# ==========================================

@dataclass
class Box:
    center: np.ndarray
    size: np.ndarray

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.size = np.asarray(self.size, dtype=float)

        if self.center.shape != (3,) or self.size.shape != (3,):
            raise ValueError("Box must be 3D")

        if np.any(self.size <= 0):
            raise ValueError("Box size must be positive")

    def signed_distance(self, point):
        """
        计算点到长方体表面的有符号距离。
        负数表示点在长方体内部。
        """
        point = np.asarray(point, dtype=float)

        q = np.abs(point - self.center) - self.size / 2

        outside = np.linalg.norm(np.maximum(q, 0))

        inside = min(float(np.max(q)), 0.0)

        return float(outside + inside)


# ==========================================
# 3. 圆柱形障碍物
# ==========================================

@dataclass
class Cylinder:
    center: np.ndarray
    radius: float
    height: float

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)

        if self.center.shape != (3,):
            raise ValueError("Cylinder center must be 3D")

        if self.radius <= 0 or self.height <= 0:
            raise ValueError("Invalid cylinder parameters")

    def signed_distance(self, point):
        point = np.asarray(point, dtype=float)

        horizontal = (
            np.linalg.norm(point[:2] - self.center[:2])
            - self.radius
        )

        vertical = (
            abs(point[2] - self.center[2])
            - self.height / 2
        )

        d = np.array([horizontal, vertical])

        outside = np.linalg.norm(np.maximum(d, 0))

        inside = min(float(np.max(d)), 0.0)

        return float(outside + inside)


# ==========================================
# 4. 动态球形障碍物
# ==========================================

@dataclass
class MovingSphere(Sphere):
    velocity: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray

    def __post_init__(self):
        super().__post_init__()

        self.velocity = np.asarray(self.velocity, dtype=float)
        self.lower_bound = np.asarray(self.lower_bound, dtype=float)
        self.upper_bound = np.asarray(self.upper_bound, dtype=float)

        if any(a.shape != (3,) for a in [
            self.velocity,
            self.lower_bound,
            self.upper_bound
        ]):
            raise ValueError("Movement parameters must be 3D")

        if np.any(
            self.upper_bound - self.lower_bound <= 2 * self.radius
        ):
            raise ValueError("Movement bounds are too small")

    def step(self, dt):
        """
        更新障碍物位置。

        匀速运动：
            position = position + velocity * dt

        碰到边界：
            速度方向反转
        """
        if dt <= 0:
            raise ValueError("dt must be positive")

        self.center = self.center + self.velocity * dt

        for i in range(3):

            lower = self.lower_bound[i] + self.radius
            upper = self.upper_bound[i] - self.radius

            if self.center[i] < lower:
                self.center[i] = lower
                self.velocity[i] = abs(self.velocity[i])

            elif self.center[i] > upper:
                self.center[i] = upper
                self.velocity[i] = -abs(self.velocity[i])


# ==========================================
# 5. 无人机碰撞检测
# ==========================================

def check_collision(
    obstacle,
    drone_position,
    drone_radius=0.25
):
    """
    无人机暂时近似为半径 0.25m 的球体。
    """
    if drone_radius < 0:
        raise ValueError("Drone radius must be non-negative")

    distance = obstacle.signed_distance(drone_position)

    return distance <= drone_radius
