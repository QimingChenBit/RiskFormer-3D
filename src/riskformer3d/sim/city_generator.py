
"""
RiskFormer-3D: Structured Urban Environment V0.2

结构化三维城市生成器

主要功能：
1. 道路、街区和建筑地块规划
2. 商业区、住宅区和工业区
3. 裙楼、塔楼、屋顶设备
4. 与真实建筑连接的空中连廊
5. 狭窄巷道、施工封路与绿地
6. 动态飞行障碍物
7. 随机种子和难度控制

坐标单位：米
"""

from dataclasses import dataclass, field
import numpy as np

from .obstacles import Box, MovingSphere


# ==========================================
# 1. 城市数据结构
# ==========================================

@dataclass
class CityScene:
    size: float
    max_height: float
    roads: tuple
    road_width: float

    buildings: dict
    bridges: list
    barriers: list
    movers: list

    start: np.ndarray
    goal: np.ndarray

    # 新增城市细节
    roof_equipment: list = field(default_factory=list)
    parks: list = field(default_factory=list)
    lane_markings: list = field(default_factory=list)

    @property
    def static_obstacles(self):
        """
        返回真正需要进行碰撞检测的物体。

        道路标线和草坪不作为飞行障碍。
        """
        obstacles = []

        for group in self.buildings.values():
            obstacles.extend(group)

        obstacles.extend(self.bridges)
        obstacles.extend(self.barriers)
        obstacles.extend(self.roof_equipment)

        return obstacles


# ==========================================
# 2. 城市道路与街区
# ==========================================

def make_blocks(size, roads, width):
    """根据主干道划分街区边界。"""

    blocks = []
    left = 0.0

    for center in roads:
        right = center - width / 2

        if right > left:
            blocks.append((left, right))

        left = center + width / 2

    if left < size:
        blocks.append((left, size))

    return blocks


def make_lots(low, high, alley_width=2.0):
    """
    将街区划分成两个建筑地块。

    alley_width 越小，巷道越狭窄。
    """
    setback = 1.1

    usable = high - low - 2 * setback
    lot_size = (usable - alley_width) / 2

    if lot_size < 2.5:
        return []

    first = low + setback + lot_size / 2
    second = high - setback - lot_size / 2

    return [
        (first, lot_size),
        (second, lot_size)
    ]


def make_lane_markings(size, roads, road_width):
    """
    创建道路中央的虚线标识。

    这些仅用于可视化，不参与碰撞检测。
    """
    markings = []

    positions = np.arange(3.0, size - 2, 5.0)

    for center in roads:
        for p in positions:

            # 十字路口内不画普通道路虚线
            near_intersection = any(
                abs(p - crossing) <
                road_width / 2 + 2.0
                for crossing in roads
            )

            if near_intersection:
                continue

            # 沿Y方向的道路
            markings.append(Box(
                center=[center, p, 0.12],
                size=[0.14, 2.2, 0.02]
            ))

            # 沿X方向的道路
            markings.append(Box(
                center=[p, center, 0.12],
                size=[2.2, 0.14, 0.02]
            ))

    return markings


# ==========================================
# 3. 建筑结构生成
# ==========================================

def create_building(
    rng,
    cx,
    cy,
    lot_x,
    lot_y,
    zone,
    buildings,
    roof_equipment
):
    """
    根据城区类型创建建筑。

    downtown:
        低层裙楼 + 高层塔楼

    residential:
        中低层住宅建筑

    industrial:
        宽体工业厂房

    返回主要建筑体，用于后续连廊连接。
    """

    # 建筑占地面积小于地块面积
    sx = lot_x * rng.uniform(0.82, 0.92)
    sy = lot_y * rng.uniform(0.82, 0.92)

    if zone == "downtown":

        total_height = rng.uniform(21.0, 34.0)
        podium_height = rng.uniform(5.0, 7.5)

        # 底部裙楼
        podium = Box(
            center=[cx, cy, podium_height / 2],
            size=[sx, sy, podium_height]
        )

        # 上部塔楼采用较小的占地面积
        tower_height = total_height - podium_height

        tower = Box(
            center=[
                cx,
                cy,
                podium_height + tower_height / 2
            ],
            size=[
                sx * 0.72,
                sy * 0.72,
                tower_height
            ]
        )

        buildings[zone].extend([podium, tower])
        primary = tower

    elif zone == "residential":

        height = rng.uniform(8.0, 17.0)

        primary = Box(
            center=[cx, cy, height / 2],
            size=[sx, sy, height]
        )

        buildings[zone].append(primary)

    else:
        # 工业厂房通常较低、占地较宽
        height = rng.uniform(7.0, 14.0)

        primary = Box(
            center=[cx, cy, height / 2],
            size=[sx, sy, height]
        )

        buildings[zone].append(primary)

    # ======================================
    # 屋顶设备
    # ======================================

    if rng.random() < 0.45:

        top = (
            primary.center[2]
            + primary.size[2] / 2
        )

        # 屋顶设备与建筑顶部接触
        equipment_height = 1.4

        equipment = Box(
            center=[
                primary.center[0],
                primary.center[1],
                top + equipment_height / 2
            ],
            size=[
                min(primary.size[0] * 0.4, 2.0),
                min(primary.size[1] * 0.4, 2.0),
                equipment_height
            ]
        )

        roof_equipment.append(equipment)

    return primary


# ==========================================
# 4. 建筑间空中连廊
# ==========================================

def connect_buildings_x(left, right):
    """
    沿X方向连接两栋真实存在的建筑。

    连廊必须：
    1. 两栋建筑在Y方向具有重叠范围
    2. 连廊高度不能超过建筑顶部
    3. 连廊底部离地足够高
    4. 两端伸入建筑表面少量距离
    """

    left_face = left.center[0] + left.size[0] / 2
    right_face = right.center[0] - right.size[0] / 2

    gap = right_face - left_face

    if gap <= 0:
        return None

    y_min = max(
        left.center[1] - left.size[1] / 2,
        right.center[1] - right.size[1] / 2
    )

    y_max = min(
        left.center[1] + left.size[1] / 2,
        right.center[1] + right.size[1] / 2
    )

    if y_max - y_min < 2.2:
        return None

    left_top = left.center[2] + left.size[2] / 2
    right_top = right.center[2] + right.size[2] / 2

    z = min(14.0, left_top - 2.5, right_top - 2.5)

    if z < 9.0:
        return None

    return Box(
        center=[
            (left_face + right_face) / 2,
            (y_min + y_max) / 2,
            z
        ],
        size=[
            gap + 0.36,
            1.8,
            2.2
        ]
    )


def connect_buildings_y(bottom, top):
    """沿Y方向连接两栋建筑。"""

    bottom_face = (
        bottom.center[1] + bottom.size[1] / 2
    )

    top_face = (
        top.center[1] - top.size[1] / 2
    )

    gap = top_face - bottom_face

    if gap <= 0:
        return None

    x_min = max(
        bottom.center[0] - bottom.size[0] / 2,
        top.center[0] - top.size[0] / 2
    )

    x_max = min(
        bottom.center[0] + bottom.size[0] / 2,
        top.center[0] + top.size[0] / 2
    )

    if x_max - x_min < 2.2:
        return None

    bottom_top = (
        bottom.center[2] + bottom.size[2] / 2
    )

    top_top = top.center[2] + top.size[2] / 2

    z = min(14.0, bottom_top - 2.5, top_top - 2.5)

    if z < 9.0:
        return None

    return Box(
        center=[
            (x_min + x_max) / 2,
            (bottom_face + top_face) / 2,
            z
        ],
        size=[
            1.8,
            gap + 0.36,
            2.2
        ]
    )


# ==========================================
# 5. 主城市生成函数
# ==========================================

def generate_city(seed=42, difficulty="hard"):
    """
    创建可复现的结构化三维城市场景。

    easy:
        较低建筑密度、宽巷道、少量动态目标

    medium:
        中等建筑密度、空中连廊、施工障碍

    hard:
        高密度建筑、狭窄通道、双向连廊、
        路障以及多个动态飞行目标
    """

    if difficulty not in ("easy", "medium", "hard"):
        raise ValueError("Unknown difficulty")

    rng = np.random.default_rng(seed)

    size = 80.0
    max_height = 45.0

    # 规则化城市主干道
    roads = (16.0, 40.0, 64.0)
    road_width = 7.0

    settings = {
        "easy": {
            "density": 0.65,
            "alley": 3.0,
            "movers": 1
        },
        "medium": {
            "density": 0.82,
            "alley": 2.1,
            "movers": 2
        },
        "hard": {
            "density": 0.97,
            "alley": 1.3,
            "movers": 3
        }
    }[difficulty]

    buildings = {
        "downtown": [],
        "residential": [],
        "industrial": []
    }

    roof_equipment = []
    parks = []
    bridges = []
    barriers = []

    # 记录每个地块上的主要建筑
    sites = {}

    blocks = make_blocks(
        size,
        roads,
        road_width
    )

    # 连廊两端的地块必须存在建筑
    bridge_sites = {
        (1, 1, 1, 1),
        (2, 1, 0, 1),
        (1, 1, 0, 1),
        (1, 2, 0, 0)
    }

    # 保留一个公共绿地地块
    park_site = (2, 2, 0, 0)

    # ======================================
    # 5.1 生成城区建筑
    # ======================================

    for i, (x0, x1) in enumerate(blocks):
        for j, (y0, y1) in enumerate(blocks):

            if i in (1, 2) and j in (1, 2):
                zone = "downtown"

            elif i == 0 or j == 3:
                zone = "industrial"

            else:
                zone = "residential"

            x_lots = make_lots(
                x0, x1, settings["alley"]
            )

            y_lots = make_lots(
                y0, y1, settings["alley"]
            )

            for xi, (cx, lx) in enumerate(x_lots):
                for yi, (cy, ly) in enumerate(y_lots):

                    key = (i, j, xi, yi)

                    # 一个实际存在的开放式绿地
                    if key == park_site:

                        parks.append(Box(
                            center=[cx, cy, 0.055],
                            size=[lx * 0.9, ly * 0.9, 0.09]
                        ))

                        continue

                    # 在中高难度下确保连廊有支撑建筑
                    forced = (
                        difficulty in ("medium", "hard")
                        and key in bridge_sites
                    )

                    if not forced:
                        if rng.random() > settings["density"]:
                            continue

                    building = create_building(
                        rng=rng,
                        cx=cx,
                        cy=cy,
                        lot_x=lx,
                        lot_y=ly,
                        zone=zone,
                        buildings=buildings,
                        roof_equipment=roof_equipment
                    )

                    sites[key] = building

    # ======================================
    # 5.2 连接真实存在的建筑
    # ======================================

    if difficulty in ("medium", "hard"):

        bridge = connect_buildings_x(
            sites[(1, 1, 1, 1)],
            sites[(2, 1, 0, 1)]
        )

        if bridge is not None:
            bridges.append(bridge)

    if difficulty == "hard":

        bridge = connect_buildings_y(
            sites[(1, 1, 0, 1)],
            sites[(1, 2, 0, 0)]
        )

        if bridge is not None:
            bridges.append(bridge)

    # ======================================
    # 5.3 道路施工与封闭
    # ======================================

    if difficulty in ("medium", "hard"):

        # 封闭中央纵向道路的一个低空路段
        barriers.append(Box(
            center=[40.0, 52.0, 2.25],
            size=[6.6, 1.5, 4.5]
        ))

    if difficulty == "hard":

        # 另一个方向的道路封闭
        barriers.append(Box(
            center=[52.0, 16.0, 2.25],
            size=[1.5, 6.6, 4.5]
        ))

    # ======================================
    # 5.4 动态飞行障碍物
    # ======================================

    movers = [
        MovingSphere(
            center=[16, 29, 3],
            radius=0.75,
            velocity=[0, 0.85, 0],
            lower_bound=[14, 21, 1],
            upper_bound=[18, 35, 7]
        ),
        MovingSphere(
            center=[29, 64, 4],
            radius=0.8,
            velocity=[0.8, 0, 0],
            lower_bound=[21, 62, 1],
            upper_bound=[35, 66, 7]
        ),
        MovingSphere(
            center=[64, 51, 3],
            radius=0.75,
            velocity=[0, -0.7, 0],
            lower_bound=[62, 45, 1],
            upper_bound=[66, 59, 7]
        )
    ]

    movers = movers[:settings["movers"]]

    # ======================================
    # 5.5 起点、终点与道路细节
    # ======================================

    start = np.array(
        [16.0, 16.0, 3.0],
        dtype=float
    )

    goal = np.array(
        [64.0, 64.0, 3.0],
        dtype=float
    )

    lane_markings = make_lane_markings(
        size,
        roads,
        road_width
    )

    return CityScene(
        size=size,
        max_height=max_height,
        roads=roads,
        road_width=road_width,
        buildings=buildings,
        bridges=bridges,
        barriers=barriers,
        movers=movers,
        start=start,
        goal=goal,
        roof_equipment=roof_equipment,
        parks=parks,
        lane_markings=lane_markings
    )
