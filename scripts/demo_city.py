
"""
RiskFormer-3D
三维结构化城市环境可视化 V0.3

新增功能：
1. 建筑物六面分色
2. 同城区建筑具有轻微色差
3. 建筑物真实棱线描绘
4. 柔和科研风格配色
5. 减少道路交叉处的重叠
6. 正交投影与真实空间比例
7. 动态三维球体动画
8. 独立 HTML 输出

运行：
    python scripts/demo_city.py

输出：
    outputs/city_demo.html
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from riskformer3d.sim.obstacles import Box
from riskformer3d.sim.city_generator import generate_city


# =========================================================
# 1. 场景参数
# =========================================================

CITY_SEED = 42
CITY_DIFFICULTY = "hard"

SIM_DT = 0.2
ANIMATION_FRAMES = 81
FRAME_DURATION_MS = 100


# 柔和科研仿真配色
COLORS = {
    "ground": "#E4E9E2",
    "road": "#77818B",
    "marking": "#F5E8B9",

    "downtown": "#8CA9C5",
    "residential": "#9BBDAA",
    "industrial": "#B8ADA0",

    "rooftop": "#CED6DD",
    "bridge": "#D5B783",
    "construction": "#D88F71",
    "park": "#9DCC9C",

    "moving": "#F3A340",
    "start": "#16A085",
    "goal": "#DB526A",
}


# =========================================================
# 2. 长方体几何定义
# =========================================================

BOX_VERTICES = np.array([
    [-1, -1, -1],
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [ 1, -1,  1],
    [ 1,  1,  1],
    [-1,  1,  1],
], dtype=float)


# 每个长方体由 12 个三角形面组成
# 顺序：底面、顶面、四个侧面
BOX_FACES = np.array([
    [0, 2, 1],
    [0, 3, 2],

    [4, 5, 6],
    [4, 6, 7],

    [0, 1, 5],
    [0, 5, 4],

    [1, 2, 6],
    [1, 6, 5],

    [2, 3, 7],
    [2, 7, 6],

    [3, 0, 4],
    [3, 4, 7],
], dtype=int)


# 长方体的 12 条真实棱线
# 不绘制三角形面的内部对角线
BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]


# =========================================================
# 3. 颜色调整函数
# =========================================================

def adjust_color(hex_color, factor):
    """
    调整十六进制颜色亮度。

    factor > 0：变亮
    factor < 0：变暗

    示例：
        adjust_color("#7799BB", 0.2)
    """

    rgb = np.array([
        int(hex_color[i:i + 2], 16)
        for i in (1, 3, 5)
    ], dtype=float)

    if factor >= 0:
        rgb = rgb + (255 - rgb) * factor
    else:
        rgb = rgb * (1 + factor)

    rgb = np.clip(rgb, 0, 255).astype(int)

    return "#{:02X}{:02X}{:02X}".format(*rgb)


# =========================================================
# 4. 长方体绘制函数
# =========================================================

def draw_boxes(
    fig,
    boxes,
    color,
    name,
    opacity=1.0,
    show_edges=False,
    vary_color=False,
    edge_width=2
):
    """
    批量绘制长方体。

    与旧版本相比：

    1. 六个面使用不同亮度
    2. 同类建筑可以具有轻微色差
    3. 可以显示建筑物的 12 条棱线
    4. 使用统一的 Mesh3d 减少图形对象数量

    参数：
        show_edges:
            是否绘制建筑边缘

        vary_color:
            是否为同类建筑添加颜色变化

        edge_width:
            棱线宽度，单位为屏幕像素
    """

    if not boxes:
        return

    all_vertices = []
    all_faces = []
    all_face_colors = []

    edge_x = []
    edge_y = []
    edge_z = []

    # 不同建筑的亮度变化
    variations = [
        -0.09,
        -0.04,
         0.03,
         0.09,
         0.00,
         0.06
    ]

    for index, box in enumerate(boxes):

        center = np.asarray(
            box.center,
            dtype=float
        )

        size = np.asarray(
            box.size,
            dtype=float
        )

        # 计算当前长方体的真实顶点
        vertices = (
            center
            + BOX_VERTICES * (size / 2.0)
        )

        # 修正顶点索引
        faces = BOX_FACES + index * 8

        all_vertices.append(vertices)
        all_faces.append(faces)

        # -------------------------------------
        # 建筑颜色变化
        # -------------------------------------

        if vary_color:
            factor = variations[
                index % len(variations)
            ]
        else:
            factor = 0.0

        base_color = adjust_color(
            color,
            factor
        )

        # -------------------------------------
        # 六个面分别设置亮度
        # -------------------------------------

        face_colors = [
            adjust_color(base_color, -0.25),  # 底面
            adjust_color(base_color,  0.27),  # 屋顶
            adjust_color(base_color,  0.06),  # 侧面1
            adjust_color(base_color, -0.06),  # 侧面2
            adjust_color(base_color, -0.16),  # 侧面3
            adjust_color(base_color,  0.00),  # 侧面4
        ]

        # 每个长方体有6个面
        # 每个面包含2个三角形
        for shade in face_colors:
            all_face_colors.extend([
                shade,
                shade
            ])

        # -------------------------------------
        # 收集建筑棱线
        # -------------------------------------

        if show_edges:

            # 轻微向外偏移棱线
            # 减少棱线与建筑表面的深度冲突
            outline_vertices = (
                center
                + BOX_VERTICES * (
                    size / 2.0 + 0.015
                )
            )

            for a, b in BOX_EDGES:

                edge_x.extend([
                    outline_vertices[a, 0],
                    outline_vertices[b, 0],
                    None
                ])

                edge_y.extend([
                    outline_vertices[a, 1],
                    outline_vertices[b, 1],
                    None
                ])

                edge_z.extend([
                    outline_vertices[a, 2],
                    outline_vertices[b, 2],
                    None
                ])

    # =========================================
    # 合并所有几何模型
    # =========================================

    vertices = np.concatenate(
        all_vertices,
        axis=0
    )

    faces = np.concatenate(
        all_faces,
        axis=0
    )

    # =========================================
    # 绘制建筑实体
    # =========================================

    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],

            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],

            facecolor=all_face_colors,

            opacity=opacity,
            flatshading=True,

            # 主要依靠分面颜色表达立体感
            lighting=dict(
                ambient=0.92,
                diffuse=0.15,
                specular=0.0,
                roughness=1.0,
                fresnel=0.0
            ),

            lightposition=dict(
                x=150,
                y=-100,
                z=200
            ),

            name=name,
            legendgroup=name,
            showlegend=True,
            hoverinfo="name"
        )
    )

    # =========================================
    # 绘制建筑轮廓线
    # =========================================

    if show_edges:

        fig.add_trace(
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,

                mode="lines",

                line=dict(
                    color="rgba(49,61,75,0.68)",
                    width=edge_width
                ),

                name=name + " Edges",
                legendgroup=name,
                showlegend=False,
                hoverinfo="skip"
            )
        )


# =========================================================
# 5. 三维球体绘制
# =========================================================

def draw_sphere(
    center,
    radius,
    color,
    name="Moving Obstacle"
):
    """
    按真实三维半径绘制动态球体。

    与固定像素大小的 Scatter3d marker 不同，
    球体几何尺寸使用米作为单位。
    """

    center = np.asarray(
        center,
        dtype=float
    )

    theta = np.linspace(
        0,
        2 * np.pi,
        18
    )

    phi = np.linspace(
        0,
        np.pi,
        12
    )

    x = (
        center[0]
        + radius * np.outer(
            np.sin(phi),
            np.cos(theta)
        )
    )

    y = (
        center[1]
        + radius * np.outer(
            np.sin(phi),
            np.sin(theta)
        )
    )

    z = (
        center[2]
        + radius * np.outer(
            np.cos(phi),
            np.ones_like(theta)
        )
    )

    return go.Surface(
        x=x,
        y=y,
        z=z,

        surfacecolor=np.zeros_like(x),

        colorscale=[
            [0, color],
            [1, color]
        ],

        cmin=0,
        cmax=1,

        showscale=False,
        opacity=1.0,

        name=name,
        showlegend=False,
        hoverinfo="skip",

        lighting=dict(
            ambient=0.7,
            diffuse=0.55,
            specular=0.15,
            roughness=0.65
        )
    )


# =========================================================
# 6. 道路绘制
# =========================================================

def add_roads(fig, city):
    """
    绘制城市道路。

    纵向道路贯穿城市，
    横向道路只绘制在纵向道路之间。

    这样可以减少十字路口的
    共面重叠和闪烁。
    """

    road_boxes = []

    road_centers = sorted(city.roads)
    width = city.road_width

    # -------------------------------------
    # 纵向道路
    # -------------------------------------

    for c in road_centers:

        road_boxes.append(
            Box(
                center=[
                    c,
                    city.size / 2,
                    0.055
                ],

                size=[
                    width,
                    city.size,
                    0.08
                ]
            )
        )

    # -------------------------------------
    # 计算纵向道路之间的横向区间
    # -------------------------------------

    x_segments = []

    previous_end = 0.0

    for c in road_centers:

        left_edge = c - width / 2

        if left_edge > previous_end:
            x_segments.append(
                (previous_end, left_edge)
            )

        previous_end = c + width / 2

    if previous_end < city.size:
        x_segments.append(
            (previous_end, city.size)
        )

    # -------------------------------------
    # 绘制横向道路
    # -------------------------------------

    for y in road_centers:

        for x0, x1 in x_segments:

            segment_length = x1 - x0

            if segment_length <= 0:
                continue

            road_boxes.append(
                Box(
                    center=[
                        (x0 + x1) / 2,
                        y,
                        0.055
                    ],

                    size=[
                        segment_length,
                        width,
                        0.08
                    ]
                )
            )

    # 道路主体
    draw_boxes(
        fig=fig,
        boxes=road_boxes,
        color=COLORS["road"],
        name="Urban Roads",
        show_edges=False,
        vary_color=False
    )

    # 道路标线
    draw_boxes(
        fig=fig,
        boxes=city.lane_markings,
        color=COLORS["marking"],
        name="Road Markings",
        show_edges=False,
        vary_color=False
    )


# =========================================================
# 7. 城市建筑与附属设施
# =========================================================

def add_buildings(fig, city):

    zone_names = {
        "downtown": "Commercial District",
        "residential": "Residential District",
        "industrial": "Industrial District"
    }

    # -------------------------------------
    # 主要建筑
    # -------------------------------------

    for zone, buildings in city.buildings.items():

        draw_boxes(
            fig=fig,
            boxes=buildings,
            color=COLORS[zone],
            name=zone_names.get(zone, zone),

            # 绘制建筑棱线
            show_edges=True,

            # 同城区建筑有轻微色差
            vary_color=True,

            edge_width=2
        )

    # -------------------------------------
    # 屋顶设备
    # -------------------------------------

    draw_boxes(
        fig=fig,
        boxes=city.roof_equipment,
        color=COLORS["rooftop"],
        name="Rooftop Equipment",
        show_edges=True,
        vary_color=False,
        edge_width=1
    )

    # -------------------------------------
    # 空中连廊
    # -------------------------------------

    draw_boxes(
        fig=fig,
        boxes=city.bridges,
        color=COLORS["bridge"],
        name="Skybridges",
        show_edges=True,
        vary_color=False,
        edge_width=2
    )

    # -------------------------------------
    # 施工障碍物
    # -------------------------------------

    draw_boxes(
        fig=fig,
        boxes=city.barriers,
        color=COLORS["construction"],
        name="Construction Barriers",
        show_edges=True,
        vary_color=False,
        edge_width=2
    )


# =========================================================
# 8. 绘制绿地
# =========================================================

def add_parks(fig, city):

    draw_boxes(
        fig=fig,
        boxes=city.parks,
        color=COLORS["park"],
        name="Urban Parks",
        show_edges=False,
        vary_color=False
    )


# =========================================================
# 9. 起点与终点
# =========================================================

def add_navigation_points(fig, city):

    points = [
        (
            city.start,
            "START",
            COLORS["start"]
        ),
        (
            city.goal,
            "GOAL",
            COLORS["goal"]
        )
    ]

    for position, label, color in points:

        fig.add_trace(
            go.Scatter3d(
                x=[position[0]],
                y=[position[1]],
                z=[position[2]],

                mode="markers+text",

                marker=dict(
                    size=8,
                    color=color,
                    symbol="circle"
                ),

                text=[label],
                textposition="top center",

                textfont=dict(
                    size=12,
                    color=color
                ),

                name=label,
                showlegend=True
            )
        )


# =========================================================
# 10. 动态障碍物及动画
# =========================================================

def add_dynamic_obstacles(fig, city):
    """
    仅更新动态障碍物，
    静态建筑物不重新绘制。

    使用独立 Trace 实现动画。
    """

    dynamic_indices = []

    # 保存动态障碍物原始状态
    original_states = [
        (
            mover.center.copy(),
            mover.velocity.copy()
        )
        for mover in city.movers
    ]

    # -------------------------------------
    # 初始障碍物
    # -------------------------------------

    for i, mover in enumerate(city.movers):

        dynamic_indices.append(
            len(fig.data)
        )

        fig.add_trace(
            draw_sphere(
                center=mover.center,
                radius=mover.radius,
                color=COLORS["moving"],
                name=f"Moving Obstacle {i + 1}"
            )
        )

    # -------------------------------------
    # 生成动画
    # -------------------------------------

    frames = []

    for t in range(ANIMATION_FRAMES):

        if t > 0:
            for mover in city.movers:
                mover.step(SIM_DT)

        updated_spheres = []

        for i, mover in enumerate(city.movers):

            updated_spheres.append(
                draw_sphere(
                    center=mover.center,
                    radius=mover.radius,
                    color=COLORS["moving"],
                    name=f"Moving Obstacle {i + 1}"
                )
            )

        frames.append(
            go.Frame(
                data=updated_spheres,
                traces=dynamic_indices,
                name=str(t)
            )
        )

    fig.frames = frames

    # 将城市对象中的动态障碍物
    # 恢复到初始时刻状态
    for mover, state in zip(
        city.movers,
        original_states
    ):
        mover.center = state[0].copy()
        mover.velocity = state[1].copy()


# =========================================================
# 11. 三维相机与页面布局
# =========================================================

def configure_scene(fig, city):

    fig.update_layout(

        title=dict(
            text=(
                "RiskFormer-3D | "
                "Urban Environment V0.3"
            ),
            x=0.5,
            font=dict(
                size=21,
                color="#283747"
            )
        ),

        paper_bgcolor="white",

        scene=dict(

            bgcolor="white",

            # X轴
            xaxis=dict(
                title="X (m)",
                range=[0, city.size],
                showbackground=False,
                showgrid=False,
                zeroline=False
            ),

            # Y轴
            yaxis=dict(
                title="Y (m)",
                range=[0, city.size],
                showbackground=False,
                showgrid=False,
                zeroline=False
            ),

            # Z轴
            zaxis=dict(
                title="Z (m)",
                range=[0, city.max_height],
                showbackground=False,
                showgrid=False,
                zeroline=False
            ),

            # 保持真实三维比例
            aspectmode="manual",

            aspectratio=dict(
                x=1.0,
                y=1.0,
                z=city.max_height / city.size
            ),

            camera=dict(

                projection=dict(
                    type="orthographic"
                ),

                # 降低相机角度
                # 避免画面过于俯视
                eye=dict(
                    x=1.8,
                    y=-1.8,
                    z=1.05
                ),

                up=dict(
                    x=0,
                    y=0,
                    z=1
                ),

                center=dict(
                    x=0,
                    y=0,
                    z=-0.05
                )
            )
        ),

        scene_dragmode="orbit",

        legend=dict(
            x=0.01,
            y=0.98,
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=11)
        ),

        # =====================================
        # 动画按钮
        # =====================================

        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.42,
                y=0.02,

                buttons=[

                    dict(
                        label="Play",
                        method="animate",

                        args=[
                            None,
                            dict(
                                frame=dict(
                                    duration=FRAME_DURATION_MS,
                                    redraw=True
                                ),
                                transition=dict(
                                    duration=0
                                ),
                                fromcurrent=True,
                                mode="immediate"
                            )
                        ]
                    ),

                    dict(
                        label="Pause",
                        method="animate",

                        args=[
                            [None],
                            dict(
                                frame=dict(
                                    duration=0,
                                    redraw=False
                                ),
                                transition=dict(
                                    duration=0
                                ),
                                mode="immediate"
                            )
                        ]
                    )
                ]
            )
        ],

        margin=dict(
            l=0,
            r=0,
            t=65,
            b=40
        ),

        height=850,
        hovermode="closest"
    )


# =========================================================
# 12. 保存交互式 HTML
# =========================================================

def save_figure(fig):

    root = Path(__file__).resolve().parents[1]

    output_dir = root / "outputs"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir / "city_demo.html"
    )

    fig.write_html(
        str(output_file),
        include_plotlyjs=True,
        full_html=True,
        auto_open=True
    )

    return output_file


# =========================================================
# 13. 主程序
# =========================================================

def main():

    print("=" * 55)
    print("RiskFormer-3D | Urban Simulation V0.3")
    print("=" * 55)

    # ======================================
    # 生成城市
    # ======================================

    city = generate_city(
        seed=CITY_SEED,
        difficulty=CITY_DIFFICULTY
    )

    fig = go.Figure()

    # ======================================
    # 地面
    # ======================================

    ground = Box(
        center=[
            city.size / 2,
            city.size / 2,
            -0.05
        ],

        size=[
            city.size,
            city.size,
            0.1
        ]
    )

    draw_boxes(
        fig=fig,
        boxes=[ground],
        color=COLORS["ground"],
        name="Ground",
        show_edges=False,
        vary_color=False
    )

    # ======================================
    # 城市道路
    # ======================================

    add_roads(fig, city)

    # ======================================
    # 城市绿地
    # ======================================

    add_parks(fig, city)

    # ======================================
    # 建筑及附属结构
    # ======================================

    add_buildings(fig, city)

    # ======================================
    # 起点和终点
    # ======================================

    add_navigation_points(fig, city)

    # ======================================
    # 动态障碍物
    # ======================================

    add_dynamic_obstacles(fig, city)

    # ======================================
    # 相机与页面设置
    # ======================================

    configure_scene(fig, city)

    # ======================================
    # 保存 HTML
    # ======================================

    output = save_figure(fig)

    building_count = sum(
        len(group)
        for group in city.buildings.values()
    )

    print(f"Seed: {CITY_SEED}")
    print(f"Difficulty: {CITY_DIFFICULTY}")
    print(f"Buildings: {building_count}")

    print(
        f"Rooftop equipment: "
        f"{len(city.roof_equipment)}"
    )

    print(f"Skybridges: {len(city.bridges)}")
    print(f"Road markings: {len(city.lane_markings)}")
    print(f"Parks: {len(city.parks)}")
    print(f"Moving obstacles: {len(city.movers)}")

    print(
        f"Static collision objects: "
        f"{len(city.static_obstacles)}"
    )

    print("-" * 55)

    print("Urban environment generated successfully!")
    print(f"Output: {output}")

    print("=" * 55)


if __name__ == "__main__":
    main()
