"""不规则房间设备摆放：单文件命令行程序。

输入完整 JSON 或 JSON 文件地址，输出可行性与每件物品的中心和角度。
代码依次实现输入规范化、几何验证、快速构造、MILP 回退和命令行接口。
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

# ===== 输入与几何模型 =====

@dataclass
class Item:
    """设备尺寸及可选规则；未声明 ``supported_by`` 时设备落在地面。"""

    name: str
    length: float
    width: float
    height: float = 1.0
    supported_by: Optional[str] = None
    front_side: Optional[str] = None


@dataclass
class Placement:
    """设备中心和方向；``z`` 仅供内部检查架上关系，不属于题目二维输出。"""

    center: Tuple[float, float]
    angle_deg: float
    z: float = 0.0


@dataclass
class Problem:
    """经检查的房间问题；内开门才生成 ``door_keepout``。"""

    boundary: Polygon
    door: LineString
    is_open_inward: bool
    items: Tuple[Item, ...]
    door_keepout: Optional[Polygon] = None


def _linear_tolerance(feature_size: float, *coordinates: float) -> float:
    """按局部尺寸和坐标浮点精度设置容差，避免写死某个毫米数。"""
    magnitude = max([abs(value) for value in coordinates] + [feature_size])
    return max(feature_size * 1e-12, 32 * sys.float_info.epsilon * magnitude)


def _point(value: object, label: str) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("%s must be an [x, y] pair" % label)
    result = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise ValueError("%s coordinates must be finite numbers" % label)
        number = float(coordinate)
        if not math.isfinite(number):
            raise ValueError("%s coordinates must be finite numbers" % label)
        result.append(number)
    return result[0], result[1]


def _positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a positive finite number" % label)
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("%s must be a positive finite number" % label)
    return number


def _normalize_vertices(
    vertices: Sequence[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """删除重复点与 A-B-A 形式的原路折返。

    样例 4 有这种折返。更复杂的自交会交给 ``Polygon.is_valid`` 拒绝，
    避免擅自改变原房间形状。
    """
    result = list(vertices)
    changed = True
    while changed and len(result) >= 3:
        changed = False
        for index in range(len(result)):
            previous = result[index - 1]
            current = result[index]
            following = result[(index + 1) % len(result)]
            if current == following or previous == following:
                del result[index]
                changed = True
                break
    return result


def _door_keepout(room: Polygon, door: LineString) -> Polygon:
    """沿门宽 N 向室内延伸 N，构造题目要求的 N×N 门扇禁放区。"""

    (x0, y0), (x1, y1) = door.coords
    width = door.length
    wall_points = list(room.exterior.coords)
    wall = max(
        (LineString((start, end)) for start, end in zip(wall_points, wall_points[1:])),
        key=lambda edge: edge.intersection(door).length,
    )
    if wall.intersection(door).length == 0:
        raise ValueError("door must lie along a boundary edge")
    (wx0, wy0), (wx1, wy1) = wall.coords
    wall_length = wall.length
    # 边界可能顺时针或逆时针输入，须据此选择指向房间内部的法线。
    if room.exterior.is_ccw:
        normal = (-(wy1 - wy0) / wall_length, (wx1 - wx0) / wall_length)
    else:
        normal = ((wy1 - wy0) / wall_length, -(wx1 - wx0) / wall_length)
    offset = (normal[0] * width, normal[1] * width)
    return Polygon(
        [
            (x0, y0),
            (x1, y1),
            (x1 + offset[0], y1 + offset[1]),
            (x0 + offset[0], y0 + offset[1]),
        ]
    )


def _check_support_graph(items: Sequence[Item]) -> None:
    """检查可选承载关系的引用及环，防止设备间互相承托。"""

    by_name = {item.name: item for item in items}
    # 深度优先搜索状态：0 未访问、1 当前路径、2 已完成。
    states = {}  # type: Dict[str, int]

    def visit(name: str) -> None:
        state = states.get(name, 0)
        if state == 1:
            raise ValueError("support relationships contain a cycle")
        if state == 2:
            return
        states[name] = 1
        parent = by_name[name].supported_by
        if parent is not None:
            if parent not in by_name:
                raise ValueError("unknown support item: %s" % parent)
            visit(parent)
        states[name] = 2

    for item in items:
        visit(item.name)


def load_problem(data: Mapping[str, object]) -> Problem:
    """读取题目输入；几何含义不明确的数据直接报错。"""
    if not isinstance(data, dict):
        raise ValueError("input must be a JSON object")

    raw_boundary = data.get("boundary")
    if not isinstance(raw_boundary, (list, tuple)) or len(raw_boundary) < 3:
        raise ValueError("boundary needs at least three vertices")
    # 仅修复无面积的重复边，不自动修复一般自交多边形。
    vertices = _normalize_vertices(
        [_point(raw, "boundary vertex") for raw in raw_boundary]
    )
    if len(vertices) < 3:
        raise ValueError("boundary needs at least three distinct vertices")
    room = Polygon(vertices)
    if not room.is_valid or room.area <= 0:
        raise ValueError("boundary must be a simple polygon with positive area")

    raw_door = data.get("door")
    if not isinstance(raw_door, (list, tuple)) or len(raw_door) != 2:
        raise ValueError("door needs two endpoints")
    door = LineString([_point(raw, "door endpoint") for raw in raw_door])
    min_room_span = min(
        room.bounds[2] - room.bounds[0], room.bounds[3] - room.bounds[1]
    )
    door_tolerance = _linear_tolerance(
        min(door.length, min_room_span),
        *(coordinate for point in door.coords for coordinate in point),
    )
    if door.length <= 0 or not room.boundary.buffer(door_tolerance).covers(door):
        raise ValueError("door must have positive width and lie on the wall")

    inward = data.get("isOpenInward")
    if not isinstance(inward, bool):
        raise ValueError("isOpenInward must be a boolean")
    # 外开门无室内门扇禁区，但门口线段本身仍须保持畅通。
    keepout = _door_keepout(room, door) if inward else None

    raw_items = data.get("algoToPlace")
    if not isinstance(raw_items, dict):
        raise ValueError("algoToPlace must be an object")
    raw_rules = data.get("itemRules", {})
    # 高度、承载和开门边只作为通用扩展，原始 algoToPlace 格式保持不变。
    if not isinstance(raw_rules, dict):
        raise ValueError("itemRules must be an object")
    unknown_rules = set(raw_rules).difference(raw_items)
    if unknown_rules:
        raise ValueError(
            "itemRules refers to an unknown item: %s" % sorted(unknown_rules)[0]
        )

    items = []
    for name, dimensions in raw_items.items():
        if not isinstance(name, str) or not name:
            raise ValueError("item names must be nonempty strings")
        if not isinstance(dimensions, (list, tuple)) or len(dimensions) != 2:
            raise ValueError("item %s needs [length, width]" % name)
        length = _positive(dimensions[0], "%s length" % name)
        width = _positive(dimensions[1], "%s width" % name)
        rules = raw_rules.get(name, {})
        if not isinstance(rules, dict):
            raise ValueError("itemRules.%s must be an object" % name)
        unknown_fields = set(rules).difference({"height", "supportedBy", "frontSide"})
        if unknown_fields:
            raise ValueError(
                "unknown rule for %s: %s" % (name, sorted(unknown_fields)[0])
            )
        height = _positive(rules.get("height", 1.0), "%s height" % name)
        supported_by = rules.get("supportedBy")
        if supported_by is not None and (
            not isinstance(supported_by, str) or not supported_by
        ):
            raise ValueError("%s supportedBy must name another item" % name)
        # 冰箱默认以本地 +width 边为开门边；180° 反向会改变这条边的位置。
        default_front = (
            "+width" if name == "fridge" or name.startswith("fridge-") else None
        )
        front_side = rules.get("frontSide", default_front)
        if front_side not in (None, "+width", "-width", "+length", "-length"):
            raise ValueError("%s frontSide must identify one rectangle edge" % name)
        items.append(Item(name, length, width, height, supported_by, front_side))

    _check_support_graph(items)
    return Problem(room, door, inward, tuple(items), keepout)

# ===== 独立几何校验 =====

ANGLE_CLUSTER_TOLERANCE = 1e-5
ANGLE_VALIDATION_TOLERANCE = 1e-8


def _oriented_point(
    local: Tuple[float, float], placement: Placement
) -> Tuple[float, float]:
    angle = math.radians(placement.angle_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    x, y = local
    return (
        placement.center[0] + cosine * x - sine * y,
        placement.center[1] + sine * x + cosine * y,
    )


def _wall_angles(problem: Problem, rounded: bool) -> Tuple[float, ...]:
    """从每段墙提取平行及垂直方向；无开门边的矩形只需模 180°。"""

    wall_points = list(problem.boundary.exterior.coords)
    raw_angles = []
    for start, end in zip(wall_points, wall_points[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        if dx == 0 and dy == 0:
            continue
        wall_angle = math.degrees(math.atan2(dy, dx))
        for angle in (wall_angle, wall_angle + 90):
            normalized = angle % 180
            if rounded:
                normalized = round(normalized, 9) % 180
            raw_angles.append(float(normalized))
    return tuple(sorted(set(raw_angles)))


def _directed_angles(
    base_angles: Tuple[float, ...], item: Item, rounded: bool
) -> Tuple[float, ...]:
    """有指定开门边时补充反向角；同一矩形朝向可能对应不同开门方向。"""

    if item.front_side is None:
        return base_angles
    reversed_angles = set()
    for angle in base_angles:
        opposite = (angle + 180) % 360
        reversed_angles.add(float(round(opposite, 9) if rounded else opposite))
    return tuple(sorted(set(base_angles).union(reversed_angles)))


def exact_legal_angles(problem: Problem, item: Item) -> Tuple[float, ...]:
    """保留全部不同墙方向，供最终校验与完整方向搜索。"""
    return _directed_angles(_wall_angles(problem, rounded=False), item, rounded=False)


def legal_angles(problem: Problem, item: Item) -> Tuple[float, ...]:
    """合并近乎平行的 CAD 墙边角度，减少初轮搜索规模。"""
    raw_angles = _wall_angles(problem, rounded=True)
    # 坐标端点分别取整后，同一方向可能相差几个百万分之一度。
    base_angles = []
    for angle in raw_angles:
        if not any(
            min((angle - known) % 180, (known - angle) % 180) <= ANGLE_CLUSTER_TOLERANCE
            for known in base_angles
        ):
            base_angles.append(angle)
    return _directed_angles(tuple(base_angles), item, rounded=True)


def rectangle_polygon(item: Item, placement: Placement) -> Polygon:
    """生成设备水平投影；0° 时长沿本地 +x、宽沿本地 +y。"""
    half_length, half_width = item.length / 2, item.width / 2
    local = [
        (-half_length, -half_width),
        (half_length, -half_width),
        (half_length, half_width),
        (-half_length, half_width),
    ]
    return Polygon([_oriented_point(point, placement) for point in local])


def front_segment(item: Item, placement: Placement) -> Optional[LineString]:
    """返回旋转后的开门边；无 ``front_side`` 的设备返回 ``None``。"""
    half_length, half_width = item.length / 2, item.width / 2
    local_edges = {
        "+width": ((-half_length, half_width), (half_length, half_width)),
        "-width": ((-half_length, -half_width), (half_length, -half_width)),
        "+length": ((half_length, -half_width), (half_length, half_width)),
        "-length": ((-half_length, -half_width), (-half_length, half_width)),
    }
    if item.front_side is None:
        return None
    if item.front_side not in local_edges:
        raise ValueError("unknown front side for %s" % item.name)
    return LineString(
        [_oriented_point(point, placement) for point in local_edges[item.front_side]]
    )


def _finite_placement(placement: Placement) -> bool:
    if not isinstance(placement, Placement):
        return False
    coordinates = placement.center
    if not isinstance(coordinates, (tuple, list)) or len(coordinates) != 2:
        return False
    values = (coordinates[0], coordinates[1], placement.angle_deg, placement.z)
    return all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in values
    )


def _item_tolerance(problem: Problem, item: Item, placement: Placement) -> float:
    """依据设备短边、房间跨度及坐标量级估计几何误差。"""

    bounds = problem.boundary.bounds
    feature_size = min(
        item.length, item.width, bounds[2] - bounds[0], bounds[3] - bounds[1]
    )
    return _linear_tolerance(feature_size, *bounds, *placement.center)


def _height_tolerance(
    first_height: float, second_height: float, first_z: float, second_z: float
) -> float:
    return _linear_tolerance(
        min(first_height, second_height),
        first_z,
        second_z,
        first_z + first_height,
        second_z + second_height,
    )


def _area_tolerance(
    distance_tolerance: float, first_span: float, second_span: float
) -> float:
    # 浮点误差形成的细条面积不超过容差宽度乘较短轮廓尺度。
    return distance_tolerance * min(first_span, second_span)


def _angle_is_legal(
    angle: float, allowed: Tuple[float, ...], directional: bool
) -> bool:
    """开门边区分 360° 朝向，其余矩形按 180° 周期比较。"""

    period = 360 if directional else 180
    normalized = angle % period
    return any(
        min((normalized - candidate) % period, (candidate - normalized) % period)
        <= ANGLE_VALIDATION_TOLERANCE
        for candidate in allowed
    )


def _z_intervals_overlap(
    first: Item, first_place: Placement, second: Item, second_place: Placement
) -> bool:
    """只有设备占据的高度区间相交，水平投影重叠才构成碰撞。"""

    tolerance = _height_tolerance(
        first.height, second.height, first_place.z, second_place.z
    )
    return max(first_place.z, second_place.z) + tolerance < min(
        first_place.z + first.height, second_place.z + second.height
    )


def validate_solution(
    problem: Problem, placements: Mapping[str, Placement]
) -> List[str]:
    """独立检查全部放置结果，返回违反约束的原因列表。

    非开门边允许贴靠；XY 投影相交且占据高度重叠时才算实体碰撞。
    开门边要求与墙及同高度设备保持正间隙，不臆造固定净空深度。
    """
    errors = []  # type: List[str]
    if not isinstance(placements, dict):
        return ["placements must be a mapping from item names to placements"]
    by_name = {item.name: item for item in problem.items}
    for name in placements:
        if name not in by_name:
            errors.append("unknown item in placements: %s" % name)

    footprints = {}  # type: Dict[str, Polygon]
    tolerances = {}  # type: Dict[str, float]
    # 先逐件检查边界、门口、内开门 N×N 禁区及开门边靠墙情况。
    for item in problem.items:
        placement = placements.get(item.name)
        if placement is None:
            errors.append("missing placement for %s" % item.name)
            continue
        if not _finite_placement(placement):
            errors.append("invalid placement for %s" % item.name)
            continue
        z_tolerance = _height_tolerance(item.height, item.height, placement.z, 0)
        if placement.z < -z_tolerance:
            errors.append("%s is below floor level" % item.name)
        if item.supported_by is None and abs(placement.z) > z_tolerance:
            errors.append("%s has no support at its height" % item.name)
        if not _angle_is_legal(
            placement.angle_deg,
            exact_legal_angles(problem, item),
            item.front_side is not None,
        ):
            errors.append("%s is not parallel or perpendicular to a wall" % item.name)

        footprint = rectangle_polygon(item, placement)
        footprints[item.name] = footprint
        tolerance = _item_tolerance(problem, item, placement)
        tolerances[item.name] = tolerance
        if not problem.boundary.buffer(tolerance).covers(footprint):
            errors.append("%s extends outside the room" % item.name)
        if problem.door.intersection(footprint).length > tolerance:
            errors.append("%s covers the doorway" % item.name)
        if problem.door_keepout is not None and problem.door_keepout.intersection(
            footprint
        ).area > _area_tolerance(
            tolerance, max(item.length, item.width), problem.door.length
        ):
            errors.append("%s blocks the inward door swing" % item.name)
        front = front_segment(item, placement)
        if front is not None and front.distance(problem.boundary.boundary) <= tolerance:
            errors.append("%s opening edge touches a wall" % item.name)

    # 架上设备必须完全落在承载设备顶面，且底面高度恰等于顶面高度。
    for item in problem.items:
        parent_name = item.supported_by
        if parent_name is None or item.name not in footprints:
            continue
        parent = by_name.get(parent_name)
        if parent is None or parent_name not in footprints:
            errors.append("%s has no placed support" % item.name)
            continue
        child_placement, parent_placement = (
            placements[item.name],
            placements[parent_name],
        )
        z_tolerance = _height_tolerance(
            item.height, parent.height, child_placement.z, parent_placement.z
        )
        if abs(child_placement.z - parent_placement.z - parent.height) > z_tolerance:
            errors.append("%s is not on top of %s" % (item.name, parent_name))
        tolerance = max(tolerances[item.name], tolerances[parent_name])
        if not footprints[parent_name].buffer(tolerance).covers(footprints[item.name]):
            errors.append("%s does not fit on %s" % (item.name, parent_name))

    # 对所有同时占据相同高度的设备，分别检查实体相交和开门边受阻。
    placed_items = [item for item in problem.items if item.name in footprints]
    for index, first in enumerate(placed_items):
        first_place = placements[first.name]
        first_front = front_segment(first, first_place)
        for second in placed_items[index + 1 :]:
            second_place = placements[second.name]
            if not _z_intervals_overlap(first, first_place, second, second_place):
                continue
            first_shape, second_shape = footprints[first.name], footprints[second.name]
            tolerance = max(tolerances[first.name], tolerances[second.name])
            area_tolerance = _area_tolerance(
                tolerance,
                max(first.length, first.width),
                max(second.length, second.width),
            )
            if first_shape.intersection(second_shape).area > area_tolerance:
                errors.append("%s overlaps %s" % (first.name, second.name))
            second_front = front_segment(second, second_place)
            if (
                first_front is not None
                and first_front.distance(second_shape) <= tolerance
            ):
                errors.append(
                    "%s opening edge is blocked by %s" % (first.name, second.name)
                )
            if (
                second_front is not None
                and second_front.distance(first_shape) <= tolerance
            ):
                errors.append(
                    "%s opening edge is blocked by %s" % (second.name, first.name)
                )

    return errors

# ===== 快速构造搜索 =====

_MAX_CANDIDATES_PER_ITEM = 256
_MAX_GENERATED_PER_ITEM = 20000
_MAX_SEARCH_NODES = 4096


def _dot(first, second):
    return first[0] * second[0] + first[1] * second[1]


def _rotated_corners(item: Item, angle: float):
    radians = math.radians(angle)
    cosine, sine = math.cos(radians), math.sin(radians)
    half_length, half_width = item.length / 2, item.width / 2
    return tuple(
        (cosine * x - sine * y, sine * x + cosine * y)
        for x, y in (
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width),
        )
    )


def _wall_data(problem: Problem):
    points = list(problem.boundary.exterior.coords)
    walls = []
    for start, end in zip(points, points[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 0:
            continue
        tangent = (dx / length, dy / length)
        inward = (-tangent[1], tangent[0])
        if not problem.boundary.exterior.is_ccw:
            inward = (-inward[0], -inward[1])
        walls.append((start, tangent, inward, length))
    return walls


def _contacted_wall_length(item, placement, walls, tolerance):
    """返回有正长度贴合的最长墙段全长，点接触不计入。"""

    radians = math.radians(placement.angle_deg)
    along = (math.cos(radians), math.sin(radians))
    across = (-along[1], along[0])
    longest = 0.0
    for first, tangent, inward, length in walls:
        if abs(_cross(along, tangent)) > 1e-7 and abs(_dot(along, tangent)) > 1e-7:
            continue
        half_normal = (
            item.length * abs(_dot(along, inward))
            + item.width * abs(_dot(across, inward))
        ) / 2
        gap = _dot(inward, placement.center) - half_normal - _dot(inward, first)
        if abs(gap) > tolerance:
            continue
        half_tangent = (
            item.length * abs(_dot(along, tangent))
            + item.width * abs(_dot(across, tangent))
        ) / 2
        center_t = _dot(tangent, placement.center)
        start_t = _dot(tangent, first)
        shared = max(
            0.0,
            min(center_t + half_tangent, start_t + length)
            - max(center_t - half_tangent, start_t),
        )
        if shared > tolerance:
            longest = max(longest, length)
    return longest


def _search_item_order(item):
    """先安放开门设备与较难挪动的设备，再安放成组货架。"""

    if item.front_side is not None:
        category = 0
    elif item.name.startswith("iceMaker"):
        category = 1
    elif item.name.startswith("overShelf"):
        category = 2
    else:
        category = 3
    return category, -item.length * item.width, item.name


def _room_long_axis(problem):
    """用房间的最小外接矩形确定主方向，兼容整体旋转的房间。"""

    points = list(problem.boundary.minimum_rotated_rectangle.exterior.coords)
    edges = [
        (math.hypot(b[0] - a[0], b[1] - a[1]), a, b)
        for a, b in zip(points, points[1:])
    ]
    length, first, second = max(edges, key=lambda edge: edge[0])
    return ((second[0] - first[0]) / length, (second[1] - first[1]) / length)


def _fixed_vertices(problem: Problem, placed):
    vertices = list(problem.boundary.exterior.coords)[:-1]
    vertices.extend(problem.door.coords)
    if problem.door_keepout is not None:
        vertices.extend(list(problem.door_keepout.exterior.coords)[:-1])
    for _, shape, _, _, _ in placed.values():
        vertices.extend(list(shape.exterior.coords)[:-1])
    return vertices


def _centers_for_angle(item, angle, corners, vertices, walls):
    # 顶点接触覆盖普通角落以及沿已放物件边缘紧排的情况。
    for vertex in vertices:
        for corner in corners:
            yield vertex[0] - corner[0], vertex[1] - corner[1]

    radians = math.radians(angle)
    along = (math.cos(radians), math.sin(radians))
    across = (-along[1], along[0])
    for start, tangent, inward, wall_length in walls:
        if min(abs(_dot(tangent, along)), abs(_dot(tangent, across))) > 1e-5:
            continue
        half_normal = max(abs(_dot(corner, inward)) for corner in corners)
        half_tangent = max(abs(_dot(corner, tangent)) for corner in corners)
        # 法向贴墙；切向接触点由墙端点、门区及已放物件的顶点生成。
        positions = [half_tangent, wall_length - half_tangent, wall_length / 2]
        for vertex in vertices:
            projection = _dot(
                (vertex[0] - start[0], vertex[1] - start[1]), tangent
            )
            positions.extend((projection - half_tangent, projection + half_tangent))
        for position in positions:
            if position < -half_tangent or position > wall_length + half_tangent:
                continue
            yield (
                start[0] + inward[0] * half_normal + tangent[0] * position,
                start[1] + inward[1] * half_normal + tangent[1] * position,
            )


def _candidate_is_room_valid(problem, item, shape, front, tolerance):
    if not problem.boundary.buffer(tolerance).covers(shape):
        return False
    if problem.door.intersection(shape).length > tolerance:
        return False
    if problem.door_keepout is not None:
        area_tolerance = _area_tolerance(
            tolerance, max(item.length, item.width), problem.door.length
        )
        if problem.door_keepout.intersection(shape).area > area_tolerance:
            return False
    if front is not None and front.distance(problem.boundary.boundary) <= tolerance:
        return False
    return True


def _candidates(problem, item, placed, walls, wall_band, deadline):
    vertices = _fixed_vertices(problem, placed)
    bounds = problem.boundary.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    quantum = max(1e-9, min(item.length, item.width) * 1e-8)
    seen = set()
    results = []
    generated = 0

    extent = max(width, height)
    contact_tolerance = max(1e-7, extent * 1e-8)
    is_over_shelf = item.name.startswith("overShelf")
    if is_over_shelf:
        long_axis = _room_long_axis(problem)
        group_span = sum(
            min(other.length, other.width)
            for other in problem.items
            if other.name.startswith("overShelf")
        )
        group_margin = min(
            problem.door.length * 0.15,
            min(item.length, item.width) * 0.4,
        )
        wall_preferences = []
        for index, (first, tangent, _, length) in enumerate(walls):
            end = (first[0] + tangent[0] * length,
                   first[1] + tangent[1] * length)
            free_end_wall = (
                abs(_dot(tangent, long_axis)) < 0.5
                and length >= group_span + group_margin
            )
            group_on_wall = sum(
                1
                for other_item, _, _, _, other_pose in placed.values()
                if other_item.name.startswith("overShelf")
                and _contacted_wall_length(
                    other_item, other_pose, [walls[index]], contact_tolerance
                ) > 0
            )
            wall_preferences.append((
                group_on_wall,
                free_end_wall,
                LineString((first, end)).distance(problem.door),
                length,
            ))

    def candidate_order(candidate):
        """用房间墙段和设备开门深度排序，不依赖样例坐标。"""

        original_score, placement, shape = candidate[:3]
        wall_length = _contacted_wall_length(
            item, placement, walls, contact_tolerance
        )
        if item.front_side is not None:
            # 大型开门设备先占可贴墙、门前有空间的位置；其余几何
            # 条件仍由候选检查及最终独立校验保证。
            front_depth, _ = _front_room_depth(problem, item, placement)
            door_distance = shape.distance(problem.door)
            opening_score = (
                (wall_length / extent) ** 2
                * (front_depth / extent)
                * (max(door_distance, 1e-9) / extent) ** 0.7
            )
            return -opening_score, original_score
        if is_over_shelf:
            contacts = [
                index for index, wall in enumerate(walls)
                if _contacted_wall_length(
                    item, placement, [wall], contact_tolerance
                ) > 0
            ]
            if contacts:
                same_group, end_wall, door_gap, wall_length = max(
                    (wall_preferences[index] for index in contacts),
                    key=lambda choice: (choice[0], choice[1], choice[2], choice[3]),
                )
                # 一段足以容纳整组、远离入口的端墙是优先区；随后
                # 接续已有同类设备。同类分组只是偏好，不能阻止回溯。
                return (-same_group, -int(end_wall), -door_gap,
                        -wall_length, original_score)
            return (0, 0, 0.0, 0.0, original_score)
        # 同样贴住一件架子的长边时，优先保留长墙给连续的架子，
        # 避免由绝对坐标打破平局而把架子堆在短墙端部。
        return -wall_length, original_score

    for angle in legal_angles(problem, item):
        corners = _rotated_corners(item, angle)
        for center in _centers_for_angle(item, angle, corners, vertices, walls):
            generated += 1
            if generated > _MAX_GENERATED_PER_ITEM or (
                generated % 64 == 0 and time.monotonic() >= deadline
            ):
                return sorted(results, key=candidate_order)[
                    :_MAX_CANDIDATES_PER_ITEM
                ]
            key = (
                round(center[0] / quantum),
                round(center[1] / quantum),
                round(angle, 7),
            )
            if key in seen:
                continue
            seen.add(key)
            placement = Placement(center, angle)
            shape = rectangle_polygon(item, placement)
            tolerance = _item_tolerance(problem, item, placement)
            front = front_segment(item, placement)
            if not _candidate_is_room_valid(
                problem, item, shape, front, tolerance
            ):
                continue
            # 先排除与已有物件冲突的位置，再取排序前 256 个。
            # 否则高分位置可能全被占据，挤掉真正可用的候选。
            if _conflicts(item, shape, front, tolerance, placed):
                continue

            shared_wall = shape.boundary.intersection(wall_band).length
            relative_x = (center[0] - bounds[0]) / width
            relative_y = (center[1] - bounds[1]) / height
            # 优先贴墙；位置次序仅用于保持搜索确定性。
            score = (
                -round(shared_wall, 2),
                relative_x + relative_y,
                relative_x,
                relative_y,
            )
            results.append((score, placement, shape, front, tolerance))
    return sorted(results, key=candidate_order)[
        :_MAX_CANDIDATES_PER_ITEM
    ]


def _conflicts(item, shape, front, tolerance, placed):
    for other_item, other_shape, other_front, other_tolerance, _ in placed.values():
        pair_tolerance = max(tolerance, other_tolerance)
        area_tolerance = _area_tolerance(
            pair_tolerance,
            max(item.length, item.width),
            max(other_item.length, other_item.width),
        )
        if shape.intersection(other_shape).area > area_tolerance:
            return True
        if front is not None and front.distance(other_shape) <= pair_tolerance:
            return True
        if other_front is not None and other_front.distance(shape) <= pair_tolerance:
            return True
    return False


def find_feasible_layout(
    problem: Problem, time_limit: float = 2.0
) -> Optional[Dict[str, Placement]]:
    """构造全部地面物品的可行摆放；失败时返回 ``None``。

    承载关系交给完整 MILP 处理。搜索只返回经 ``validate_solution``
    独立检查的布局，不会把候选耗尽误报为不可行。
    """

    if any(item.supported_by is not None for item in problem.items):
        return None
    if not math.isfinite(time_limit) or time_limit <= 0:
        return None
    if not problem.items:
        return {} if not validate_solution(problem, {}) else None

    deadline = time.monotonic() + time_limit
    walls = _wall_data(problem)
    room_extent = max(
        problem.boundary.bounds[2] - problem.boundary.bounds[0],
        problem.boundary.bounds[3] - problem.boundary.bounds[1],
    )
    wall_band = problem.boundary.boundary.buffer(max(1e-5, room_extent * 1e-9))
    items = sorted(
        problem.items,
        key=_search_item_order,
    )
    # item 名称 -> (item, footprint, front, tolerance, placement)
    placed = {}
    nodes = 0

    def search(index):
        nonlocal nodes
        if time.monotonic() >= deadline or nodes >= _MAX_SEARCH_NODES:
            return None
        if index == len(items):
            solution = {name: data[4] for name, data in placed.items()}
            return solution if not validate_solution(problem, solution) else None

        item = items[index]
        for _, placement, shape, front, tolerance in _candidates(
            problem, item, placed, walls, wall_band, deadline
        ):
            if time.monotonic() >= deadline or nodes >= _MAX_SEARCH_NODES:
                break
            if _conflicts(item, shape, front, tolerance, placed):
                continue
            placed[item.name] = (item, shape, front, tolerance, placement)
            nodes += 1
            solution = search(index + 1)
            if solution is not None:
                return solution
            del placed[item.name]
        return None

    return search(0)


def _layout_quality(problem, placements, entrance, walls, wall_tolerance):
    """比较已验证布局的入口占用、贴墙数和开门设备周围的留白。

    净距仅用于在同一题目内部排序，不被解释为法规或厂家要求。
    按从小到大的距离逐项比较，优先改善最拥挤的位置。
    """

    footprints = {
        item.name: rectangle_polygon(item, placements[item.name])
        for item in problem.items
    }
    # 外开门虽然不占室内门扇区，人仍需从门口进入。用门宽构造的
    # 入口区域作为质量目标，不改变原题对外开门的硬约束含义。
    entrance_overlap = sum(
        shape.intersection(entrance).area
        for item in problem.items
        if placements[item.name].z == 0
        for shape in (footprints[item.name],)
    )
    wall_count, wall_length = _wall_score(
        problem, placements, walls, wall_tolerance
    )
    front_gaps = []
    effective_open_space = []
    useful_open_space = []
    room_depths = []
    body_gaps = []
    for item in problem.items:
        if item.front_side is None:
            continue
        pose = placements[item.name]
        front = front_segment(item, pose)
        scale = min(item.length, item.width)
        median_depth, shallow_depth = _front_room_depth(problem, item, pose)
        room_depths.append((median_depth / scale, shallow_depth / scale))
        item_front_gaps = []
        for other in problem.items:
            if other.name == item.name or not _z_intervals_overlap(
                item, pose, other, placements[other.name]
            ):
                continue
            gap = front.distance(footprints[other.name]) / scale
            front_gaps.append(gap)
            item_front_gaps.append(gap)
            body_gaps.append(
                footprints[item.name].distance(footprints[other.name]) / scale
            )
        # 物品很远但门朝向尽端时，实际可用深度仍受墙限制。
        nearest = min(item_front_gaps) if item_front_gaps else float("inf")
        effective = min(nearest, median_depth / scale)
        effective_open_space.append(effective)
        # 开门边半宽只是质量排序的饱和尺度，不是施工净空标准。
        # 未达到它时可牺牲一次贴墙；达到后优先多贴墙。
        useful_open_space.append(min(effective, front.length / (2 * scale)))
    return (
        -round(entrance_overlap / max(entrance.area, 1.0), 7),
        tuple(round(value, 7) for value in sorted(useful_open_space)),
        wall_count,
        tuple(round(value, 7) for value in sorted(effective_open_space)),
        tuple(
            (round(median, 7), round(shallow, 7))
            for median, shallow in sorted(room_depths)
        ),
        tuple(round(value, 7) for value in sorted(front_gaps)),
        tuple(round(value, 7) for value in sorted(body_gaps)),
        round(wall_length, 7),
    )


def _front_room_depth(problem, item, pose):
    """沿开门方向测三条射线，返回中位和最短的可用房间深度。

    这只评估房间几何；其他设备的阻挡由 `_layout_quality` 中的
    开门边净距另行衡量。采样避开边缘，减少侧墙对整扇门的误判。
    """

    front = front_segment(item, pose)
    first, second = front.coords
    middle = ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)
    outward = (middle[0] - pose.center[0], middle[1] - pose.center[1])
    norm = math.hypot(*outward)
    normal = (outward[0] / norm, outward[1] / norm)
    bounds = problem.boundary.bounds
    extent = 2 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])

    def hit_distances(geometry, start):
        if geometry.is_empty:
            return []
        if hasattr(geometry, "geoms"):
            values = []
            for part in geometry.geoms:
                values.extend(hit_distances(part, start))
            return values
        return [
            (point[0] - start[0]) * normal[0]
            + (point[1] - start[1]) * normal[1]
            for point in geometry.coords
        ]

    depths = []
    for fraction in (0.2, 0.5, 0.8):
        start = (
            first[0] + fraction * (second[0] - first[0]),
            first[1] + fraction * (second[1] - first[1]),
        )
        end = (start[0] + extent * normal[0],
               start[1] + extent * normal[1])
        crossings = hit_distances(
            LineString((start, end)).intersection(problem.boundary.boundary),
            start,
        )
        positive = [distance for distance in crossings if distance > 1e-7]
        depths.append(min(positive) if positive else extent)
    return sorted(depths)[1], min(depths)


def _door_front_radius(problem, placements, item, steps=6):
    """估计人从门走到一段开门边时，路径允许的最大圆形半径。

    将房间内缩、物件外扩同一半径，检查门口与开门边采样点是否
    落在同一自由区域。结果只用于相对排序，不代表人体或规范尺寸。
    """

    front = front_segment(item, placements[item.name])
    first, second = front.coords
    middle = ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)
    outward = (
        middle[0] - placements[item.name].center[0],
        middle[1] - placements[item.name].center[1],
    )
    norm = math.hypot(*outward)
    outward = (outward[0] / norm, outward[1] / norm)
    door_first, door_second = problem.door.coords
    door_middle = (
        (door_first[0] + door_second[0]) / 2,
        (door_first[1] + door_second[1]) / 2,
    )
    inside = _door_keepout(problem.boundary, problem.door).centroid
    inward = (inside.x - door_middle[0], inside.y - door_middle[1])
    norm = math.hypot(*inward)
    inward = (inward[0] / norm, inward[1] / norm)
    footprints = [
        rectangle_polygon(other, placements[other.name])
        for other in problem.items if placements[other.name].z == 0
    ]
    limit = min(problem.door.length, front.length) / 2
    margin = max(
        _linear_tolerance(limit, *door_middle, *middle),
        limit * 1e-5,
    )

    def connected(radius):
        free_space = problem.boundary.buffer(-radius).difference(
            unary_union([shape.buffer(radius) for shape in footprints])
        )
        if free_space.is_empty:
            return False
        offset = radius + margin
        door_points = [
            Point(
                door_first[0] + fraction * (door_second[0] - door_first[0])
                + inward[0] * offset,
                door_first[1] + fraction * (door_second[1] - door_first[1])
                + inward[1] * offset,
            )
            for fraction in (0.25, 0.5, 0.75)
        ]
        front_points = [
            Point(
                first[0] + fraction * (second[0] - first[0])
                + outward[0] * offset,
                first[1] + fraction * (second[1] - first[1])
                + outward[1] * offset,
            )
            for fraction in (0.25, 0.5, 0.75)
        ]
        pieces = getattr(free_space, "geoms", (free_space,))
        return any(
            piece.area > 0
            and any(piece.covers(point) for point in door_points)
            and sum(piece.covers(point) for point in front_points) >= 2
            for piece in pieces
        )

    lower, upper = 0.0, limit
    for _ in range(steps):
        midpoint = (lower + upper) / 2
        if connected(midpoint):
            lower = midpoint
        else:
            upper = midpoint
    return lower


_SERVICE_RADIUS_FRACTIONS = (
    0.001, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.45
)


def _service_access_profile(problem, placements):
    """按逐渐增大的通行半径，统计从门可到达操作边的设备数。

    有指定开门边的设备只检查该边；普通架子尝试两条长边。共享
    每个半径下的房间自由区域，避免逐件重复求几何差集。这里的半径
    仅是布局间的相对指标，不是人体或消防通道的合格尺寸。
    """

    door_first, door_second = problem.door.coords
    door_middle = (
        (door_first[0] + door_second[0]) / 2,
        (door_first[1] + door_second[1]) / 2,
    )
    inside = _door_keepout(problem.boundary, problem.door).centroid
    inward = (inside.x - door_middle[0], inside.y - door_middle[1])
    norm = math.hypot(*inward)
    inward = (inward[0] / norm, inward[1] / norm)
    obstacles = [
        rectangle_polygon(item, placements[item.name])
        for item in problem.items if placements[item.name].z == 0
    ]
    service_edges = []
    for item in problem.items:
        pose = placements[item.name]
        if item.front_side is not None:
            edges = (front_segment(item, pose),)
        else:
            corners = list(rectangle_polygon(item, pose).exterior.coords)[:4]
            if item.length >= item.width:
                pairs = ((corners[0], corners[1]), (corners[2], corners[3]))
            else:
                pairs = ((corners[1], corners[2]), (corners[3], corners[0]))
            edges = tuple(LineString(pair) for pair in pairs)
        options = []
        for edge in edges:
            first, second = edge.coords
            middle = ((first[0] + second[0]) / 2,
                      (first[1] + second[1]) / 2)
            normal = (middle[0] - pose.center[0],
                      middle[1] - pose.center[1])
            size = math.hypot(*normal)
            options.append((first, second, (normal[0] / size, normal[1] / size)))
        service_edges.append(options)

    counts = []
    for fraction in _SERVICE_RADIUS_FRACTIONS:
        radius = problem.door.length * fraction
        free = problem.boundary.buffer(-radius).difference(
            unary_union([shape.buffer(radius) for shape in obstacles])
        )
        if free.is_empty:
            counts.append(0)
            continue
        margin = max(_linear_tolerance(radius, *door_middle), radius * 1e-5)
        offset = radius + margin
        door_points = [
            Point(
                door_first[0] + part * (door_second[0] - door_first[0])
                + inward[0] * offset,
                door_first[1] + part * (door_second[1] - door_first[1])
                + inward[1] * offset,
            )
            for part in (0.25, 0.5, 0.75)
        ]
        pieces = [
            piece for piece in getattr(free, "geoms", (free,))
            if piece.area > 0 and any(piece.covers(point) for point in door_points)
        ]
        count = 0
        for options in service_edges:
            if any(
                any(
                    sum(
                        piece.covers(
                            Point(
                                first[0] + part * (second[0] - first[0])
                                + normal[0] * offset,
                                first[1] + part * (second[1] - first[1])
                                + normal[1] * offset,
                            )
                        )
                        for part in (0.25, 0.5, 0.75)
                    ) >= 2
                    for piece in pieces
                )
                for first, second, normal in options
            ):
                count += 1
        counts.append(count)
    return tuple(counts)


def _path_layout_quality(problem, placements, entrance, walls, wall_tolerance):
    """先保证设备能从门口到达，再优先贴墙，最后比较通路宽度。

    题目要求空间允许时优先贴墙。若两个布局在最小测试半径下
    可达件数相同，不能仅因较大的测试半径多通过几件，就把一件
    本可贴墙的货架移到过道中间。
    """

    base = _layout_quality(problem, placements, entrance, walls, wall_tolerance)
    service_profile = _service_access_profile(problem, placements)
    relative_radii = []
    radii = []
    front_gaps = []
    for item in problem.items:
        if item.front_side is None:
            continue
        radius = _door_front_radius(problem, placements, item)
        radii.append(radius)
        front = front_segment(item, placements[item.name])
        # 连通的通道优先于整条开门边上的最近点净距：一个角落可能
        # 离货架很近，但另一段开门边仍可从门口顺畅抵达。
        limit = min(problem.door.length, front.length) / 2
        relative_radii.append(radius / limit)
        front_gaps.append(min(
            (
                front.distance(rectangle_polygon(other, placements[other.name]))
                for other in problem.items
                if other.name != item.name and _z_intervals_overlap(
                    item, placements[item.name], other, placements[other.name]
                )
            ),
            default=problem.door.length,
        ))
    return (
        base[0],
        service_profile[0],
        base[2],
        service_profile[1:],
        tuple(round(value, 6) for value in sorted(relative_radii)),
        tuple(round(value / problem.door.length, 6)
              for value in sorted(front_gaps)),
        tuple(round(radius, 6) for radius in sorted(radii)), base,
    )


def _find_accessible_layout(problem, initial, time_limit):
    """从不同的首件贴墙位置重新摆满房间，避免陷在首个可行解。

    单件挪动无法交换冰箱和货架占据的整段墙。这里固定首件的一个
    候选位置，重新构造其余设备，并以入口到操作面的连通宽度比较。
    超时返回已知的有效布局，不把未搜索完的分支当成无解。
    """

    if time_limit <= 0 or any(item.supported_by for item in problem.items):
        return initial
    if not any(item.front_side for item in problem.items):
        return initial
    deadline = time.monotonic() + time_limit
    items = sorted(
        problem.items,
        key=_search_item_order,
    )
    walls = _wall_data(problem)
    extent = max(
        problem.boundary.bounds[2] - problem.boundary.bounds[0],
        problem.boundary.bounds[3] - problem.boundary.bounds[1],
    )
    wall_band = problem.boundary.boundary.buffer(max(1e-5, extent * 1e-9))
    score_walls = _wall_geometry(problem)
    wall_tolerance = max(1e-7, extent * 1e-8)
    entrance = _door_keepout(problem.boundary, problem.door)
    best = dict(initial)
    best_score = _path_layout_quality(
        problem, best, entrance, score_walls, wall_tolerance
    )

    def complete(index, placed, nodes):
        if time.monotonic() >= deadline or nodes[0] >= _MAX_SEARCH_NODES:
            return None
        if index == len(items):
            candidate = {name: data[4] for name, data in placed.items()}
            return candidate if not validate_solution(problem, candidate) else None
        item = items[index]
        for _, pose, shape, front, tolerance in _candidates(
            problem, item, placed, walls, wall_band, deadline
        ):
            if time.monotonic() >= deadline or nodes[0] >= _MAX_SEARCH_NODES:
                break
            placed[item.name] = (item, shape, front, tolerance, pose)
            nodes[0] += 1
            candidate = complete(index + 1, placed, nodes)
            if candidate is not None:
                return candidate
            del placed[item.name]
        return None

    first = items[0]
    for _, pose, shape, front, tolerance in _candidates(
        problem, first, {}, walls, wall_band, deadline
    ):
        if time.monotonic() >= deadline:
            break
        placed = {first.name: (first, shape, front, tolerance, pose)}
        candidate = complete(1, placed, [0])
        if candidate is None:
            continue
        score = _path_layout_quality(
            problem, candidate, entrance, score_walls, wall_tolerance
        )
        if score > best_score:
            best, best_score = candidate, score
    return best


def _refine_layout(problem, placements, time_limit):
    """在不破坏硬约束的前提下，重新安置妨碍入口或开门的设备。

    每次只移动一件设备并重新校验。承载组件保持不动；与它们无
    承载关系的地面设备仍可改进，不使一台架上机器禁用整个改进阶段。
    """

    if time_limit <= 0:
        return placements
    support_names = {
        item.supported_by for item in problem.items if item.supported_by is not None
    }
    movable = [
        item for item in problem.items
        if item.supported_by is None and item.name not in support_names
    ]
    if not movable:
        return placements
    entrance = _door_keepout(problem.boundary, problem.door)
    if not any(item.front_side for item in problem.items) and all(
        rectangle_polygon(item, placements[item.name]).intersection(entrance).area
        <= 1e-9
        for item in problem.items
    ):
        return placements

    deadline = time.monotonic() + time_limit
    current = dict(placements)
    walls = _wall_data(problem)
    score_walls = _wall_geometry(problem)
    extent = max(
        problem.boundary.bounds[2] - problem.boundary.bounds[0],
        problem.boundary.bounds[3] - problem.boundary.bounds[1],
    )
    wall_tolerance = max(1e-7, extent * 1e-8)
    wall_band = problem.boundary.boundary.buffer(max(1e-5, extent * 1e-9))
    current_score = _layout_quality(
        problem, current, entrance, score_walls, wall_tolerance
    )

    for _ in range(min(len(movable), 6)):
        if time.monotonic() >= deadline:
            break
        best_score = current_score
        best_move = None
        for item in movable:
            if time.monotonic() >= deadline:
                break
            placed = {}
            for other in problem.items:
                if other.name == item.name:
                    continue
                pose = current[other.name]
                footprint = rectangle_polygon(other, pose)
                placed[other.name] = (
                    other,
                    footprint,
                    front_segment(other, pose),
                    _item_tolerance(problem, other, pose),
                    pose,
                )
            for _, pose, _, _, _ in _candidates(
                problem, item, placed, walls, wall_band, deadline
            ):
                if time.monotonic() >= deadline:
                    break
                trial = dict(current)
                trial[item.name] = pose
                score = _layout_quality(
                    problem, trial, entrance, score_walls, wall_tolerance
                )
                if score > best_score:
                    best_score = score
                    best_move = (item.name, pose)
        if best_move is None:
            break
        trial = dict(current)
        trial[best_move[0]] = best_move[1]
        if validate_solution(problem, trial):
            break
        current, current_score = trial, best_score
    return current


def _refine_access_path(problem, placements, time_limit):
    """在几何摆放之后改善门至设备正面的连续可通行宽度。"""

    front_items = [item for item in problem.items if item.front_side is not None]
    if time_limit <= 0 or not front_items:
        return placements

    support_names = {
        item.supported_by for item in problem.items if item.supported_by is not None
    }
    movable = [
        item for item in problem.items
        if item.supported_by is None and item.name not in support_names
    ]
    if not movable:
        return placements
    deadline = time.monotonic() + time_limit
    current = dict(placements)
    entrance = _door_keepout(problem.boundary, problem.door)
    walls = _wall_data(problem)
    score_walls = _wall_geometry(problem)
    extent = max(
        problem.boundary.bounds[2] - problem.boundary.bounds[0],
        problem.boundary.bounds[3] - problem.boundary.bounds[1],
    )
    wall_tolerance = max(1e-7, extent * 1e-8)
    wall_band = problem.boundary.boundary.buffer(max(1e-5, extent * 1e-9))
    current_score = _path_layout_quality(
        problem, current, entrance, score_walls, wall_tolerance
    )
    for _ in range(2):
        if time.monotonic() >= deadline:
            break
        best_score, best_move = current_score, None
        for item in movable:
            if time.monotonic() >= deadline:
                break
            placed = {}
            for other in problem.items:
                if other.name == item.name:
                    continue
                pose = current[other.name]
                placed[other.name] = (
                    other,
                    rectangle_polygon(other, pose),
                    front_segment(other, pose),
                    _item_tolerance(problem, other, pose),
                    pose,
                )
            for _, pose, _, _, _ in _candidates(
                problem, item, placed, walls, wall_band, deadline
            ):
                if time.monotonic() >= deadline:
                    break
                trial = dict(current)
                trial[item.name] = pose
                score = _path_layout_quality(
                    problem, trial, entrance, score_walls, wall_tolerance
                )
                if score > best_score:
                    best_score, best_move = score, (item.name, pose)
        if best_move is None:
            break
        trial = dict(current)
        trial[best_move[0]] = best_move[1]
        if validate_solution(problem, trial):
            break
        current, current_score = trial, best_score
    return current

# ===== MILP 回退与求解流程 =====

@dataclass
class Solution:
    """求解结果：可行、已证明不可行，或限时内未能判定。"""

    status: str
    placements: Dict[str, Placement]
    message: str = ""


class _Affine:
    """稀疏仿射表达式，供约束和目标函数复用。"""

    def __init__(self, terms=None, constant=0.0):
        self.terms = dict(terms or {})
        self.constant = float(constant)

    def __add__(self, other):
        if not isinstance(other, _Affine):
            other = _Affine(constant=other)
        terms = dict(self.terms)
        for index, coefficient in other.terms.items():
            terms[index] = terms.get(index, 0.0) + coefficient
        return _Affine(terms, self.constant + other.constant)

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self * -1.0

    def __mul__(self, factor):
        return _Affine(
            {index: factor * value for index, value in self.terms.items()},
            factor * self.constant,
        )

    __rmul__ = __mul__


class _Model:
    """组装稀疏 MILP 约束，并保存变量上下界以计算尽量小的 big-M。"""

    def __init__(self):
        self.lower = []  # type: List[float]
        self.upper = []  # type: List[float]
        self.integrality = []  # type: List[int]
        self.cost = []  # type: List[float]
        self.rows = []  # type: List[Dict[int, float]]
        self.row_lower = []  # type: List[float]
        self.row_upper = []  # type: List[float]

    def variable(self, low=0.0, high=1.0, binary=False, cost=0.0):
        index = len(self.lower)
        self.lower.append(float(low))
        self.upper.append(float(high))
        self.integrality.append(1 if binary else 0)
        self.cost.append(float(cost))
        return index

    def row(self, expression, low=-math.inf, high=math.inf):
        self.rows.append(expression.terms)
        self.row_lower.append(low - expression.constant)
        self.row_upper.append(high - expression.constant)

    def max_value(self, expression):
        value = expression.constant
        for index, coefficient in expression.terms.items():
            value += coefficient * (
                self.upper[index] if coefficient >= 0 else self.lower[index]
            )
        return value

    def implies_at_most(self, selector, expression, high=0.0):
        # selector=1 时强制 expression<=high；=0 时用变量界计算的 M 关闭该约束。
        # 不写任意的巨大常数，能减轻 MILP 的数值误差。
        big_m = max(0.0, self.max_value(expression) - high) + 1e-8
        self.row(expression + _Affine({selector: big_m}), high=high + big_m)

    def implies_at_least(self, selector, expression, low=0.0):
        self.implies_at_most(selector, -expression, high=-low)

    def optimize(self, seconds):
        row_indices = []
        column_indices = []
        data = []
        for row_index, row in enumerate(self.rows):
            for column_index, coefficient in row.items():
                if coefficient:
                    row_indices.append(row_index)
                    column_indices.append(column_index)
                    data.append(coefficient)
        matrix = coo_matrix(
            (data, (row_indices, column_indices)),
            shape=(len(self.rows), len(self.lower)),
        ).tocsc()
        return milp(
            c=np.asarray(self.cost),
            integrality=np.asarray(self.integrality),
            bounds=Bounds(self.lower, self.upper),
            constraints=LinearConstraint(matrix, self.row_lower, self.row_upper),
            options={"time_limit": max(0.01, seconds), "presolve": True},
        )


class _Shape:
    """凸形状的中心及支撑函数；支撑函数给出指定轴上的半宽。"""

    def __init__(self, center_x, center_y, support):
        self.center_x = center_x
        self.center_y = center_y
        self.support = support

    def projection(self, axis):
        return axis[0] * self.center_x + axis[1] * self.center_y


def _dot(left, right):
    return left[0] * right[0] + left[1] * right[1]


def _cross(left, right):
    return left[0] * right[1] - left[1] * right[0]


def _unit_axes(points, parallel_tolerance=1e-9):
    axes = []
    for first, second in zip(points, points[1:] + points[:1]):
        dx, dy = second[0] - first[0], second[1] - first[1]
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        for axis in ((dx / length, dy / length), (-dy / length, dx / length)):
            if axis[0] < -1e-10 or (abs(axis[0]) <= 1e-10 and axis[1] < 0):
                axis = (-axis[0], -axis[1])
            if not any(abs(_cross(axis, old)) < parallel_tolerance for old in axes):
                axes.append(axis)
    return tuple(axes)


def _triangulate_simple_polygon(points):
    """耳切法把简单多边形分成三角形，用于表达凹房间的中心位置。"""

    remaining = list(points)
    if sum(_cross(a, b) for a, b in zip(remaining, remaining[1:] + remaining[:1])) < 0:
        remaining.reverse()
    # 中间的共线顶点不改变房间区域，却会妨碍耳切法稳定选出三角形。
    changed = True
    while changed and len(remaining) > 3:
        changed = False
        for index in range(len(remaining)):
            before = remaining[index - 1]
            current = remaining[index]
            after = remaining[(index + 1) % len(remaining)]
            if (
                abs(
                    _cross(
                        (current[0] - before[0], current[1] - before[1]),
                        (after[0] - current[0], after[1] - current[1]),
                    )
                )
                < 1e-11
            ):
                del remaining[index]
                changed = True
                break

    triangles = []
    while len(remaining) > 3:
        outline = Polygon(remaining)
        chosen = None
        for index in range(len(remaining)):
            before = remaining[index - 1]
            current = remaining[index]
            after = remaining[(index + 1) % len(remaining)]
            if (
                _cross(
                    (current[0] - before[0], current[1] - before[1]),
                    (after[0] - current[0], after[1] - current[1]),
                )
                <= 1e-11
            ):
                continue
            triangle = Polygon((before, current, after))
            if not outline.covers(triangle):
                continue
            if any(
                triangle.contains(Point(other))
                for other_index, other in enumerate(remaining)
                if other_index not in (index - 1, index, (index + 1) % len(remaining))
            ):
                continue
            chosen = index
            triangles.append((before, current, after))
            break
        if chosen is None:
            raise ValueError("The normalized room outline could not be triangulated")
        del remaining[chosen]
    triangles.append(tuple(remaining))
    return triangles


def _separate(model, first, second, axes, clearance=0.0):
    """要求两个凸形状在至少一条分离轴的某一方向上不相交。"""

    selectors = []
    # 对每条候选轴建立两个方向的投影不等式；二元变量只需选中其中一个。
    # clearance 默认为零，表示设备边界相贴合法，内部面积不能重叠。
    for base in axes:
        for direction in (1.0, -1.0):
            axis = (direction * base[0], direction * base[1])
            opposite = (-axis[0], -axis[1])
            expression = (
                first.projection(axis)
                - second.projection(axis)
                + first.support(axis)
                + second.support(opposite)
            )
            selector = model.variable(binary=True)
            model.implies_at_most(selector, expression, high=-clearance)
            selectors.append(selector)
    model.row(_Affine({selector: 1.0 for selector in selectors}), low=1.0)


def _fixed_shape(vertices):
    center = (
        sum(vertex[0] for vertex in vertices) / len(vertices),
        sum(vertex[1] for vertex in vertices) / len(vertices),
    )

    def support(axis):
        return _Affine(
            constant=max(
                _dot(axis, (point[0] - center[0], point[1] - center[1]))
                for point in vertices
            )
        )

    return _Shape(_Affine(constant=center[0]), _Affine(constant=center[1]), support)


def _rectangle_shape(x_index, y_index, choices, item, scale):
    def support(axis):
        terms = {}
        for angle, selector in choices:
            radians = math.radians(angle)
            along = (math.cos(radians), math.sin(radians))
            across = (-along[1], along[0])
            terms[selector] = (
                0.5 * item.length * abs(_dot(axis, along))
                + 0.5 * item.width * abs(_dot(axis, across))
            ) / scale
        return _Affine(terms)

    return _Shape(_Affine({x_index: 1.0}), _Affine({y_index: 1.0}), support)


def _front_shape(x_index, y_index, choices, item, scale):
    """将可开启的正面建模为随角度旋转的线段，而非臆造固定净空深度。"""

    x_terms = {x_index: 1.0}
    y_terms = {y_index: 1.0}
    directions = []
    for angle, selector in choices:
        radians = math.radians(angle)
        along = (math.cos(radians), math.sin(radians))
        across = (-along[1], along[0])
        side = item.front_side or "+width"
        if side in ("+width", "-width"):
            sign = 1.0 if side[0] == "+" else -1.0
            offset = (
                sign * item.width * across[0] / (2 * scale),
                sign * item.width * across[1] / (2 * scale),
            )
            segment_direction = along
            half_length = item.length / (2 * scale)
        else:
            sign = 1.0 if side[0] == "+" else -1.0
            offset = (
                sign * item.length * along[0] / (2 * scale),
                sign * item.length * along[1] / (2 * scale),
            )
            segment_direction = across
            half_length = item.width / (2 * scale)
        x_terms[selector] = offset[0]
        y_terms[selector] = offset[1]
        directions.append((selector, segment_direction, half_length))

    def support(axis):
        return _Affine(
            {
                selector: half_length * abs(_dot(axis, direction))
                for selector, direction, half_length in directions
            }
        )

    return _Shape(_Affine(x_terms), _Affine(y_terms), support)


def _root_of(item, by_name):
    while item.supported_by is not None:
        item = by_name[item.supported_by]
    return item.name


def _support_depth(item, by_name):
    depth = 0
    while item.supported_by is not None:
        item = by_name[item.supported_by]
        depth += 1
    return depth


def _support_layers(problem):
    """按显式承载关系分组，每组只用最底层设备占用地面。"""

    by_name = {item.name: item for item in problem.items}
    roots = []
    members = {}
    for item in problem.items:
        root = _root_of(item, by_name)
        if root not in members:
            members[root] = []
            roots.append(by_name[root])
        members[root].append(item)
    return roots, members, ""


def _assembly_states(problem, root, members, angle_provider):
    """枚举整组合法角度，并筛掉尺寸无法放进承载面的组合。"""

    by_name = {item.name: item for item in members}

    children = sorted(
        (item for item in members if item.name != root.name),
        key=lambda item: _support_depth(item, by_name),
    )
    states = [{root.name: angle} for angle in angle_provider(problem, root)]
    for child in children:
        parent = by_name[child.supported_by]
        next_states = []
        for state in states:
            parent_radians = math.radians(state[parent.name])
            for angle in angle_provider(problem, child):
                # 投影到架子的长、宽轴；子设备可采用自己的合法墙角方向，
                # 因此不要求它与架子的长边平行。
                difference = math.radians(angle) - parent_radians
                cosine = abs(math.cos(difference))
                sine = abs(math.sin(difference))
                span_along = child.length * cosine + child.width * sine
                span_across = child.length * sine + child.width * cosine
                if (
                    span_along <= parent.length + 1e-9
                    and span_across <= parent.width + 1e-9
                ):
                    next_state = dict(state)
                    next_state[child.name] = angle
                    next_states.append(next_state)
        states = next_states
        if len(states) > 256:
            return None
        if not states:
            break
    return states


def _make_placements(
    problem,
    roots,
    members,
    choices,
    state_angles,
    member_centers,
    heights,
    values,
    origin,
    scale,
):
    placements = {}
    for root in roots:
        options = choices[root.name]
        _, selector = max(options, key=lambda pair: values[pair[1]])
        angles = state_angles[root.name][selector]
        for member in members[root.name]:
            x_index, y_index = member_centers[member.name]
            center = (
                origin[0] + scale * values[x_index],
                origin[1] + scale * values[y_index],
            )
            placements[member.name] = Placement(
                center, angles[member.name], heights[member.name]
            )
    return placements


def _wall_geometry(problem):
    """统一墙边方向，使房间内部总在每条有向墙的左侧。"""

    points = list(problem.boundary.exterior.coords)[:-1]
    if sum(_cross(a, b) for a, b in zip(points, points[1:] + points[:1])) < 0:
        points.reverse()
    walls = []
    for first, second in zip(points, points[1:] + points[:1]):
        difference = (second[0] - first[0], second[1] - first[1])
        length = math.hypot(*difference)
        if length <= 1e-10:
            continue
        tangent = (difference[0] / length, difference[1] / length)
        walls.append((first, tangent, (-tangent[1], tangent[0]), length))
    return walls


def _wall_score(problem, placements, walls, tolerance):
    """统计有效贴墙设备，容忍 CAD 坐标与浮点运算带来的微小误差。"""

    count = 0
    shared_total = 0.0
    for item in problem.items:
        placement = placements[item.name]
        radians = math.radians(placement.angle_deg)
        along = (math.cos(radians), math.sin(radians))
        across = (-along[1], along[0])
        best_shared = 0.0
        for first, tangent, inward, length in walls:
            if abs(_cross(along, tangent)) > 1e-7 and abs(_dot(along, tangent)) > 1e-7:
                continue
            half_normal = (
                item.length * abs(_dot(along, inward))
                + item.width * abs(_dot(across, inward))
            ) / 2
            signed_gap = (
                _dot(inward, placement.center) - half_normal - _dot(inward, first)
            )
            if abs(signed_gap) > tolerance:
                continue
            half_tangent = (
                item.length * abs(_dot(along, tangent))
                + item.width * abs(_dot(across, tangent))
            ) / 2
            center_t = _dot(tangent, placement.center)
            start_t = _dot(tangent, first)
            shared = max(
                0.0,
                min(center_t + half_tangent, start_t + length)
                - max(center_t - half_tangent, start_t),
            )
            best_shared = max(best_shared, shared)
        # 任意超过数值误差的正长度重合都算贴墙；角点相碰不算。
        overlap_tolerance = _linear_tolerance(
            min(item.length, item.width), *placement.center
        )
        if best_shared > overlap_tolerance:
            count += 1
            shared_total += best_shared
    return count, shared_total


def _snap_to_walls(problem, placements):
    """尝试将合法布局平移到墙边，仅保留通过几何校验且贴墙更好的移动。

    MILP 的等式容差可能留下微小缝隙。移动承载设备时，架上的所有后代
    一起平移；旋转角度保持不变。
    """

    if not placements:
        return placements, 0
    walls = _wall_geometry(problem)
    if not walls:
        return placements, 0
    room_extent = max(
        problem.boundary.bounds[2] - problem.boundary.bounds[0],
        problem.boundary.bounds[3] - problem.boundary.bounds[1],
    )
    tolerance = max(1e-7, room_extent * 1e-8)
    current = dict(placements)
    initial_count, _ = _wall_score(problem, current, walls, tolerance)
    current_score = _wall_score(problem, current, walls, tolerance)
    by_name = {item.name: item for item in problem.items}

    def moves_with(name, ancestor):
        while name != ancestor:
            item = by_name[name]
            if item.supported_by is None:
                return False
            name = item.supported_by
        return True

    for _ in range(2):
        changed = False
        for item in problem.items:
            placement = current[item.name]
            radians = math.radians(placement.angle_deg)
            along = (math.cos(radians), math.sin(radians))
            across = (-along[1], along[0])
            moving = [name for name in current if moves_with(name, item.name)]
            best = None
            best_score = current_score
            for first, tangent, inward, length in walls:
                if (
                    abs(_cross(along, tangent)) > 1e-7
                    and abs(_dot(along, tangent)) > 1e-7
                ):
                    continue
                half_normal = (
                    item.length * abs(_dot(along, inward))
                    + item.width * abs(_dot(across, inward))
                ) / 2
                half_tangent = (
                    item.length * abs(_dot(along, tangent))
                    + item.width * abs(_dot(across, tangent))
                ) / 2
                target_n = _dot(inward, first) + half_normal
                current_n = _dot(inward, placement.center)
                current_t = _dot(tangent, placement.center)
                start_t = _dot(tangent, first)
                trial_tangents = (
                    current_t,
                    start_t + length / 2,
                    start_t + half_tangent,
                    start_t + length - half_tangent,
                )
                for target_t in trial_tangents:
                    delta_n = target_n - current_n
                    delta_t = target_t - current_t
                    if math.hypot(delta_n, delta_t) <= tolerance / 2:
                        continue
                    translation = (
                        delta_n * inward[0] + delta_t * tangent[0],
                        delta_n * inward[1] + delta_t * tangent[1],
                    )
                    trial = dict(current)
                    for name in moving:
                        old = current[name]
                        trial[name] = Placement(
                            (
                                old.center[0] + translation[0],
                                old.center[1] + translation[1],
                            ),
                            old.angle_deg,
                            old.z,
                        )
                    # 每次吸附都重验完整约束，不能为了贴墙破坏门区或支撑关系。
                    if validate_solution(problem, trial):
                        continue
                    score = _wall_score(problem, trial, walls, tolerance)
                    if score[0] > best_score[0] or (
                        score[0] == best_score[0] and score[1] > best_score[1] + 1e-5
                    ):
                        best = trial
                        best_score = score
            if best is not None:
                current = best
                current_score = best_score
                changed = True
        if not changed:
            break
    return current, current_score[0] - initial_count


def _solve_once(
    problem: Problem,
    time_limit: float,
    clearance: float,
    maximize_walls: bool = False,
    angle_provider=None,
) -> Solution:
    """按指定的数值间隙建立并求解一次 MILP。"""

    if angle_provider is None:
        angle_provider = legal_angles
    roots, members, support_error = _support_layers(problem)
    if support_error:
        return Solution("infeasible", {}, support_error)
    if not roots:
        return Solution("feasible", {}, "There are no items to place")

    min_x, min_y, max_x, max_y = problem.boundary.bounds
    # 统一缩放到约 0～1 的坐标，降低毫米级输入和布尔变量混用时的病态性。
    scale = max(max_x - min_x, max_y - min_y)
    if scale <= 0:
        return Solution("unknown", {}, "The room has no positive extent")
    origin = (min_x, min_y)

    def local(point):
        return ((point[0] - min_x) / scale, (point[1] - min_y) / scale)

    outline = [local(point) for point in list(problem.boundary.exterior.coords)[:-1]]
    if sum(_cross(a, b) for a, b in zip(outline, outline[1:] + outline[:1])) < 0:
        outline.reverse()
    exact_mode = angle_provider is not legal_angles
    axis_tolerance = 1e-13 if exact_mode else 1e-9
    axes = _unit_axes(outline, axis_tolerance)
    if not axes:
        return Solution("unknown", {}, "The room has no usable wall directions")
    normalized_bounds = (0.0, 0.0, (max_x - min_x) / scale, (max_y - min_y) / scale)

    model = _Model()
    member_centers = {}
    choices = {}
    member_choices = {}
    state_angles = {}
    bodies = {}
    member_bodies = {}
    heights = {}
    for root in roots:
        # 中心是连续变量；整组朝向由恰好一个二元选项确定。
        x_index = model.variable(0.0, normalized_bounds[2])
        y_index = model.variable(0.0, normalized_bounds[3])
        member_centers[root.name] = (x_index, y_index)
        heights[root.name] = 0.0
        states = _assembly_states(problem, root, members[root.name], angle_provider)
        if states is None:
            return Solution(
                "unknown", {}, "Too many support orientations for %s" % root.name
            )
        if not states:
            return Solution(
                "infeasible", {}, "No shared legal angle for %s" % root.name
            )
        options = []
        state_angles[root.name] = {}
        for state in states:
            selector = model.variable(binary=True)
            options.append((state[root.name], selector))
            state_angles[root.name][selector] = state
        choices[root.name] = options
        member_choices[root.name] = {
            member.name: [
                (state[member.name], selector)
                for _, selector in options
                for state in (state_angles[root.name][selector],)
            ]
            for member in members[root.name]
        }
        model.row(
            _Affine({selector: 1.0 for _, selector in options}), low=1.0, high=1.0
        )
        body = _rectangle_shape(x_index, y_index, options, root, scale)
        bodies[root.name] = body
        member_bodies[root.name] = body
        for axis, low, high in (
            ((1.0, 0.0), normalized_bounds[0], normalized_bounds[2]),
            ((0.0, 1.0), normalized_bounds[1], normalized_bounds[3]),
        ):
            model.row(body.projection(axis) + body.support(axis), high=high)
            model.row(
                body.projection(axis) - body.support((-axis[0], -axis[1])), low=low
            )

    if exact_mode:
        # 精确角度回退时，把矩形自身的法向也加入分离轴集合。
        # 墙角四舍五入后可能与原始墙方向略有差异，不能漏检该方向。
        exact_axes = list(axes)
        for root in roots:
            for options in member_choices[root.name].values():
                for angle, _ in options:
                    radians = math.radians(angle)
                    for axis in (
                        (math.cos(radians), math.sin(radians)),
                        (-math.sin(radians), math.cos(radians)),
                    ):
                        if axis[0] < -1e-10 or (abs(axis[0]) <= 1e-10 and axis[1] < 0):
                            axis = (-axis[0], -axis[1])
                        if not any(
                            abs(_cross(axis, old)) < axis_tolerance
                            for old in exact_axes
                        ):
                            exact_axes.append(axis)
        axes = tuple(exact_axes)

    # 架上设备有独立中心。每个角度选项激活承载面长、宽两轴上的四个界，
    # 使子设备整体落在父设备顶面内，而非强制两者中心重合。
    by_name = {item.name: item for item in problem.items}
    for root in roots:
        for child in sorted(
            members[root.name], key=lambda item: _support_depth(item, by_name)
        ):
            if child.name == root.name:
                continue
            parent = by_name[child.supported_by]
            heights[child.name] = heights[parent.name] + parent.height
            x_index = model.variable(0.0, normalized_bounds[2])
            y_index = model.variable(0.0, normalized_bounds[3])
            member_centers[child.name] = (x_index, y_index)
            child_body = _rectangle_shape(
                x_index, y_index, member_choices[root.name][child.name], child, scale
            )
            member_bodies[child.name] = child_body
            parent_body = member_bodies[parent.name]
            for _, selector in choices[root.name]:
                state = state_angles[root.name][selector]
                parent_angle = math.radians(state[parent.name])
                along = (math.cos(parent_angle), math.sin(parent_angle))
                across = (-along[1], along[0])
                child_angle = math.radians(state[child.name])
                child_along = (math.cos(child_angle), math.sin(child_angle))
                child_across = (-child_along[1], child_along[0])
                for axis, parent_extent in (
                    (along, parent.length),
                    (across, parent.width),
                ):
                    child_extent = child.length * abs(
                        _dot(axis, child_along)
                    ) + child.width * abs(_dot(axis, child_across))
                    allowance = (parent_extent - child_extent) / (2 * scale)
                    difference = child_body.projection(axis) - parent_body.projection(
                        axis
                    )
                    model.implies_at_most(selector, difference, high=allowance)
                    model.implies_at_least(selector, difference, low=-allowance)

        # 同一架上的多件设备可处于相同高度，需要彼此避让；
        # 架面恰好在设备底部，二者接触不构成体积碰撞。
        assembly = members[root.name]
        for index, first in enumerate(assembly):
            for second in assembly[index + 1 :]:
                overlap_height = min(
                    heights[first.name] + first.height,
                    heights[second.name] + second.height,
                ) - max(heights[first.name], heights[second.name])
                if overlap_height > 1e-9:
                    _separate(
                        model,
                        member_bodies[first.name],
                        member_bodies[second.name],
                        axes,
                    )

    # 凸边直接用半平面约束；凹角处则阻止墙段穿过设备，
    # 再用下面的三角形析取约束保证设备中心位于房间内。
    recessed = False
    walls = []
    for first, second in zip(outline, outline[1:] + outline[:1]):
        vector = (second[0] - first[0], second[1] - first[1])
        length = math.hypot(*vector)
        if length < 1e-12:
            continue
        tangent = (vector[0] / length, vector[1] / length)
        inward = (-tangent[1], tangent[0])
        supporting = all(
            _dot(inward, (point[0] - first[0], point[1] - first[1])) >= -1e-9
            for point in outline
        )
        walls.append((first, second, tangent, inward, length, supporting))
        wall_shape = _fixed_shape((first, second))
        for root in roots:
            body = bodies[root.name]
            if supporting:
                expression = -body.projection(inward) + body.support(
                    (-inward[0], -inward[1])
                )
                model.row(expression, high=-_dot(inward, first))
            else:
                recessed = True
                _separate(model, body, wall_shape, axes)

    if recessed:
        # 凹多边形可由三角形的并表示；至少选择一个容纳中心的三角形。
        try:
            triangles = _triangulate_simple_polygon(outline)
        except ValueError as error:
            return Solution("unknown", {}, str(error))
        for root in roots:
            body = bodies[root.name]
            alternatives = []
            for triangle in triangles:
                selector = model.variable(binary=True)
                alternatives.append(selector)
                for first, second in zip(triangle, triangle[1:] + triangle[:1]):
                    edge = (second[0] - first[0], second[1] - first[1])
                    expression = edge[0] * (body.center_y - first[1]) - edge[1] * (
                        body.center_x - first[0]
                    )
                    model.implies_at_least(selector, expression)
            model.row(_Affine({selector: 1.0 for selector in alternatives}), low=1.0)

    for position, first in enumerate(roots):
        for second in roots[position + 1 :]:
            _separate(model, bodies[first.name], bodies[second.name], axes)

    # 内开门的 N×N 方区作为面积障碍；外开门仍须避开门口线段。
    # 正间隙仅用于浮点误差修复，不擅自增加设计净空要求。
    if problem.door_keepout is not None:
        door_square = _fixed_shape(
            tuple(
                local(point)
                for point in list(problem.door_keepout.exterior.coords)[:-1]
            )
        )
        for root in roots:
            _separate(model, bodies[root.name], door_square, axes)
    door_segment = _fixed_shape(tuple(local(point) for point in problem.door.coords))
    for root in roots:
        _separate(model, bodies[root.name], door_segment, axes, clearance)

    # 冰箱不设虚构的固定开门深度；实际正面线段必须避开其他设备和墙。
    for root in roots:
        for member in members[root.name]:
            if member.front_side is None:
                continue
            x_index, y_index = member_centers[member.name]
            front = _front_shape(
                x_index, y_index, member_choices[root.name][member.name], member, scale
            )
            for other in roots:
                if other.name != root.name:
                    for other_member in members[other.name]:
                        overlap_height = min(
                            heights[member.name] + member.height,
                            heights[other_member.name] + other_member.height,
                        ) - max(heights[member.name], heights[other_member.name])
                        if overlap_height > 1e-9:
                            _separate(
                                model,
                                front,
                                member_bodies[other_member.name],
                                axes,
                                clearance,
                            )
            for first, second, _, _, _, _ in walls:
                _separate(model, front, _fixed_shape((first, second)), axes, clearance)
            for other in members[root.name]:
                if other.name == member.name:
                    continue
                if (
                    min(
                        heights[member.name] + member.height,
                        heights[other.name] + other.height,
                    )
                    - max(heights[member.name], heights[other.name])
                    > 1e-9
                ):
                    _separate(model, front, member_bodies[other.name], axes, clearance)

    # 贴墙计分只属于次级优化；可行性搜索不建立这些二元变量和约束。
    if maximize_walls:
        # 二元 witness 证明边与墙有正长度重合，角点相碰不计分。
        for root in roots:
            for member in members[root.name]:
                body = member_bodies[member.name]
                touch = model.variable(binary=True, cost=-1.0)
                witnesses = []
                for first, _, tangent, inward, length, _ in walls:
                    for angle, orientation in member_choices[root.name][member.name]:
                        direction = (
                            math.cos(math.radians(angle)),
                            math.sin(math.radians(angle)),
                        )
                        if (
                            abs(_cross(direction, tangent)) > 1e-8
                            and abs(_dot(direction, tangent)) > 1e-8
                        ):
                            continue
                        witness = model.variable(binary=True)
                        witnesses.append(witness)
                        model.row(
                            _Affine({witness: 1.0, orientation: -1.0}), high=0.0
                        )
                        distance = (
                            body.projection(inward)
                            - body.support(inward)
                            - _dot(inward, first)
                        )
                        model.implies_at_most(witness, distance)
                        model.implies_at_least(witness, distance)
                        start = (
                            body.projection(tangent)
                            + body.support(tangent)
                            - _dot(tangent, first)
                        )
                        end = (
                            body.projection(tangent)
                            - body.support(tangent)
                            - _dot(tangent, first)
                        )
                        model.implies_at_least(witness, start, low=1e-7)
                        model.implies_at_most(witness, end, high=length - 1e-7)
                if witnesses:
                    model.row(
                        _Affine({touch: 1.0, **{witness: -1.0 for witness in witnesses}}),
                        high=0.0,
                    )
                else:
                    model.row(_Affine({touch: 1.0}), high=0.0)
    try:
        result = model.optimize(time_limit)
    except (ValueError, RuntimeError) as error:
        return Solution("unknown", {}, "MILP failed: %s" % error)
    if result.x is not None:
        placements = _make_placements(
            problem,
            roots,
            members,
            choices,
            state_angles,
            member_centers,
            heights,
            result.x,
            origin,
            scale,
        )
        errors = validate_solution(problem, placements)
        if not errors:
            message = (
                "Wall-contact optimum proved"
                if maximize_walls and result.status == 0
                else "Valid layout found; wall-contact optimum not proved"
            )
            return Solution("feasible", placements, message)
        return Solution(
            "unknown",
            {},
            "Solver incumbent failed geometry check: " + "; ".join(errors[:3]),
        )
    if result.status == 2:
        return Solution("infeasible", {}, "MILP proved the layout impossible")
    return Solution(
        "unknown", {}, "MILP ended without a checked layout: " + str(result.message)
    )


def solve(problem: Problem, time_limit: float = 30.0) -> Solution:
    """放置全部设备，再改善门口与开门设备附近的空间。

    限时内找不到经校验的布局返回 unknown，不等于证明不可行。
    所有 feasible 结果离开本函数前都经过独立几何校验。
    """

    def finalize(solution):
        if solution.status == "feasible" and solution.placements:
            snapped, added = _snap_to_walls(problem, solution.placements)
            solution.placements = snapped
            if added:
                solution.message += "; wall snapping added %d contact(s)" % added
        return solution

    def refine(solution):
        if solution.status != "feasible" or not solution.placements:
            return solution
        remaining = time_limit - (time.monotonic() - started)
        if remaining <= 0.1:
            return solution
        if any(item.front_side for item in problem.items):
            improved = _find_accessible_layout(
                problem, solution.placements, min(15.0, remaining * 0.55)
            )
            remaining = time_limit - (time.monotonic() - started)
            if remaining > 0.1:
                improved = _refine_access_path(problem, improved, remaining)
        else:
            improved = _refine_layout(
                problem, solution.placements, min(10.0, remaining)
            )
        if improved != solution.placements:
            solution.placements = improved
            solution.message += "; entrance, opening and access path refined"
        return solution

    if time_limit < 0.01:
        return Solution("unknown", {}, "Time limit too short to start the solver")
    started = time.monotonic()
    # 有限的几何接触候选通常能迅速给出可行布局；完整几何校验通过后，
    # 即使后续 MILP 未在限时内结束，也保留这个已验证的候选。
    # 给几何构造留出主要时间。某些 IDE 的调试器会显著放慢 Python 层搜索；
    # 仍保留至少 20% 的时间给 MILP 回退。
    quick_budget = min(25.0, time_limit * 0.8)
    quick_placements = find_feasible_layout(problem, quick_budget)
    if quick_placements is not None:
        candidate = finalize(
            Solution("feasible", quick_placements, "Validated constructive layout")
        )
        walls = _wall_geometry(problem)
        room_extent = max(
            problem.boundary.bounds[2] - problem.boundary.bounds[0],
            problem.boundary.bounds[3] - problem.boundary.bounds[1],
        )
        tolerance = max(1e-7, room_extent * 1e-8)
        candidate_score = _wall_score(problem, candidate.placements, walls, tolerance)
        if candidate_score[0] == len(problem.items):
            return refine(candidate)
        remaining = time_limit - (time.monotonic() - started)
        if remaining > 0.1:
            improved = _solve_once(
                problem, remaining, 1e-5, maximize_walls=True
            )
            if improved.status == "feasible":
                improved = finalize(improved)
                if (
                    _wall_score(problem, improved.placements, walls, tolerance)
                    > candidate_score
                ):
                    return refine(improved)
        return refine(candidate)

    remaining = time_limit - (time.monotonic() - started)
    if remaining <= 0.01:
        return Solution("unknown", {}, "No validated layout within the time limit")
    initial_budget = min(8.0, remaining, max(0.01, time_limit * 0.4))
    first = _solve_once(problem, initial_budget, clearance=0.0)
    search_angles = legal_angles
    if first.status == "infeasible":
        # 快速角度集会合并近似平行的 CAD 墙边；仅凭它不能证明原问题无解。
        # 回退到未合并的精确方向后，才可报告真正的不可行。

        if all(
            legal_angles(problem, item) == exact_legal_angles(problem, item)
            for item in problem.items
        ):
            return first
        remaining = time_limit - (time.monotonic() - started)
        if remaining <= 0:
            return Solution(
                "unknown", {}, "Coarse angles were infeasible; exact angles not checked"
            )
        exact = _solve_once(
            problem,
            remaining,
            clearance=0.0,
            angle_provider=exact_legal_angles,
        )
        if exact.status == "infeasible":
            return exact
        if exact.status != "feasible":
            return Solution("unknown", {}, "Exact-angle feasibility was not resolved")
        first = exact
        search_angles = exact_legal_angles

    candidate = first if first.status == "feasible" else None
    repair_margin = 0.0
    if candidate is None:
        # MILP 容差可能把冰箱正面或门口的零距离接触误判为可行。
        # 正间隙只修复搜索；带间隙模型无解不代表原问题无解。
        remaining = time_limit - (time.monotonic() - started)
        if remaining > 0:
            retry_budget = min(8.0, remaining)
            repaired = _solve_once(
                problem,
                retry_budget,
                clearance=1e-5,
                angle_provider=search_angles,
            )
            if repaired.status == "feasible":
                candidate = repaired
                repair_margin = 1e-5
        if candidate is None:
            # 带间隙搜索只是数值修复，不能代替原题的零间隙可行性搜索。
            # 用剩余时间继续求原模型，避免两次短轮失败后提前返回。
            remaining = time_limit - (time.monotonic() - started)
            if remaining > 0.01:
                final_attempt = _solve_once(
                    problem,
                    remaining,
                    clearance=0.0,
                    angle_provider=search_angles,
                )
                if final_attempt.status == "feasible":
                    candidate = final_attempt
            if candidate is None:
                return Solution("unknown", {}, "No validated layout within the time limit")

    if repair_margin:
        candidate.message = (
            "Valid layout found after numerical contact repair; "
            "original wall-contact optimum not proved"
        )
    candidate = finalize(candidate)
    walls = _wall_geometry(problem)
    room_extent = max(
        problem.boundary.bounds[2] - problem.boundary.bounds[0],
        problem.boundary.bounds[3] - problem.boundary.bounds[1],
    )
    tolerance = max(1e-7, room_extent * 1e-8)
    candidate_score = _wall_score(problem, candidate.placements, walls, tolerance)

    remaining = time_limit - (time.monotonic() - started)
    if remaining > 0.1:
        # 微小间隙避免质量优化阶段反复选择正面或门口恰好相碰的候选解。
        # 它只引导贴墙搜索；不可行性证明仍以零间隙模型为准。
        improved = _solve_once(
            problem,
            remaining,
            1e-5,
            maximize_walls=True,
            angle_provider=search_angles,
        )
        if improved.status == "feasible":
            improved = finalize(improved)
            if (
                _wall_score(problem, improved.placements, walls, tolerance)
                > candidate_score
            ):
                improved.message = (
                    "Valid layout found with wall-contact search; "
                    "original wall-contact optimum not proved"
                )
                return refine(improved)
    return refine(candidate)

# ===== 命令行入口 =====

def export_dxf(problem, placements, path):
    """把已验证的二维布局写成简单的 AutoCAD R12 DXF。"""

    entities = [
        "0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009",
        "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES",
    ]

    def line(first, second, layer, color):
        entities.extend([
            "0", "LINE", "8", layer, "62", str(color),
            "10", str(first[0]), "20", str(first[1]), "30", "0",
            "11", str(second[0]), "21", str(second[1]), "31", "0",
        ])

    boundary = list(problem.boundary.exterior.coords)
    for first, second in zip(boundary, boundary[1:]):
        line(first, second, "ROOM", 4)
    door = list(problem.door.coords)
    line(door[0], door[1], "DOOR", 1)

    for item in problem.items:
        pose = placements[item.name]
        corners = list(rectangle_polygon(item, pose).exterior.coords)
        if item.name.startswith("fridge"):
            layer, color = "FRIDGE", 30
        elif item.name.startswith("iceMaker"):
            layer, color = "ICE_MAKER", 2
        elif item.name.startswith("overShelf"):
            layer, color = "OVER_SHELF", 6
        else:
            layer, color = "SHELF", 5
        for first, second in zip(corners, corners[1:]):
            line(first, second, layer, color)
        # R12 标签使用 ASCII；非 ASCII 名称只在图中文字中被替换。
        label = item.name.encode("ascii", "replace").decode("ascii")
        entities.extend([
            "0", "TEXT", "8", "LABELS", "62", "7",
            "10", str(pose.center[0]), "20", str(pose.center[1]), "30", "0",
            "40", "70", "1", label,
        ])
        front = front_segment(item, pose)
        if front is not None:
            first, second = front.coords
            line(first, second, "OPENING_FRONT", 1)

    entities.extend(["0", "ENDSEC", "0", "EOF"])
    path.write_text("\n".join(entities) + "\n", encoding="ascii")


def _resolve_json_path(value):
    """解析完整路径、脚本目录相对路径及迁移前的样例路径。"""
    raw = str(value).strip().strip("\ufeff")
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        raw = raw[1:-1]
    path = Path(raw).expanduser()
    if path.is_file():
        return path

    script_dir = Path(__file__).resolve().parent
    if not path.is_absolute():
        beside_script = script_dir / path
        if beside_script.is_file():
            return beside_script
    # 目录整理前用户可能保存了“项目根目录/example1.json”地址。
    if path.parent == script_dir or (not path.is_absolute() and len(path.parts) == 1):
        relocated = script_dir / "输入示例" / path.name
        if relocated.is_file():
            return relocated
    raise FileNotFoundError(
        "JSON file not found: %s (examples are in %s)"
        % (path, script_dir / "输入示例")
    )


def _read_json_file(path):
    """按 UTF-8 读取指定文件，兼容旧样例地址。"""
    with _resolve_json_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _decode_stdin_line(raw):
    """兼容 Windows 控制台 GBK 和 PowerShell 管道 UTF-8。"""
    if isinstance(raw, str):
        return [raw]
    choices = []
    for encoding in ("utf-8-sig", sys.stdin.encoding or "gbk", "gbk"):
        try:
            value = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        if value not in choices:
            choices.append(value)
    if not choices:
        raise ValueError("standard input is not valid UTF-8 or GBK text")
    return choices


def _read_stdin_input():
    """接受粘贴的文件路径或多行 JSON，并返回解析后的题目数据。"""
    print(
        "Paste problem JSON or a .json file path, then press Enter:",
        file=sys.stderr,
        flush=True,
    )
    lines = []
    stream = getattr(sys.stdin, "buffer", sys.stdin)
    for raw_line in stream:
        decoded = _decode_stdin_line(raw_line)
        line = decoded[0]
        if not lines:
            value = line.strip()
            if not value:
                continue
            if not value.startswith(("{", "[")):
                # 同一组原始字节可能有两种解码；选择确实存在的路径。
                for candidate in decoded:
                    try:
                        return _read_json_file(candidate.strip())
                    except FileNotFoundError:
                        continue
                return _read_json_file(value)
        lines.append(line)
        try:
            return json.loads("".join(lines))
        except json.JSONDecodeError:
            # JSON 可能尚未输入完毕；若到达 EOF，下面会报告实际格式错误。
            continue
    if not lines:
        raise ValueError("standard input is empty")
    return json.loads("".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Place rectangular items in a room.")
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Input JSON file (omit to enter JSON or a path on standard input)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=30.0,
        help="Maximum solve time in seconds (default: 30)",
    )
    parser.add_argument(
        "--output", type=Path, help="Write the result JSON to this file"
    )
    parser.add_argument(
        "--cad-output", type=Path,
        help="Write a feasible layout as a 2D DXF file",
    )
    args = parser.parse_args(argv)

    try:
        if not math.isfinite(args.time_limit) or args.time_limit <= 0:
            raise ValueError("--time-limit must be a positive finite number")
        if args.input is None:
            data = _read_stdin_input()
        else:
            data = _read_json_file(args.input)
        problem = load_problem(data)
        if args.input is None:
            print(
                "Input loaded; solving (time limit: %g seconds)..." % args.time_limit,
                file=sys.stderr,
                flush=True,
            )
        result = solve(problem, time_limit=args.time_limit)
        if result.status == "feasible":
            # 求解器可能有浮点误差；对外公布前用独立几何规则再校验一次。
            errors = validate_solution(problem, result.placements)
            if errors:
                raise RuntimeError(
                    "solver produced an invalid layout: " + "; ".join(errors)
                )
        if result.status == "unknown":
            # 限时未判定不等于不可行，不能输出 feasible=false。
            print(
                result.message or "solver could not decide feasibility", file=sys.stderr
            )
            return 3
        if result.status not in ("feasible", "infeasible"):
            raise RuntimeError("unexpected solver status: " + result.status)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("invalid input: " + str(exc), file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print("internal error: " + str(exc), file=sys.stderr)
        return 4

    # 高度只用于内部承载和碰撞判断；公开结果严格采用题目的二维格式。
    payload = {
        "feasible": result.status == "feasible",
        "placements": {
            name: {
                "center": [float(pose.center[0]), float(pose.center[1])],
                "angleDeg": float(pose.angle_deg),
            }
            for name, pose in result.placements.items()
        },
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        if args.cad_output and result.status == "feasible":
            export_dxf(problem, result.placements, args.cad_output)
        if args.output:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
    except OSError as exc:
        print("cannot write output: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
