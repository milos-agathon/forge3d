"""Geometric predicates for C07's registered heightfield and solid envelopes."""

from functools import cached_property
from itertools import product
import math

import numpy as np


_GRID = 512
_CORNERS = np.asarray(tuple(product((0, 1), repeat=3)))
_BOX_FACES = np.asarray([
    (0, 1, 3), (0, 3, 2), (4, 5, 7), (4, 7, 6),
    (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
    (0, 2, 6), (0, 6, 4), (1, 3, 7), (1, 7, 5),
])


def _clip(vertices, normal, offset, error=0.):
    """Clip an ordered convex polygon (or segment) to normal · x >= offset."""
    if not len(vertices):
        return vertices
    sides = vertices @ normal - offset
    sides[np.abs(sides) <= error] = 0
    result = []
    for a, b, da, db in zip(np.roll(vertices, 1, axis=0), vertices,
                             np.roll(sides, 1), sides):
        if (da < 0 < db) or (db < 0 < da):
            result.append(a + (b - a) * (da / (da - db)))
        if db >= 0:
            result.append(b)
    return np.asarray(result).reshape(-1, vertices.shape[1])


def _clip_polygon(vertices, planes, error=0.):
    for normal, offset in planes:
        vertices = _clip(vertices, normal, offset, error)
        if not len(vertices):
            break
    return vertices


def _rectangle_planes(lo, hi):
    return [(axis, lower) for axis, lower in zip(np.eye(2), lo)] + [
        (-axis, -upper) for axis, upper in zip(np.eye(2), hi)
    ]


def intersects_rectangle(vertices, rect):
    return bool(len(_clip_polygon(vertices, _rectangle_planes(rect[:2], rect[2:]))))


def mesh_height(heights, xy):
    col, row = np.clip(xy, 0, _GRID - 1).T
    i = np.minimum(col.astype(int), _GRID - 2)
    j = np.minimum(row.astype(int), _GRID - 2)
    u, v = col - i, row - j
    sw, se = heights[j, i], heights[j, i + 1]
    nw, ne = heights[j + 1, i], heights[j + 1, i + 1]
    return np.where(
        u + v <= 1, (1 - u - v) * sw + u * se + v * nw,
        (u + v - 1) * ne + (1 - v) * se + (1 - u) * nw,
    )


def segment_cuts(a, b):
    """Parameters on every grid triangle edge a segment can cross."""
    cuts = [np.array([0., 1.])]
    for start, end in zip((*a[:2], sum(a[:2])), (*b[:2], sum(b[:2]))):
        lo, hi = sorted((start, end))
        lines = np.arange(math.floor(lo) + 1, math.ceil(hi))
        if lines.size:
            cuts.append((lines - start) / (end - start))
    return np.unique(np.concatenate(cuts))


class StoryMesh:
    def __init__(self, heights, span):
        self.heights = heights
        self.span = span
        self.step = span / (_GRID - 1)

    def grid(self, xy):
        return (xy / self.span + 0.5) * (_GRID - 1)

    def world(self, xy):
        return (xy / (_GRID - 1) - 0.5) * self.span

    def lift(self, xy):
        return np.column_stack((self.world(xy), mesh_height(self.heights, xy)))

    def triangles(self, lo, hi):
        # Include cells on either side when a query lies exactly on an edge.
        i, j = np.maximum(np.ceil(lo).astype(int) - 1, 0)
        k, l = np.minimum(np.floor(hi).astype(int), _GRID - 2)
        x, y = np.meshgrid(np.arange(i, k + 1), np.arange(j, l + 1))
        return _cell_faces(np.column_stack((x.ravel(), y.ravel())))

    def drape(self, line, faces):
        if len(line):
            for a, b in zip(line[:-1], line[1:]):
                cuts = segment_cuts(a, b)
                points = self.lift(a + cuts[:, None] * (b - a))
                yield from (np.stack((p, q)) for p, q in zip(points[:-1], points[1:]))
        for face in faces:
            for cell in self.triangles(face.min(axis=0), face.max(axis=0)):
                # Both native triangles are wound counterclockwise.
                edges = np.roll(cell, -1, axis=0) - cell
                normals = np.column_stack((-edges[:, 1], edges[:, 0]))
                polygon = _clip_polygon(face, zip(normals, np.sum(cell * normals, axis=1)))
                if len(polygon) >= 3:
                    points = self.lift(polygon)
                    for i in range(1, len(points) - 1):
                        a, b, c = points[[0, i, i + 1]]
                        if np.any(np.cross(b - a, c - a)):
                            yield np.stack((a, b, c))

    def max_height(self, lo, hi):
        """Exact maximum on the clipped horizontal box of a clearance bound."""
        lo, hi = self.grid(lo), self.grid(hi)
        i, j = np.ceil(lo).astype(int)
        k, l = np.floor(hi).astype(int)
        maximum = self.heights[j:l + 1, i:k + 1].max(initial=-np.inf)
        corners = np.asarray([(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])])
        for a, b in zip(corners, np.roll(corners, -1, axis=0)):
            xy = a + segment_cuts(a, b)[:, None] * (b - a)
            maximum = max(maximum, mesh_height(self.heights, xy).max())
        return float(maximum)

    @cached_property
    def tile_heights(self):
        # Dyadic hierarchy of conservative min/max bounds over terrain cells.
        # Cell bounds include all four vertices, including shared tile edges.
        h = self.heights
        corners = (h[:-1, :-1], h[1:, :-1], h[:-1, 1:], h[1:, 1:])
        lo, hi = np.minimum.reduce(corners), np.maximum.reduce(corners)
        levels = [(lo, hi)]
        while lo.shape[0] > 1:
            indices = np.arange(0, lo.shape[0], 2)
            lo = np.minimum.reduceat(np.minimum.reduceat(lo, indices, axis=0), indices, axis=1)
            hi = np.maximum.reduceat(np.maximum.reduceat(hi, indices, axis=0), indices, axis=1)
            levels.append((lo, hi))
        return levels

    def candidates(self, eye, target, planes):
        xy = np.vstack((eye, target))[:, :2]
        xy_min, xy_max = xy.min(axis=0), xy.max(axis=0)
        tiles = np.array([[0, 0]])
        for level in reversed(range(len(self.tile_heights))):
            low, high = self.tile_heights[level]
            tiles = tiles[np.all(tiles < low.shape[0], axis=1)]
            i, j = tiles.T
            span = 2**level
            lo = np.column_stack((self.world(tiles * span), low[j, i]))
            hi = np.column_stack((self.world(np.minimum((tiles + 1) * span, _GRID - 1)), high[j, i]))
            error = _plane_error(lo - eye, hi - eye)
            keep = np.all((lo[:, :2] <= xy_max + error) & (hi[:, :2] >= xy_min - error), axis=1)
            for normal, offset in planes:
                furthest = np.where(normal >= 0, hi, lo) - eye
                keep &= furthest @ normal >= offset - error
            tiles = tiles[keep]
            if not len(tiles):
                return np.empty((0, 3, 3))
            if level:
                tiles = (2 * tiles[:, None] + _TILE_CHILDREN).reshape(-1, 2)
        grid_triangles = _cell_faces(tiles)
        return self.lift(grid_triangles.reshape(-1, 2)).reshape(-1, 3, 3)


_TILE_CHILDREN = np.asarray(tuple(product((0, 1), repeat=2)))


def _cell_faces(sw):
    se, nw, ne = sw + (1, 0), sw + (0, 1), sw + (1, 1)
    return np.concatenate((np.stack((sw, se, nw), axis=1),
                           np.stack((ne, nw, se), axis=1)))


def box_triangles(lo, hi, eye, forward, near):
    corners = np.where(_CORNERS, hi, lo)
    faces = corners[_BOX_FACES]
    signed = (corners - eye) @ forward - near
    if not (signed.min() < 0 < signed.max()):
        return faces
    # An opaque solid can span the near plane with its entry face clipped.
    # Add the slice there, so an entire sight cone inside the solid is blocked.
    edges = [(i, j) for i, point in enumerate(_CORNERS) for j in range(i + 1, len(_CORNERS))
             if np.count_nonzero(point != _CORNERS[j]) == 1]
    cap = list(corners[signed == 0])
    for i, j in edges:
        if (signed[i] < 0 < signed[j]) or (signed[j] < 0 < signed[i]):
            cap.append(corners[i] + (corners[j] - corners[i]) * (signed[i] / (signed[i] - signed[j])))
    center = np.mean(cap, axis=0)
    right = _unit(np.cross(forward, np.eye(3)[np.argmin(np.abs(forward))]))
    up = np.cross(forward, right)
    order = np.argsort(np.arctan2((cap - center) @ up, (cap - center) @ right))
    cap = np.asarray(cap)[order]
    return np.concatenate((faces, np.stack([cap[[0, i, i + 1]] for i in range(1, len(cap) - 1)])))


def obstacle_at_eye(eye, obstacles, minimum):
    for obstacle, lo, hi in obstacles:
        if (np.all(lo[:2] <= eye[:2]) and np.all(eye[:2] <= hi[:2])
                and lo[2] <= eye[2] <= hi[2] + minimum):
            return obstacle
    return None


def point_blocker(eye, target, obstacles, near_fraction):
    for obstacle, lo, hi in obstacles:
        delta = target - eye
        parallel = delta == 0
        if np.any(parallel & ((eye < lo) | (eye > hi))):
            continue
        a, b = (lo[~parallel] - eye[~parallel]) / delta[~parallel], (hi[~parallel] - eye[~parallel]) / delta[~parallel]
        start = max(near_fraction, np.minimum(a, b).max(initial=-np.inf))
        end = min(1, np.maximum(a, b).min(initial=np.inf))
        if start <= end and start < 1:
            return obstacle
    return None


def _unit(vector):
    length = np.linalg.norm(vector)
    return vector / length if length else vector


def _occlusion_planes(target):
    """Convex sight region in eye-relative coordinates, before the feature."""
    if len(target) == 2:
        a, b = target
        line = _unit(b - a)
        side = _unit(np.cross(a, b - a))
        front = _unit(a - line * (a @ line))
        if not np.any(side):
            return None
        planes = [(side, 0.), (-side, 0.)]
        for normal in (np.cross(side, a), np.cross(b, side)):
            planes.append((_unit(normal), 0.))
    else:
        a, b, c = target
        front = _unit(np.cross(b - a, c - a))
        if front @ a < 0:
            front = -front
        if not np.any(front) or front @ a == 0:
            return None
        planes = []
        for v, w, inside in ((a, b, c), (b, c, a), (c, a, b)):
            normal = _unit(np.cross(v, w - v))
            planes.append((normal if normal @ inside >= 0 else -normal, 0.))
    return planes, front, float(front @ target[0])


def _plane_error(*vectors):
    # Conservative double-precision allowance for local dot/cross and clipping
    # arithmetic. It does not depend on viewport resolution.
    return 32 * np.spacing(max(1., *(float(np.abs(v).max(initial=0.)) for v in vectors)))


def _triangles_block(triangles, eye, planes, front, depth):
    if not len(triangles):
        return False
    relative = triangles - eye
    error = _plane_error(relative, np.asarray(depth))
    candidates = (relative @ front).min(axis=1) < depth - error
    for normal, offset in planes:
        candidates &= (relative @ normal).max(axis=1) >= offset - error
    for triangle in relative[candidates]:
        overlap = _clip_polygon(triangle, planes, error)
        if len(overlap) and (overlap @ front).min() < depth - error:
            return True
    return False


def whole_blocker(mesh, eye, target, forward, near, obstacle_faces, obstacles):
    relative = target - eye
    region = _occlusion_planes(relative)
    if region is None:
        # A polygon viewed edge-on projects onto its edges. A collinear
        # segment projects to a single ray, checked up to its farther end.
        if len(target) == 3:
            for edge in ((0, 1), (1, 2), (2, 0)):
                if np.any(target[edge[0]] != target[edge[1]]):
                    blocked = whole_blocker(mesh, eye, target[list(edge)], forward,
                                            near, obstacle_faces, obstacles)
                    if blocked:
                        return blocked
            return None
        farthest = target[np.argmax(relative @ forward)]
        depth = (farthest - eye) @ forward
        blocker = point_blocker(eye, farthest, obstacles, near / depth)
        if blocker:
            return blocker
        start = eye + (farthest - eye) * (near / depth)
        a = np.array([*mesh.grid(start[:2]), start[2]])
        b = np.array([*mesh.grid(farthest[:2]), farthest[2]])
        visible_segment = a + segment_cuts(a, b)[:-1, None] * (b - a)
        if len(visible_segment) and np.any(visible_segment[:, 2] <= mesh_height(mesh.heights, visible_segment[:, :2])):
            return "terrain"
        return None
    planes, front, depth = region
    planes.append((forward, near))
    for obstacle, triangles in obstacle_faces:
        if _triangles_block(triangles, eye, planes, front, depth):
            return obstacle
    terrain = mesh.candidates(eye, target, [*planes, (-front, -depth)])
    if _triangles_block(terrain, eye, planes, front, depth):
        return "terrain"
    return None
