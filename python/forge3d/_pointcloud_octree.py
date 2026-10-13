"""Shared spatial rules for point-cloud hierarchy adapters.

File parsing, hierarchy-page loading, counts and spacing belong to the adapters.
This module only subdivides bounds and enumerates present children in octant order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Collection, List, Optional

if TYPE_CHECKING:
    from .pointcloud import OctreeBounds, OctreeKey, OctreeNode


def bounds_for_key(root_bounds: Optional[OctreeBounds], key: OctreeKey) -> OctreeBounds:
    """Subdivide each axis using the key's bits, from root to leaf.

    Keep midpoint arithmetic (including its rounding) and the existing missing-
    bounds fallback shared by COPC and EPT.
    """
    # Spatial types retain their public module and pickle identity in pointcloud.
    from .pointcloud import OctreeBounds

    if root_bounds is None:
        return OctreeBounds((0, 0, 0), (1, 1, 1))

    min_pt = list(root_bounds.min)
    max_pt = list(root_bounds.max)
    coordinates = (key.x, key.y, key.z)

    for depth in range(key.depth):
        shift = key.depth - depth - 1
        mid = [(min_pt[axis] + max_pt[axis]) / 2 for axis in range(3)]
        for axis, coordinate in enumerate(coordinates):
            if (coordinate >> shift) & 1:
                min_pt[axis] = mid[axis]
            else:
                max_pt[axis] = mid[axis]

    return OctreeBounds(tuple(min_pt), tuple(max_pt))


def child_nodes(
    key: OctreeKey,
    present_keys: Collection[str],
    make_node: Callable[[OctreeKey], OctreeNode],
) -> List[OctreeNode]:
    """Enumerate immediate children, independent of hierarchy storage order.

    The adapter constructs node metadata so file-format spacing and point counts
    stay local to it. Enumeration never parses or loads a hierarchy page.
    """
    children = []
    for octant in range(8):
        child_key = key.child(octant)
        if str(child_key) in present_keys:
            children.append(make_node(child_key))
    return children
