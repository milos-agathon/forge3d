"""Seeded polygon corpora and deterministic failure shrinking for EUCLIDEA."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import dataclass

Geometry = dict[str, object]
Pair = tuple[Geometry, Geometry]


def polygon(points: list[tuple[float, float]]) -> Geometry:
    ring = [[x, y] for x, y in points]
    if ring[0] != ring[-1]:
        ring.append(ring[0].copy())
    return {"type": "Polygon", "coordinates": [ring]}


def rectangle(x: float, y: float, width: float, height: float) -> Geometry:
    return polygon(
        [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
    )


@dataclass(frozen=True)
class SeededPolygonCorpus:
    seed: int = 0x4555434C49444541

    def pairs(self, count: int) -> Iterator[Pair]:
        rng = random.Random(self.seed)
        for index in range(count):
            x = rng.randrange(-1024, 1024) / 16
            y = rng.randrange(-1024, 1024) / 16
            width = 1 + rng.randrange(1, 64) / 16
            height = 1 + rng.randrange(1, 64) / 16
            left = rectangle(x, y, width, height)
            mode = index % 6
            if mode == 0:  # random simple polygon: polar order cannot cross itself
                cx, cy = x + width / 2, y + height / 2
                points = []
                for vertex in range(7):
                    angle = 2 * math.pi * vertex / 7
                    radius = width * (0.35 + 0.25 * rng.random())
                    points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
                right = polygon(points)
            elif mode == 1:  # exactly shared edge
                right = rectangle(x + width, y, width, height)
            elif mode == 2:  # 0--4 ULP near-coincident edge
                edge = x + width
                for _ in range(index % 5):
                    edge = math.nextafter(edge, math.inf)
                right = rectangle(edge, y + height / 4, width, height)
            elif mode == 3:  # collinear boundary chain
                right = polygon(
                    [
                        (x + width / 2, y - height),
                        (x + width * 1.5, y - height),
                        (x + width * 1.5, y),
                        (x + width, y),
                        (x + width / 2, y),
                    ]
                )
            elif mode == 4:  # valid sliver
                right = polygon(
                    [
                        (x - width, y + height / 2),
                        (x + width * 2, y + height / 2),
                        (x + width * 2, math.nextafter(y + height / 2, math.inf)),
                    ]
                )
            else:  # simple concave star-shaped polygon
                right = polygon(
                    [
                        (x - width, y - height),
                        (x + width * 2, y - height),
                        (x + width * 2, y + height * 2),
                        (x + width / 2, y + height * 0.75),
                        (x - width, y + height * 2),
                    ]
                )
            yield left, right

    def invalid_stars(self, count: int) -> Iterator[Geometry]:
        """Yield replayable self-intersecting stars for validity rejection."""

        rng = random.Random(self.seed ^ 0xBAD5_7A12)
        for _ in range(count):
            cx = rng.randrange(-128, 128) / 8
            cy = rng.randrange(-128, 128) / 8
            radius = 1 + rng.randrange(1, 32) / 8
            outer = [
                (
                    cx + radius * math.cos(2 * math.pi * index / 5),
                    cy + radius * math.sin(2 * math.pi * index / 5),
                )
                for index in range(5)
            ]
            yield polygon([outer[index] for index in (0, 2, 4, 1, 3)])


def shrink_failure(pair: Pair, still_fails: Callable[[Pair], bool]) -> Pair:
    """Deterministically remove vertices while preserving a reported failure."""

    current = deepcopy(pair)
    changed = True
    while changed:
        changed = False
        for side in (0, 1):
            ring = current[side]["coordinates"][0]  # type: ignore[index]
            if len(ring) <= 4:
                continue
            for index in range(1, len(ring) - 1):
                candidate = deepcopy(current)
                candidate_ring = candidate[side]["coordinates"][0]  # type: ignore[index]
                candidate_ring.pop(index)
                candidate_ring[-1] = candidate_ring[0].copy()
                if still_fails(candidate):
                    current = candidate
                    changed = True
                    break
            if changed:
                break
    return current
