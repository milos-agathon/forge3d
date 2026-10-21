"""Exact wall-colour rules and the Noord data invariant; no GPU rendering."""
import numpy as np
import pytest

import render


def colours(labels, dtype=np.float32):
    """Distinct RGB values make each hand-written label grid unambiguous."""
    labels = np.asarray(labels, dtype=dtype)
    return np.stack((labels, labels * 2, labels * 3), axis=-1)


@pytest.mark.parametrize("status_value", [1, 2, 3])
def test_ground_including_diagonal_takes_building_colour(status_value):
    height = np.zeros((3, 3), np.float32)
    status = np.zeros((3, 3), np.uint8)
    height[1, 1], status[1, 1] = 30, status_value
    rgb = colours([[0, 0, 0], [0, 7, 0], [0, 0, 0]])
    expected = colours([[7, 7, 7], [7, 7, 7], [7, 7, 7]])
    np.testing.assert_array_equal(render.wall_colours(rgb, height, status), expected)


def test_lower_building_edges_change_but_interior_stays():
    height = np.full((5, 5), 12, np.float32)
    height[:, 0] = 30
    status = np.full((5, 5), 2, np.uint8)
    rgb = colours([[9, 2, 2, 2, 2]] * 5)
    expected = colours([[9, 9, 2, 2, 2]] * 5)
    np.testing.assert_array_equal(render.wall_colours(rgb, height, status), expected)


def test_touching_equal_height_buildings_are_unchanged():
    height = np.full((2, 2), 30, np.float32)
    status = np.array([[1, 2], [3, 2]], np.uint8)
    rgb = colours([[1, 2], [1, 2]])
    np.testing.assert_array_equal(render.wall_colours(rgb, height, status), rgb)


def test_water_next_to_ground_is_unchanged():
    height = np.array([[0, 0, 0], [0, -1.5, 0], [0, 0, 0]], np.float32)
    status = np.zeros((3, 3), np.uint8)
    rgb = colours([[1, 1, 1], [1, 2, 1], [1, 1, 1]])
    np.testing.assert_array_equal(render.wall_colours(rgb, height, status), rgb)


def test_tallest_neighbour_wins_using_original_colours():
    height = np.array([[12, 0, 30]], np.float32)
    status = np.array([[1, 0, 2]], np.uint8)
    rgb = colours([[2, 0, 9]])
    np.testing.assert_array_equal(
        render.wall_colours(rgb, height, status), colours([[2, 9, 9]])
    )
    # The middle building takes the tall colour, but must still donate its own
    # original colour to its lower neighbour in this same call.
    height = np.array([[0, 12, 30]], np.float32)
    status = np.array([[0, 1, 2]], np.uint8)
    rgb = colours([[0, 2, 9]])
    np.testing.assert_array_equal(
        render.wall_colours(rgb, height, status), colours([[2, 9, 9]])
    )


def test_equal_height_tie_uses_fixed_neighbour_order():
    height = np.full((3, 3), 30, np.float32)
    height[1, 1] = 0
    status = np.ones((3, 3), np.uint8)
    status[1, 1] = 0
    labels = np.array([[1, 2, 3], [4, 0, 5], [6, 7, 8]])
    # Remove candidates in order, proving every successive neighbour's priority.
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                   (0, 1), (1, -1), (1, 0), (1, 1)):
        expected = labels.copy()
        expected[1, 1] = labels[1 + dy, 1 + dx]
        np.testing.assert_array_equal(
            render.wall_colours(colours(labels), height, status), colours(expected)
        )
        status[1 + dy, 1 + dx] = 0


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_border_building_does_not_wrap(turns):
    height = np.rot90(np.array([[30, 0, 0], [0, 0, 0], [0, 0, 0]], np.float32), turns)
    status = (height > 0).astype(np.uint8)
    rgb = colours(np.rot90([[9, 0, 0], [0, 0, 0], [0, 0, 0]], turns))
    expected = colours(np.rot90([[9, 9, 0], [9, 9, 0], [0, 0, 0]], turns))
    np.testing.assert_array_equal(render.wall_colours(rgb, height, status), expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
def test_inputs_unchanged_and_output_is_new_with_same_shape_and_dtype(dtype):
    height = np.array([[30, 0], [0, 0]], np.float32)
    status = np.array([[2, 0], [0, 0]], np.uint8)
    rgb = colours([[9, 0], [0, 0]], dtype)
    originals = [a.copy() for a in (rgb, height, status)]
    for array in (rgb, height, status):
        array.flags.writeable = False
    out = render.wall_colours(rgb, height, status)
    assert out.shape == rgb.shape
    assert out.dtype == rgb.dtype
    assert not np.shares_memory(out, rgb)
    np.testing.assert_array_equal(out, colours([[9, 9], [9, 9]], dtype))
    for array, original in zip((rgb, height, status), originals):
        np.testing.assert_array_equal(array, original)


def test_noord_2026_taller_building_colour_invariant():
    height, rgb, status, _ = render.layers(2026)
    out = render.wall_colours(rgb, height, status)
    rows, cols = height.shape
    # Check one real row at a time, without another full RGB grid or a stack
    # of eight neighbours. This independently checks every real-data texel.
    xs = np.arange(cols)
    for y in range(rows):
        has_taller = np.zeros(cols, bool)
        matches_taller = np.zeros(cols, bool)
        for dy in (-1, 0, 1):
            qy = y + dy
            if not 0 <= qy < rows:
                continue
            for dx in (-1, 0, 1):
                if dy == dx == 0:
                    continue
                valid = (xs + dx >= 0) & (xs + dx < cols)
                px = xs[valid]
                qx = px + dx
                taller = (status[qy, qx] > 0) & (height[qy, qx] > height[y, px])
                has_taller[px] |= taller
                matches_taller[px] |= taller & np.all(out[y, px] == rgb[qy, qx], axis=-1)
        changed = np.any(out[y] != rgb[y], axis=-1)
        assert not np.any(changed & ~has_taller), f"Changed without taller building: row {y}"
        assert np.all(matches_taller[changed]), f"Changed to a non-neighbour colour: row {y}"
