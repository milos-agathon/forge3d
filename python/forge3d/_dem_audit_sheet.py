"""Cartographic inspection sheet. JSON geometries remain at source resolution."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def sheet(z, transform, findings, source, grid_shape=None):
    h, w = z.shape
    grid_h, grid_w = grid_shape or (h, w)
    # Preserve the grid aspect ratio inside each of the six printable panels.
    scale = min(420 / w, 290 / h)
    size = max(1, round(w * scale)), max(1, round(h * scale))
    sx, sy = size[0] / w, size[1] / h
    finite = np.isfinite(z)
    if min(z.shape) > 1:
        dy, dx = np.gradient(z)
        gx = (transform.e * dx - transform.d * dy) / transform.determinant
        gy = (transform.a * dy - transform.b * dx) / transform.determinant
        light = (-gx + gy + 1) / np.sqrt(3 * (1 + gx * gx + gy * gy))
    else:
        light = np.zeros_like(z)
    shade = np.where(finite, 138 + 92 * np.nan_to_num(light), 38).astype("uint8")
    base = Image.fromarray(shade).resize(size, Image.Resampling.BILINEAR).convert("RGB")
    panels = [
        ("Missing data", {"void"}, "#00d6e6"),
        ("Spikes and pits", {"spike", "pit"}, "#ff5429"),
        ("Periodic striping", {"striping"}, "#ffcd38"),
        ("Straight seams", {"seam"}, "#b852ff"),
        ("Elevation quantization", {"quantization"}, "#3dd782"),
        ("Flat fills", {"flat_fill"}, "#ff71a5"),
    ]
    page = Image.new("RGB", (1336, 748), "#fcfaf4")
    draw = ImageDraw.Draw(page)
    draw.text(
        (16, 10),
        "DEM inspection  /  " + (Path(source).name if source else "array"),
        fill="black",
    )
    draw.text(
        (16, 30),
        f"Candidates for inspection   |   {grid_w} columns x {grid_h} rows   |   origin at upper left, column increases right, row increases down",
        fill="black",
    )
    for index, (name, classes, color) in enumerate(panels):
        x0, y0 = 16 + index % 3 * 443, 58 + index // 3 * 340
        image = base.copy()
        ink = ImageDraw.Draw(image)
        selected = [f for f in findings if f["class"] in classes]
        for f in selected:
            if "cell" in f:
                c, r = f["cell"]
                x, y = (c + 0.5) * sx, (r + 0.5) * sy
                ink.ellipse(
                    (x - 2, y - 2, x + 2, y + 2),
                    fill=color if f["class"] == "spike" else "#3889ff",
                )
            elif "support_runs" in f:
                support = Image.new("L", size)
                raster = ImageDraw.Draw(support)
                for row, first, end in f["support_runs"]:
                    x, y = int(first * sx), int(row * sy)
                    raster.rectangle(
                        (
                            x,
                            y,
                            max(x, int(end * sx) - 1),
                            max(y, int((row + 1) * sy) - 1),
                        ),
                        fill=255,
                    )
                tint = Image.blend(image, Image.new("RGB", size, color), 0.45)
                image.paste(tint, mask=support)
            elif "rings" in f:
                for ring in f["rings"]:
                    ink.line(
                        [
                            (min(size[0] - 1, x * sx), min(size[1] - 1, y * sy))
                            for x, y in ring
                        ],
                        fill=color,
                        width=1,
                    )
            else:
                left, top, right, bottom = (
                    f["analysis_bounds"] if "analysis_bounds" in f else f["bounds"]
                )
                ink.rectangle(
                    (
                        left * sx,
                        top * sy,
                        max(left * sx, right * sx - 1),
                        max(top * sy, bottom * sy - 1),
                    ),
                    outline=color,
                )
        draw.text((x0, y0), f"{name}   /   {len(selected)}", fill="black")
        page.paste(image, (x0, y0 + 20))
        note = ""
        if classes == {"spike", "pit"}:
            note = "Orange: spike   Blue: pit"
        elif "striping" in classes and selected:
            note = "Supported windows  /  " + ", ".join(
                f"{f['axis']} {f['period_cells']:.2f} px" for f in selected
            )
        elif "quantization" in classes and selected:
            note = f"Aligned cells  /  height step {selected[0]['step']:g}"
        # Two corner labels tie every panel to the JSON pixel coordinates.
        draw.text((x0, y0 + 21 + size[1]), "0", fill="black")
        draw.text(
            (x0 + size[0], y0 + 21 + size[1]),
            f"column {grid_w - 1}",
            fill="black",
            anchor="ra",
        )
        draw.text((x0, y0 + 35 + size[1]), note, fill="black")
    return page
