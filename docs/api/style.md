# Mapbox Style Spec Import

forge3d supports importing [Mapbox GL Style Spec](https://docs.mapbox.com/mapbox-gl-js/style-spec/) JSON files to style vector overlays and labels. This enables ecosystem compatibility with existing Mapbox/MapLibre styles.

## Quick Start

```python
from forge3d import load_style, apply_style

# Load a Mapbox style
spec = load_style("mapbox-streets.json")
print(f"Loaded {len(spec.layers)} layers")

# Apply style to GeoJSON features
features = [
    {"type": "Feature", "properties": {"class": "motorway"}, "geometry": {...}},
]
styled = apply_style(spec, features, source_layer="road")

for feature, style in styled:
    print(f"Fill: {style.fill_color}, Stroke: {style.stroke_color}")
```

## Supported Properties (v1)

### Fill Layers

| Property | Type | Description |
|----------|------|-------------|
| `fill-color` | color | Fill color |
| `fill-opacity` | number | Fill opacity (0-1) |
| `fill-outline-color` | color | Outline stroke color |

### Line Layers

| Property | Type | Description |
|----------|------|-------------|
| `line-color` | color | Line color |
| `line-width` | number | Line width in pixels |
| `line-opacity` | number | Line opacity (0-1) |

### Symbol Layers

| Property | Type | Description |
|----------|------|-------------|
| `text-field` | string | Text content (e.g., `"{name}"`) |
| `text-size` | number | Font size in pixels |
| `text-color` | color | Text color |
| `text-halo-color` | color | Text outline color |
| `text-halo-width` | number | Outline width in pixels |
| `text-offset` | array | Offset from anchor [x, y] in ems |

### Filters

Supported filter expressions:

| Expression | Example | Description |
|------------|---------|-------------|
| `==` | `["==", "class", "road"]` | Equality |
| `!=` | `["!=", "type", "bridge"]` | Inequality |
| `all` | `["all", [...], [...]]` | All conditions must match |
| `any` | `["any", [...], [...]]` | Any condition matches |
| `has` | `["has", "name"]` | Property exists |
| `!` | `["!", [...]]` | Negation |
| `in` | `["in", "class", "a", "b"]` | Value in list |

## API Reference

### `load_style(path)`

Load a Mapbox GL Style Spec JSON file.

```python
spec = load_style("style.json")
```

**Parameters:**
- `path`: Path to style.json file

**Returns:** `StyleSpec` object

**Raises:**
- `ValueError`: If style version is not 8
- `FileNotFoundError`: If file does not exist

### `parse_style(data)`

Parse a style from a dictionary (already loaded JSON).

```python
import json
with open("style.json") as f:
    data = json.load(f)
spec = parse_style(data)
```

### `apply_style(spec, features, source_layer=None, zoom=10.0)`

Apply style spec to GeoJSON features.

```python
result = apply_style(spec, features, source_layer="water", zoom=12.0)
for feature, style in result:
    # style is a VectorStyle with fill_color, stroke_color, etc.
    pass
```

**Parameters:**
- `spec`: Parsed `StyleSpec`
- `features`: List of GeoJSON feature dictionaries
- `source_layer`: Filter to layers matching this source-layer
- `zoom`: Current zoom level for zoom-dependent filtering

**Returns:** List of `(feature, VectorStyle)` tuples

### `parse_color(color_str)`

Parse a CSS color string to RGBA tuple.

```python
rgba = parse_color("#ff0000")  # (1.0, 0.0, 0.0, 1.0)
rgba = parse_color("rgb(255, 128, 0)")  # (1.0, 0.5, 0.0, 1.0)
rgba = parse_color("hsl(120, 100%, 50%)")  # green
rgba = parse_color("blue")  # (0.0, 0.0, 1.0, 1.0)
```

Supported formats:
- Hex: `#RGB`, `#RGBA`, `#RRGGBB`, `#RRGGBBAA`
- RGB: `rgb(r, g, b)`, `rgba(r, g, b, a)`
- HSL: `hsl(h, s%, l%)`, `hsla(h, s%, l%, a)`
- Named: `black`, `white`, `red`, `green`, `blue`, etc.

### Style Classes

#### `StyleSpec`

```python
spec.version       # Style version (always 8)
spec.name          # Style name
spec.layers        # List of StyleLayer objects

# Filter methods
spec.fill_layers()              # Get all fill layers
spec.line_layers()              # Get all line layers
spec.symbol_layers()            # Get all symbol layers
spec.layer_by_id("water")       # Find layer by ID
spec.layers_for_source_layer("road")  # Get layers for source-layer
```

#### `StyleLayer`

```python
layer.id           # Unique layer ID
layer.layer_type   # "fill", "line", "symbol", "background"
layer.source_layer # Source-layer name (for vector tiles)
layer.paint        # PaintProps object
layer.layout       # LayoutProps object
layer.filter       # Filter expression (list)
layer.minzoom      # Minimum zoom level
layer.maxzoom      # Maximum zoom level

layer.is_visible()           # Check visibility
layer.in_zoom_range(12.0)    # Check zoom range
layer.matches_filter(props)  # Evaluate filter
```

#### `VectorStyle`

Returned by `paint_to_vector_style()` and `apply_style()`:

```python
style.fill_color    # (r, g, b, a) tuple
style.stroke_color  # (r, g, b, a) tuple
style.stroke_width  # float
style.point_size    # float
```

#### `LabelStyle`

Returned by `layout_to_label_style()`:

```python
style.size        # Font size in pixels
style.color       # (r, g, b, a) tuple
style.halo_color  # (r, g, b, a) tuple
style.halo_width  # Halo width in pixels
style.offset      # (x, y) offset in pixels
```

## Interactive Viewer

Use the style viewer example for interactive exploration:

```bash
python examples/style_viewer_interactive.py \
    --dem terrain.tif \
    --style mapbox-streets.json \
    --vectors roads.geojson
```

Interactive commands:
- `style <path>` - Load new style
- `layers` - List all layers
- `layer <id>` - Show layer details
- `snap <path>` - Take screenshot

## Expression Evaluation

Full support for data-driven styling expressions:

### Interpolation

```python
# Zoom-dependent line width
["interpolate", ["linear"], ["zoom"],
    5, 1,
    15, 4,
    20, 8
]

# Exponential interpolation
["interpolate", ["exponential", 1.5], ["zoom"],
    10, 1,
    20, 100
]
```

### Step Function

```python
# Discrete values at breakpoints
["step", ["get", "height"],
    "small",
    10, "medium",
    50, "large"
]
```

### Match Expression

```python
# Pattern matching on property values
["match", ["get", "class"],
    "motorway", "#ff0000",
    "primary", "#ffa500",
    ["secondary", "tertiary"], "#ffff00",
    "#888888"  # default
]
```

### Case Expression

```python
# Conditional logic
["case",
    [">", ["get", "population"], 1000000], "large",
    [">", ["get", "population"], 100000], "medium",
    "small"  # default
]
```

### Other Expressions

- **Property access**: `get`, `has`, `at`, `length`
- **Comparison**: `==`, `!=`, `<`, `<=`, `>`, `>=`
- **Logic**: `all`, `any`, `!`
- **Math**: `+`, `-`, `*`, `/`, `%`, `^`, `sqrt`, `sin`, `cos`, etc.
- **String**: `concat`, `upcase`, `downcase`
- **Color**: `rgb`, `rgba`
- **Special**: `zoom`, `coalesce`, `literal`

## Sprite Atlas Loading

Load sprite atlases for icon rendering:

```python
from forge3d.style import load_sprite_atlas

# Load sprite atlas (expects sprite.json and sprite.png)
atlas = load_sprite_atlas("path/to/sprite")

# Access sprite entries
entry = atlas.get("marker-icon")
print(f"Size: {entry.width}x{entry.height}")

# Get UV coordinates (normalized 0-1)
uvs = atlas.get_uvs("marker-icon")  # [u0, v0, u1, v1]
```

## Glyph/Font Loading

Support for PBF glyph ranges (Mapbox font format):

```python
from forge3d.style import FontStack, GlyphRange

# Create font stack
fonts = FontStack(["Open Sans Regular", "Arial Unicode MS"])

# Get required glyph ranges for text
ranges = fonts.required_ranges("Hello 世界")
for r in ranges:
    print(f"Need to load: {r.filename()}")  # e.g., "0-255.pbf"
```

## Limitations

Current limitations:

- **Raster/Hillshade**: Raster and hillshade layers are parsed but not rendered
- **3D**: 3D extrusion properties are not supported
- **Sprite images**: Atlas metadata loaded; full image decoding requires external library
