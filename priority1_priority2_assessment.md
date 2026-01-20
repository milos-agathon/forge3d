# Priority 1 and 2 Implementation Assessment

## Scope
- Compare Priority 1 (Scene Bundle) and Priority 2 (Mapbox Style Spec Import) against `docs/plan.md`.
- Evidence from repository inspection only; no runtime tests or renders executed.

## Summary
- Priority 1: Partial. Rust schema, Python bundle workflow, and CLI flags exist, but IPC Save/Load is stubbed, schema differs from plan, and acceptance tests do not render or hash images.
- Priority 2: Partial. Python parser/converters exist, but the Rust `style` module is missing, render pipeline integration is absent, and tests do not perform pixel diffs.

## Priority 1 Findings (Scene Bundle)
- Plan items 1.1-1.4 are not fully satisfied as envisioned (see evidence checklist for details).
- Scene bundle docs are missing (`docs/api/bundle.md` referenced by examples is not present).
  ```text
  docs\api\render_bundles.md
  ```
- Rust unit test exists in `src/bundle/manifest.rs`, which conflicts with the rule "tests are written in Python, never in Rust".
  ```text
  188-}
  189-
  190:#[cfg(test)]
  191-mod tests {
  192-    use super::*;
  ```
- Plan acceptance test expects PNG hash comparison; current bundle roundtrip test does not mention PNG or hashing.
  ```text
  NO_MATCH: png|hashlib|md5 in tests/test_bundle_roundtrip.py
  ```
- Bundle file layout diverges from plan (`overlays/vectors.json` and `assets/` root, not `vectors.geojson` or `assets/hdri`).
  ```text
  9-- render/ - preset configuration
  10:- assets/ - fonts, HDRI, etc.
  11-"""
  --
  211-    (bundle_path / "render").mkdir(exist_ok=True)
  212:    (bundle_path / "assets").mkdir(exist_ok=True)
  213-    
  --
  235-            json.dump(overlays, f, indent=2)
  236:        manifest.checksums["overlays/vectors.json"] = _compute_sha256(overlays_path)
  237-    
  --
  242-            json.dump(labels, f, indent=2)
  243:        manifest.checksums["overlays/labels.json"] = _compute_sha256(labels_path)
  244-    
  --
  255-        hdr_src = Path(hdr_path)
  256:        hdr_dst = bundle_path / "assets" / hdr_src.name
  257-        shutil.copy2(hdr_src, hdr_dst)
  258:        rel_path = f"assets/{hdr_src.name}"
  259-        manifest.checksums[rel_path] = _compute_sha256(hdr_dst)
  --
  345-    # Find HDR file
  346:    assets_dir = bundle_path / "assets"
  347:    if assets_dir.exists():
  348-        for ext in (".hdr", ".exr"):
  349:            for hdr_file in assets_dir.glob(f"*{ext}"):
  350-                result.hdr_path = hdr_file
  ```
- Rust bundle schema appears isolated (no references outside `src/bundle`).
  ```text
  src\bundle\mod.rs:13:pub use manifest::{BundleManifest, BundleError, BundleResult};
  src\bundle\manifest.rs:33:pub struct BundleManifest {
  src\bundle\manifest.rs:108:impl BundleManifest {
  src\bundle\manifest.rs:196:        let mut manifest = BundleManifest::new("test_bundle");
  src\bundle\manifest.rs:207:        let loaded: BundleManifest = serde_json::from_str(&json).unwrap();
  ```

## Priority 2 Findings (Mapbox Style Spec Import)
- Plan items 2.1-2.4 are not fully satisfied as envisioned (see evidence checklist for details).
- No Mapbox style fixture exists in the repo, so the "parse Mapbox Streets v8" stop-condition is unverified.
  ```text
  NO_MATCH: *mapbox*
  ```
- Style docs claim sprite/glyph APIs that are not implemented in `python/forge3d`.
  ```text
  272:## Sprite Atlas Loading
  290:## Glyph/Font Loading
  ```
  ```text
  NO_MATCH: load_sprite_atlas|FontStack|GlyphRange in python/forge3d
  ```
- "Visual diff" tests rely on MD5 hashes of style values, not pixel diffs.
  ```text
  397-            styled_colors.append(color)
  398-        
  399:        styled_hash = hashlib.md5(str(styled_colors).encode()).hexdigest()
  400-        
  401-        # Default output
  402-        default = VectorStyle()
  403-        default_colors = [default.fill_color] * len(features)
  404:        default_hash = hashlib.md5(str(default_colors).encode()).hexdigest()
  405-        
  406-        assert styled_hash != default_hash, "Styled hash should differ from default"
  --
  414-        for zoom in [5, 10, 15, 20]:
  415-            width = evaluate_number_expr(roads.paint.line_width, {}, zoom=float(zoom))
  416:            h = hashlib.md5(str(width).encode()).hexdigest()
  417-            hashes.append(h)
  418-        
  ```
- `src/lib.rs` declares `pub mod style;` even though no Rust style module exists.
  ```text
  1123-pub mod pointcloud; // P5: Point Cloud support (COPC, EPT)
  1124-pub mod bundle; // Scene bundle (.forge3d) for portable scene packages
  1125:pub mod style; // Mapbox Style Spec import for vector/label styling
  1126-
  1127-// Re-export commonly used types
  ```

## Evidence checklist (Plan items)

### Priority 1
- [x] 1.1 Define bundle schema in Rust (BundleManifest exists).
  ```text
  13-#[derive(Debug, thiserror::Error)]
  14-pub enum BundleError {
  15-    #[error("IO error: {0}")]
  16-    Io(#[from] std::io::Error),
  17-    
  18-    #[error("JSON error: {0}")]
  19-    Json(#[from] serde_json::Error),
  20-    
  21-    #[error("Invalid bundle: {0}")]
  22-    Invalid(String),
  23-    
  24-    #[error("Version mismatch: expected {expected}, got {got}")]
  25-    VersionMismatch { expected: u32, got: u32 },
  26-    
  27-    #[error("Checksum mismatch for {path}")]
  28-    ChecksumMismatch { path: String },
  29-}
  30-
  31-/// Bundle manifest containing metadata and checksums.
  32-#[derive(Debug, Clone, Serialize, Deserialize)]
  33:pub struct BundleManifest {
  34-    /// Schema version (currently 1)
  35-    pub version: u32,
  36-    
  37-    /// Human-readable bundle name
  38-    pub name: String,
  39-    
  40-    /// ISO 8601 creation timestamp
  41-    pub created_at: String,
  42-    
  43-    /// Optional description
  44-    #[serde(default, skip_serializing_if = "Option::is_none")]
  45-    pub description: Option<String>,
  46-    
  47-    /// SHA-256 checksums for bundle files (path -> hex digest)
  48-    #[serde(default)]
  49-    pub checksums: HashMap<String, String>,
  50-    
  51-    /// Terrain metadata
  52-    #[serde(default, skip_serializing_if = "Option::is_none")]
  53-    pub terrain: Option<TerrainMeta>,
  ```

- [ ] 1.2 Python SceneBundle API matches plan signature and roundtrip stop-condition.
  ```text
  84-
  85-@dataclass
  86:class BundleManifest:
  87-    """Bundle manifest containing metadata and checksums."""
  88-    version: int
  --
  168-
  169-
  170:def save_bundle(
  171-    path: Union[str, Path],
  172-    *,
  --
  285-
  286-
  287:def load_bundle(path: Union[str, Path], verify_checksums: bool = True) -> LoadedBundle:
  288-    """Load a scene bundle from disk.
  289-    
  ```
  ```text
  92-    checksums: Dict[str, str] = field(default_factory=dict)
  93-    terrain: Optional[TerrainMeta] = None
  94-    camera_bookmarks: List[CameraBookmark] = field(default_factory=list)
  95:    preset: Optional[Dict[str, Any]] = None
  96-
  97-    @classmethod
  98-    def new(cls, name: str) -> "BundleManifest":
  ```

- [ ] 1.3 IPC SaveBundle/LoadBundle round-trip restores viewer state.
  ```text
  814-                }
  815-                self.snapshot_request = Some(path);
  816-            }
  817:            ViewerCmd::SaveBundle { path, name } => {
  818-                // Bundle saving is handled via Python; this just acknowledges the command
  819-                let bundle_name = name.as_deref().unwrap_or("scene");
  820:                println!("SaveBundle requested: {} (name: {})", path, bundle_name);
  821-                // Store the request for Python-side handling
  822-                self.pending_bundle_save = Some((path, name));
  823-            }
  ```
  ```text
  821-                // Store the request for Python-side handling
  822-                self.pending_bundle_save = Some((path, name));
  823-            }
  824:            ViewerCmd::LoadBundle { path } => {
  825-                // Bundle loading is handled via Python; this just acknowledges the command
  826:                println!("LoadBundle requested: {}", path);
  827-                // Store the request for Python-side handling
  828-                self.pending_bundle_load = Some(path);
  829-            }
  ```
  ```text
  src\viewer\cmd\handler.rs:822:                self.pending_bundle_save = Some((path, name));
  src\viewer\cmd\handler.rs:828:                self.pending_bundle_load = Some(path);
  src\viewer\viewer_struct.rs:255:    pub(crate) pending_bundle_save: Option<(String, Option<String>)>;
  src\viewer\viewer_struct.rs:256:    pub(crate) pending_bundle_load: Option<String>;
  src\viewer\init\viewer_new.rs:303:            pending_bundle_save: None,
  src\viewer\init\viewer_new.rs:304:            pending_bundle_load: None,
  ```

- [ ] 1.4 CLI flags + precedence + render smoke stop-condition.
  ```text
  422-    # Scene Bundle (.forge3d)
  423-    parser.add_argument(
  424:        "--save-bundle",
  425-        type=Path,
  426-        default=None,
  --
  428-    )
  429-    parser.add_argument(
  430:        "--load-bundle",
  431-        type=Path,
  432-        default=None,
  --
  446-    args = _parse_args()
  447-
  448:    # Handle --load-bundle: load scene from bundle, apply preset from bundle
  449-    if args.load_bundle:
  450-        from forge3d.bundle import load_bundle, is_bundle
  --
  554-    result = _impl.run(args)
  555-    
  556:    # Handle --save-bundle: save scene to bundle after rendering
  557-    if args.save_bundle:
  558-        from forge3d.bundle import save_bundle
  ```
  ```text
  26-
  27-def test_save_bundle_flag_exists():
  28:    """--save-bundle flag is recognized by argparse."""
  29-    mod = _load_terrain_demo()
  30-    # Create a minimal parser and check the flag exists
  31:    import argparse
  32-    import sys
  33-    
  --
  36-    try:
  37-        sys.argv = ["terrain_demo.py", "--save-bundle", "/tmp/test.forge3d"]
  38:        args = mod._parse_args()
  39-        assert args.save_bundle == Path("/tmp/test.forge3d")
  40-    finally:
  --
  43-
  44-def test_load_bundle_flag_exists():
  45:    """--load-bundle flag is recognized by argparse."""
  46-    mod = _load_terrain_demo()
  47-    import sys
  --
  50-    try:
  51-        sys.argv = ["terrain_demo.py", "--load-bundle", "/tmp/test.forge3d"]
  52:        args = mod._parse_args()
  53-        assert args.load_bundle == Path("/tmp/test.forge3d")
  54-    finally:
  ```

### Priority 2
- [ ] 2.1 Style spec parser in Rust (`src/style`).
  ```text
  NO_MATCH: src/style/*.rs
  ```

- [ ] 2.2 Property converters in Rust.
  ```text
  NO_MATCH: src/style/*.rs
  ```

- [x] 2.3 Python API (`load_style`, `apply_style`) exists.
  ```text
  81-
  82-@dataclass
  83:class StyleLayer:
  84-    """A single style layer."""
  85-    id: str
  --
  148-
  149-
  150:def load_style(path: Path | str) -> StyleSpec:
  151-    """Load a Mapbox GL Style Spec JSON file.
  152-
  --
  533-
  534-
  535:def apply_style(
  536-    spec: StyleSpec,
  537-    features: list[dict],
  ```

- [ ] 2.4 Integration with render pipeline (`python/forge3d/render.py`, `src/labels/layer.rs`).
  ```text
  NO_MATCH: style integration in python/forge3d/render.py
  ```
  ```text
  NO_MATCH: Mapbox|StyleSpec in src/labels/layer.rs
  ```

## What remains / risks
- P1 risk: schema drift between Rust and Python bundle manifests and divergence from planned directory layout.
- P1 risk: viewer IPC Save/Load does not restore viewer state; interactive round-trip is unverified.
- P2 risk: `src/lib.rs` declares `pub mod style` but no Rust module exists, which is a build blocker unless gated.
- P2 risk: style pipeline is not wired into rendering; tests do not prove user-facing visual changes.
- P2 risk: docs advertise sprite/glyph APIs that are missing.

### Remaining checklist
- [ ] Run bundle tests with pytest.
  ```text
  NOT RUN: python -m pytest tests/test_bundle_roundtrip.py tests/test_bundle_cli.py
  ```
- [ ] Run style tests with pytest.
  ```text
  NOT RUN: python -m pytest tests/test_style_parser.py tests/test_style_render.py tests/test_style_visual.py
  ```
- [ ] Run viewer IPC SaveBundle/LoadBundle end-to-end check.
  ```text
  NOT RUN: viewer IPC SaveBundle/LoadBundle end-to-end check (bundle save/load restores viewer state)
  ```
