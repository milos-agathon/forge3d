#!/usr/bin/env python3
"""
P0-10 Acceptance Demo: Renderer Config Pipeline

Demonstrates the complete renderer configuration pipeline:
1. Load preset
2. Apply CLI-style overrides
3. Validate configuration
4. Serialize and print config

This script validates P0 config plumbing without requiring GPU or rendering.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add parent to path to import forge3d
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from forge3d.config import load_renderer_config
from forge3d import presets


def main() -> int:
    print("=" * 80)
    print("P0-10 Acceptance Test: Renderer Config Pipeline")
    print("=" * 80)
    print()

    # Simulated CLI flags from P0-10 requirement
    cli_flags = {
        "preset": "outdoor_sun",
        "brdf": "cooktorrance-ggx",
        "shadows": "pcf",
        "cascades": 2,
        "hdr": "assets/snow_field_4k.hdr",
    }

    print("Step 1: Load preset")
    print("-" * 80)
    preset_name = cli_flags["preset"]
    print(f"Loading preset: {preset_name}")
    
    preset_config = presets.get(preset_name)
    print(f"Preset loaded: {preset_name}")
    print()

    print("Step 2: Apply CLI overrides")
    print("-" * 80)
    overrides = {
        "brdf": cli_flags["brdf"],
        "shadows": cli_flags["shadows"],
        "cascades": cli_flags["cascades"],
        "hdr": cli_flags["hdr"],
    }
    print(f"CLI overrides:")
    for key, value in overrides.items():
        print(f"  --{key} {value}")
    print()

    print("Step 3: Build and validate renderer config")
    print("-" * 80)
    print("Calling load_renderer_config(preset, overrides)...")
    config = load_renderer_config(preset_config, overrides)
    
    print("Validating configuration...")
    config.validate()
    print("✓ Configuration is valid")
    print()

    print("Step 4: Serialize configuration")
    print("-" * 80)
    config_dict = config.to_dict()
    
    print("Configuration structure:")
    print(f"  - Lighting: {len(config_dict['lighting']['lights'])} light(s)")
    print(f"  - Shading: {config_dict['shading']['brdf']}")
    print(f"  - Shadows: {config_dict['shadows']['technique']} (cascades={config_dict['shadows']['cascades']})")
    print(f"  - GI modes: {config_dict['gi']['modes']}")
    print(f"  - Atmosphere: sky={config_dict['atmosphere']['sky']}, hdr={config_dict['atmosphere'].get('hdr_path', 'None')}")
    print()

    print("Step 5: Verify preset + override merge")
    print("-" * 80)
    
    # Verify preset values that were NOT overridden
    assert config_dict["shading"]["brdf"] == "cooktorrance-ggx", "BRDF override not applied"
    print(f"✓ BRDF override applied: cooktorrance-ggx")
    
    assert config_dict["shadows"]["technique"] == "pcf", "Shadow technique override not applied"
    print(f"✓ Shadow technique override applied: pcf")
    
    assert config_dict["shadows"]["cascades"] == 2, "Cascades override not applied"
    print(f"✓ Cascades override applied: 2")
    
    # Note: outdoor_sun preset has CSM by default, but we overrode to PCF
    assert config_dict["shadows"]["technique"] == "pcf", "Shadow override precedence failed"
    print(f"✓ Override precedence correct: pcf (not CSM from preset)")
    
    assert config_dict["atmosphere"].get("hdr_path") == "assets/snow_field_4k.hdr", "HDR path not applied"
    print(f"✓ HDR path applied: assets/snow_field_4k.hdr")
    
    # Verify preset values that WERE preserved
    assert len(config_dict["lighting"]["lights"]) == 1, "Expected 1 light from preset"
    light = config_dict["lighting"]["lights"][0]
    assert light["type"] == "directional", "Expected directional light from preset"
    print(f"✓ Preset light preserved: directional (intensity={light['intensity']})")
    
    assert config_dict["atmosphere"]["sky"] == "hosek-wilkie", "Expected Hosek-Wilkie sky from preset"
    print(f"✓ Preset sky preserved: hosek-wilkie")
    
    print()
    print("Step 6: Full configuration JSON")
    print("-" * 80)
    print(json.dumps(config_dict, indent=2))
    print()

    print("=" * 80)
    print("P0-10 ACCEPTANCE TEST PASSED ✓")
    print("=" * 80)
    print()
    print("Summary:")
    print("  ✓ Preset loaded successfully (outdoor_sun)")
    print("  ✓ CLI overrides applied (brdf, shadows, cascades, hdr)")
    print("  ✓ Override precedence correct (CLI > preset)")
    print("  ✓ Configuration validates")
    print("  ✓ Serialization works (to_dict())")
    print()
    print("The renderer configuration pipeline is working correctly.")
    print("All P0 config plumbing requirements are satisfied.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
