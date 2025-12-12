#!/usr/bin/env python3
"""Show effective toggle values after preset application for P6.1 verification."""
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))

def show_toggles(preset_path=None, force_colormap_srgb=False, force_output_srgb_eotf=False):
    """Simulate the preset loading and CLI override to show effective values."""
    
    # Start with defaults
    colormap_srgb = False
    output_srgb_eotf = False
    
    # Apply preset if provided
    if preset_path:
        with open(preset_path) as f:
            preset = json.load(f)
        cli_params = preset.get("cli_params", {})
        colormap_srgb = cli_params.get("colormap_srgb", False)
        output_srgb_eotf = cli_params.get("output_srgb_eotf", False)
        print(f"[PRESET] Loaded: {preset_path}")
        print(f"[PRESET] colormap_srgb={colormap_srgb}, output_srgb_eotf={output_srgb_eotf}")
    
    # Apply CLI overrides (these always win)
    if force_colormap_srgb:
        colormap_srgb = True
        print(f"[CLI OVERRIDE] colormap_srgb forced to True")
    if force_output_srgb_eotf:
        output_srgb_eotf = True
        print(f"[CLI OVERRIDE] output_srgb_eotf forced to True")
    
    print(f"\n=== EFFECTIVE VALUES ===")
    print(f"colormap_srgb = {colormap_srgb}")
    print(f"output_srgb_eotf = {output_srgb_eotf}")
    return colormap_srgb, output_srgb_eotf

if __name__ == "__main__":
    print("\n" + "="*60)
    print("SCENARIO 1: P5 preset WITHOUT forced flags")
    print("="*60)
    show_toggles(preset_path="presets/p5_gore_shader_only.json")
    
    print("\n" + "="*60)
    print("SCENARIO 2: P5 preset WITH forced flags")
    print("="*60)
    show_toggles(preset_path="presets/p5_gore_shader_only.json", 
                 force_colormap_srgb=True, force_output_srgb_eotf=True)
    
    print("\n" + "="*60)
    print("SCENARIO 3: P6 preset")
    print("="*60)
    show_toggles(preset_path="presets/p6_gore_detail_normals.json")
