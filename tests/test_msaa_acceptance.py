"""Acceptance test for MSAA downgrade.

This test verifies that the terrain_demo.py script works with --msaa 8
and successfully downgrades to MSAA 4 without validation errors.
"""
import subprocess
import sys
from pathlib import Path


def test_terrain_demo_msaa8_no_panic():
    """Test that terrain_demo.py --msaa 8 succeeds without panic."""
    demo_script = Path(__file__).parent.parent / "examples" / "terrain_demo.py"
    output_path = Path(__file__).parent.parent / "examples" / "out" / "test_msaa8.png"

    # Clean up output if it exists
    if output_path.exists():
        output_path.unlink()

    # Run terrain_demo.py with MSAA 8
    result = subprocess.run(
        [
            sys.executable,
            str(demo_script),
            "--dem", "assets/Gore_Range_Albers_1m.tif",
            "--hdr", "assets/snow_field_4k.hdr",
            "--size", "640", "360",
            "--render-scale", "1.0",
            "--msaa", "8",  # Request MSAA 8 - should downgrade to 4
            "--z-scale", "1.5",
            "--exposure", "1.0",
            "--colormap-domain", "200.0", "2200.0",
            "--output", str(output_path),
            "--overwrite",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    # Print output for debugging
    if result.stdout:
        print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    # Verify the command succeeded
    assert result.returncode == 0, f"terrain_demo.py failed with code {result.returncode}: {result.stderr}"

    # Verify output file was created
    assert output_path.exists(), f"Output file not created: {output_path}"

    # Verify no validation errors in stderr
    assert "Validation Error" not in result.stderr, "MSAA validation error occurred"
    assert "Sample count 8 is not supported" not in result.stderr, "MSAA 8 not downgraded properly"

    # Clean up
    if output_path.exists():
        output_path.unlink()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
