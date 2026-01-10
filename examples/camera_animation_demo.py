#!/usr/bin/env python3
"""Camera Animation Demo - Keyframe-based camera flyovers for terrain.

This example demonstrates Feature C (Plan 1 MVP) camera animation capabilities:
- CameraAnimation with keyframe storage
- Cubic Hermite (Catmull-Rom) interpolation for smooth camera paths
- Real-time preview in interactive viewer
- Offline frame export to PNG sequence

Usage:
    # Interactive preview (orbit animation in viewer)
    python examples/camera_animation_demo.py

    # Export frames to disk
    python examples/camera_animation_demo.py --export ./frames --fps 30

    # Custom DEM file
    python examples/camera_animation_demo.py --dem assets/tif/Mount_Fuji_30m.tif
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from forge3d.animation import CameraAnimation, RenderConfig

# Asset paths
ASSETS_DIR = Path(__file__).parent.parent / "assets"
DEFAULT_DEM = ASSETS_DIR / "tif" / "dem_rainier.tif"


def create_orbit_animation(duration: float = 10.0) -> CameraAnimation:
    """Create a 360-degree orbit animation around the terrain.
    
    Args:
        duration: Animation duration in seconds
        
    Returns:
        CameraAnimation with keyframes for a complete orbit
    """
    anim = CameraAnimation()
    
    # Keyframes for smooth 360° orbit with elevation changes
    # phi: azimuth (horizontal rotation), theta: elevation angle
    keyframes = [
        # time, phi, theta, radius, fov
        (0.0,    0.0,  45.0, 2500.0, 55.0),   # Start: front view
        (2.5,   90.0,  35.0, 2200.0, 55.0),   # Quarter turn, lower angle, closer
        (5.0,  180.0,  50.0, 2800.0, 55.0),   # Half turn, higher angle, farther
        (7.5,  270.0,  40.0, 2400.0, 55.0),   # Three-quarter turn
        (duration, 360.0, 45.0, 2500.0, 55.0), # Complete loop (same as start)
    ]
    
    for t, phi, theta, radius, fov in keyframes:
        anim.add_keyframe(time=t, phi=phi, theta=theta, radius=radius, fov=fov)
    
    return anim


def create_flyover_animation(duration: float = 8.0) -> CameraAnimation:
    """Create a dramatic flyover animation with zoom.
    
    Args:
        duration: Animation duration in seconds
        
    Returns:
        CameraAnimation with keyframes for a flyover sequence
    """
    anim = CameraAnimation()
    
    # Dramatic flyover: start far and high, zoom in, pan around, zoom out
    keyframes = [
        # time, phi, theta, radius, fov
        (0.0,   45.0, 60.0, 4000.0, 45.0),   # Start: distant overview
        (2.0,   60.0, 45.0, 2000.0, 55.0),   # Zoom in, lower angle
        (4.0,   90.0, 30.0, 1500.0, 60.0),   # Close-up, low angle
        (6.0,  135.0, 40.0, 2500.0, 55.0),   # Pan and pull back
        (duration, 180.0, 55.0, 3500.0, 50.0), # End: distant side view
    ]
    
    for t, phi, theta, radius, fov in keyframes:
        anim.add_keyframe(time=t, phi=phi, theta=theta, radius=radius, fov=fov)
    
    return anim


def create_sunrise_animation(duration: float = 12.0) -> CameraAnimation:
    """Create a slow sunrise reveal animation.
    
    Args:
        duration: Animation duration in seconds
        
    Returns:
        CameraAnimation with keyframes for sunrise reveal
    """
    anim = CameraAnimation()
    
    # Slow pan with gentle movements for sunrise timelapse feel
    keyframes = [
        # time, phi, theta, radius, fov
        (0.0,   90.0, 25.0, 3000.0, 50.0),   # Low angle, east-facing
        (4.0,  100.0, 30.0, 2800.0, 52.0),   # Gentle rise
        (8.0,  115.0, 38.0, 2600.0, 54.0),   # Continue rising
        (duration, 135.0, 45.0, 2500.0, 55.0), # Morning view
    ]
    
    for t, phi, theta, radius, fov in keyframes:
        anim.add_keyframe(time=t, phi=phi, theta=theta, radius=radius, fov=fov)
    
    return anim


def preview_animation_interpolation(anim: CameraAnimation, fps: int = 10):
    """Print interpolated camera states to preview the animation.
    
    Args:
        anim: CameraAnimation to preview
        fps: Samples per second for preview
    """
    print(f"\n{'='*60}")
    print(f"Animation Preview: {anim.keyframe_count} keyframes, {anim.duration:.1f}s duration")
    print(f"{'='*60}")
    
    total_frames = anim.get_frame_count(fps)
    print(f"At {fps} fps: {total_frames} frames\n")
    
    # Sample every 0.5 seconds
    sample_interval = 0.5
    t = 0.0
    while t <= anim.duration:
        state = anim.evaluate(t)
        if state:
            print(f"t={t:5.2f}s | phi={state.phi_deg:7.2f}° theta={state.theta_deg:5.2f}° "
                  f"radius={state.radius:7.1f} fov={state.fov_deg:5.1f}°")
        t += sample_interval
    print()


def calculate_sun_direction(camera_phi_deg: float, sun_offset_deg: float = 120.0) -> tuple[float, float]:
    """Calculate sun direction based on camera position.
    
    Args:
        camera_phi_deg: Camera azimuth angle in degrees
        sun_offset_deg: Offset angle for sun relative to camera (default: 120° for side lighting)
    
    Returns:
        Tuple of (sun_azimuth_deg, sun_elevation_deg)
    """
    # Position sun at an offset from camera for dramatic side/back lighting
    sun_azimuth = (camera_phi_deg + sun_offset_deg) % 360.0
    
    # Keep sun at moderate elevation for good shadows (varies slightly with camera angle)
    # This creates more dramatic lighting changes as camera moves
    sun_elevation = 45.0 + 10.0 * ((camera_phi_deg % 180.0) / 180.0 - 0.5)
    
    return sun_azimuth, sun_elevation


def run_interactive_preview(
    dem_path: Path,
    anim: CameraAnimation,
    z_scale: float = 0.15,
    dynamic_sun: bool = True,
    sun_offset: float = 120.0,
    sun_intensity: float = 1.0,
):
    """Run animation preview in the interactive viewer.
    
    Args:
        dem_path: Path to DEM file
        anim: CameraAnimation to preview
        z_scale: Terrain height exaggeration
    """
    from forge3d.viewer import open_viewer_async
    
    print(f"\nOpening interactive viewer with terrain: {dem_path.name}")
    print("Animation will loop continuously. Close window to exit.")
    print("-" * 60)
    
    # Find and launch viewer
    try:
        # Use terrain viewer IPC for orbit camera control
        import subprocess
        import socket
        import json
        import re
        
        from forge3d.viewer import _find_viewer_binary
        
        binary = _find_viewer_binary()
        
        # Launch viewer with terrain
        cmd = [
            binary,
            "--terrain", str(dem_path),
            "--ipc-host", "127.0.0.1",
            "--ipc-port", "0",
            "--size", "1280x720",
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        # Wait for viewer ready message
        port = None
        ready_pattern = re.compile(r"FORGE3D_VIEWER_READY port=(\d+)")
        
        for line in iter(process.stdout.readline, ""):
            print(f"[viewer] {line.rstrip()}")
            match = ready_pattern.search(line)
            if match:
                port = int(match.group(1))
                break
        
        # Drain any remaining stdout to prevent pipe buffer blocking
        import threading
        def drain_output():
            try:
                for line in iter(process.stdout.readline, ""):
                    pass  # Discard output
            except:
                pass
        drain_thread = threading.Thread(target=drain_output, daemon=True)
        drain_thread.start()
        
        if port is None:
            print("ERROR: Viewer did not report ready")
            process.terminate()
            return
        
        print(f"\nViewer ready on port {port}")
        print("Starting animation loop (Ctrl+C to stop)...")
        
        # Connect to viewer
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10.0)
        sock.connect(("127.0.0.1", port))
        
        def send_cmd(cmd_dict):
            request = json.dumps(cmd_dict) + "\n"
            sock.sendall(request.encode("utf-8"))
            response = b""
            while b"\n" not in response:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
            return json.loads(response.split(b"\n")[0].decode("utf-8"))
        
        # Set terrain parameters
        send_cmd({"cmd": "set_terrain", "zscale": z_scale})
        
        # Set initial camera position before waiting
        first_state = anim.evaluate(0.0)
        if first_state:
            send_cmd({
                "cmd": "set_terrain_camera",
                "phi_deg": first_state.phi_deg,
                "theta_deg": first_state.theta_deg,
                "radius": first_state.radius,
                "fov_deg": first_state.fov_deg,
            })
        
        # Wait for terrain to load and first frame to render
        print("Waiting for terrain to load and render", end="", flush=True)
        for i in range(80):  # 8 seconds total
            time.sleep(0.1)
            if i % 10 == 0:
                print(".", end="", flush=True)
        print(" ready!", flush=True)
        
        # Animation loop
        fps = 30
        frame_time = 1.0 / fps
        start_time = time.time()
        
        try:
            while process.poll() is None:
                # Calculate animation time (loop)
                elapsed = time.time() - start_time
                t = elapsed % anim.duration
                
                # Get interpolated camera state
                state = anim.evaluate(t)
                if state:
                    # Update camera
                    send_cmd({
                        "cmd": "set_terrain_camera",
                        "phi_deg": state.phi_deg,
                        "theta_deg": state.theta_deg,
                        "radius": state.radius,
                        "fov_deg": state.fov_deg,
                    })
                    
                    # Update sun direction if dynamic lighting enabled
                    if dynamic_sun:
                        sun_azimuth, sun_elevation = calculate_sun_direction(state.phi_deg, sun_offset)
                        send_cmd({
                            "cmd": "set_terrain_sun",
                            "azimuth_deg": sun_azimuth,
                            "elevation_deg": sun_elevation,
                            "intensity": sun_intensity,
                        })
                
                time.sleep(frame_time)
                
        except KeyboardInterrupt:
            print("\nAnimation stopped by user")
        finally:
            try:
                send_cmd({"cmd": "close"})
            except Exception:
                pass
            sock.close()
            process.terminate()
            
    except FileNotFoundError:
        print("ERROR: Could not find interactive_viewer binary.")
        print("Build with: cargo build --release --bin interactive_viewer")
    except Exception as e:
        print(f"ERROR: {e}")


def encode_frames_to_mp4(
    frames_dir: Path,
    output_path: Path,
    fps: int = 30,
    quality: str = "high",
    cleanup_frames: bool = False,
    title: Optional[str] = None,
) -> bool:
    """Encode PNG frames to MP4 video using ffmpeg.
    
    Args:
        frames_dir: Directory containing frame_*.png files
        output_path: Output MP4 file path
        fps: Frames per second
        quality: Quality preset ('high', 'medium', 'low')
        cleanup_frames: Delete PNG frames after encoding
        title: Optional title text to overlay at top of video
    
    Returns:
        True if encoding succeeded, False otherwise
    """
    import subprocess
    import shutil
    
    # Check if ffmpeg is available
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found. Install with: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)")
        return False
    
    # Quality presets
    crf_map = {"high": 18, "medium": 23, "low": 28}
    crf = crf_map.get(quality, 23)
    
    print(f"\nEncoding MP4 video...")
    print(f"Quality: {quality} (CRF={crf})")
    if title:
        print(f"Title: {title}")
    print(f"Output: {output_path}")
    
    try:
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-framerate", str(fps),
            "-pattern_type", "glob",
            "-i", str(frames_dir / "frame_*.png"),
        ]
        
        # Add title overlay if specified
        if title:
            # Escape special characters for drawtext filter
            title_escaped = title.replace(":", "\\:").replace("'", "'\\'")
            drawtext_filter = (
                f"drawtext="
                f"text='{title_escaped}':"
                f"fontsize=72:"
                f"fontcolor=white:"
                f"bordercolor=black:"
                f"borderw=3:"
                f"x=(w-text_w)/2:"
                f"y=60"
            )
            cmd.extend(["-vf", drawtext_filter])
        
        cmd.extend([
            "-c:v", "libx264",
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-movflags", "+faststart",  # Web optimization
            str(output_path),
        ])
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        
        if result.returncode == 0:
            file_size = output_path.stat().st_size / (1024 * 1024)
            print(f"✓ MP4 created successfully ({file_size:.1f} MB)")
            
            if cleanup_frames:
                print("Cleaning up PNG frames...")
                for frame_file in frames_dir.glob("frame_*.png"):
                    frame_file.unlink()
                print("✓ Frames deleted")
            
            return True
        else:
            print(f"ERROR: ffmpeg failed with code {result.returncode}")
            if result.stderr:
                print(f"ffmpeg output: {result.stderr[-500:]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("ERROR: ffmpeg encoding timed out")
        return False
    except Exception as e:
        print(f"ERROR: Failed to encode MP4: {e}")
        return False


def export_animation_frames(
    dem_path: Path,
    anim: CameraAnimation,
    output_dir: Path,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
    z_scale: float = 0.15,
    create_mp4: bool = False,
    mp4_output: Optional[Path] = None,
    mp4_quality: str = "high",
    cleanup_frames: bool = False,
    dynamic_sun: bool = True,
    sun_offset: float = 120.0,
    sun_intensity: float = 1.0,
    title: Optional[str] = None,
):
    """Export animation frames to PNG files and optionally encode to MP4.
    
    Args:
        dem_path: Path to DEM file
        anim: CameraAnimation to render
        output_dir: Directory for output frames
        fps: Frames per second
        width: Output width in pixels
        height: Output height in pixels
        z_scale: Terrain height exaggeration
        create_mp4: Create MP4 video after frame export
        mp4_output: MP4 output path (default: output_dir.mp4)
        mp4_quality: MP4 quality preset ('high', 'medium', 'low')
        cleanup_frames: Delete PNG frames after MP4 creation
    """
    from forge3d.viewer import open_viewer_async
    
    print(f"\nExporting animation frames to: {output_dir}")
    print(f"Settings: {width}x{height} @ {fps} fps")
    print(f"Total frames: {anim.get_frame_count(fps)}")
    print("-" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use terrain viewer for rendering
        import subprocess
        import socket
        import json
        import re
        
        from forge3d.viewer import _find_viewer_binary
        
        binary = _find_viewer_binary()
        
        # Launch viewer with terrain (headless-ish, just for rendering)
        cmd = [
            binary,
            "--terrain", str(dem_path),
            "--ipc-host", "127.0.0.1",
            "--ipc-port", "0",
            "--size", f"{width}x{height}",
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        
        # Wait for ready
        port = None
        ready_pattern = re.compile(r"FORGE3D_VIEWER_READY port=(\d+)")
        
        for line in iter(process.stdout.readline, ""):
            match = ready_pattern.search(line)
            if match:
                port = int(match.group(1))
                break
        
        if port is None:
            print("ERROR: Viewer did not start")
            process.terminate()
            return
        
        # Connect
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(30.0)
        sock.connect(("127.0.0.1", port))
        
        def send_cmd(cmd_dict):
            request = json.dumps(cmd_dict) + "\n"
            sock.sendall(request.encode("utf-8"))
            response = b""
            while b"\n" not in response:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
            return json.loads(response.split(b"\n")[0].decode("utf-8"))
        
        # Set terrain params
        send_cmd({"cmd": "set_terrain", "zscale": z_scale})
        time.sleep(0.5)  # Let terrain load
        
        # Render each frame
        total_frames = anim.get_frame_count(fps)
        start_time = time.time()
        
        for frame in range(total_frames):
            t = frame / fps
            state = anim.evaluate(t)
            
            if state:
                # Set camera
                send_cmd({
                    "cmd": "set_terrain_camera",
                    "phi_deg": state.phi_deg,
                    "theta_deg": state.theta_deg,
                    "radius": state.radius,
                    "fov_deg": state.fov_deg,
                })
                
                # Set sun direction if dynamic lighting enabled
                if dynamic_sun:
                    sun_azimuth, sun_elevation = calculate_sun_direction(state.phi_deg, sun_offset)
                    send_cmd({
                        "cmd": "set_terrain_sun",
                        "azimuth_deg": sun_azimuth,
                        "elevation_deg": sun_elevation,
                        "intensity": sun_intensity,
                    })
                
                # Snapshot
                frame_path = output_dir / f"frame_{frame:04d}.png"
                send_cmd({
                    "cmd": "snapshot",
                    "path": str(frame_path),
                    "width": width,
                    "height": height,
                })
                time.sleep(0.3)  # Wait for snapshot
            
            # Progress
            progress = (frame + 1) / total_frames * 100
            elapsed = time.time() - start_time
            eta = elapsed / (frame + 1) * (total_frames - frame - 1) if frame > 0 else 0
            print(f"\rFrame {frame + 1}/{total_frames} ({progress:.1f}%) ETA: {eta:.1f}s", end="", flush=True)
        
        print(f"\n\nExport complete! Frames saved to: {output_dir}")
        
        send_cmd({"cmd": "close"})
        sock.close()
        process.wait(timeout=5)
        
        # Encode to MP4 if requested
        if create_mp4:
            if mp4_output is None:
                mp4_output = output_dir.with_suffix(".mp4")
            
            success = encode_frames_to_mp4(
                frames_dir=output_dir,
                output_path=mp4_output,
                fps=fps,
                quality=mp4_quality,
                cleanup_frames=cleanup_frames,
                title=title,
            )
            
            if success:
                print(f"\n✓ Animation exported to: {mp4_output}")
            else:
                print(f"\n✗ MP4 encoding failed. Frames are still in: {output_dir}")
        else:
            print(f"\nTo create MP4: ffmpeg -framerate {fps} -i {output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4")
        
    except FileNotFoundError:
        print("ERROR: Could not find interactive_viewer binary.")
        print("Build with: cargo build --release --bin interactive_viewer")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Camera Animation Demo - Keyframe-based camera flyovers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Animation Types:
  orbit    - 360° orbit around terrain (default)
  flyover  - Dramatic zoom and pan sequence
  sunrise  - Slow sunrise reveal animation

Examples:
  # Preview orbit animation in viewer
  python examples/camera_animation_demo.py
  
  # Preview flyover animation
  python examples/camera_animation_demo.py --animation flyover
  
  # Export frames at 60fps
  python examples/camera_animation_demo.py --export ./frames --fps 60
  
  # Export directly to MP4 video
  python examples/camera_animation_demo.py --export ./frames --mp4 --fps 30
  
  # Export MP4 with custom output path and cleanup frames
  python examples/camera_animation_demo.py --export ./frames --mp4 --mp4-output animation.mp4 --cleanup-frames
  
  # Use different terrain
  python examples/camera_animation_demo.py --dem assets/tif/Mount_Fuji_30m.tif
        """,
    )
    
    parser.add_argument(
        "--dem", type=Path, default=DEFAULT_DEM,
        help=f"Path to DEM file (default: {DEFAULT_DEM.name})"
    )
    parser.add_argument(
        "--animation", choices=["orbit", "flyover", "sunrise"], default="orbit",
        help="Animation type (default: orbit)"
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Animation duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--export", type=Path, metavar="DIR",
        help="Export frames to directory (instead of interactive preview)"
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Frames per second for export (default: 30)"
    )
    parser.add_argument(
        "--width", type=int, default=1920,
        help="Output width for export (default: 1920)"
    )
    parser.add_argument(
        "--height", type=int, default=1080,
        help="Output height for export (default: 1080)"
    )
    parser.add_argument(
        "--z-scale", type=float, default=0.15,
        help="Terrain height exaggeration (default: 0.15)"
    )
    parser.add_argument(
        "--preview-only", action="store_true",
        help="Only print interpolation preview, don't launch viewer"
    )
    parser.add_argument(
        "--mp4", action="store_true",
        help="Create MP4 video after frame export (requires ffmpeg)"
    )
    parser.add_argument(
        "--mp4-output", type=Path, metavar="FILE",
        help="MP4 output path (default: <export_dir>.mp4)"
    )
    parser.add_argument(
        "--mp4-quality", choices=["high", "medium", "low"], default="high",
        help="MP4 quality preset (default: high)"
    )
    parser.add_argument(
        "--cleanup-frames", action="store_true",
        help="Delete PNG frames after MP4 creation"
    )
    parser.add_argument(
        "--static-sun", action="store_true",
        help="Disable dynamic sun movement (sun stays fixed)"
    )
    parser.add_argument(
        "--sun-offset", type=float, default=120.0, metavar="DEGREES",
        help="Sun offset angle from camera in degrees (default: 120)"
    )
    parser.add_argument(
        "--sun-intensity", type=float, default=1.0, metavar="FLOAT",
        help="Sun intensity multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--title", type=str, metavar="TEXT",
        help="Title text to overlay at top of video (MP4 export only)"
    )
    
    args = parser.parse_args()
    
    # Validate DEM path
    if not args.dem.exists():
        print(f"ERROR: DEM file not found: {args.dem}")
        sys.exit(1)
    
    # Create animation
    print(f"Camera Animation Demo")
    print(f"=" * 60)
    print(f"DEM: {args.dem.name}")
    print(f"Animation: {args.animation}")
    print(f"Duration: {args.duration}s")
    
    if args.animation == "orbit":
        anim = create_orbit_animation(args.duration)
    elif args.animation == "flyover":
        anim = create_flyover_animation(args.duration)
    else:  # sunrise
        anim = create_sunrise_animation(args.duration)
    
    # Preview interpolation
    preview_animation_interpolation(anim, fps=2)
    
    if args.preview_only:
        return
    
    # Run interactive or export
    if args.export:
        export_animation_frames(
            dem_path=args.dem,
            anim=anim,
            output_dir=args.export,
            fps=args.fps,
            width=args.width,
            height=args.height,
            z_scale=args.z_scale,
            create_mp4=args.mp4,
            mp4_output=args.mp4_output,
            mp4_quality=args.mp4_quality,
            cleanup_frames=args.cleanup_frames,
            dynamic_sun=not args.static_sun,
            sun_offset=args.sun_offset,
            sun_intensity=args.sun_intensity,
            title=args.title,
        )
    else:
        run_interactive_preview(
            dem_path=args.dem,
            anim=anim,
            z_scale=args.z_scale,
            dynamic_sun=not args.static_sun,
            sun_offset=args.sun_offset,
            sun_intensity=args.sun_intensity,
        )


if __name__ == "__main__":
    main()
