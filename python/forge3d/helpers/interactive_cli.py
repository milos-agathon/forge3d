from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time
from typing import Iterable, Optional, Tuple

import numpy as np

try:
    import forge3d as f3d
except Exception as exc:  # pragma: no cover
    raise ImportError("forge3d Python API is required. Ensure package is installed or built.") from exc


def _command_reader(
    cmd_queue: mp.Queue,
    resp_queue: mp.Queue,
    stop_flag: mp.Event,
) -> None:
    """Dedicated process/thread to read interactive commands from stdin.

    Mirrors the behavior used by examples to provide a single-terminal prompt.
    """
    try:
        print("\nInteractive commands ready. Type 'help' for options. Close viewer or type 'quit' to exit.", flush=True)
        while not stop_flag.is_set():
            try:
                # Print prompt and flush explicitly to ensure it appears when running in a child process
                print("forge3d> ", end="", flush=True)
                line = input()
            except EOFError:
                break
            except KeyboardInterrupt:
                print("^C", flush=True)
                continue

            if line is None:
                continue

            line = line.strip()
            if not line:
                continue

            try:
                cmd_queue.put(line, timeout=0.1)
            except Exception:
                break

            # Try to get a response with a timeout so we can provide helpful
            # feedback even if the control loop isn't running yet.
            try:
                response = resp_queue.get(timeout=2.0)
            except EOFError:
                break
            except queue.Empty:
                # If the control loop hasn't started, provide a helpful fallback.
                lower = line.lower()
                if lower in {"help", "h"}:
                    fallback = [
                        "\nAvailable commands:",
                        "  camera [distance] [theta] [phi] | distance=<d> theta=<t> phi=<p>",
                        "      Set absolute camera parameters (leave blank to keep current value)",
                        "  orbit <dtheta> <dphi>",
                        "      Rotate camera incrementally (degrees)",
                        "  zoom <delta>",
                        "      Adjust camera distance (negative delta zooms in)",
                        "  snapshot <path> [width] [height]",
                        "      Save PNG snapshot (defaults to 1920x1080)",
                        "  get_camera",
                        "      Print current camera state",
                        "  raytrace <output> [width] [height] [frames]",
                        "      Path trace current view to PNG",
                        "  help | quit",
                        "      Show this help or exit",
                        "\nNote: Viewer is still initializing; some commands will run once ready.",
                    ]
                    for item in fallback:
                        print(item, flush=True)
                    continue
                else:
                    print("Viewer still initializing; command queued. It will run once ready.", flush=True)
                    continue
            except Exception:
                response = None

            if response is None:
                break

            if isinstance(response, Iterable) and not isinstance(response, (str, bytes)):
                for item in response:
                    print(item, flush=True)
            else:
                text = "" if response is None else str(response)
                if text:
                    print(text, flush=True)
    finally:
        stop_flag.set()


def _wait_for_viewer_ready(max_wait: float = 10.0, poll_interval: float = 0.2) -> bool:
    """Poll until the native viewer is ready to accept camera commands."""
    deadline = time.monotonic() + max(float(max_wait), 0.0)
    last_error: Optional[Exception] = None

    while True:
        try:
            f3d.get_camera()
            return True
        except Exception as exc:
            last_error = exc
            if time.monotonic() >= deadline:
                break
            time.sleep(max(float(poll_interval), 0.01))

    if last_error is not None:
        print(f"Warning: Viewer not ready yet ({last_error}). Commands may fail until initialization completes.")
    return False


def interactive_control_loop(
    dem_path: Optional[str] = None,
    dem_data: Optional[np.ndarray] = None,
    spacing: Optional[Tuple[float, float]] = None,
    *,
    raytrace_helper: Optional[object] = None,
    default_size: Tuple[int, int] = (1920, 1080),
    stop_event: Optional[threading.Event] = None,
    command_queue: Optional[mp.Queue] = None,
    response_queue: Optional[mp.Queue] = None,
    command_stop: Optional[mp.Event] = None,
) -> None:
    """Interactive command loop for viewer control.

    Exposes commands: camera, orbit, zoom, snapshot, get_camera, raytrace, help, quit
    """
    print("\n" + "=" * 60, flush=True)
    print("INTERACTIVE VIEWER CONTROL", flush=True)
    print("=" * 60, flush=True)
    print("The 3D viewer is running. You can interact with it using mouse/keyboard.", flush=True)
    print("Type commands below to control camera and capture snapshots.", flush=True)
    print("Type 'help' for available commands or 'quit' to exit.", flush=True)
    print("=" * 60 + "\n", flush=True)

    viewer_ready = False
    print("Viewer still launching... commands will retry until the viewer is ready.", flush=True)

    while True:
        if stop_event and stop_event.is_set():
            print("\nViewer closed. Ending interactive control session.")
            break
        try:
            if command_queue is not None:
                try:
                    cmd_input = command_queue.get(timeout=0.25)
                except queue.Empty:
                    continue
            else:
                cmd_input = input("forge3d> ")

            if cmd_input is None:
                break

            cmd_input = str(cmd_input).strip()
            if not cmd_input:
                if response_queue is not None:
                    response_queue.put(())
                continue

            parts = cmd_input.split()
            cmd = parts[0].lower()

            messages: list[str] = []

            def emit(msg: str = "") -> None:
                messages.append(str(msg))

            def ensure_viewer_ready(timeout: float = 6.0) -> bool:
                nonlocal viewer_ready
                if viewer_ready:
                    return True
                viewer_ready = _wait_for_viewer_ready(max_wait=timeout)
                if viewer_ready:
                    emit("Viewer ready. Commands are now live.")
                else:
                    emit("Viewer still initialising; please retry in a moment.")
                return viewer_ready

            if cmd in {"quit", "exit", "q"}:
                emit("Exiting...")
                if stop_event:
                    stop_event.set()
                if command_stop is not None:
                    command_stop.set()
                if command_queue is not None:
                    try:
                        command_queue.put(None, timeout=0.1)
                    except Exception:
                        pass
                if response_queue is not None:
                    response_queue.put(messages)
                else:
                    for msg in messages:
                        print(msg)
                break

            elif cmd in {"help", "h"}:
                emit("\nAvailable commands:")
                emit("  camera [distance] [theta] [phi] | distance=<d> theta=<t> phi=<p>")
                emit("      Set absolute camera parameters (leave blank to keep current value)")
                emit("  orbit <dtheta> <dphi>")
                emit("      Rotate camera incrementally (degrees)")
                emit("  zoom <delta>")
                emit("      Adjust camera distance (negative delta zooms in)")
                emit("  snapshot <path> [width] [height]")
                emit("      Save PNG snapshot (defaults to 1920x1080)")
                emit("  get_camera")
                emit("      Print current camera state")
                emit("  raytrace <output> [width] [height] [frames]")
                emit("      Path trace current view to PNG")
                emit("  help | quit")
                emit("      Show this help or exit")
                emit("\nExamples:")
                emit("  camera 2000 45 30")
                emit("  camera distance=2000 theta=45 phi=30")
                emit("  orbit 10 20")
                emit("  zoom -100")
                emit("  snapshot my_view.png 1920 1080")
                emit("  get_camera")
                if response_queue is not None:
                    response_queue.put(messages)
                else:
                    for msg in messages:
                        print(msg)
                continue

            elif cmd in {"camera", "cam"}:
                if len(parts) == 1:
                    emit("Usage: camera [distance] [theta] [phi] | distance=<d> theta=<t> phi=<p>")
                    if response_queue is not None:
                        response_queue.put(messages)
                    else:
                        for msg in messages:
                            print(msg)
                    continue

                named: dict[str, float] = {}
                positional: list[float] = []
                parse_error = False
                for token in parts[1:]:
                    try:
                        if "=" in token:
                            key, value = token.split("=", 1)
                            named[key.strip().lower()] = float(value)
                        else:
                            positional.append(float(token))
                    except ValueError:
                        emit(f"Error: could not parse value '{token}'")
                        parse_error = True
                        break
                if parse_error:
                    if response_queue is not None:
                        response_queue.put(messages)
                    else:
                        for msg in messages:
                            print(msg)
                    continue

                distance: Optional[float] = None
                theta: Optional[float] = None
                phi: Optional[float] = None

                if positional:
                    try:
                        if len(positional) >= 1:
                            distance = positional[0]
                        if len(positional) >= 2:
                            theta = positional[1]
                        if len(positional) >= 3:
                            phi = positional[2]
                        if len(positional) > 3:
                            raise ValueError("Too many positional values")
                    except ValueError as exc:
                        emit(f"Error: {exc}")
                        if response_queue is not None:
                            response_queue.put(messages)
                        else:
                            for msg in messages:
                                print(msg)
                        continue

                distance = named.get("distance", distance)
                theta = named.get("theta", theta)
                phi = named.get("phi", phi)

                if distance is None and theta is None and phi is None:
                    emit("No camera values provided; use distance/theta/phi")
                    if response_queue is not None:
                        response_queue.put(messages)
                    else:
                        for msg in messages:
                            print(msg)
                    continue

                if not viewer_ready and not ensure_viewer_ready(timeout=6.0):
                    if response_queue is not None:
                        response_queue.put(messages)
                    else:
                        for msg in messages:
                            print(msg)
                    continue

                try:
                    f3d.set_camera(
                        distance=float(distance) if distance is not None else None,
                        theta=float(theta) if theta is not None else None,
                        phi=float(phi) if phi is not None else None,
                    )
                    current = f3d.get_camera()
                    emit(
                        "✓ Camera set: distance={:.1f}, theta={:.1f}°, phi={:.1f}°".format(
                            current["distance"], current["theta"], current["phi"],
                        )
                    )
                except Exception as e:
                    emit(f"Error: {e}")
                if response_queue is not None:
                    response_queue.put(messages)
                else:
                    for msg in messages:
                        print(msg)
                continue

            elif cmd in {"orbit", "rotate"}:
                if len(parts) != 3:
                    emit("Usage: orbit <dtheta> <dphi>")
                    if response_queue is not None:
                        response_queue.put(messages)
                    else:
                        for msg in messages:
                            print(msg)
                    continue
                try:
                    dtheta = float(parts[1])
                    dphi = float(parts[2])
                    if not ensure_viewer_ready(timeout=6.0):
                        if response_queue is not None:
                            response_queue.put(messages)
                        else:
                            for msg in messages:
                                print(msg)
                        continue
                    cam = f3d.get_camera()
                    new_theta = cam["theta"] + dtheta
                    new_phi = float(np.clip(cam["phi"] + dphi, -89.5, 89.5))
                    f3d.set_camera(theta=new_theta, phi=new_phi)
                    emit(f"✓ Orbit: theta={new_theta:.1f}°, phi={new_phi:.1f}°")
                except Exception as e:
                    emit(f"Error: {e}")
                if response_queue is not None:
                    response_queue.put(messages)
                else:
                    for msg in messages:
                        print(msg)
                continue

            elif cmd == "zoom":
                if len(parts) != 2:
                    emit("Usage: zoom <delta>  (negative delta zooms in)")
                    if response_queue is not None:
                        response_queue.put(messages)
                    else:
                        for msg in messages:
                            print(msg)
                    continue
                try:
                    delta = float(parts[1])
                    if not ensure_viewer_ready(timeout=6.0):
                        if response_queue is not None:
                            response_queue.put(messages)
                        else:
                            for msg in messages:
                                print(msg)
                        continue
                    cam = f3d.get_camera()
                    new_distance = float(max(1.0, cam["distance"] + delta))
                    f3d.set_camera(distance=new_distance)
                    direction = "in" if delta < 0 else "out"
                    emit(f"✓ Zoomed {direction}: distance={new_distance:.1f}")
                except Exception as e:
                    emit(f"Error: {e}")
                if response_queue is not None:
                    response_queue.put(messages)
                else:
                    for msg in messages:
                        print(msg)
                continue

            elif cmd in {"snapshot", "snap", "capture"}:
                if len(parts) < 2:
                    emit("Usage: snapshot <path> [width] [height]")
                    if response_queue is not None:
                        response_queue.put(messages)
                    else:
                        for msg in messages:
                            print(msg)
                    continue
                try:
                    path = parts[1]
                    width = int(parts[2]) if len(parts) > 2 else 1920
                    height = int(parts[3]) if len(parts) > 3 else 1080
                    if ensure_viewer_ready(timeout=6.0):
                        f3d.snapshot(path, width=width, height=height)
                        emit(f"✓ Snapshot saved: {path} ({width}x{height})")
                except Exception as e:
                    emit(f"Error: {e}")
                if response_queue is not None:
                    response_queue.put(messages)
                else:
                    for msg in messages:
                        print(msg)
                continue

            elif cmd in {"get_camera", "getcam", "cam_info"}:
                try:
                    if not ensure_viewer_ready(timeout=3.0):
                        if response_queue is not None:
                            response_queue.put(messages)
                        else:
                            for msg in messages:
                                print(msg)
                        continue
                    cam = f3d.get_camera()
                    emit("\nCamera State:")
                    emit(f"  Position (eye):  [{cam['eye'][0]:.1f}, {cam['eye'][1]:.1f}, {cam['eye'][2]:.1f}]")
                    emit(f"  Target:          [{cam['target'][0]:.1f}, {cam['target'][1]:.1f}, {cam['target'][2]:.1f}]")
                    emit(f"  Distance:        {cam['distance']:.1f}")
                    emit(f"  Theta (yaw):     {cam['theta']:.1f}°")
                    emit(f"  Phi (pitch):     {cam['phi']:.1f}°")
                    emit(f"  FOV:             {cam['fov']:.1f}°")
                except Exception as e:
                    emit(f"Error: {e}")
                if response_queue is not None:
                    response_queue.put(messages)
                else:
                    for msg in messages:
                        print(msg)
                continue

            elif cmd in {"raytrace", "rt"}:
                if len(parts) < 2:
                    emit("Usage: raytrace <output> [width] [height] [frames]")
                    if response_queue is not None:
                        response_queue.put(messages)
                    else:
                        for msg in messages:
                            print(msg)
                    continue
                if raytrace_helper is None:
                    emit("Raytracing unavailable: DEM mesh not initialized.")
                    if response_queue is not None:
                        response_queue.put(messages)
                    else:
                        for msg in messages:
                            print(msg)
                    continue

                output_path = parts[1]
                try:
                    width = int(parts[2]) if len(parts) > 2 else int(default_size[0])
                    height = int(parts[3]) if len(parts) > 3 else int(default_size[1])
                    frames = int(parts[4]) if len(parts) > 4 else 8
                    if width <= 0 or height <= 0:
                        raise ValueError("width/height must be positive")
                    if frames <= 0:
                        raise ValueError("frames must be positive")
                except ValueError as exc:
                    emit(f"Error: {exc}")
                    if response_queue is not None:
                        response_queue.put(messages)
                    else:
                        for msg in messages:
                            print(msg)
                    continue

                try:
                    if not ensure_viewer_ready(timeout=6.0):
                        if response_queue is not None:
                            response_queue.put(messages)
                        else:
                            for msg in messages:
                                print(msg)
                        continue
                    vertices, indices = raytrace_helper.get_mesh()
                    cam_state = f3d.get_camera()
                    eye = np.asarray(cam_state["eye"], dtype=np.float32)
                    target = np.asarray(cam_state["target"], dtype=np.float32)
                    up_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)

                    # Avoid degenerate up vector when camera is near poles
                    forward = target - eye
                    if np.linalg.norm(forward) > 1e-6:
                        forward_n = forward / (np.linalg.norm(forward) + 1e-8)
                        if abs(float(np.dot(forward_n, up_vec))) > 0.98:
                            up_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)

                    aspect = float(width) / float(height)
                    camera = {
                        "origin": tuple(map(float, eye.tolist())),
                        "look_at": tuple(map(float, target.tolist())),
                        "up": tuple(map(float, up_vec.tolist())),
                        "fov_y": float(cam_state.get("fov", 45.0)),
                        "aspect": aspect,
                        "exposure": 1.0,
                    }

                    image, meta = f3d.render_raytrace_mesh(
                        (vertices, indices),
                        size=(int(width), int(height)),
                        frames=int(frames),
                        seed=int(seed),
                        sampling_mode=sampling_mode,
                        debug_mode=int(debug_mode) if isinstance(debug_mode, (int, float)) else 0,
                        normalize=False,
                        preview=False,
                        outfile=None,
                        verbose=False,
                    )

                    f3d.numpy_to_png(output_path, image)
                    engine = "GPU" if meta.get("gpu_used") else "CPU fallback"
                    emit(
                        f"✓ Raytraced {output_path} ({width}x{height}, frames={frames}) via {engine}."
                    )
                except Exception as e:
                    emit(f"Raytrace failed: {e}")
                if response_queue is not None:
                    response_queue.put(messages)
                else:
                    for msg in messages:
                        print(msg)
                continue

            # Unknown command
            emit(f"Unknown command: {cmd}. Type 'help' for available commands.")
            if response_queue is not None:
                response_queue.put(messages)
            else:
                for msg in messages:
                    print(msg)

        except KeyboardInterrupt:
            if response_queue is not None:
                response_queue.put(["Interrupted. Type 'quit' to exit."])
            else:
                print("\nInterrupted. Type 'quit' to exit.")
        except EOFError:
            if response_queue is not None:
                response_queue.put(["Exiting..."])
            else:
                print("\nExiting...")
            break
        except Exception as e:
            if response_queue is not None:
                response_queue.put([f"Error: {e}"])
            else:
                print(f"Error: {e}")


def demo_snapshots(output_dir, width: int = 1920, height: int = 1080) -> None:
    views = [
        {"name": "North", "distance": 2000, "theta": 0, "phi": 30, "delay": 1.5},
        {"name": "Northeast", "distance": 2000, "theta": 45, "phi": 30, "delay": 1.5},
        {"name": "East", "distance": 2000, "theta": 90, "phi": 30, "delay": 1.5},
        {"name": "Southeast", "distance": 2000, "theta": 135, "phi": 30, "delay": 1.5},
        {"name": "South", "distance": 2000, "theta": 180, "phi": 30, "delay": 1.5},
        {"name": "Southwest", "distance": 2000, "theta": 225, "phi": 30, "delay": 1.5},
        {"name": "West", "distance": 2000, "theta": 270, "phi": 30, "delay": 1.5},
        {"name": "Northwest", "distance": 2000, "theta": 315, "phi": 30, "delay": 1.5},
        {"name": "Overhead", "distance": 3000, "theta": 0, "phi": 85, "delay": 1.5},
        {"name": "Close", "distance": 1000, "theta": 45, "phi": 20, "delay": 1.5},
    ]

    print(f"\nCapturing {len(views)} views at {width}x{height}...\n")

    for i, view in enumerate(views, 1):
        print(f"[{i}/{len(views)}] {view['name']} View")
        print(f"        Setting camera: distance={view['distance']}, theta={view['theta']}°, phi={view['phi']}°")

        # Set camera position
        f3d.set_camera(distance=view['distance'], theta=view['theta'], phi=view['phi'])

        # Wait for camera to update
        time.sleep(view['delay'])

        # Get current camera state (demonstrates get_camera)
        cam = f3d.get_camera()
        print(f"        Camera at: [{cam['eye'][0]:.1f}, {cam['eye'][1]:.1f}, {cam['eye'][2]:.1f}]")

        # Capture snapshot
        output_path = output_dir / f"{i:02d}_{view['name'].lower()}_view.png"
        f3d.snapshot(str(output_path), width=width, height=height)
        print(f"        Saved: {output_path}")
        print()

    print("=" * 60)
    print(f"✓ Demo complete! {len(views)} snapshots saved to: {output_dir}")
    print("=" * 60)
