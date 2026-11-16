"""Build the isolated C07 level, evaluate Sequencer and run its legacy MRQ preset."""

import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from time import perf_counter
import traceback

import unreal


ASSET_API = unreal.get_editor_subsystem(unreal.EditorAssetSubsystem)
ACTORS = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
LEVELS = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
MOVIE = unreal.MoviePipelineLibrary


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def xyz(value):
    return [value.x, value.y, value.z]


def asset(name, cls, factory):
    result = unreal.AssetToolsHelpers.get_asset_tools().create_asset(name, ASSETS, cls, factory)
    if result is None:
        raise RuntimeError(f"cannot create {name}")
    return result


def require(value, message):
    if not value:
        raise RuntimeError(message)
    return value


def project_assets(sequences, presets):
    root = Path(unreal.SystemLibrary.get_project_directory())
    paths = [root / "C07Acceptance.uproject", root / "Config/DefaultEngine.ini"]
    assets = {LEVEL, f"{ASSETS}/SwissHeight"}
    assets.update(value.get_path_name().split(".")[0] for value in (*sequences.values(), *presets.values()))
    for path in sorted(assets):
        relative = path.removeprefix("/Game/")
        extension = ".umap" if path == LEVEL else ".uasset"
        paths.append(root / "Content" / (relative + extension))
    return {path.relative_to(root).as_posix(): digest(path) for path in paths}


def terrain_material():
    material = asset("SwissHeight", unreal.Material, unreal.MaterialFactoryNew())
    material.set_editor_property("two_sided", True)
    material.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    lib = unreal.MaterialEditingLibrary
    vertex = lib.create_material_expression(material, unreal.MaterialExpressionVertexColor, -200, 0)
    require(lib.connect_material_property(vertex, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR),
            "cannot connect terrain vertex colours")
    lib.recompile_material(material)
    require(ASSET_API.save_loaded_asset(material), "cannot save terrain material")
    return material


def import_terrain(kit, job, meta):
    reg = job["registration"]
    require(reg["origin_enu_m"] == meta["mesh_origin_e_n_z_m"]
            and reg["rotation_neu_to_world"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            and reg["translation_cm"] == [0, 0, 0], "unexpected isolated project registration")
    with (kit / "terrain-north-first.csv").open(newline="", encoding="utf-8") as stream:
        heights = [[float(value) for value in row] for row in csv.reader(stream)]
    west, south, east, north = meta["mesh_bounds_wsen"]
    oe, on, oz = reg["origin_enu_m"]
    rows, cols = len(heights), len(heights[0])
    de, dn = (east - west) / (cols - 1), (north - south) / (rows - 1)
    lo, hi = meta["height_domain_m"]
    vertices, colours, uv = [], [], []
    for r, row in enumerate(heights):
        for c, z in enumerate(row):
            vertices.append(unreal.Vector((north - r * dn - on) * 100, (west + c * de - oe) * 100, (z - oz) * 100))
            uv.append(unreal.Vector2D(c / (cols - 1), r / (rows - 1)))
            # Fixed height tint and directional relief for the technical fixture.
            a = (z - lo) / (hi - lo)
            slope = (row[min(c + 1, cols - 1)] - heights[max(r - 1, 0)][c]) / de
            light = 0.62 + 0.38 * (0.5 + 0.5 * math.tanh(slope))
            colours.append(unreal.LinearColor(light * (0.05 + 0.68 * a),
                                              light * (0.13 + 0.57 * a),
                                              light * (0.07 + 0.56 * a), 1))
    triangles = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            nw = r * cols + c
            ne, sw, se = nw + 1, nw + cols, nw + cols + 1
            triangles.extend((unreal.IntVector(sw, nw, se), unreal.IntVector(ne, se, nw)))
    buffers = unreal.GeometryScriptSimpleMeshBuffers()
    buffers.set_editor_properties({"vertices": vertices, "triangles": triangles, "vertex_colors": colours,
                                   "uv0": uv, "normals": [unreal.Vector(0, 0, 1)] * len(vertices)})
    actor = ACTORS.spawn_actor_from_class(unreal.DynamicMeshActor, unreal.Vector())
    actor.set_actor_label("swissALTI3D 2025 LV95 LN02")
    component = actor.get_dynamic_mesh_component()
    mesh = component.get_dynamic_mesh()
    unreal.GeometryScript_MeshEdits.append_buffers_to_mesh(mesh, buffers)
    require(mesh.get_triangle_count() == len(triangles), "Unreal terrain triangle count differs from source")
    component.configure_material_set([terrain_material()], True)
    return actor, len(vertices), len(triangles)


def register_landmarks(jobs, meta, terrain):
    oe, on, oz = meta["mesh_origin_e_n_z_m"]
    mesh = terrain.get_dynamic_mesh_component().get_dynamic_mesh()
    mesh_spatial = unreal.GeometryScript_MeshSpatial
    _, bvh = mesh_spatial.build_bvh_for_mesh(mesh)
    transform = terrain.get_actor_transform()
    references = {p["feature_id"]: p for job in jobs for p in job["landmarks"]}
    features = meta["shots"]["eiger"]["features"]
    probes = {}
    for key, row in features.items():
        e, n, h = row["easting_northing_height_m"]
        marker = ACTORS.spawn_actor_from_class(unreal.TargetPoint,
                                               unreal.Vector((n - on) * 100, (e - oe) * 100, (h - oz) * 100))
        marker.set_actor_label(row["name"] + " " + key)
        actual = marker.get_actor_location()
        probe = {"source_enu_m": [e, n, h], "target_point_cm": xyz(actual)}
        if key in references:
            expected = references[key]["world_cm"]
            above = unreal.Vector(actual.x, actual.y, (meta["height_domain_m"][1] - oz) * 100 + 100)
            local_origin = unreal.MathLibrary.inverse_transform_location(transform, above)
            local_dir = unreal.MathLibrary.inverse_transform_direction(transform, unreal.Vector(0, 0, -1))
            _, hit, _ = mesh_spatial.find_nearest_ray_intersection_with_mesh(
                mesh, bvh, local_origin, local_dir, unreal.GeometryScriptSpatialQueryOptions())
            require(hit.hit, f"terrain ray missed {row['name']}")
            point = unreal.MathLibrary.transform_location(transform, hit.hit_position)
            probe.update({"reference_terrain_cm": expected, "trace_cm": xyz(point),
                          "trace_height_residual_cm": point.z - expected[2]})
        probes[key] = probe
    return probes


def add_keys(channel, values):
    for index, value in enumerate(values):
        channel.add_key(unreal.FrameNumber(index), value, interpolation=unreal.MovieSceneKeyInterpolation.LINEAR)


def camera_sequence(job, name):
    count = job["frame_count"]
    sequence = asset(name, unreal.LevelSequence, unreal.LevelSequenceFactoryNew())
    sequence.set_display_rate(unreal.FrameRate(job["fps"], 1))
    sequence.set_playback_start(0)
    sequence.set_playback_end(count)
    camera = ACTORS.spawn_actor_from_class(unreal.CineCameraActor, unreal.Vector(*job["frames"][0]["eye_cm"]))
    camera.set_actor_label(f"C07 {name} Cine Camera")
    lens = camera.get_cine_camera_component()
    w, h = job["filmback_mm"]
    lengths = [f["focal_length_mm"] for f in job["frames"]]
    optics = lens.get_editor_property("lens_settings")
    optics.set_editor_properties({"min_focal_length": min(lengths), "max_focal_length": max(lengths)})
    lens.set_editor_properties({
        "filmback": unreal.CameraFilmbackSettings(sensor_width=w, sensor_height=h),
        "lens_settings": optics,
        "focus_settings": unreal.CameraFocusSettings(focus_method=unreal.CameraFocusMethod.DISABLE),
        "constrain_aspect_ratio": True,
        "override_custom_near_clipping_plane": True,
        "custom_near_clipping_plane": job["frames"][0]["reference_clip_cm"][0],
        "current_focal_length": job["frames"][0]["focal_length_mm"],
    })
    actor_binding = sequence.add_possessable(camera)
    transforms = actor_binding.add_track(unreal.MovieScene3DTransformTrack).add_section()
    transforms.set_range(0, count)
    rotation_keys = []
    previous = None
    for frame in job["frames"]:
        rot = unreal.MathLibrary.make_rot_from_xz(unreal.Vector(*frame["forward"]), unreal.Vector(*frame["up"]))
        angles = (rot.roll, rot.pitch, rot.yaw)
        if previous is not None:
            angles = tuple(old + math.remainder(new - old, 360) for old, new in zip(previous, angles))
        rotation_keys.append([*frame["eye_cm"], *angles])
        previous = angles
    for channel, values in zip(transforms.get_all_channels()[:6], zip(*rotation_keys), strict=True):
        add_keys(channel, values)
    component_binding = sequence.add_possessable(lens)
    component_binding.set_parent(actor_binding)
    focal_track = component_binding.add_track(unreal.MovieSceneFloatTrack)
    focal_track.set_property_name_and_path("CurrentFocalLength", "CurrentFocalLength")
    focal = focal_track.add_section()
    focal.set_range(0, count)
    add_keys(focal.get_all_channels()[0], [f["focal_length_mm"] for f in job["frames"]])
    cut = sequence.add_track(unreal.MovieSceneCameraCutTrack).add_section()
    cut.set_range(0, count)
    cut.set_camera_binding_id(sequence.get_binding_id(actor_binding))
    require(ASSET_API.save_loaded_asset(sequence), f"cannot save sequence {name}")
    return sequence


def camera_sample(lens):
    transform = lens.get_world_transform()
    rot = transform.rotation.rotator()
    return {"eye_cm": xyz(transform.translation),
            "forward": xyz(unreal.MathLibrary.get_forward_vector(rot)),
            "right": xyz(unreal.MathLibrary.get_right_vector(rot)),
            "up": xyz(unreal.MathLibrary.get_up_vector(rot)),
            "rotation_deg": [rot.roll, rot.pitch, rot.yaw],
            "filmback_mm": [lens.filmback.sensor_width, lens.filmback.sensor_height],
            "focal_length_mm": lens.current_focal_length,
            "horizontal_fov_deg": lens.get_horizontal_field_of_view()}


def configure_colour(config, neutral):
    if neutral:
        setting = config.find_or_add_setting_by_class(unreal.MoviePipelineColorSetting)
        setting.set_editor_property("disable_tone_curve", True)
        require(setting.get_editor_property("disable_tone_curve"), "cannot disable the filmic tone curve")
        require(ASSET_API.save_loaded_asset(config), "cannot save the neutral render preset")


def configure_neutral_exposure():
    if not NEUTRAL:
        return None
    label = "C07 neutral exposure"
    volume = next((actor for actor in ACTORS.get_all_level_actors()
                   if isinstance(actor, unreal.PostProcessVolume) and actor.get_actor_label() == label), None)
    if volume is None:
        volume = ACTORS.spawn_actor_from_class(unreal.PostProcessVolume, unreal.Vector())
        volume.set_actor_label(label)
    values = {"auto_exposure_method": unreal.AutoExposureMethod.AEM_MANUAL,
              "auto_exposure_apply_physical_camera_exposure": False,
              "auto_exposure_bias": 0.0, "bloom_intensity": 0.0, "vignette_intensity": 0.0,
              "local_exposure_highlight_contrast_scale": 1.0, "local_exposure_shadow_contrast_scale": 1.0}
    settings = volume.get_editor_property("settings")
    for name, value in values.items():
        settings.set_editor_property("override_" + name, True)
        settings.set_editor_property(name, value)
    volume.set_editor_properties({"unbound": True, "settings": settings})
    require(LEVELS.save_current_level(), "cannot save fixed neutral exposure")
    return {**values, "auto_exposure_method": "manual EV100 0", "unbound": True}


def create_preset(shot, neutral):
    name = shot["registration"]["preset_id"] + "_" + shot["shot"]
    config = asset(name, unreal.MoviePipelinePrimaryConfig, unreal.MoviePipelinePrimaryConfigFactory())
    output = config.find_or_add_setting_by_class(unreal.MoviePipelineOutputSetting)
    output.set_editor_properties({
        "output_resolution": unreal.IntPoint(*shot["size"]),
        "file_name_format": "frame_{frame_number_rel}",
        "zero_pad_frame_numbers": 8, "override_existing_output": False,
        "use_custom_frame_rate": True, "output_frame_rate": unreal.FrameRate(shot["fps"], 1),
    })
    aa = config.find_or_add_setting_by_class(unreal.MoviePipelineAntiAliasingSetting)
    aa.set_editor_properties({
        "temporal_sample_count": 1, "spatial_sample_count": 1,
        "override_anti_aliasing": True, "anti_aliasing_method": unreal.AntiAliasingMethod.AAM_NONE,
        "engine_warm_up_count": 0, "render_warm_up_count": 0,
    })
    camera = config.find_or_add_setting_by_class(unreal.MoviePipelineCameraSetting)
    camera.set_editor_property("shutter_timing", unreal.MoviePipelineShutterTiming.FRAME_CENTER)
    config.find_or_add_setting_by_class(unreal.MoviePipelineDeferredPassBase)
    config.find_or_add_setting_by_class(unreal.MoviePipelineImageSequenceOutput_PNG)
    configure_colour(config, neutral)
    require(ASSET_API.save_loaded_asset(config), f"cannot save render preset {name}")
    return config


def observe(_):
    if not CURRENT or not CURRENT.get("executor"):
        return
    executor = CURRENT["executor"]
    try:
        for pipeline in unreal.ObjectIterator(unreal.MoviePipeline):
            if MOVIE.get_pipeline_state(pipeline) != unreal.MovieRenderPipelineState.PRODUCING_FRAMES:
                continue
            if MOVIE.get_current_segment_state(pipeline) != unreal.MovieRenderShotState.RENDERING:
                continue
            frame = MOVIE.get_root_frame_number(pipeline).value
            if frame in CURRENT["samples"]:
                continue
            for player in unreal.ObjectIterator(unreal.LevelSequencePlayer):
                if "UEDPIE_" not in player.get_path_name():
                    continue
                camera = player.get_active_camera_component()
                if not isinstance(camera, unreal.CineCameraComponent):
                    continue
                time = player.get_current_time()
                pose = camera_sample(camera)
                pose.update({"mrq_root_frame": frame, "sequence_frame": time.time.frame_number.value,
                             "sequence_sub_frame": time.time.sub_frame,
                             "sequence_rate": [time.rate.numerator, time.rate.denominator]})
                CURRENT["samples"][frame] = pose
                if CURRENT["mode"] == "cancel":
                    CURRENT["cancel_requested"] = True
                    executor.cancel_current_job()
                break
    except Exception:
        if CURRENT.get("executor") == executor:
            CURRENT["error"] = traceback.format_exc()
            executor.cancel_all_jobs()


def finished_work(data):
    CURRENT["job_finished"] = data.success
    CURRENT["outputs"] = [str(path) for shot in data.shot_data
                          for passes in shot.render_pass_data.values() for path in passes.file_paths]


def executor_error(_executor, _pipeline, fatal, message):
    CURRENT["errors"].append({"fatal": fatal, "message": str(message)})


def finish(executor, ok):
    current = CURRENT
    if current.get("executor") != executor or current.get("terminal"):
        return
    current["terminal"] = True
    job, folder = current["job"], current["folder"]
    result = {
        "run_id": folder.name, "job_sha256": digest(current["job_path"]),
        "renderer": {"id": "unreal-mrq-external", "engine_version": unreal.SystemLibrary.get_engine_version()},
        "sequence": current["sequence_path"], "registration": job["registration"],
        "preset": current["preset_path"],
        "settings": {"temporal_samples": 1, "spatial_samples": 1, "shutter": "frame centre",
                     "motion_blur": False, "aa": "none", "focus": "disabled", "overscan": 0,
                     "interpolation": "linear channels with unwrapped rotation", "png": "sRGB",
                     "filmic_tone_curve_disabled": current["filmic_tone_curve_disabled"],
                     "exposure": "manual EV100 0" if NEUTRAL else "project auto-exposure disabled",
                     "console_overrides": []},
        "executor_success": ok, "job_finished": current.get("job_finished"), "executor_errors": current["errors"],
        "input_sequence_exists": current["input_sequence_exists"],
        "scene_sha256": digest(BASE / "scene.json"),
        "render_seconds": perf_counter() - current["started"], "status": "error",
    }
    try:
        write(folder / "samples.json", list(current["samples"].values()))
        result["evaluated_samples_sha256"] = digest(folder / "samples.json")
        if current.get("error"):
            raise RuntimeError(current["error"])
        if current.get("cancel_requested"):
            result.update(status="cancelled", diagnostic="PIE executor cancel_current_job called during rendering")
        else:
            require(current["input_sequence_exists"] or current.get("job_finished"),
                    f"MRQ completed without work for missing sequence {current['sequence_path']}")
            require(ok and current.get("job_finished") and not current["errors"],
                    f"MRQ did not report successful work: {current['errors']}")
            paths = [Path(p).resolve() for p in current.get("outputs", [])]
            expected = [folder / f"frame_{i:08d}.png" for i in range(job["frame_count"])]
            require(sorted(paths) == expected and sorted(folder.glob("*.png")) == expected,
                    "MRQ did not return the exact frame set")
            require(sorted(current["samples"]) == list(range(job["frame_count"])),
                    "MRQ did not yield an evaluated camera sample for each output frame")
            result["frames"] = [{"index": i, "file": p.name, "sha256": digest(p)} for i, p in enumerate(expected)]
            result["status"] = "succeeded"
    except Exception as exc:
        result["diagnostic"] = str(exc)
    write(folder / "result.json", result)
    current["executor"] = None
    unreal.log(f"C07 {folder.name}: {result['status']}")


def run_next(_):
    if CURRENT.get("executor") or QUEUE.is_rendering():
        return
    try:
        item = next(RUNS, None)
        if item is None:
            unreal.unregister_slate_post_tick_callback(advance_handle)
            unreal.unregister_slate_post_tick_callback(sample_handle)
            write(BASE / "queue-finished.json", {"status": "finished"})
            return
        mode, job_path, folder, sequence, preset = item
        folder.mkdir()
        sequence_path = f"{ASSETS}/MissingSequence.MissingSequence" if mode == "error" else sequence.get_path_name()
        CURRENT.clear()
        CURRENT.update(mode=mode, job_path=job_path, job=read(job_path), folder=folder,
                       samples={}, errors=[], sequence_path=sequence_path, preset_path=preset.get_path_name(),
                       input_sequence_exists=ASSET_API.does_asset_exist(sequence_path))
        queue = QUEUE.get_queue()
        queue.delete_all_jobs()
        job = queue.allocate_new_job(unreal.MoviePipelineExecutorJob)
        job.job_name = folder.name
        job.sequence = unreal.SoftObjectPath(sequence_path)
        job.map = unreal.SoftObjectPath(f"{LEVEL}.SwissTerrain")
        job.set_configuration(preset)
        config = job.get_configuration()
        colour = config.find_setting_by_class(unreal.MoviePipelineColorSetting)
        CURRENT["filmic_tone_curve_disabled"] = bool(colour and colour.disable_tone_curve)
        require(CURRENT["filmic_tone_curve_disabled"] == NEUTRAL, "job colour setting differs from request")
        output = config.find_setting_by_class(unreal.MoviePipelineOutputSetting)
        output.set_editor_property("output_directory", unreal.DirectoryPath(path=folder.as_posix()))
        config.initialize_transient_settings()
        executor = unreal.MoviePipelinePIEExecutor()
        executor.set_is_rendering_offscreen(True)
        executor.on_executor_finished_delegate.add_callable_unique(finish)
        executor.on_executor_errored_delegate.add_callable_unique(executor_error)
        executor.on_individual_job_work_finished_delegate.add_callable_unique(finished_work)
        CURRENT.update(executor=executor, started=perf_counter())
        QUEUE.render_queue_with_executor_instance(executor)
    except Exception:
        write(BASE / "queue-failure.json", {"diagnostic": traceback.format_exc()})
        unreal.unregister_slate_post_tick_callback(advance_handle)
        unreal.unregister_slate_post_tick_callback(sample_handle)
        raise


def prepare_scene(request):
    global ASSETS, LEVEL, NEUTRAL
    NEUTRAL = request.get("neutral", False)
    kit = Path(request["kit"])
    meta = read(kit / "comparison.json")
    started = perf_counter()
    paths = {name: Path(path) for name, path in request["jobs"].items()}
    jobs = [read(path) for path in paths.values()]
    require(all(job["schema"] == "c07.unreal-perspective-job/1"
                and job["projection"] == "perspective" and job["temporal_samples"] == 1
                for job in jobs), "only perspective, frame-centred, single-temporal-sample jobs are supported")
    eiger, ostegg = read(paths["eiger"]), read(paths["ostegg"])
    LEVEL = eiger["registration"]["level_id"]
    ASSETS = LEVEL.rsplit("/", 1)[0]
    require(ASSETS.startswith("/Game/C07") and LEVEL.endswith("/SwissTerrain")
            and eiger["registration"] == ostegg["registration"],
            "C07 requires a dedicated registered Swiss level")
    for filename, key in (("terrain.npy", "terrain_sha256"), ("terrain-north-first.csv", "terrain_csv_sha256"),
                          ("features.json", "feature_sha256")):
        require(all(digest(kit / filename) == job[key] for job in jobs), f"changed source input: {filename}")
    if "scene_from" in request:
        scene = read(Path(request["scene_from"]))
        require(scene["level"] == LEVEL
                and scene["project_directory"] == unreal.SystemLibrary.get_project_directory(),
                "saved scene does not match this registered project")
        require(LEVELS.load_level(LEVEL), "cannot reopen the registered level")
        sequences = {name: ASSET_API.load_asset(info["path"])
                     for name, info in scene["sequences"].items()}
        presets = {job["shot"]: ASSET_API.load_asset(
            f"{ASSETS}/{job['registration']['preset_id']}_{job['shot']}")
            for job in (eiger, ostegg)}
        for preset in presets.values():
            require(preset, "cannot load the registered preset")
            configure_colour(preset, NEUTRAL)
        exposure = configure_neutral_exposure()
        write(BASE / "scene.json", {**scene, "scene_from": request["scene_from"],
                                   "editor_pid": os.getpid(),
                                   "neutral_postprocess": exposure,
                                   "asset_sha256": project_assets(sequences, presets),
                                   "setup_seconds": perf_counter() - started})
        return planned_runs(request, paths, sequences, presets)
    require(LEVELS.new_level(LEVEL), "cannot create isolated Swiss level")
    unreal.log(f"C07 created level {LEVEL}")
    terrain, vertices, triangles = import_terrain(kit, eiger, meta)
    unreal.log(f"C07 imported {vertices} terrain vertices and {triangles} triangles")
    registrations = register_landmarks((eiger, ostegg), meta, terrain)
    sequences = {name: camera_sequence(read(path), name) for name, path in paths.items()}
    presets = {name: create_preset(shot, NEUTRAL) for name, shot in (("eiger", eiger), ("ostegg", ostegg))}
    exposure = configure_neutral_exposure()
    require(LEVELS.save_current_level(), "cannot save isolated level")
    defaults = {name: unreal.SystemLibrary.get_console_variable_bool_value(name)
                for name in ("r.DefaultFeature.MotionBlur", "r.DefaultFeature.AutoExposure")}
    require(not any(defaults.values()), "the C07 preset requires disabled default motion blur and auto-exposure")
    write(BASE / "scene.json", {"engine": unreal.SystemLibrary.get_engine_version(),
                               "editor_pid": os.getpid(),
                               "neutral_postprocess": exposure,
                               "project_directory": unreal.SystemLibrary.get_project_directory(),
                               "render_defaults": defaults,
                               "level": LEVEL, "vertices": vertices, "triangles": triangles,
                               "asset_sha256": project_assets(sequences, presets),
                               "landmarks": registrations, "setup_seconds": perf_counter() - started,
                               "sequences": {name: {"path": seq.get_path_name(),
                                                    "tick_rate": [seq.get_tick_resolution().numerator,
                                                                  seq.get_tick_resolution().denominator]}
                                             for name, seq in sequences.items()}})
    return planned_runs(request, paths, sequences, presets)


def planned_runs(request, paths, sequences, presets):
    for item in request["runs"]:
        shot = item["shot"]
        sequence, preset = sequences[shot], presets[read(paths[shot])["shot"]]
        require(sequence and preset, f"cannot load sequence and preset for {shot}")
        yield item["mode"], paths[shot], BASE / item["name"], sequence, preset


def main(base):
    global BASE, CURRENT, QUEUE, RUNS, sample_handle, advance_handle
    BASE = base
    try:
        CURRENT = {}
        QUEUE = unreal.get_editor_subsystem(unreal.MoviePipelineQueueSubsystem)
        RUNS = prepare_scene(read(BASE / "request.json"))
        sample_handle = unreal.register_slate_post_tick_callback(observe)
        advance_handle = unreal.register_slate_post_tick_callback(run_next)
    except Exception:
        write(BASE / "scene-failure.json", {"diagnostic": traceback.format_exc()})
        raise
