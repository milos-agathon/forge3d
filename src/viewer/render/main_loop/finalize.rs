use crate::viewer::viewer_enums::{CaptureKind, VizMode};
use crate::viewer::Viewer;

impl Viewer {
    pub(super) fn finish_render_frame(
        &mut self,
        encoder: wgpu::CommandEncoder,
        output: wgpu::SurfaceTexture,
    ) {
        // Submit rendering
        self.queue.submit(std::iter::once(encoder.finish()));

        // Auto-snapshot once if env var is set and we haven't requested yet
        if self.snapshot_request.is_none() && !self.auto_snapshot_done {
            if let Some(ref p) = self.auto_snapshot_path {
                self.snapshot_request = Some(p.clone());
                self.auto_snapshot_done = true;
            }
        }

        // Snapshot if requested (read back the texture before present)
        // CRITICAL: Only write the file when we have the offscreen texture.
        // The Python script waits for the file to appear and stabilize, so writing
        // a black swapchain texture would cause it to exit prematurely before
        // the retry logic can render the proper offscreen texture.
        if self.snapshot_request.is_some() {
            if let Some(tex) = self.pending_snapshot_tex.take() {
                let path = self.snapshot_request.clone().unwrap();
                if let Err(e) = self.snapshot_swapchain_to_png(&tex, &path) {
                    eprintln!("Snapshot failed: {}", e);
                } else {
                    println!("Saved snapshot to {}", path);
                }
                // Clear the request only after successful offscreen snapshot
                self.snapshot_request = None;
            }
            // If no offscreen texture yet, keep the request and don't write any file.
            // This forces the waiting script to continue waiting while we retry.
        }
        output.present();

        // Optionally dump P5 artifacts after finishing all passes
        if self.dump_p5_requested {
            if let Err(e) = self.dump_gbuffer_artifacts() {
                eprintln!("[P5] dump failed: {}", e);
            }
            self.dump_p5_requested = false;
        }
        self.frame_count += 1;
        if let Some(fps) = self.fps_counter.tick() {
            let viz = match self.viz_mode {
                VizMode::Material => "material",
                VizMode::Normal => "normal",
                VizMode::Depth => "depth",
                VizMode::Gi => "gi",
                VizMode::Lit => "lit",
            };
            self.window.set_title(&format!(
                "{} | FPS: {:.1} | Mode: {:?} | Viz: {}",
                self.view_config.title,
                fps,
                self.camera.mode(),
                viz
            ));
        }

        // Process any pending P5 capture requests using current frame data.
        // Handle all queued captures before the viewer exits so that scripts
        // like p5_golden.sh (which enqueue multiple :p5 commands followed by
        // :quit) still produce every expected artifact.
        while let Some(kind) = self.pending_captures.pop_front() {
            match kind {
                CaptureKind::P51CornellSplit => {
                    if let Err(e) = self.capture_p51_cornell_with_scene() {
                        eprintln!("[P5.1] Cornell split failed: {}", e);
                    }
                }
                CaptureKind::P51AoGrid => {
                    if let Err(e) = self.capture_p51_ao_grid() {
                        eprintln!("[P5.1] AO grid failed: {}", e);
                    }
                }
                CaptureKind::P51ParamSweep => {
                    if let Err(e) = self.capture_p51_param_sweep() {
                        eprintln!("[P5.1] AO sweep failed: {}", e);
                    }
                }
                CaptureKind::P52SsgiCornell => {
                    if let Err(e) = self.capture_p52_ssgi_cornell() {
                        eprintln!("[P5.2] SSGI Cornell capture failed: {}", e);
                    }
                }
                CaptureKind::P52SsgiTemporal => {
                    if let Err(e) = self.capture_p52_ssgi_temporal() {
                        eprintln!("[P5.2] SSGI temporal compare failed: {}", e);
                    }
                }
                CaptureKind::P53SsrGlossy => {
                    if let Err(e) = self.capture_p53_ssr_glossy() {
                        eprintln!("[P5.3] SSR glossy capture failed: {}", e);
                    }
                }
                CaptureKind::P53SsrThickness => {
                    if let Err(e) = self.capture_p53_ssr_thickness_ablation() {
                        eprintln!("[P5.3] SSR thickness capture failed: {}", e);
                    }
                }
                CaptureKind::P54GiStack => {
                    if let Err(e) = self.capture_p54_gi_stack_ablation() {
                        eprintln!("[P5.4] GI stack ablation capture failed: {}", e);
                    }
                }
            }
        }
    }
}
