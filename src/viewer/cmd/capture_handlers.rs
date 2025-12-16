// src/viewer/cmd/capture_handlers.rs
// P5 capture command handlers for the interactive viewer

use super::super::viewer_enums::CaptureKind;
use std::collections::VecDeque;

/// Queue a P5.1 Cornell split capture
pub fn queue_p51_cornell(pending_captures: &mut VecDeque<CaptureKind>) {
    pending_captures.push_back(CaptureKind::P51CornellSplit);
    println!("[P5.1] capture: Cornell OFF/ON split queued");
}

/// Queue a P5.1 AO grid capture
pub fn queue_p51_grid(pending_captures: &mut VecDeque<CaptureKind>) {
    pending_captures.push_back(CaptureKind::P51AoGrid);
    println!("[P5.1] capture: AO buffers grid queued");
}

/// Queue a P5.1 parameter sweep capture
pub fn queue_p51_sweep(pending_captures: &mut VecDeque<CaptureKind>) {
    pending_captures.push_back(CaptureKind::P51ParamSweep);
    println!("[P5.1] capture: AO parameter sweep queued");
}

/// Queue a P5.2 SSGI Cornell capture
pub fn queue_p52_ssgi_cornell(pending_captures: &mut VecDeque<CaptureKind>) {
    pending_captures.push_back(CaptureKind::P52SsgiCornell);
    println!("[P5.2] capture: SSGI Cornell split queued");
}

/// Queue a P5.2 SSGI temporal compare capture
pub fn queue_p52_ssgi_temporal(pending_captures: &mut VecDeque<CaptureKind>) {
    pending_captures.push_back(CaptureKind::P52SsgiTemporal);
    println!("[P5.2] capture: SSGI temporal compare queued");
}

/// Queue a P5.3 SSR glossy spheres capture
pub fn queue_p53_ssr_glossy(pending_captures: &mut VecDeque<CaptureKind>) {
    pending_captures.push_back(CaptureKind::P53SsrGlossy);
    println!("[P5.3] capture: SSR glossy spheres queued");
}

/// Queue a P5.3 SSR thickness ablation capture
pub fn queue_p53_ssr_thickness(pending_captures: &mut VecDeque<CaptureKind>) {
    pending_captures.push_back(CaptureKind::P53SsrThickness);
    println!("[P5.3] capture: SSR thickness ablation queued");
}

/// Queue a P5.4 GI stack ablation capture
pub fn queue_p54_gi_stack(pending_captures: &mut VecDeque<CaptureKind>) {
    pending_captures.push_back(CaptureKind::P54GiStack);
    println!("[P5.4] capture: GI stack ablation queued");
}
