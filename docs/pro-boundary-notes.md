# Pro Boundary Notes

Living notes for the planned open-core split. The Phase 2 implementation keeps
the open workflow centered on learning, exploration, and automation, while the
likely Pro boundary clusters around production output and premium packaging.

| Date | Area | Lean | Rationale |
| --- | --- | --- | --- |
| 2026-03-10 | Viewer launch, terrain loading, camera, sun, snapshots | Open | The first-render loop must stay frictionless or the rest of the platform never gets used. |
| 2026-03-10 | Bundled datasets and dataset fetch helpers | Open | Tutorials, notebooks, and CI all depend on a stable sample-data story. |
| 2026-03-10 | Notebook widget wrapper (`ViewerWidget` plus inline fallback preview) | Open | Notebook exploration is part of the base developer experience, not an upsell. |
| 2026-03-10 | Raster overlays, labels, and vector overlay IPC | Open | Styling and annotation need to stay available so users can evaluate the core scene model. |
| 2026-03-10 | Point-cloud loading and parameter control | Open | This is still core scene exploration, similar to terrain and overlays. |
| 2026-03-10 | Map plate composition (`MapPlate`, legend, scale bar, north arrow) | Pro | This is packaging for deliverables rather than basic scene inspection. |
| 2026-03-10 | SVG and PDF export | Pro | Print-grade export has clear commercial value and little effect on initial adoption. |
| 2026-03-10 | Rich buildings workflows beyond inspection | Pro | Metadata parsing is useful in open; premium value starts when the workflow becomes production visualization. |
| 2026-03-10 | Scene bundles for shareable project artifacts | Pro | Bundles fit team handoff, repeatability, and downstream delivery workflows. |

## Working rule

Keep the entire "load data, inspect scene, automate camera, save a screenshot"
loop open. Put the paywall, if any, around polished deliverables and reusable
project packaging rather than around basic understanding of the scene.

## Current implementation note

The repository does not yet enforce a runtime license gate around the Pro-leaning
features listed above. These notes are product guidance for the next phase, not
an already-enforced contract.
