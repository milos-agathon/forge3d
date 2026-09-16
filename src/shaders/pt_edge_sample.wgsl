// src/shaders/pt_edge_sample.wgsl
// DIFFERENTIA: the reparameterized boundary estimator.
//
// The shading pass detaches ray-scene visibility; the image jump across a
// boundary pixel pair still moves when the boundary moves, and its gradient
// is a distribution-free one-sided term (Liang et al. edge sampling /
// reparameterization). This kernel:
//
//   * DETECTS candidate boundary pixels from the Depth/Normal AOVs the
//     primal wrote on frame 0 — the same rasterizer-style pixel-center
//     record the spec requires — rather than tracing a fresh ray for every
//     neighbor pair. Slot coverage is deterministic: exactly two slots per
//     pixel (east and south neighbor), dispatched 1-D over 2*pixel_count.
//   * EVALUATES the boundary velocity for the parameters that actually move
//     the boundary under the current parameter set — the sun direction only.
//     (Geometry is detached: albedo/intensity/turbidity do not move any
//     boundary, so their edge terms are identically zero.)
//
// Two boundary types, per Liang et al.:
//   * silhouette (hit <-> miss): moves only with geometry/camera — both
//     detached here, so the term is exactly zero and the pair is skipped.
//   * cast-shadow edge: one side lit, the other occluded along omega while
//     still front-facing. The occluded side's shadow boundary moves with
//     omega through the blocker position; the image-pixel velocity is
//     projected through the pinhole Jacobian. Terminator pairs (shadowed
//     side already back-facing, n.omega <= 0) carry NO image jump — the
//     Lambertian term is continuous through the n.omega = 0 contour — so
//     they contribute exactly zero.
//
// The jump is evaluated against the exact AOV geometry (depth+normal ->
// world position), bilinear albedo through the shared map fetch, and the
// shared Beer transmittance — no arbitrary damping factor; the scalar
// magnitude is reported separately for diagnostics.
// RELEVANT FILES: src/shaders/pt_inverse_loss.wgsl, src/path_tracing/inverse/mod.rs

// gid.y folds 1-D dispatches wider than the 65535-workgroup x limit:
// INV_DISPATCH_X (declared in pt_inverse_loss.wgsl, same module) scales it.

@compute @workgroup_size(256)
fn main_inverse_edge(@builtin(global_invocation_id) gid: vec3<u32>) {
    if ((inv_params.ctrl.x & INV_FLAG_EDGE) == 0u) { return; }
    let W = uniforms.width;
    let H = uniforms.height;
    let px = W * H;
    let pair = gid.y * INV_DISPATCH_X + gid.x;
    if (pair >= 2u * px) { return; }
    let pix = pair / 2u;
    let axis = pair % 2u;                  // 0: +x pair, 1: +y pair
    let gx = pix % W;
    let gy = pix / W;
    let nx = gx + select(0u, 1u, axis == 0u);
    let ny = gy + select(1u, 0u, axis == 0u);
    if (nx >= W || ny >= H) { return; }
    let npix = ny * W + nx;

    // --- detection from the frame-0 AOVs (Depth.a = NaN on miss) ---
    let da = textureLoad(inv_aov_depth, vec2<i32>(i32(gx), i32(gy)), 0).r;
    let db = textureLoad(inv_aov_depth, vec2<i32>(i32(nx), i32(ny)), 0).r;
    let hita = da == da;
    let hitb = db == db;
    if (!hita && !hitb) { return; }
    let silh = hita != hitb;
    let omega = normalize(lighting.light_dir);

    var pa = vec3<f32>(0.0);
    var pb = vec3<f32>(0.0);
    var na = vec3<f32>(0.0, 1.0, 0.0);
    var nb = vec3<f32>(0.0, 1.0, 0.0);
    var va = 0.0;
    var vb = 0.0;
    if (hita) {
        na = textureLoad(inv_aov_normal, vec2<i32>(i32(gx), i32(gy)), 0).rgb;
        pa = uniforms.cam_origin + inv_center_dir(gx, gy) * da;
        if (dot(na, omega) <= 0.0) {
            va = 0.0;
        } else if (lighting.shadows_enabled != 0u
            && intersect_shadow_ray(Ray(pa + na * 1e-3, 1e-3, omega, 1e30), 1e30)) {
            va = 0.0;
        } else {
            va = 1.0;
        }
    }
    if (hitb) {
        nb = textureLoad(inv_aov_normal, vec2<i32>(i32(nx), i32(ny)), 0).rgb;
        pb = uniforms.cam_origin + inv_center_dir(nx, ny) * db;
        if (dot(nb, omega) <= 0.0) {
            vb = 0.0;
        } else if (lighting.shadows_enabled != 0u
            && intersect_shadow_ray(Ray(pb + nb * 1e-3, 1e-3, omega, 1e30), 1e30)) {
            vb = 0.0;
        } else {
            vb = 1.0;
        }
    }
    let shad = hita && hitb && (va != vb);
    if (!silh && !shad) { return; }
    if (silh) {
        // Boundary moves only with the detached geometry: d v/d sun = 0.
        return;
    }

    // --- boundary evaluation: the shadowed side is the boundary surface ---
    // lit_first=1 when pixel a is lit (image jump oriented a -> b).
    let lit_first = va > 0.5;
    let p_s = select(pa, pb, lit_first);   // shadowed side's surface point
    let n_s = select(na, nb, lit_first);
    let p_l = select(pb, pa, lit_first);   // lit side's surface point
    let n_l = select(nb, na, lit_first);
    let m_s = dot(n_s, omega);

    // Image jump across the pair: sun integrand at the lit surface minus the
    // shadowed side's (zero) sun term. Albedo through the SHARED map fetch.
    let alb = terrain_albedo_at(p_l);
    let S = lighting.light_color * terrain_sun_transmit(omega.y);
    let f_jump = alb * S * max(dot(n_l, omega), 0.0);
    let adj_mix = 0.5 * (inv_v4[pix].rgb + inv_v4[npix].rgb);
    // sgn orients the jump so v_px > 0 always means "boundary moves so the
    // lit region grows" — consistent for both axes and both configurations.
    let sgn = select(-1.0, 1.0, lit_first);
    let jump = sgn * dot(adj_mix, f_jump);
    if (!inv_is_finite(jump)) {
        inv_add_scalar(INV_GRAD_NONFINITE, 1.0);
        return;
    }
    inv_add_scalar(INV_GRAD_EDGE_MASS, abs(jump));
    if (m_s <= 0.0) {
        // Terminator pair: the shadowed side is back-facing, so the image
        // function is CONTINUOUS across the boundary (the sun term
        // alb*S*max(n.omega,0) vanishes at the n.omega=0 contour from both
        // sides). A reparameterized boundary term needs a value
        // discontinuity — a shading crease carries none — so the correct
        // contribution is zero.
        return;
    }

    // Cast-shadow branch: follow the omega-ray from the shadowed point to
    // the blocker. The shadow boundary on the receiver is the receiver's
    // intersection with the shadow plane spanned by the blocker's grazing
    // silhouette edge and omega. The silhouette tangent is approximated by
    // the horizontal contour direction at the contact (exact for ridge,
    // rim and cliff silhouettes — the dominant heightfield case). As omega
    // rotates by dω the shadow plane rotates about the edge; only the
    // boundary-NORMAL motion flips pixels — the along-line component is
    // contact sliding, not boundary motion. For a point p on the boundary:
    //   (p - B) . N = 0,  N = e x omega,  dN = e x dω
    //   dp . N = (B - p) . dN = -t_b (dω . N)   (B - p = t_b * omega)
    // projected into the receiver tangent plane via m = N - (N.n_s) n_s:
    //   v_vec = -t_b * (dω . N) * m / |m|^2 .
    let bray = Ray(p_s + n_s * 1e-3, 1e-3, omega, 1e30);
    let blocker = intersect_hybrid(bray);
    if (blocker.hit == 0u) {
        return;
    }
    let n_b = blocker.normal;
    let e2 = vec3<f32>(n_b.z, 0.0, -n_b.x);   // horizontal contour tangent
    if (dot(e2, e2) <= 1e-8) {                // flat-face contact: no edge
        return;
    }
    let e_dir = normalize(e2);
    let n_pl = cross(e_dir, omega);           // shadow-plane normal
    var m_dir = n_pl - dot(n_pl, n_s) * n_s;  // receiver-plane boundary normal
    let m2 = dot(m_dir, m_dir);
    if (m2 <= 1e-8) {
        return;
    }
    let t_b = length(blocker.point - p_s);

    // Pixel-space projection Jacobian at p_s (world -> ndc).
    let cf = uniforms.cam_forward;
    let rel = p_s - uniforms.cam_origin;
    let z = dot(rel, cf);
    if (z <= 1e-6) {
        return;
    }
    let half_h = tan(0.5 * uniforms.cam_fov_y);
    let half_w = uniforms.cam_aspect * half_h;
    let wpp = f32(W) * 0.5 / half_w;
    let hpp = f32(H) * 0.5 / half_h;
    // v_px = J * (dp/d omega_i), projected onto the pair's axis.
    for (var i = 0u; i < 3u; i = i + 1u) {
        let v_vec = m_dir * (-t_b * n_pl[i] / m2);
        // world delta -> ndc velocity (pinhole, y axis flipped in ndc).
        let xw = dot(v_vec, uniforms.cam_right);
        let yw = dot(v_vec, uniforms.cam_up);
        let zw = dot(v_vec, cf);
        let v_ndc_x = (xw * z - dot(rel, uniforms.cam_right) * zw) / (z * z);
        let v_ndc_y = (yw * z - dot(rel, uniforms.cam_up) * zw) / (z * z);
        let v_px = select(v_ndc_y * (-hpp), v_ndc_x * wpp, axis == 0u);
        inv_add_scalar(INV_GRAD_SUN_X + i, jump * v_px);
    }
}
