@compute @workgroup_size(1)
fn unguarded_zero_div() {
    let denom: f32 = 0.0;
    let x = 1.0 / denom;
    _ = x;
}
