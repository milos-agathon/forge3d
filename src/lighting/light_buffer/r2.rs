pub fn r2_sample(index: u64) -> [f32; 2] {
    const PHI: f64 = 1.324_717_957_244_746;
    const A1: f64 = 1.0 / PHI;
    const A2: f64 = 1.0 / (PHI * PHI);
    let idx = index as f64;
    [frac(0.5 + A1 * idx) as f32, frac(0.5 + A2 * idx) as f32]
}

fn frac(x: f64) -> f64 {
    x - x.floor()
}
