use forge3d::codec::f3dz::{decode_dem, encode_dem, gpu::benchmark_decode_gpu, EncodeOptions};
use std::time::{Duration, Instant};

fn terrain(side: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(side * side);
    for y in 0..side {
        for x in 0..side {
            let local_x = x % 64;
            let local_y = y % 64;
            let wave =
                ((local_x * 17 + local_y * 31 + (local_x ^ local_y) * 3) % 97) as f32 * 0.03125;
            values.push(local_x as f32 * 0.125 + local_y as f32 * 0.0625 + wave);
        }
    }
    values
}

fn median_duration(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn main() {
    let side = 2_048usize;
    let values = terrain(side);
    // Dyadic epsilon keeps the synthetic 1/32-metre terrain on the integer
    // lattices, so this measures the normal low-barrier path rather than exact
    // finite-value escape handling.
    let mut options = EncodeOptions::new(0.125);
    options.progressive = true;
    let warm = encode_dem(&values, side as u32, side as u32, &options).unwrap();

    let mut encode_times = Vec::new();
    for _ in 0..5 {
        let started = Instant::now();
        let encoded = encode_dem(&values, side as u32, side as u32, &options).unwrap();
        assert_eq!(encoded, warm);
        encode_times.push(started.elapsed());
    }
    let encode_seconds = median_duration(encode_times).as_secs_f64();
    let cpu_encode_mpx_s = (side * side) as f64 / encode_seconds / 1.0e6;

    let mut decode_times = Vec::new();
    for _ in 0..5 {
        let started = Instant::now();
        let decoded = decode_dem(&warm, Some(options.epsilon)).unwrap();
        assert_eq!(decoded.values.len(), values.len());
        decode_times.push(started.elapsed());
    }
    let decode_seconds = median_duration(decode_times).as_secs_f64();
    let cpu_decode_mpx_s = (side * side) as f64 / decode_seconds / 1.0e6;

    // The GPU gate uses the public 64x64 page default. It measures the
    // non-progressive single-layer decode used when a caller requests final
    // quality directly; progressive base/refinement identity is tested
    // independently.
    let gpu_side = side;
    let mut gpu_options = options.clone();
    gpu_options.progressive = false;
    let gpu_stream = encode_dem(&values, gpu_side as u32, gpu_side as u32, &gpu_options).unwrap();
    println!(
        "F3DZ_BENCH cpu_adapter={} grid={}x{} eps={} progressive=true bytes={} cpu_encode_mpx_s={:.3} cpu_decode_mpx_s={:.3}",
        std::env::consts::ARCH,
        side,
        side,
        options.epsilon,
        warm.len(),
        cpu_encode_mpx_s,
        cpu_decode_mpx_s,
    );
    match benchmark_decode_gpu(&gpu_stream, 128) {
        Ok(result) => println!(
            "F3DZ_BENCH gpu_adapter={:?} grid={}x{} eps={} progressive=false page={}x{} pages={} iterations={} fast_path={} direct_tokens={} timing_source={} gpu_elapsed_s={:.6} wall_s={:.6} gpu_decode_gpx_s={:.3}",
            result.adapter,
            gpu_side,
            gpu_side,
            gpu_options.epsilon,
            gpu_options.tile_size,
            gpu_options.tile_size,
            (gpu_side / usize::from(gpu_options.tile_size)).pow(2),
            result.iterations,
            result.fast_path,
            result.direct_tokens,
            result.timing_source,
            result.elapsed_seconds,
            result.wall_seconds,
            result.gigapixels_per_second,
        ),
        Err(error) => println!("F3DZ_BENCH gpu_status=NOT_AVAILABLE reason={error}"),
    }
}
