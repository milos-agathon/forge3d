use forge3d::codec::f3dz::{decode_dem, encode_dem, gpu::benchmark_decode_gpu, EncodeOptions};
use std::time::{Duration, Instant};

fn terrain(side: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(side * side);
    for y in 0..side {
        for x in 0..side {
            let wave = ((x * 17 + y * 31 + (x ^ y) * 3) % 97) as f32 * 0.03125;
            values.push(1_000.0 + x as f32 * 0.125 + y as f32 * 0.0625 + wave);
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
    let mut options = EncodeOptions::new(0.1);
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

    let gpu_side = 512usize;
    let gpu_values = terrain(gpu_side);
    let gpu_stream = encode_dem(&gpu_values, gpu_side as u32, gpu_side as u32, &options).unwrap();
    println!(
        "F3DZ_BENCH cpu_adapter={} grid={}x{} progressive=true bytes={} cpu_encode_mpx_s={:.3} cpu_decode_mpx_s={:.3}",
        std::env::consts::ARCH,
        side,
        side,
        warm.len(),
        cpu_encode_mpx_s,
        cpu_decode_mpx_s,
    );
    match benchmark_decode_gpu(&gpu_stream, 2_048) {
        Ok(result) => println!(
            "F3DZ_BENCH gpu_adapter={:?} grid={}x{} iterations={} elapsed_s={:.6} gpu_decode_gpx_s={:.3}",
            result.adapter,
            gpu_side,
            gpu_side,
            result.iterations,
            result.elapsed_seconds,
            result.gigapixels_per_second,
        ),
        Err(error) => println!("F3DZ_BENCH gpu_status=NOT_AVAILABLE reason={error}"),
    }
}
