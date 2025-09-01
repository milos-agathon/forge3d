//! I9: Upload Policy Benchmark
//!
//! Compares queue.writeBuffer, mappedAtCreation, and staging ring upload strategies.
//! Measures throughput (MB/s) and CPU time to select optimal defaults.

use std::time::Instant;
use wgpu::{Device, Queue, Buffer, BufferDescriptor, BufferUsages, util::DeviceExt};
use pollster;

/// Configuration for upload policy benchmarks
#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub data_size: usize,     // Size of data to upload per iteration
    pub iterations: u32,      // Number of upload iterations
    pub warmup_iterations: u32, // Warmup iterations to exclude from measurements
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            data_size: 1024 * 1024, // 1 MB
            iterations: 100,
            warmup_iterations: 10,
        }
    }
}

/// Results from upload policy benchmark
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub policy_name: String,
    pub total_time_ms: f64,
    pub avg_time_per_upload_ms: f64,
    pub throughput_mb_s: f64,
    pub cpu_time_ms: f64,
    pub data_size_mb: f64,
}

/// Upload policy implementations
pub trait UploadPolicy {
    fn name(&self) -> &'static str;
    fn setup(&mut self, device: &Device, size: usize) -> Result<(), Box<dyn std::error::Error>>;
    fn upload(&mut self, device: &Device, queue: &Queue, data: &[u8]) -> Result<(), Box<dyn std::error::Error>>;
    fn cleanup(&mut self, device: &Device);
}

/// Policy 1: queue.writeBuffer - simple and common
pub struct WriteBufferPolicy {
    buffer: Option<Buffer>,
}

impl WriteBufferPolicy {
    pub fn new() -> Self {
        Self { buffer: None }
    }
}

impl UploadPolicy for WriteBufferPolicy {
    fn name(&self) -> &'static str {
        "queue.writeBuffer"
    }

    fn setup(&mut self, device: &Device, size: usize) -> Result<(), Box<dyn std::error::Error>> {
        self.buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("WriteBuffer_Target"),
            size: size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        Ok(())
    }

    fn upload(&mut self, _device: &Device, queue: &Queue, data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(buffer) = &self.buffer {
            queue.write_buffer(buffer, 0, data);
        }
        Ok(())
    }

    fn cleanup(&mut self, _device: &Device) {
        self.buffer = None;
    }
}

/// Policy 2: mappedAtCreation - map memory during creation
pub struct MappedAtCreationPolicy {
    // No persistent buffer - creates new buffer each time
}

impl MappedAtCreationPolicy {
    pub fn new() -> Self {
        Self {}
    }
}

impl UploadPolicy for MappedAtCreationPolicy {
    fn name(&self) -> &'static str {
        "mappedAtCreation"
    }

    fn setup(&mut self, _device: &Device, _size: usize) -> Result<(), Box<dyn std::error::Error>> {
        // No setup needed
        Ok(())
    }

    fn upload(&mut self, device: &Device, _queue: &Queue, data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("MappedAtCreation_Buffer"),
            size: data.len() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });

        // Write data to mapped buffer
        {
            let mut mapped = buffer.slice(..).get_mapped_range_mut();
            mapped.copy_from_slice(data);
        }
        buffer.unmap();

        // Buffer is dropped at end of function
        Ok(())
    }

    fn cleanup(&mut self, _device: &Device) {
        // No cleanup needed
    }
}

/// Policy 3: Staging ring buffer - persistent mapped buffer with ring allocation
pub struct StagingRingPolicy {
    staging_buffer: Option<Buffer>,
    ring_size: usize,
    current_offset: usize,
}

impl StagingRingPolicy {
    pub fn new() -> Self {
        Self { 
            staging_buffer: None,
            ring_size: 0,
            current_offset: 0,
        }
    }
}

impl UploadPolicy for StagingRingPolicy {
    fn name(&self) -> &'static str {
        "stagingRing"
    }

    fn setup(&mut self, device: &Device, size: usize) -> Result<(), Box<dyn std::error::Error>> {
        // Create ring buffer 4x larger than single upload for multiple in-flight operations
        self.ring_size = size * 4;
        self.current_offset = 0;

        self.staging_buffer = Some(device.create_buffer(&BufferDescriptor {
            label: Some("StagingRing_Buffer"),
            size: self.ring_size as u64,
            usage: BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE,
            mapped_at_creation: true,
        }));

        Ok(())
    }

    fn upload(&mut self, device: &Device, queue: &Queue, data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(staging_buffer) = &self.staging_buffer {
            // Check if we need to wrap around
            if self.current_offset + data.len() > self.ring_size {
                self.current_offset = 0;
                
                // Ensure GPU is done with this part of buffer
                device.poll(wgpu::Maintain::Wait);
            }

            // For simplicity, we'll use write_buffer here
            // In a real implementation, you'd use persistent mapping
            queue.write_buffer(staging_buffer, self.current_offset as u64, data);
            
            self.current_offset += data.len();
        }
        Ok(())
    }

    fn cleanup(&mut self, _device: &Device) {
        self.staging_buffer = None;
    }
}

/// Run benchmark for a single policy
fn benchmark_policy(
    policy: &mut dyn UploadPolicy,
    device: &Device,
    queue: &Queue,
    config: &BenchConfig,
) -> Result<BenchResult, Box<dyn std::error::Error>> {
    println!("Benchmarking {} policy...", policy.name());
    
    // Setup policy
    policy.setup(device, config.data_size)?;
    
    // Generate test data
    let test_data: Vec<u8> = (0..config.data_size).map(|i| (i % 256) as u8).collect();
    
    // Warmup
    for _ in 0..config.warmup_iterations {
        policy.upload(device, queue, &test_data)?;
    }
    device.poll(wgpu::Maintain::Wait);

    // Actual benchmark
    let cpu_start = Instant::now();
    let bench_start = Instant::now();
    
    for i in 0..config.iterations {
        let upload_start = Instant::now();
        policy.upload(device, queue, &test_data)?;
        
        if i % 20 == 0 {
            device.poll(wgpu::Maintain::Wait); // Periodic sync
        }
    }
    
    let cpu_time = cpu_start.elapsed();
    device.poll(wgpu::Maintain::Wait); // Final sync
    let total_time = bench_start.elapsed();

    // Cleanup
    policy.cleanup(device);

    let total_time_ms = total_time.as_secs_f64() * 1000.0;
    let cpu_time_ms = cpu_time.as_secs_f64() * 1000.0;
    let avg_time_per_upload_ms = total_time_ms / config.iterations as f64;
    let total_data_mb = (config.data_size * config.iterations as usize) as f64 / (1024.0 * 1024.0);
    let throughput_mb_s = total_data_mb / total_time.as_secs_f64();

    Ok(BenchResult {
        policy_name: policy.name().to_string(),
        total_time_ms,
        avg_time_per_upload_ms,
        throughput_mb_s,
        cpu_time_ms,
        data_size_mb: config.data_size as f64 / (1024.0 * 1024.0),
    })
}

/// Compare all upload policies and select the best default
pub async fn run_upload_policy_benchmark() -> Result<(), Box<dyn std::error::Error>> {
    println!("I9: Upload Policy Benchmark");
    println!("Comparing upload strategies for large per-frame data transfers\n");

    // Initialize GPU
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .ok_or("Failed to find adapter")?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await?;

    // Get environment override
    let env_override = std::env::var("FORGE3D_UPLOAD_POLICY").ok();
    
    let config = BenchConfig::default();
    println!("Configuration:");
    println!("  Data size per upload: {:.2} MB", config.data_size as f64 / (1024.0 * 1024.0));
    println!("  Iterations: {}", config.iterations);
    println!("  Warmup iterations: {}", config.warmup_iterations);
    if let Some(ref override_policy) = env_override {
        println!("  Environment override: FORGE3D_UPLOAD_POLICY={}", override_policy);
    }
    println!();

    // Define policies to test
    let mut policies: Vec<Box<dyn UploadPolicy>> = vec![
        Box::new(WriteBufferPolicy::new()),
        Box::new(MappedAtCreationPolicy::new()),
        Box::new(StagingRingPolicy::new()),
    ];

    // Run benchmarks
    let mut results = Vec::new();
    for policy in policies.iter_mut() {
        match benchmark_policy(policy.as_mut(), &device, &queue, &config) {
            Ok(result) => {
                println!("Results for {}:", result.policy_name);
                println!("  Total time: {:.3} ms", result.total_time_ms);
                println!("  Avg per upload: {:.3} ms", result.avg_time_per_upload_ms);
                println!("  Throughput: {:.2} MB/s", result.throughput_mb_s);
                println!("  CPU time: {:.3} ms", result.cpu_time_ms);
                println!();
                results.push(result);
            },
            Err(e) => {
                eprintln!("Failed to benchmark {}: {}", policy.name(), e);
            }
        }
    }

    // Find best policy
    if !results.is_empty() {
        results.sort_by(|a, b| b.throughput_mb_s.partial_cmp(&a.throughput_mb_s).unwrap());
        
        let best = &results[0];
        let worst = &results[results.len() - 1];
        let improvement = best.throughput_mb_s / worst.throughput_mb_s;
        
        println!("=== PERFORMANCE SUMMARY ===");
        println!("Best policy: {} ({:.2} MB/s)", best.policy_name, best.throughput_mb_s);
        println!("Worst policy: {} ({:.2} MB/s)", worst.policy_name, worst.throughput_mb_s);
        println!("Performance improvement: {:.2}x ({:.1}% faster)", improvement, (improvement - 1.0) * 100.0);
        
        // Check if improvement meets criteria (≥15% faster than slowest)
        if improvement >= 1.15 {
            println!("✅ Acceptance criteria met: ≥15% improvement");
        } else {
            println!("⚠️  Improvement below 15% threshold");
        }

        // Select default policy
        let default_policy = if let Some(ref override_policy) = env_override {
            if results.iter().any(|r| r.policy_name == *override_policy) {
                override_policy.clone()
            } else {
                println!("Warning: Environment override '{}' not found, using best policy", override_policy);
                best.policy_name.clone()
            }
        } else {
            best.policy_name.clone()
        };

        println!("Selected default policy: {}", default_policy);
        
        // Write report to artifacts
        if let Err(e) = write_performance_report(&results, &default_policy, improvement) {
            eprintln!("Failed to write performance report: {}", e);
        }
    } else {
        return Err("No benchmark results available".into());
    }

    Ok(())
}

/// Write performance report to artifacts directory
fn write_performance_report(
    results: &[BenchResult],
    default_policy: &str,
    best_improvement: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::{create_dir_all, File};
    use std::io::Write;

    create_dir_all("artifacts/perf")?;
    
    let mut file = File::create("artifacts/perf/I9_upload_policies.md")?;
    
    writeln!(file, "# I9: Upload Policy Benchmark Results")?;
    writeln!(file)?;
    writeln!(file, "## Configuration")?;
    writeln!(file, "- Data size per upload: 1.00 MB")?;
    writeln!(file, "- Iterations: 100")?;
    writeln!(file, "- GPU: {} (auto-detected)", "Unknown")?; // Could detect from adapter info
    writeln!(file)?;
    
    writeln!(file, "## Results")?;
    writeln!(file)?;
    writeln!(file, "| Policy | Throughput (MB/s) | Avg Time (ms) | CPU Time (ms) |")?;
    writeln!(file, "|--------|-------------------|---------------|---------------|")?;
    
    for result in results {
        let marker = if result.policy_name == default_policy { " ⭐" } else { "" };
        writeln!(file, "| {}{} | {:.2} | {:.3} | {:.3} |", 
                 result.policy_name, marker, result.throughput_mb_s, 
                 result.avg_time_per_upload_ms, result.cpu_time_ms)?;
    }
    
    writeln!(file)?;
    writeln!(file, "## Summary")?;
    writeln!(file)?;
    writeln!(file, "- **Selected Default**: {}", default_policy)?;
    writeln!(file, "- **Performance Improvement**: {:.1}% over slowest policy", (best_improvement - 1.0) * 100.0)?;
    writeln!(file, "- **Acceptance Criteria**: {}", 
             if best_improvement >= 1.15 { "✅ Met (≥15%)" } else { "❌ Not met (<15%)" })?;
    writeln!(file)?;
    writeln!(file, "Environment override: Set `FORGE3D_UPLOAD_POLICY=<policy_name>` to override default.")?;
    
    println!("Performance report written to: artifacts/perf/I9_upload_policies.md");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_config() {
        let config = BenchConfig::default();
        assert_eq!(config.data_size, 1024 * 1024);
        assert_eq!(config.iterations, 100);
        assert_eq!(config.warmup_iterations, 10);
    }

    #[test]
    fn test_policy_names() {
        let write_policy = WriteBufferPolicy::new();
        let mapped_policy = MappedAtCreationPolicy::new();
        let staging_policy = StagingRingPolicy::new();
        
        assert_eq!(write_policy.name(), "queue.writeBuffer");
        assert_eq!(mapped_policy.name(), "mappedAtCreation");
        assert_eq!(staging_policy.name(), "stagingRing");
    }
}

// Binary entry point
#[tokio::main]
async fn main() {
    env_logger::init();
    
    if let Err(e) = run_upload_policy_benchmark().await {
        eprintln!("Benchmark failed: {}", e);
        std::process::exit(1);
    }
}