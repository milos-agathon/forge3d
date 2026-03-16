use super::*;

impl GpuBvhBuilder {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Result<Self> {
        // Load and compile WGSL shaders
        let morton_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LBVH Morton"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/lbvh_morton.wgsl").into()),
        });

        let sort_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Radix Sort"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/radix_sort_pairs.wgsl").into(),
            ),
        });

        let link_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LBVH Link"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/lbvh_link.wgsl").into()),
        });

        let refit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("BVH Refit"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/bvh_refit.wgsl").into()),
        });

        // Create compute pipelines
        let morton_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LBVH Morton Pipeline"),
            layout: None,
            module: &morton_shader,
            entry_point: "main",
        });

        let sort_count_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Radix Sort Count Pipeline"),
                layout: None,
                module: &sort_shader,
                entry_point: "count_pass",
            });

        let sort_scan_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Radix Sort Scan Pipeline"),
            layout: None,
            module: &sort_shader,
            entry_point: "scan_pass",
        });

        let sort_scatter_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Radix Sort Scatter Pipeline"),
                layout: None,
                module: &sort_shader,
                entry_point: "scatter_pass",
            });

        let sort_clear_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Radix Sort Clear Pipeline"),
                layout: None,
                module: &sort_shader,
                entry_point: "clear_hist",
            });

        let sort_bitonic_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Bitonic Sort Pipeline"),
                layout: None,
                module: &sort_shader,
                entry_point: "bitonic_sort",
            });

        let link_nodes_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LBVH Link Nodes Pipeline"),
                layout: None,
                module: &link_shader,
                entry_point: "link_nodes",
            });

        let init_leaves_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LBVH Init Leaves Pipeline"),
                layout: None,
                module: &link_shader,
                entry_point: "init_leaves",
            });

        let refit_leaves_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("BVH Refit Leaves Pipeline"),
                layout: None,
                module: &refit_shader,
                entry_point: "refit_leaves",
            });

        let refit_internal_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("BVH Refit Internal Pipeline"),
                layout: None,
                module: &refit_shader,
                entry_point: "refit_internal",
            });

        let refit_iterative_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("BVH Refit Iterative Pipeline"),
                layout: None,
                module: &refit_shader,
                entry_point: "refit_iterative",
            });

        Ok(Self {
            device,
            queue,
            morton_pipeline,
            sort_count_pipeline,
            sort_scan_pipeline,
            sort_scatter_pipeline,
            sort_clear_pipeline,
            sort_bitonic_pipeline,
            link_nodes_pipeline,
            init_leaves_pipeline,
            refit_leaves_pipeline,
            refit_internal_pipeline,
            refit_iterative_pipeline,
        })
    }
}
