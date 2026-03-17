pub(crate) struct TimestampResources {
    query_set: Option<wgpu::QuerySet>,
    buffer: Option<wgpu::Buffer>,
    readback: Option<wgpu::Buffer>,
}

impl TimestampResources {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return Self {
                query_set: None,
                buffer: None,
                readback: None,
            };
        }

        Self {
            query_set: Some(device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("brdf_tile.timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            })),
            buffer: Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("brdf_tile.timestamp_buffer"),
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })),
            readback: Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("brdf_tile.timestamp_readback"),
                size: 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })),
        }
    }

    pub(crate) fn write_begin(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(query_set) = &self.query_set {
            encoder.write_timestamp(query_set, 0);
        }
    }

    pub(crate) fn timestamp_writes(&self) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        self.query_set
            .as_ref()
            .map(|query_set| wgpu::RenderPassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            })
    }

    pub(crate) fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if let (Some(query_set), Some(buffer), Some(readback)) =
            (&self.query_set, &self.buffer, &self.readback)
        {
            encoder.resolve_query_set(query_set, 0..2, buffer, 0);
            encoder.copy_buffer_to_buffer(buffer, 0, readback, 0, 16);
        }
    }
}
