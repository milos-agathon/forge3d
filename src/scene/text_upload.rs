//! Shared atlas upload and validated glyph geometry for native text paths.
use crate::core::resource_tracker::AllocationOwner;
use crate::core::text_overlay::{TextInstance, TextOverlayRenderer};
use numpy::{PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;

pub(super) fn upload_atlas(
    allocation_owner: &AllocationOwner,
    renderer: Option<&mut TextOverlayRenderer>,
    atlas: &pyo3::types::PyAny,
    channels: Option<u32>,
    smoothing: Option<f32>,
) -> PyResult<()> {
    let _allocation_scope = allocation_owner.activate();
    let (h, w, c, data) = if let Ok(arr) = atlas.extract::<PyReadonlyArray3<u8>>() {
        let shape = arr.shape();
        if shape.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "atlas must be HxWxC uint8",
            ));
        }
        let h = shape[0] as u32;
        let w = shape[1] as u32;
        let c = shape[2] as u32;
        if c != 1 && c != 3 && c != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "atlas channels must be 1, 3, or 4",
            ));
        }
        (h, w, c, arr.as_array().to_owned().into_raw_vec())
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected numpy uint8 array HxWxC",
        ));
    };
    let g = crate::core::gpu::try_ctx()?;
    // Convert to RGBA8
    let mut rgba: Vec<u8> = Vec::with_capacity((h * w * 4) as usize);
    if c == 4 {
        rgba = data;
    } else if c == 3 {
        let mut idx = 0usize;
        while idx < data.len() {
            rgba.push(data[idx]);
            rgba.push(data[idx + 1]);
            rgba.push(data[idx + 2]);
            rgba.push(255);
            idx += 3;
        }
    } else {
        // c == 1 (SDF)
        for v in data.iter() {
            rgba.push(*v);
            rgba.push(*v);
            rgba.push(*v);
            rgba.push(255);
        }
    }
    let row_bytes = w * 4;
    let padded_bpr = crate::core::gpu::align_copy_bpr(row_bytes);
    let mut padded = vec![0u8; (padded_bpr * h) as usize];
    for y in 0..h as usize {
        let s = y * row_bytes as usize;
        let d = y * padded_bpr as usize;
        padded[d..d + row_bytes as usize].copy_from_slice(&rgba[s..s + row_bytes as usize]);
    }
    let tex = crate::core::resource_tracker::tracked_create_texture(
        &g.device,
        &wgpu::TextureDescriptor {
            label: Some("text_msdf_atlas"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
    )?;
    g.queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &padded,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(std::num::NonZeroU32::new(padded_bpr).unwrap().into()),
            rows_per_image: Some(std::num::NonZeroU32::new(h).unwrap().into()),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

    // Update text overlay renderer state
    if let Some(tr) = renderer {
        tr.set_atlas(tex, view);
        tr.recreate_bind_group(&g.device, None)?;
        if let Some(ch) = channels {
            tr.set_channels(ch);
        }
        if let Some(sm) = smoothing {
            tr.set_smoothing(sm);
        }
        tr.upload_uniforms(&g.queue);
    }

    Ok(())
}

pub(super) fn glyph_instance(
    rect: [f32; 4],
    uv: [f32; 4],
    color: [f32; 4],
    halo_color: [f32; 4],
    halo_width: f32,
    rotation: f32,
) -> PyResult<TextInstance> {
    let [x, y, w, h] = rect;
    let [u0, v0, u1, v1] = uv;
    let [r, g, b, a] = color;
    let [halo_r, halo_g, halo_b, halo_a] = halo_color;

    if !rotation.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "text rotation must be finite",
        ));
    }
    // A rotated quad can be entirely visible even when its unrotated
    // rectangle crosses an edge. Let the GPU clip after rotation.
    let (rect_min, rect_max) = if rotation == 0.0 {
        (
            [x.max(0.0), y.max(0.0)],
            [(x + w).max(0.0), (y + h).max(0.0)],
        )
    } else {
        ([x, y], [x + w, y + h])
    };
    let uv_min = [u0, v0];
    let uv_max = [u1, v1];
    let color = [
        r.clamp(0.0, 1.0),
        g.clamp(0.0, 1.0),
        b.clamp(0.0, 1.0),
        a.clamp(0.0, 1.0),
    ];
    let halo_color = [
        halo_r.clamp(0.0, 1.0),
        halo_g.clamp(0.0, 1.0),
        halo_b.clamp(0.0, 1.0),
        halo_a.clamp(0.0, 1.0),
    ];
    let mut instance =
        crate::core::text_overlay::TextInstance::new(rect_min, rect_max, uv_min, uv_max, color)
            .with_halo(halo_color, halo_width);
    instance.rotation = rotation;
    Ok(instance)
}
