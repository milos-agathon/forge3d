//! Terrain-only shadow-boundary certificates on one receiver slice.
//! The producer proves the ideal-real event set; the post-dispatch audit then
//! conditionally certifies the executed WGSL branches and records numerical
//! discrepancies from the actual f32 inputs reported by the GPU.

use crate::core::resource_tracker::{tracked_host_allocation, ResourceHandle};
use std::alloc::{alloc, Layout};
use std::ops::{Deref, DerefMut};
use std::ptr;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Iv {
    pub lo: f64,
    pub hi: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Uncertifiable {
    InvalidInput,
    UnresolvedRoot,
    CoincidentEvent,
    FirstHit,
    Camera,
    Density,
    Memory,
}

/// Owns the host-visible reservation for the exact backing capacity. Growth
/// reserves the replacement buffer while the previous buffer remains live.
/// No backing Vec escapes without its reservation.
#[derive(Debug)]
pub(crate) struct TrackedVec<T> {
    values: Vec<T>,
    allocation: Option<ResourceHandle>,
    label: &'static str,
}

impl<T> TrackedVec<T> {
    fn new(label: &'static str) -> Self {
        Self {
            values: Vec::new(),
            allocation: None,
            label,
        }
    }

    fn with_capacity(capacity: usize, label: &'static str) -> Result<Self, Uncertifiable> {
        let mut result = Self::new(label);
        result.reserve(capacity)?;
        Ok(result)
    }

    fn reserve(&mut self, needed: usize) -> Result<(), Uncertifiable> {
        if needed <= self.values.capacity() {
            return Ok(());
        }
        let doubled = self
            .values
            .capacity()
            .checked_mul(2)
            .ok_or(Uncertifiable::Memory)?;
        let target = needed.max(doubled).max(1);
        let layout = Layout::array::<T>(target).map_err(|_| Uncertifiable::Memory)?;
        let bytes = u64::try_from(layout.size()).map_err(|_| Uncertifiable::Memory)?;
        let replacement =
            tracked_host_allocation(bytes, self.label).map_err(|_| Uncertifiable::Memory)?;
        // Vec::reserve_exact is permitted to overallocate. Allocate the
        // exact Layout instead, so the reservation precedes and equals the
        // backing allocation. T is nonzero-sized for every use in this file;
        // zero-sized T already has Vec::capacity()==usize::MAX above.
        let ptr = unsafe { alloc(layout) } as *mut T;
        if ptr.is_null() {
            return Err(Uncertifiable::Memory);
        }
        let len = self.values.len();
        // SAFETY: `ptr` is a fresh global-allocator block with exactly the
        // alignment and size for `target` T values; target>old capacity>=len.
        // Copying initialized values then setting the old Vec length to zero
        // transfers ownership without dropping either value twice. Both Vec
        // allocations use the same global allocator and exact layouts.
        unsafe {
            ptr::copy_nonoverlapping(self.values.as_ptr(), ptr, len);
            self.values.set_len(0);
            let old = std::mem::replace(&mut self.values, Vec::from_raw_parts(ptr, len, target));
            drop(old);
        }
        self.allocation = Some(replacement);
        Ok(())
    }

    fn push(&mut self, value: T) -> Result<(), Uncertifiable> {
        self.reserve(
            self.values
                .len()
                .checked_add(1)
                .ok_or(Uncertifiable::Memory)?,
        )?;
        self.values.push(value);
        Ok(())
    }

    fn pop(&mut self) -> Option<T> {
        self.values.pop()
    }
}

impl<T> Deref for TrackedVec<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}
impl<T> DerefMut for TrackedVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}

impl Iv {
    fn p(x: f64) -> Self {
        Self { lo: x, hi: x }
    }
    fn finite(self) -> Result<Self, Uncertifiable> {
        if self.lo.is_finite() && self.hi.is_finite() {
            Ok(self)
        } else {
            Err(Uncertifiable::UnresolvedRoot)
        }
    }
    fn add(self, b: Self) -> Result<Self, Uncertifiable> {
        // Zero is exactly representable.  Retaining an f64 outward-rounding
        // subnormal around two exact zero operands would manufacture an
        // impossible f32 sign ambiguity in a later WGSL operation.
        if self.lo == 0.0 && self.hi == 0.0 && b.lo == 0.0 && b.hi == 0.0 {
            return Ok(Self::p(0.0));
        }
        Self {
            lo: down(self.lo + b.lo),
            hi: up(self.hi + b.hi),
        }
        .finite()
    }
    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
    fn sub(self, b: Self) -> Result<Self, Uncertifiable> {
        self.add(b.neg())
    }
    fn mul(self, b: Self) -> Result<Self, Uncertifiable> {
        // As above, a point-zero operand has an exact zero product for every
        // value in the other interval; no f64 rounding enclosure is needed.
        if (self.lo == 0.0 && self.hi == 0.0) || (b.lo == 0.0 && b.hi == 0.0) {
            return Ok(Self::p(0.0));
        }
        let p = [
            self.lo * b.lo,
            self.lo * b.hi,
            self.hi * b.lo,
            self.hi * b.hi,
        ];
        if p.iter().any(|v| !v.is_finite()) {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        Self {
            lo: down(p.iter().copied().fold(f64::INFINITY, f64::min)),
            hi: up(p.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
        }
        .finite()
    }
    fn inv(self) -> Result<Self, Uncertifiable> {
        if self.lo <= 0.0 && self.hi >= 0.0 {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        Self {
            lo: down(1.0 / self.hi),
            hi: up(1.0 / self.lo),
        }
        .finite()
    }
    fn div(self, b: Self) -> Result<Self, Uncertifiable> {
        self.mul(b.inv()?)
    }
    fn sqrt(self) -> Result<Self, Uncertifiable> {
        if self.lo <= 0.0 {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        Self {
            lo: down(self.lo.sqrt()).max(0.0),
            hi: up(self.hi.sqrt()),
        }
        .finite()
    }
    fn sign(self) -> Option<i8> {
        if self.lo > 0.0 {
            Some(1)
        } else if self.hi < 0.0 {
            Some(-1)
        } else {
            None
        }
    }
    fn abs(self) -> Self {
        if self.lo >= 0.0 {
            self
        } else if self.hi <= 0.0 {
            self.neg()
        } else {
            Self {
                lo: 0.0,
                hi: self.lo.abs().max(self.hi.abs()),
            }
        }
    }
    fn interior_contains(self, b: Self) -> bool {
        self.lo < b.lo && b.hi < self.hi
    }
}

// Each basic IEEE-754 operation rounds to nearest. Moving its finite result
// one representable f64 outward encloses the corresponding exact real value.
// The module requires gradual underflow and no fast-math reassociation.
fn up(x: f64) -> f64 {
    if x == f64::INFINITY {
        x
    } else if x == 0.0 {
        f64::from_bits(1)
    } else {
        f64::from_bits(if x > 0.0 {
            x.to_bits() + 1
        } else {
            x.to_bits() - 1
        })
    }
}
fn down(x: f64) -> f64 {
    if x == f64::NEG_INFINITY {
        x
    } else if x == 0.0 {
        -f64::from_bits(1)
    } else {
        f64::from_bits(if x > 0.0 {
            x.to_bits() - 1
        } else {
            x.to_bits() + 1
        })
    }
}

// Derivative indices: dependent receiver coordinate, free receiver coordinate,
// ray parameter t, and the three independent sun-direction components.
#[derive(Clone, Copy)]
struct Ad {
    v: Iv,
    d: [Iv; 6],
}
impl Ad {
    fn c(v: Iv) -> Self {
        Self {
            v,
            d: [Iv::p(0.0); 6],
        }
    }
    fn var(v: Iv, k: usize) -> Self {
        let mut a = Self::c(v);
        a.d[k] = Iv::p(1.0);
        a
    }
    fn add(self, b: Self) -> Result<Self, Uncertifiable> {
        let mut d = [Iv::p(0.0); 6];
        for (k, v) in d.iter_mut().enumerate() {
            *v = self.d[k].add(b.d[k])?
        }
        Ok(Self {
            v: self.v.add(b.v)?,
            d,
        })
    }
    fn neg(self) -> Self {
        let mut a = self;
        a.v = a.v.neg();
        for x in &mut a.d {
            *x = x.neg()
        }
        a
    }
    fn sub(self, b: Self) -> Result<Self, Uncertifiable> {
        self.add(b.neg())
    }
    fn mul(self, b: Self) -> Result<Self, Uncertifiable> {
        let mut d = [Iv::p(0.0); 6];
        for (k, v) in d.iter_mut().enumerate() {
            *v = self.d[k].mul(b.v)?.add(self.v.mul(b.d[k])?)?
        }
        Ok(Self {
            v: self.v.mul(b.v)?,
            d,
        })
    }
    fn inv(self) -> Result<Self, Uncertifiable> {
        let v = self.v.inv()?;
        let q = v.mul(v)?;
        let mut d = [Iv::p(0.0); 6];
        for (k, x) in d.iter_mut().enumerate() {
            *x = self.d[k].mul(q)?.neg()
        }
        Ok(Self { v, d })
    }
    fn div(self, b: Self) -> Result<Self, Uncertifiable> {
        self.mul(b.inv()?)
    }
    fn sqrt(self) -> Result<Self, Uncertifiable> {
        let v = self.v.sqrt()?;
        let den = v.add(v)?;
        let mut d = [Iv::p(0.0); 6];
        for (k, x) in d.iter_mut().enumerate() {
            *x = self.d[k].div(den)?
        }
        Ok(Self { v, d })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Chart {
    DependentX,
    DependentZ,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EventClass {
    SmoothGrazing,
    CreaseX,
    CreaseZ,
    PerimeterX,
    PerimeterZ,
    Cutoff,
    CellEntryX,
    CellEntryZ,
}

#[derive(Clone, Copy)]
pub(crate) struct Dem<'a> {
    pub width: usize,
    pub height: usize,
    pub x0: f64,
    pub z0: f64,
    pub sx: f64,
    pub sz: f64,
    pub heights: &'a [f64],
}
#[derive(Clone, Copy)]
struct Cell {
    x0: Iv,
    z0: Iv,
    sx: f64,
    sz: f64,
    h: [f64; 4],
}
impl Dem<'_> {
    fn validate(&self) -> Result<(), Uncertifiable> {
        let n = self
            .width
            .checked_mul(self.height)
            .ok_or(Uncertifiable::InvalidInput)?;
        if self.width < 2
            || self.height < 2
            || n != self.heights.len()
            || self.sx <= 0.0
            || self.sz <= 0.0
            || ![self.x0, self.z0, self.sx, self.sz]
                .into_iter()
                .all(f64::is_finite)
            || !self.heights.iter().all(|v| v.is_finite())
        {
            return Err(Uncertifiable::InvalidInput);
        }
        Ok(())
    }
    fn cell(&self, cx: usize, cz: usize) -> Result<Cell, Uncertifiable> {
        let x0 = Iv::p(self.x0).add(Iv::p(self.sx).mul(Iv::p(cx as f64))?)?;
        let z0 = Iv::p(self.z0).add(Iv::p(self.sz).mul(Iv::p(cz as f64))?)?;
        let i = cz * self.width + cx;
        Ok(Cell {
            x0,
            z0,
            sx: self.sx,
            sz: self.sz,
            h: [
                self.heights[i],
                self.heights[i + 1],
                self.heights[i + self.width],
                self.heights[i + self.width + 1],
            ],
        })
    }
    fn max_h(&self) -> f64 {
        self.heights
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }
}
impl Cell {
    fn h(self, x: Ad, z: Ad) -> Result<Ad, Uncertifiable> {
        let u = x.sub(Ad::c(self.x0))?.div(Ad::c(Iv::p(self.sx)))?;
        let v = z.sub(Ad::c(self.z0))?.div(Ad::c(Iv::p(self.sz)))?;
        let a = Iv::p(self.h[1]).sub(Iv::p(self.h[0]))?;
        let b = Iv::p(self.h[2]).sub(Iv::p(self.h[0]))?;
        let c = Iv::p(self.h[3])
            .sub(Iv::p(self.h[1]))?
            .sub(Iv::p(self.h[2]))?
            .add(Iv::p(self.h[0]))?;
        Ad::c(Iv::p(self.h[0]))
            .add(u.mul(Ad::c(a))?)?
            .add(v.mul(Ad::c(b))?)?
            .add(u.mul(v)?.mul(Ad::c(c))?)
    }
    fn hx(self, z: Ad) -> Result<Ad, Uncertifiable> {
        let v = z.sub(Ad::c(self.z0))?.div(Ad::c(Iv::p(self.sz)))?;
        let a = Iv::p(self.h[1]).sub(Iv::p(self.h[0]))?;
        let c = Iv::p(self.h[3])
            .sub(Iv::p(self.h[1]))?
            .sub(Iv::p(self.h[2]))?
            .add(Iv::p(self.h[0]))?;
        Ad::c(a).add(v.mul(Ad::c(c))?)?.div(Ad::c(Iv::p(self.sx)))
    }
    fn hz(self, x: Ad) -> Result<Ad, Uncertifiable> {
        let u = x.sub(Ad::c(self.x0))?.div(Ad::c(Iv::p(self.sx)))?;
        let b = Iv::p(self.h[2]).sub(Iv::p(self.h[0]))?;
        let c = Iv::p(self.h[3])
            .sub(Iv::p(self.h[1]))?
            .sub(Iv::p(self.h[2]))?
            .add(Iv::p(self.h[0]))?;
        Ad::c(b).add(u.mul(Ad::c(c))?)?.div(Ad::c(Iv::p(self.sz)))
    }
    fn strict_contains(self, x: Iv, z: Iv) -> Result<bool, Uncertifiable> {
        let x1 = self.x0.add(Iv::p(self.sx))?;
        let z1 = self.z0.add(Iv::p(self.sz))?;
        Ok(x.lo > self.x0.hi && x.hi < x1.lo && z.lo > self.z0.hi && z.hi < z1.lo)
    }
    fn may_contain(self, x: Iv, z: Iv) -> Result<bool, Uncertifiable> {
        let x1 = self.x0.add(Iv::p(self.sx))?;
        let z1 = self.z0.add(Iv::p(self.sz))?;
        Ok(x.hi >= self.x0.lo && x.lo <= x1.hi && z.hi >= self.z0.lo && z.lo <= z1.hi)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Camera {
    pub origin: [f64; 3],
    pub right: [f64; 3],
    pub up: [f64; 3],
    pub forward: [f64; 3],
    /// Actual f32 projection operands captured from the replay UBO/preflight.
    pub fov_y: f64,
    pub half_h: f64,
    pub half_w: f64,
    pub aspect: f64,
    pub camera_exposure: f64,
    pub width: u32,
    pub height: u32,
    pub pixel_x: u32,
    pub pixel_y: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GpuEdgeProvenance {
    pub camera: Camera,
    pub sun_direction: [f32; 3],
    pub reused_lighting: bool,
    pub reservoir_wh: [f32; 2],
    /// The independently keyed edge-IBL draws, captured before the probe
    /// early-return so each final audit must use the same f32 random inputs.
    pub ibl_random: [f32; 2],
    /// Exact frame-UBO seed and event ids that generated `ibl_random`.
    pub seed: [u32; 2],
    pub frame: u32,
    pub sample: u32,
    /// The terrain/light UBO operands used by the source-chain replay.
    pub terrain_dimensions: [u32; 2],
    pub terrain_flags: u32,
    pub env_width: u32,
    pub env_height: f32,
    pub light_direction: [f32; 3],
    pub light_color: [f32; 3],
    pub turbidity_excess: f32,
    pub env_intensity: f32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CameraProposal {
    pub jitter_x: Iv,
    pub jitter_y: Iv,
    pub tent_pdf_x: Iv,
    pub tent_pdf_y: Iv,
    pub jacobian: Iv,
    pub density: Iv,
}
pub(crate) struct SliceInput<'a> {
    pub dem: Dem<'a>,
    pub receiver_cell: (usize, usize),
    pub chart: Chart,
    pub free: f64,
    pub dependent: Iv,
    pub omega: [f64; 3],
    pub tmin: f64,
    pub tmax: f64,
    pub normal_offset: f64,
    pub camera: Camera,
}

/// One draw over the whole DEM: chart probability 1/2 and a free coordinate
/// uniform on that chart's complete world-space extent. The host must call
/// this once per pixel/replicate/frame draw and must not preselect a receiver
/// cell, because that would change the proposal density.
pub(crate) struct SceneSliceInput<'a> {
    pub dem: Dem<'a>,
    pub chart: Chart,
    pub free: f64,
    pub omega: [f64; 3],
    pub camera: Camera,
    pub tmin: f64,
    pub tmax: f64,
    pub normal_offset: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CertifiedEvent {
    #[allow(dead_code)] // event class is retained for diagnostics and GPU fixture coverage
    pub class: EventClass,
    pub receiver: Iv,
    pub receiver_xyz: [Iv; 3],
    /// The DEM cell whose bilinear patch certified this receiver. The bounds
    /// retain outward rounding of the original DEM origin and spacing.
    pub receiver_cell: (usize, usize),
    pub receiver_cell_world: [Iv; 4], // x0, x1, z0, z1
    pub normal_xyz: [Iv; 3],
    pub pixel: [u32; 2],
    pub signed_sun_weight: [Iv; 3],
    pub camera_proposal: CameraProposal,
    pub sun_direction: [f64; 3],
    #[cfg(test)]
    pub lit_on_lower_side: bool,
    /// Exact camera inputs used for the support, first-hit, and density proof.
    pub camera: Camera,
}

/// One discrete draw is uniform over all pixels, replicates, frames, and
/// camera samples. The caller draws the chart with probability 1/2 and the
/// free coordinate uniformly over the entire corresponding DEM extent.
/// `replicate` is implicit in the destination replay dispatch.
#[derive(Clone, Copy)]
pub(crate) struct DiscreteDraw {
    pub replicate_count: u32,
    pub frame_count: u32,
    pub samples_per_frame: u32,
    pub frame: u32,
    pub sample: u32,
}

/// Matches the 64-byte WGSL `InvEdgeEvent` layout. `ids.w=1` is the shader's
/// certification marker. Conversion to f32 is accompanied by `absolute_error`
/// below; a projected record alone is not an ideal-real certificate.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct InvEdgeEvent {
    pub ids: [u32; 4],
    pub receiver: [f32; 4],
    pub normal: [f32; 4],
    pub grad_weight: [f32; 4],
}
const _: [(); 64] = [(); std::mem::size_of::<InvEdgeEvent>()];

#[derive(Clone, Copy, Debug)]
pub(crate) struct ProjectedEvent {
    pub record: InvEdgeEvent,
    /// Upper bounds on |ideal-real value - uploaded f32 value|, ordered as
    /// receiver xyz, normal xyz, then sun-gradient weight xyz.
    #[allow(dead_code)] // retained for numerical projection diagnostics
    pub absolute_error: [f64; 9],
    /// Ideal-real metadata retained until the GPU audit has proved that the
    /// actual f32 camera ray stayed on the certified receiver patch.
    pub receiver_xyz: [Iv; 3],
    pub receiver_cell: (usize, usize),
    pub receiver_cell_world: [Iv; 4],
    pub camera: Camera,
    pub camera_proposal: CameraProposal,
    pub sun_direction: [f64; 3],
    #[cfg(test)]
    pub class: EventClass,
}

/// Host-shareable mirror of `InvEdgeAudit` in `pt_edge_sample.wgsl`.
///
/// These are observations, not certificates.  In particular, a finite flag
/// in this record is accepted only after [`validate_gpu_audits`] proves the
/// corresponding terrain branch from the actual reported f32 ray.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GpuAuditRecord {
    pub camera_ray: [f32; 4],
    pub camera_hit: [f32; 4],
    pub ibl_ray: [f32; 4],
    pub ibl_origin: [f32; 4],
    pub lit_linear: [f32; 4],
    pub shadow_linear: [f32; 4],
    pub loss_jump: [f32; 4],
    pub gradient: [f32; 4],
    pub radiance_base: [f32; 4],
    pub radiance_sun: [f32; 4],
    pub radiance_ibl: [f32; 4],
    pub target_raw: [f32; 4],
    pub mean_linear: [f32; 4],
    pub original_linear: [f32; 4],
    pub camera_origin: [f32; 4],
    pub camera_right: [f32; 4],
    pub camera_up: [f32; 4],
    pub camera_forward: [f32; 4],
    pub camera_projection: [f32; 4],
    pub camera_dimensions: [u32; 4],
    pub sun_provenance: [f32; 4],
    pub reservoir_provenance: [f32; 4],
    pub source_seed: [u32; 4],
    pub source_terrain: [u32; 4],
    pub source_light_direction: [f32; 4],
    pub source_lighting: [f32; 4],
    pub source_environment: [f32; 4],
    pub environment_lookup: [u32; 4],
    pub surface_albedo: [f32; 4],
    pub surface_weights: [f32; 4],
    pub surface_texels: [u32; 4],
    pub environment_effective: [f32; 4],
    pub sampled_spectrum: [f32; 4],
}
const _: [(); 528] = [(); std::mem::size_of::<GpuAuditRecord>()];

/// Borrowed views of the values actually uploaded for a certified edge replay.
/// `heights` are the shared R32Float texels before the shader's f32
/// multiplication by `height_exaggeration`. Audited target values are captured
/// immediately after the shader textureLoad, so later target re-uploads remain
/// represented without retaining a duplicate host image.
#[derive(Debug)]
pub(crate) struct GpuAuditContext<'a> {
    pub terrain_width: usize,
    pub terrain_height: usize,
    pub terrain_origin: [f32; 2],
    pub terrain_spacing: [f32; 2],
    pub height_exaggeration: f32,
    pub heights: &'a [f32],
    /// The exact linear RGB values uploaded to the mapped albedo texture.
    /// Inverse scenes always bind this map, even when all values are equal.
    pub albedo: &'a [f32],
    /// The exact source environment map, if the scene bound one. A map whose
    /// angular lookup cannot be proven is rejected rather than treated as a
    /// fallback environment.
    pub env_map: Option<(&'a [f32], u32, u32)>,
    pub terrain_flags: u32,
    pub env_dimensions: [u32; 2],
    pub light_direction: [f32; 3],
    pub light_color: [f32; 3],
    pub turbidity_excess: f32,
    pub env_intensity: f32,
    pub seed: [u32; 2],
    pub provenance: GpuEdgeProvenance,
    pub normal_offset: f32,
    pub ray_tmin: f32,
    pub ray_tmax: f32,
    pub struct_weight: f32,
    /// The two actual f32 uniform operands of
    /// `inv_rfs = inv_params.rsv0.x * inv_params.fl.w`.  Retaining them
    /// separately lets the audit prove the base-image subtraction from the
    /// same rounded scale the shader used, rather than treating the captured
    /// scale as an unexplained observation.
    pub mean_image_weight: f32,
    pub replay_sample_weight: f32,
    /// Actual shader operands in `(lit-shadow)*jump_numerator/jump_denominator`.
    pub jump_numerator: f32,
    pub jump_denominator: f32,
}

/// Runtime, input-dependent numerical evidence for one accepted audit batch.
///
/// Reference differences are observations, never labelled as error bounds.
/// `gradient_total_interval` is separately constructed with interval f32
/// operations and the WGSL transcendental error allowances documented below;
/// it includes upload and order-independent accumulation rounding error.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GpuAuditNumericalBound {
    pub event_count: usize,
    pub accepted_camera_witness_count: usize,
    pub ibl_clear_count: usize,
    pub ibl_occluded_count: usize,
    /// Conservative f32 inverse-camera projection displacement from the
    /// ideal event's continuous proposal coordinates. It is retained as a
    /// measured input perturbation, separately from the uploaded weight.
    pub max_camera_jitter_projection_error: f64,
    pub max_upload_weight_error: f64,
    /// Width of the full keyed-RNG/cosine/normalize source enclosure for the
    /// actual f32 IBL direction used in visibility traversal.
    pub max_ibl_direction_source_width: f64,
    /// Width of the source-derived sun/IBL radiance operands before they
    /// enter the mean-image reconstruction.
    pub max_source_radiance_width: f64,
    pub max_radiance_reconstruction_width: f64,
    pub max_loss_reference_difference: f64,
    pub max_jump_reference_difference: f64,
    pub max_gradient_multiply_difference: f64,
    pub max_loss_interval_width: f64,
    pub gradient_sum_f64: [f64; 3],
    pub replay_order_sum_f32: [f32; 3],
    pub atomic_sum_observed: [f32; 3],
    pub atomic_sum_difference: [f64; 3],
    pub accumulation_rounding_bound: [f64; 3],
    pub gradient_total_interval: [[f64; 2]; 3],
}

/// Host-order accumulation of edge chunks which have individually passed
/// [`validate_gpu_audits`]. `combined_interval` is a real-valued proof range:
/// it includes each chunk's projected-input quantization range as well as
/// shader arithmetic.  It therefore advances with outward f64 interval
/// addition, rather than pretending those ideal-input alternatives are f32
/// host operands.  The separately checked `combined_edge_sum` follows the
/// actual f32 host additions, and `host_add_rounding_bound` contains only
/// their operation-derived rounding envelope.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ValidatedEdgeAccumulator {
    pub chunk_count: usize,
    pub combined_edge_sum: [f32; 3],
    pub combined_interval: [[f64; 2]; 3],
    pub host_add_rounding_bound: [f64; 3],
}

impl Default for ValidatedEdgeAccumulator {
    fn default() -> Self {
        Self {
            chunk_count: 0,
            combined_edge_sum: [0.0; 3],
            combined_interval: [[0.0, 0.0]; 3],
            host_add_rounding_bound: [0.0; 3],
        }
    }
}

impl ValidatedEdgeAccumulator {
    pub(crate) fn empty() -> Self {
        Self::default()
    }

    /// Append one validated chunk in the exact order used by the host scalar
    /// reduction. Unknown, non-normal, or interval-inconsistent sums reject;
    /// no tolerance is introduced.
    pub(crate) fn push_chunk(
        &mut self,
        chunk: &GpuAuditNumericalBound,
    ) -> Result<(), Uncertifiable> {
        let mut next_sum = [0.0_f32; 3];
        let mut next_interval = [[0.0_f64; 2]; 3];
        let mut next_rounding = [0.0_f64; 3];
        for k in 0..3 {
            let batch = Iv {
                lo: chunk.gradient_total_interval[k][0],
                hi: chunk.gradient_total_interval[k][1],
            };
            let prior = Iv {
                lo: self.combined_interval[k][0],
                hi: self.combined_interval[k][1],
            };
            let observed_batch = chunk.atomic_sum_observed[k];
            if !batch.lo.is_finite()
                || !batch.hi.is_finite()
                || batch.lo > batch.hi
                || !ordinary_f32(observed_batch)
                || !iv_contains(batch, observed_batch)
            {
                return Err(Uncertifiable::UnresolvedRoot);
            }
            let observed = self.combined_edge_sum[k] + observed_batch;
            if !ordinary_f32(observed) {
                return Err(Uncertifiable::UnresolvedRoot);
            }
            // `prior` and `batch` contain an ideal-real upload range in
            // addition to actual f32 execution error.  They are proof sets,
            // not operands of the host `f32` addition below; using the
            // WGSL-result helper here would incorrectly reject a valid
            // zero-weight range merely because its real endpoints are
            // subnormal.  The actual host operands are checked separately.
            let combined = prior.add(batch)?;
            if !iv_contains(combined, observed) {
                return Err(Uncertifiable::UnresolvedRoot);
            }

            // Bound only this host addition's f32 rounding from its actual
            // finite operands, then accumulate those nonnegative bounds
            // outward in host order.
            let exact = f64::from(self.combined_edge_sum[k]) + f64::from(observed_batch);
            if !exact.is_finite() {
                return Err(Uncertifiable::UnresolvedRoot);
            }
            let rounded = wgsl_add(
                Iv::p(f64::from(self.combined_edge_sum[k])),
                Iv::p(f64::from(observed_batch)),
            )?;
            let add_error = (rounded.hi - exact).abs().max((exact - rounded.lo).abs());
            let rounding = up(self.host_add_rounding_bound[k] + up(add_error));
            if !rounding.is_finite() {
                return Err(Uncertifiable::UnresolvedRoot);
            }
            next_sum[k] = observed;
            next_interval[k] = [combined.lo, combined.hi];
            next_rounding[k] = rounding;
        }
        self.chunk_count = self
            .chunk_count
            .checked_add(1)
            .ok_or(Uncertifiable::UnresolvedRoot)?;
        self.combined_edge_sum = next_sum;
        self.combined_interval = next_interval;
        self.host_add_rounding_bound = next_rounding;
        Ok(())
    }
}

/// Combine the independently-produced smooth scalar with an edge sum whose
/// chunks have all passed GPU audit. This proves only the final host f32
/// addition; it makes no claim about the smooth path's internal arithmetic.
/// The returned bound is the operation-derived rounding envelope of that one
/// addition for each component.
pub(crate) fn combine_smooth_and_validated_edge(
    smooth: [f32; 3],
    edge: &ValidatedEdgeAccumulator,
) -> Result<([f32; 3], [f64; 3]), Uncertifiable> {
    let mut combined = [0.0_f32; 3];
    let mut rounding_bound = [0.0_f64; 3];
    for k in 0..3 {
        let edge_interval = Iv {
            lo: edge.combined_interval[k][0],
            hi: edge.combined_interval[k][1],
        };
        if !ordinary_f32(smooth[k])
            || !ordinary_f32(edge.combined_edge_sum[k])
            || !edge_interval.lo.is_finite()
            || !edge_interval.hi.is_finite()
            || edge_interval.lo > edge_interval.hi
            || !iv_contains(edge_interval, edge.combined_edge_sum[k])
        {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        let result = smooth[k] + edge.combined_edge_sum[k];
        if !ordinary_f32(result) {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        let exact = f64::from(smooth[k]) + f64::from(edge.combined_edge_sum[k]);
        if !exact.is_finite() {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        let enclosed = wgsl_add(
            Iv::p(f64::from(smooth[k])),
            Iv::p(f64::from(edge.combined_edge_sum[k])),
        )?;
        if !iv_contains(enclosed, result) {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        let error = (enclosed.hi - exact).abs().max((exact - enclosed.lo).abs());
        if !error.is_finite() {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        combined[k] = result;
        rounding_bound[k] = up(error);
    }
    Ok((combined, rounding_bound))
}

/// Apply the production replicate average to the three already-combined sun
/// scalars. The enclosure follows the actual host operations: conversion of
/// `reps` to the f32 divisor, one f32 reciprocal, then one f32 multiply per
/// component. The returned bounds include both operation envelopes relative
/// to the exact quotient of the captured f32 operands.
pub(crate) fn scale_validated_sun_gradient(
    combined: [f32; 3],
    reps: u32,
) -> Result<([f32; 3], [f64; 3]), Uncertifiable> {
    if reps == 0 {
        return Err(Uncertifiable::InvalidInput);
    }
    let divisor = reps as f32;
    if !ordinary_f32(divisor) || divisor <= 0.0 {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let reciprocal = 1.0_f32 / divisor;
    if !ordinary_f32(reciprocal) || reciprocal <= 0.0 {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let reciprocal_interval = wgsl_div(Iv::p(1.0), Iv::p(f64::from(divisor)))?;
    if !iv_contains(reciprocal_interval, reciprocal) {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let mut scaled = [0.0_f32; 3];
    let mut rounding_bound = [0.0_f64; 3];
    for k in 0..3 {
        if !ordinary_f32(combined[k]) {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        let result = combined[k] * reciprocal;
        if !ordinary_f32(result) {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        let enclosed = wgsl_mul(Iv::p(f64::from(combined[k])), reciprocal_interval)?;
        if !iv_contains(enclosed, result) {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        let exact = f64::from(combined[k]) / f64::from(divisor);
        if !exact.is_finite() {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        let error = (enclosed.hi - exact).abs().max((exact - enclosed.lo).abs());
        if !error.is_finite() {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        scaled[k] = result;
        rounding_bound[k] = up(error);
    }
    Ok((scaled, rounding_bound))
}

/// Decode the exact storage-buffer stride without alignment casts or reading
/// padding. GPU storage values are little-endian on wgpu's supported hosts.
pub(crate) fn decode_gpu_audit_bytes(
    raw: &[u8],
) -> Result<TrackedVec<GpuAuditRecord>, Uncertifiable> {
    const STRIDE: usize = 528;
    if !raw.len().is_multiple_of(STRIDE) {
        return Err(Uncertifiable::InvalidInput);
    }
    let mut out = TrackedVec::with_capacity(raw.len() / STRIDE, "inverse-edge-audit-decode")?;
    for chunk in raw.chunks_exact(STRIDE) {
        let mut fields = [[0.0_f32; 4]; 33];
        for (field, dst) in fields.iter_mut().enumerate() {
            for (lane, value) in dst.iter_mut().enumerate() {
                let start = (field * 4 + lane) * 4;
                *value = f32::from_le_bytes(
                    chunk[start..start + 4]
                        .try_into()
                        .map_err(|_| Uncertifiable::InvalidInput)?,
                );
            }
        }
        out.push(GpuAuditRecord {
            camera_ray: fields[0],
            camera_hit: fields[1],
            ibl_ray: fields[2],
            ibl_origin: fields[3],
            lit_linear: fields[4],
            shadow_linear: fields[5],
            loss_jump: fields[6],
            gradient: fields[7],
            radiance_base: fields[8],
            radiance_sun: fields[9],
            radiance_ibl: fields[10],
            target_raw: fields[11],
            mean_linear: fields[12],
            original_linear: fields[13],
            camera_origin: fields[14],
            camera_right: fields[15],
            camera_up: fields[16],
            camera_forward: fields[17],
            camera_projection: fields[18],
            camera_dimensions: fields[19].map(f32::to_bits),
            sun_provenance: fields[20],
            reservoir_provenance: fields[21],
            source_seed: fields[22].map(f32::to_bits),
            source_terrain: fields[23].map(f32::to_bits),
            source_light_direction: fields[24],
            source_lighting: fields[25],
            source_environment: fields[26],
            environment_lookup: fields[27].map(f32::to_bits),
            surface_albedo: fields[28],
            surface_weights: fields[29],
            surface_texels: fields[30].map(f32::to_bits),
            environment_effective: fields[31],
            sampled_spectrum: fields[32],
        })?;
    }
    Ok(out)
}

/// Decode and validate the exact shader inputs captured by one preflight
/// audit. No value is repaired or renormalized: this object is the provenance
/// token against which every later event audit is compared bit-for-bit.
pub(crate) fn gpu_edge_provenance(
    audit: &GpuAuditRecord,
) -> Result<GpuEdgeProvenance, Uncertifiable> {
    let float_rows = [
        audit.camera_origin,
        audit.camera_right,
        audit.camera_up,
        audit.camera_forward,
        audit.camera_projection,
        audit.sun_provenance,
        audit.reservoir_provenance,
        audit.source_light_direction,
        audit.source_lighting,
        audit.source_environment,
    ];
    if !float_rows.iter().flatten().copied().all(ordinary_f32) {
        return Err(Uncertifiable::InvalidInput);
    }
    let [width, height, pixel_x, pixel_y] = audit.camera_dimensions;
    let [jx, jy, half_h, half_w] = audit.camera_projection;
    let [wh_x, wh_y, env_u1, env_u2] = audit.reservoir_provenance;
    let reused = match audit.sun_provenance[3].to_bits() {
        bits if bits == 0.0_f32.to_bits() => false,
        bits if bits == 1.0_f32.to_bits() => true,
        _ => return Err(Uncertifiable::InvalidInput),
    };
    if width == 0
        || height == 0
        || pixel_x >= width
        || pixel_y >= height
        || !(jx > -0.5 && jx < 0.5 && jy > -0.5 && jy < 0.5)
        || half_h <= 0.0
        || half_w <= 0.0
        || audit.camera_origin[3] <= 0.0
        || audit.camera_right[3] <= 0.0
        || audit.camera_up[3] <= 0.0
        || wh_x < 0.0
        || !(0.0..=3.0).contains(&wh_y)
        || wh_y.floor().to_bits() != wh_y.to_bits()
        || !(0.0..1.0).contains(&env_u1)
        || !(0.0..1.0).contains(&env_u2)
        || reused != (wh_y < 3.0)
        || audit.source_terrain[0] < 2
        || audit.source_terrain[1] < 2
        || (audit.source_terrain[3] == 0)
            != (audit.source_environment[1].to_bits() == 0.0_f32.to_bits())
        || audit.source_environment[1] < 0.0
        || audit.source_environment[1].floor().to_bits() != audit.source_environment[1].to_bits()
        || audit.source_light_direction[3].to_bits() != 0.0_f32.to_bits()
        || audit.source_lighting[3] < 0.0
        || audit.source_environment[0] < 0.0
        || audit.source_environment[2].to_bits() != 0.0_f32.to_bits()
        || audit.source_environment[3].to_bits() != 0.0_f32.to_bits()
    {
        return Err(Uncertifiable::InvalidInput);
    }
    let sun_direction = [
        audit.sun_provenance[0],
        audit.sun_provenance[1],
        audit.sun_provenance[2],
    ];
    if sun_direction.iter().all(|v| *v == 0.0) {
        return Err(Uncertifiable::InvalidInput);
    }
    Ok(GpuEdgeProvenance {
        camera: Camera {
            origin: [0, 1, 2].map(|k| f64::from(audit.camera_origin[k])),
            right: [0, 1, 2].map(|k| f64::from(audit.camera_right[k])),
            up: [0, 1, 2].map(|k| f64::from(audit.camera_up[k])),
            forward: [0, 1, 2].map(|k| f64::from(audit.camera_forward[k])),
            fov_y: f64::from(audit.camera_origin[3]),
            half_h: f64::from(half_h),
            half_w: f64::from(half_w),
            aspect: f64::from(audit.camera_right[3]),
            camera_exposure: f64::from(audit.camera_up[3]),
            width,
            height,
            pixel_x,
            pixel_y,
        },
        sun_direction,
        reused_lighting: reused,
        reservoir_wh: [wh_x, wh_y],
        ibl_random: [env_u1, env_u2],
        seed: [audit.source_seed[0], audit.source_seed[1]],
        frame: audit.source_seed[2],
        sample: audit.source_seed[3],
        terrain_dimensions: [audit.source_terrain[0], audit.source_terrain[1]],
        terrain_flags: audit.source_terrain[2],
        env_width: audit.source_terrain[3],
        env_height: audit.source_environment[1],
        light_direction: [
            audit.source_light_direction[0],
            audit.source_light_direction[1],
            audit.source_light_direction[2],
        ],
        light_color: [
            audit.source_lighting[0],
            audit.source_lighting[1],
            audit.source_lighting[2],
        ],
        turbidity_excess: audit.source_lighting[3],
        env_intensity: audit.source_environment[0],
    })
}

fn same_f32(a: f64, b: f32) -> bool {
    (a as f32).to_bits() == b.to_bits() && f64::from(b).to_bits() == a.to_bits()
}

fn audit_matches_provenance(audit: &GpuAuditRecord, expected: &GpuEdgeProvenance) -> bool {
    let camera = &expected.camera;
    camera
        .origin
        .iter()
        .zip(audit.camera_origin)
        .all(|(a, b)| same_f32(*a, b))
        && same_f32(camera.fov_y, audit.camera_origin[3])
        && camera
            .right
            .iter()
            .zip(audit.camera_right)
            .all(|(a, b)| same_f32(*a, b))
        && same_f32(camera.aspect, audit.camera_right[3])
        && camera
            .up
            .iter()
            .zip(audit.camera_up)
            .all(|(a, b)| same_f32(*a, b))
        && same_f32(camera.camera_exposure, audit.camera_up[3])
        && camera
            .forward
            .iter()
            .zip(audit.camera_forward)
            .all(|(a, b)| same_f32(*a, b))
        && audit.camera_forward[3].to_bits() == 0.0_f32.to_bits()
        && same_f32(camera.half_h, audit.camera_projection[2])
        && same_f32(camera.half_w, audit.camera_projection[3])
        && audit.camera_dimensions == [camera.width, camera.height, camera.pixel_x, camera.pixel_y]
        && expected
            .sun_direction
            .iter()
            .zip(audit.sun_provenance)
            .all(|(a, b)| a.to_bits() == b.to_bits())
        && audit.sun_provenance[3].to_bits()
            == (if expected.reused_lighting {
                1.0_f32
            } else {
                0.0_f32
            })
            .to_bits()
        && expected
            .reservoir_wh
            .iter()
            .zip(audit.reservoir_provenance[..2].iter().copied())
            .all(|(a, b)| a.to_bits() == b.to_bits())
        && expected
            .ibl_random
            .iter()
            .zip(audit.reservoir_provenance[2..].iter().copied())
            .all(|(a, b)| a.to_bits() == b.to_bits())
        && audit.source_seed
            == [
                expected.seed[0],
                expected.seed[1],
                expected.frame,
                expected.sample,
            ]
        && audit.source_terrain
            == [
                expected.terrain_dimensions[0],
                expected.terrain_dimensions[1],
                expected.terrain_flags,
                expected.env_width,
            ]
        && expected
            .light_direction
            .iter()
            .zip(audit.source_light_direction[..3].iter().copied())
            .all(|(a, b)| a.to_bits() == b.to_bits())
        && audit.source_light_direction[3].to_bits() == 0.0_f32.to_bits()
        && expected
            .light_color
            .iter()
            .zip(audit.source_lighting[..3].iter().copied())
            .all(|(a, b)| a.to_bits() == b.to_bits())
        && audit.source_lighting[3].to_bits() == expected.turbidity_excess.to_bits()
        && audit.source_environment[0].to_bits() == expected.env_intensity.to_bits()
        && audit.source_environment[1].to_bits() == expected.env_height.to_bits()
        && audit.source_environment[2].to_bits() == 0.0_f32.to_bits()
        && audit.source_environment[3].to_bits() == 0.0_f32.to_bits()
}

fn project_f32(v: Iv) -> Result<(f32, f64), Uncertifiable> {
    let m = v.lo * 0.5 + v.hi * 0.5;
    if !m.is_finite() {
        return Err(Uncertifiable::Density);
    }
    let f = m as f32;
    if !f.is_finite() {
        return Err(Uncertifiable::Density);
    }
    let error = v.sub(Iv::p(f64::from(f)))?.abs().hi;
    if !error.is_finite() {
        return Err(Uncertifiable::Density);
    }
    Ok((f, error))
}

fn exact_f32_cell_index(index: usize) -> Result<f32, Uncertifiable> {
    let projected = index as f32;
    // Compare in a wider integer type so Rust's saturating float-to-usize
    // conversion cannot make usize::MAX look exactly representable.
    if !projected.is_finite() || projected as u128 != index as u128 {
        return Err(Uncertifiable::Density);
    }
    Ok(projected)
}

fn receiver_stays_in_cell(event: &CertifiedEvent, projected: [f32; 4]) -> bool {
    let [x0, x1, z0, z1] = event.receiver_cell_world;
    let x = event.receiver_xyz[0];
    let z = event.receiver_xyz[2];
    let px = f64::from(projected[0]);
    let pz = f64::from(projected[2]);
    x.lo > x0.hi
        && x.hi < x1.lo
        && z.lo > z0.hi
        && z.hi < z1.lo
        && px > x0.hi
        && px < x1.lo
        && pz > z0.hi
        && pz < z1.lo
}

/// Project certified ideal-real events into replay records for one uniformly
/// sampled discrete tuple. The returned length is exact and may be zero;
/// there is no hidden truncation or event-count limit. `absolute_error`
/// bounds only upload quantization and root-box width, not the shader's own
/// arithmetic, visibility, or loss error. Those require measured replay
/// validation against the ideal-real certificate before accepting a run.
pub(crate) fn project_gpu_events(
    events: &[CertifiedEvent],
    camera: &Camera,
    draw: DiscreteDraw,
) -> Result<TrackedVec<ProjectedEvent>, Uncertifiable> {
    if camera.width == 0
        || camera.height == 0
        || draw.replicate_count == 0
        || draw.frame_count == 0
        || draw.samples_per_frame == 0
        || draw.frame >= draw.frame_count
        || draw.sample >= draw.samples_per_frame
    {
        return Err(Uncertifiable::InvalidInput);
    }
    let pixel_count = Iv::p(f64::from(camera.width)).mul(Iv::p(f64::from(camera.height)))?;
    let discrete_mass = pixel_count
        .mul(Iv::p(f64::from(draw.replicate_count)))?
        .mul(Iv::p(f64::from(draw.frame_count)))?
        .mul(Iv::p(f64::from(draw.samples_per_frame)))?;
    let mut out = TrackedVec::with_capacity(events.len(), "inverse-cert-projected-events")?;
    for event in events {
        if event.camera != *camera {
            return Err(Uncertifiable::InvalidInput);
        }
        if event.pixel[0] >= camera.width || event.pixel[1] >= camera.height {
            return Err(Uncertifiable::InvalidInput);
        }
        let pixel = event.pixel[1]
            .checked_mul(camera.width)
            .and_then(|v| v.checked_add(event.pixel[0]))
            .ok_or(Uncertifiable::InvalidInput)?;
        let mut record = InvEdgeEvent {
            ids: [pixel, draw.frame, draw.sample, 1],
            receiver: [0.0; 4],
            normal: [0.0; 4],
            grad_weight: [0.0; 4],
        };
        let mut absolute_error = [0.0; 9];
        for k in 0..3 {
            (record.receiver[k], absolute_error[k]) = project_f32(event.receiver_xyz[k])?;
            (record.normal[k], absolute_error[3 + k]) = project_f32(event.normal_xyz[k])?;
            (record.grad_weight[k], absolute_error[6 + k]) =
                project_f32(event.signed_sun_weight[k].mul(discrete_mass)?)?;
        }
        if !receiver_stays_in_cell(event, record.receiver) {
            return Err(Uncertifiable::Density);
        }
        record.receiver[3] = exact_f32_cell_index(event.receiver_cell.0)?;
        record.normal[3] = exact_f32_cell_index(event.receiver_cell.1)?;
        let n2 = record.normal[0] * record.normal[0]
            + record.normal[1] * record.normal[1]
            + record.normal[2] * record.normal[2];
        if !n2.is_finite() || n2 == 0.0 {
            return Err(Uncertifiable::Density);
        }
        out.push(ProjectedEvent {
            record,
            absolute_error,
            receiver_xyz: event.receiver_xyz,
            receiver_cell: event.receiver_cell,
            receiver_cell_world: event.receiver_cell_world,
            camera: event.camera,
            camera_proposal: event.camera_proposal,
            sun_direction: event.sun_direction,
            #[cfg(test)]
            class: event.class,
        })?;
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AuditRayBranch {
    Clear,
    Occluded,
}

fn ordinary_f32(v: f32) -> bool {
    v == 0.0 || v.is_normal()
}

fn f32_next_up(v: f32) -> f32 {
    if v == f32::INFINITY {
        v
    } else if v == 0.0 {
        f32::from_bits(1)
    } else {
        f32::from_bits(if v > 0.0 {
            v.to_bits() + 1
        } else {
            v.to_bits() - 1
        })
    }
}

fn f32_next_down(v: f32) -> f32 {
    if v == f32::NEG_INFINITY {
        v
    } else if v == 0.0 {
        -f32::from_bits(1)
    } else {
        f32::from_bits(if v > 0.0 {
            v.to_bits() - 1
        } else {
            v.to_bits() + 1
        })
    }
}

/// One-representable-value neighborhood of `v`, derived entirely from its
/// f32 bit pattern. WGSL requires correct rounding here but does not prescribe
/// a rounding direction.
fn f32_rounding_cell(v: f32) -> Result<Iv, Uncertifiable> {
    if !ordinary_f32(v) {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    Ok(Iv {
        lo: f64::from(f32_next_down(v)),
        hi: f64::from(f32_next_up(v)),
    })
}

/// Enclose `a * b + c` when the shader keeps the multiply and add separate.
/// Each captured operand is f32.  Start from the host's IEEE correctly-rounded
/// product, admit its two adjacent f32 rounding outcomes, then do the same
/// for each following addition.  This represents the finite set of permitted
/// basic-operation outcomes without turning cancellation near zero into an
/// f64 interval ambiguity.
fn f32_separate_mad_cell(a: f32, b: f32, c: f32) -> Result<Iv, Uncertifiable> {
    if !ordinary_f32(a) || !ordinary_f32(b) || !ordinary_f32(c) {
        return Err(Uncertifiable::Camera);
    }
    let product = a * b;
    if !ordinary_f32(product) {
        return Err(Uncertifiable::Camera);
    }
    let products = if product == 0.0 {
        [product; 3]
    } else {
        [f32_next_down(product), product, f32_next_up(product)]
    };
    let mut result = Iv {
        lo: f64::INFINITY,
        hi: f64::NEG_INFINITY,
    };
    for product in products {
        if !ordinary_f32(product) {
            return Err(Uncertifiable::Camera);
        }
        let sum = product + c;
        let cell = f32_rounding_cell(sum).map_err(|_| Uncertifiable::Camera)?;
        result = iv_hull(result, cell);
    }
    if result.lo.is_finite() && result.hi.is_finite() && result.lo <= result.hi {
        Ok(result)
    } else {
        Err(Uncertifiable::Camera)
    }
}

fn iv_hull(a: Iv, b: Iv) -> Iv {
    Iv {
        lo: a.lo.min(b.lo),
        hi: a.hi.max(b.hi),
    }
}

fn audit_context_valid(ctx: &GpuAuditContext<'_>, _count: usize) -> bool {
    let Some(texels) = ctx.terrain_width.checked_mul(ctx.terrain_height) else {
        return false;
    };
    ctx.terrain_width >= 2
        && ctx.terrain_height >= 2
        && texels == ctx.heights.len()
        && ctx.albedo.len() == texels.saturating_mul(3)
        && ctx
            .terrain_spacing
            .into_iter()
            .all(|v| ordinary_f32(v) && v > 0.0)
        && ctx.terrain_origin.into_iter().all(ordinary_f32)
        && ordinary_f32(ctx.height_exaggeration)
        && ctx.height_exaggeration > 0.0
        && ctx.heights.iter().copied().all(ordinary_f32)
        && ctx
            .albedo
            .iter()
            .copied()
            .all(|v| ordinary_f32(v) && v >= 0.0)
        && ctx.terrain_flags & 2 != 0
        && ((ctx.env_dimensions[0] == 0) == (ctx.env_dimensions[1] == 0))
        && ctx.light_direction.into_iter().all(ordinary_f32)
        && ctx.light_direction.into_iter().any(|v| v != 0.0)
        && ctx
            .light_color
            .into_iter()
            .all(|v| ordinary_f32(v) && v >= 0.0)
        && ordinary_f32(ctx.turbidity_excess)
        && ctx.turbidity_excess >= 0.0
        && ordinary_f32(ctx.env_intensity)
        && ctx.env_intensity >= 0.0
        && match ctx.env_map {
            Some((texels, width, height)) => {
                width == ctx.env_dimensions[0]
                    && height == ctx.env_dimensions[1]
                    && width > 0
                    && height > 0
                    && texels.len()
                        == (width as usize)
                            .checked_mul(height as usize)
                            .and_then(|n| n.checked_mul(3))
                            .unwrap_or(usize::MAX)
                    && texels.iter().copied().all(ordinary_f32)
            }
            None => ctx.env_dimensions == [0, 0],
        }
        && ctx
            .provenance
            .camera
            .origin
            .into_iter()
            .all(|v| ordinary_f32(v as f32) && f64::from(v as f32).to_bits() == v.to_bits())
        && ordinary_f32(ctx.normal_offset)
        && ordinary_f32(ctx.ray_tmin)
        && ordinary_f32(ctx.ray_tmax)
        && ctx.normal_offset > 0.0
        && ctx.ray_tmin >= 0.0
        && ctx.ray_tmax > ctx.ray_tmin
        && ordinary_f32(ctx.provenance.camera.camera_exposure as f32)
        && ctx.provenance.camera.camera_exposure > 0.0
        && ordinary_f32(ctx.struct_weight)
        && ctx.struct_weight >= 0.0
        && ordinary_f32(ctx.mean_image_weight)
        && ordinary_f32(ctx.replay_sample_weight)
        && ctx.mean_image_weight > 0.0
        && ctx.replay_sample_weight > 0.0
        && ordinary_f32(ctx.jump_numerator)
        && ordinary_f32(ctx.jump_denominator)
        && ctx.jump_numerator > 0.0
        && ctx.jump_denominator > 0.0
}

fn audit_cell_bounds(ctx: &GpuAuditContext<'_>, cx: usize, cz: usize) -> [f64; 4] {
    let x0 = ctx.terrain_origin[0] + ctx.terrain_spacing[0] * cx as f32;
    let x1 = ctx.terrain_origin[0] + ctx.terrain_spacing[0] * (cx + 1) as f32;
    let z0 = ctx.terrain_origin[1] + ctx.terrain_spacing[1] * cz as f32;
    let z1 = ctx.terrain_origin[1] + ctx.terrain_spacing[1] * (cz + 1) as f32;
    [f64::from(x0), f64::from(x1), f64::from(z0), f64::from(z1)]
}

fn audit_cell_heights(
    ctx: &GpuAuditContext<'_>,
    cx: usize,
    cz: usize,
) -> Result<[Iv; 4], Uncertifiable> {
    let i = cz * ctx.terrain_width + cx;
    let raw = [
        ctx.heights[i],
        ctx.heights[i + 1],
        ctx.heights[i + ctx.terrain_width],
        ctx.heights[i + ctx.terrain_width + 1],
    ];
    let mut out = [Iv::p(0.0); 4];
    for k in 0..4 {
        out[k] = wgsl_mul(
            Iv::p(f64::from(raw[k])),
            Iv::p(f64::from(ctx.height_exaggeration)),
        )?;
    }
    Ok(out)
}

fn audit_cell_gap(
    ctx: &GpuAuditContext<'_>,
    ray_origin: [f32; 3],
    ray_direction: [f32; 3],
    cx: usize,
    cz: usize,
    t: Iv,
) -> Result<Iv, Uncertifiable> {
    let bounds = audit_cell_bounds(ctx, cx, cz);
    let h = audit_cell_heights(ctx, cx, cz)?;
    let mut p = [Iv::p(0.0); 3];
    for k in 0..3 {
        p[k] = Iv::p(f64::from(ray_origin[k])).add(Iv::p(f64::from(ray_direction[k])).mul(t)?)?;
    }
    let u = p[0]
        .sub(Iv::p(bounds[0]))?
        .div(Iv::p(f64::from(ctx.terrain_spacing[0])))?;
    let v = p[2]
        .sub(Iv::p(bounds[2]))?
        .div(Iv::p(f64::from(ctx.terrain_spacing[1])))?;
    let a = h[1].sub(h[0])?;
    let b = h[2].sub(h[0])?;
    let c = h[3].sub(h[1])?.sub(h[2])?.add(h[0])?;
    let height = h[0]
        .add(a.mul(u)?)?
        .add(b.mul(v)?)?
        .add(c.mul(u)?.mul(v)?)?;
    p[1].sub(height)
}

fn ray_cell_span(
    bounds: [f64; 4],
    origin: [f32; 3],
    direction: [f32; 3],
    lo: f64,
    hi: f64,
) -> Option<(f64, f64)> {
    let mut enter = lo;
    let mut exit = hi;
    for (axis, b0, b1) in [(0, bounds[0], bounds[1]), (2, bounds[2], bounds[3])] {
        let o = f64::from(origin[axis]);
        let d = f64::from(direction[axis]);
        if d == 0.0 {
            if o < b0 || o > b1 {
                return None;
            }
        } else {
            let mut a = (b0 - o) / d;
            let mut b = (b1 - o) / d;
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            enter = enter.max(a);
            exit = exit.min(b);
        }
    }
    (enter <= exit).then_some((enter, exit))
}

fn classify_cell_segment(
    ctx: &GpuAuditContext<'_>,
    origin: [f32; 3],
    direction: [f32; 3],
    cell: (usize, usize),
    lo: f64,
    hi: f64,
) -> Result<AuditRayBranch, Uncertifiable> {
    let mid = lo * 0.5 + hi * 0.5;
    let d0 = audit_cell_gap(ctx, origin, direction, cell.0, cell.1, Iv::p(lo))?;
    let dm = audit_cell_gap(ctx, origin, direction, cell.0, cell.1, Iv::p(mid))?;
    let d1 = audit_cell_gap(ctx, origin, direction, cell.0, cell.1, Iv::p(hi))?;
    if d0.hi < 0.0 || dm.hi < 0.0 || d1.hi < 0.0 {
        return Ok(AuditRayBranch::Occluded);
    }
    // Bernstein control intervals enclose the exact quadratic over [lo,hi].
    let b1 = Iv::p(2.0).mul(dm)?.sub(Iv::p(0.5).mul(d0.add(d1)?)?)?;
    if d0.lo > 0.0 && b1.lo > 0.0 && d1.lo > 0.0 {
        return Ok(AuditRayBranch::Clear);
    }
    // Refine only when arithmetic evidence demands it.  The stop condition is
    // f64 representability, not an invented iteration or tolerance limit.
    if mid == lo || mid == hi {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let left = classify_cell_segment(ctx, origin, direction, cell, lo, mid)?;
    if left == AuditRayBranch::Occluded {
        return Ok(left);
    }
    classify_cell_segment(ctx, origin, direction, cell, mid, hi)
}

fn classify_audit_ray(
    ctx: &GpuAuditContext<'_>,
    origin: [f32; 3],
    direction: [f32; 3],
    lo: f64,
    hi: f64,
    skip_cell: Option<(usize, usize)>,
) -> Result<AuditRayBranch, Uncertifiable> {
    let mut result = AuditRayBranch::Clear;
    for cz in 0..ctx.terrain_height - 1 {
        for cx in 0..ctx.terrain_width - 1 {
            if skip_cell == Some((cx, cz)) {
                continue;
            }
            let Some((a, b)) =
                ray_cell_span(audit_cell_bounds(ctx, cx, cz), origin, direction, lo, hi)
            else {
                continue;
            };
            if a < b {
                let branch = classify_cell_segment(ctx, origin, direction, (cx, cz), a, b)?;
                if branch == AuditRayBranch::Occluded {
                    return Ok(branch);
                }
                result = branch;
            }
        }
    }
    Ok(result)
}

fn validate_camera_first_hit(
    ctx: &GpuAuditContext<'_>,
    event: &ProjectedEvent,
    audit: &GpuAuditRecord,
) -> Result<(), Uncertifiable> {
    if audit.camera_ray[3].to_bits() != 1.0_f32.to_bits() {
        return Err(Uncertifiable::Camera);
    }
    let direction = [
        audit.camera_ray[0],
        audit.camera_ray[1],
        audit.camera_ray[2],
    ];
    let hit = [
        audit.camera_hit[0],
        audit.camera_hit[1],
        audit.camera_hit[2],
    ];
    let hit_t = audit.camera_hit[3];
    if !direction.into_iter().all(ordinary_f32)
        || !hit.into_iter().all(ordinary_f32)
        || !ordinary_f32(hit_t)
        || hit_t <= ctx.ray_tmin
        || hit_t >= ctx.ray_tmax
    {
        return Err(Uncertifiable::Camera);
    }
    // Bind the replay uniforms to the camera used by the ideal-real support
    // proof. Signed zero is safely interchangeable for the subtraction; all
    // other values must have the exact host-uploaded f32 bit pattern, and the
    // original f64 value must lie in that result's permitted neighborhood.
    for k in 0..3 {
        let projected = event.camera.origin[k] as f32;
        let uploaded = ctx.provenance.camera.origin[k] as f32;
        let safely_equal =
            projected.to_bits() == uploaded.to_bits() || (projected == 0.0 && uploaded == 0.0);
        let upload_cell = f32_rounding_cell(uploaded)?;
        if !safely_equal
            || event.camera.origin[k] < upload_cell.lo
            || event.camera.origin[k] > upload_cell.hi
        {
            return Err(Uncertifiable::Camera);
        }
    }
    // Bind the uploaded receiver back to the certified ideal root box. The
    // stored upload error must contain the complete ideal interval, and the
    // f32 rounding neighborhood must intersect it; otherwise a same-cell but
    // different receiver could inherit this event's certificate.
    for k in 0..3 {
        let uploaded = f64::from(event.record.receiver[k]);
        let error = event.absolute_error[k];
        let permitted = Iv {
            lo: down(uploaded - error),
            hi: up(uploaded + error),
        };
        let ideal = event.receiver_xyz[k];
        let upload_cell = f32_rounding_cell(event.record.receiver[k])?;
        if ideal.lo < permitted.lo
            || ideal.hi > permitted.hi
            || ideal.hi < upload_cell.lo
            || ideal.lo > upload_cell.hi
        {
            return Err(Uncertifiable::Camera);
        }
    }
    // `validate_inverse_camera_mapping` runs before this guard and replays
    // `inv_cam_ray` from the captured jitter, projection, basis, and f32
    // operations.  Do not substitute `normalize(receiver-camera_origin)`:
    // the inverse mapping deliberately accepts the actual f32 pixel ray,
    // which can differ from that ideal geometric direction near nadir.
    let (cx, cz) = event.receiver_cell;
    if cx >= ctx.terrain_width - 1 || cz >= ctx.terrain_height - 1 {
        return Err(Uncertifiable::Camera);
    }
    let [x0, x1, z0, z1] = audit_cell_bounds(ctx, cx, cz);
    let hx = f64::from(hit[0]);
    let hz = f64::from(hit[2]);
    let [ix0, ix1, iz0, iz1] = event.receiver_cell_world;
    if !(hx > x0
        && hx < x1
        && hz > z0
        && hz < z1
        && hx > ix0.hi
        && hx < ix1.lo
        && hz > iz0.hi
        && hz < iz1.lo)
    {
        return Err(Uncertifiable::Camera);
    }
    // The reported hit point must be a possible result of the shader's f32
    // origin + direction*t evaluation. WGSL may contract this expression, so
    // enclose both the separately rounded multiply/add and the one-ulp cell
    // around an IEEE fused multiply-add of the same captured f32 operands.
    // A binary64 interval of the three-term exact FMA can manufacture a
    // cross-zero halo on cancellation; `mul_add` gives the correctly rounded
    // f32 fused operation directly, and the cell retains the permitted
    // rounding direction without introducing a synthetic rejection.
    for k in 0..3 {
        let origin_f32 = ctx.provenance.camera.origin[k] as f32;
        let separate = f32_separate_mad_cell(direction[k], hit_t, origin_f32)?;
        let fused = f32_rounding_cell(direction[k].mul_add(hit_t, origin_f32))?;
        let possible = iv_hull(separate, fused);
        if !iv_contains(possible, hit[k]) {
            return Err(Uncertifiable::Camera);
        }
    }
    // `terrain_leaf_intersect` chooses an f32 quadratic root and then forms
    // the reported point. That point is not reprojected onto the bilinear
    // patch, so either side of the true root can be returned after f32
    // rounding. Bracket the crossing with the exact adjacent f32 values on
    // both sides of the reported t; these are operation-defined endpoints,
    // not a tolerance.
    let root_lower = f32_next_down(hit_t);
    let root_upper = f32_next_up(hit_t);
    if !ordinary_f32(root_lower)
        || !ordinary_f32(root_upper)
        || root_lower >= hit_t
        || root_upper <= hit_t
        || root_lower <= ctx.ray_tmin
        || root_upper >= ctx.ray_tmax
    {
        return Err(Uncertifiable::FirstHit);
    }
    let span = ray_cell_span(
        audit_cell_bounds(ctx, cx, cz),
        ctx.provenance.camera.origin.map(|v| v as f32),
        direction,
        f64::from(ctx.ray_tmin),
        f64::from(ctx.ray_tmax),
    )
    .ok_or(Uncertifiable::FirstHit)?;
    if span.0 >= f64::from(root_lower) || f64::from(root_upper) >= span.1 {
        return Err(Uncertifiable::FirstHit);
    }
    if classify_audit_ray(
        ctx,
        ctx.provenance.camera.origin.map(|v| v as f32),
        direction,
        f64::from(ctx.ray_tmin),
        span.0,
        Some((cx, cz)),
    )? != AuditRayBranch::Clear
    {
        return Err(Uncertifiable::FirstHit);
    }
    let camera_origin = ctx.provenance.camera.origin.map(|v| v as f32);
    let entry_gap = audit_cell_gap(ctx, camera_origin, direction, cx, cz, Iv::p(span.0))?;
    if entry_gap.lo <= 0.0 {
        return Err(Uncertifiable::FirstHit);
    }
    // Prove a strictly descending crossing throughout the accepted part of
    // this bilinear cell. This excludes a hidden earlier root in the cell.
    let bounds = audit_cell_bounds(ctx, cx, cz);
    let h = audit_cell_heights(ctx, cx, cz)?;
    let t = Iv {
        lo: span.0,
        hi: span.1,
    };
    let x = Iv::p(ctx.provenance.camera.origin[0]).add(Iv::p(f64::from(direction[0])).mul(t)?)?;
    let z = Iv::p(ctx.provenance.camera.origin[2]).add(Iv::p(f64::from(direction[2])).mul(t)?)?;
    let u = x
        .sub(Iv::p(bounds[0]))?
        .div(Iv::p(f64::from(ctx.terrain_spacing[0])))?;
    let v = z
        .sub(Iv::p(bounds[2]))?
        .div(Iv::p(f64::from(ctx.terrain_spacing[1])))?;
    let du = Iv::p(f64::from(direction[0]) / f64::from(ctx.terrain_spacing[0]));
    let dv = Iv::p(f64::from(direction[2]) / f64::from(ctx.terrain_spacing[1]));
    let a = h[1].sub(h[0])?;
    let b = h[2].sub(h[0])?;
    let c = h[3].sub(h[1])?.sub(h[2])?.add(h[0])?;
    let dh = a
        .mul(du)?
        .add(b.mul(dv)?)?
        .add(c.mul(du.mul(v)?.add(dv.mul(u)?)?)?)?;
    let slope = Iv::p(f64::from(direction[1])).sub(dh)?;
    if slope.hi >= 0.0 {
        return Err(Uncertifiable::FirstHit);
    }
    let root_lower_gap = audit_cell_gap(
        ctx,
        camera_origin,
        direction,
        cx,
        cz,
        Iv::p(f64::from(root_lower)),
    )?;
    let root_upper_gap = audit_cell_gap(
        ctx,
        camera_origin,
        direction,
        cx,
        cz,
        Iv::p(f64::from(root_upper)),
    )?;
    // A positive entry gap, a strictly negative slope over the entire cell,
    // and opposite signs at the exact-adjacent f32 bracket prove one and only
    // one physical crossing in this receiver cell. The preceding-cell proof
    // above then makes it the actual camera first hit.
    if root_lower_gap.lo <= 0.0 || root_upper_gap.hi >= 0.0 {
        return Err(Uncertifiable::FirstHit);
    }
    // The actual camera hit is only a guard: shader radiance and normal-offset
    // IBL are evaluated from the separately certified uploaded receiver below.
    // f32 inverse-camera reconstruction need not return to that exact point,
    // but its entire one-ULP root bracket must stay in the certified cell. A
    // cross-cell result could take a different camera support branch and is
    // rejected; a same-cell displacement is recorded by the inverse-mapping
    // audit and cannot alter this guard-only branch.
    let rp = [
        f32_rounding_cell(event.record.receiver[0])?,
        f32_rounding_cell(event.record.receiver[1])?,
        f32_rounding_cell(event.record.receiver[2])?,
    ];
    let root_time = Iv {
        lo: f64::from(root_lower),
        hi: f64::from(root_upper),
    };
    let root_position = [0, 1, 2].map(|k| {
        Iv::p(ctx.provenance.camera.origin[k]).add(Iv::p(f64::from(direction[k])).mul(root_time)?)
    });
    let root_position = [root_position[0]?, root_position[1]?, root_position[2]?];
    let [ix0, ix1, iz0, iz1] = event.receiver_cell_world;
    if root_position[0].lo <= ix0.hi
        || root_position[0].hi >= ix1.lo
        || root_position[2].lo <= iz0.hi
        || root_position[2].hi >= iz1.lo
    {
        return Err(Uncertifiable::FirstHit);
    }
    let ru = rp[0]
        .sub(Iv::p(bounds[0]))?
        .div(Iv::p(f64::from(ctx.terrain_spacing[0])))?;
    let rv = rp[2]
        .sub(Iv::p(bounds[2]))?
        .div(Iv::p(f64::from(ctx.terrain_spacing[1])))?;
    let rh = h[0]
        .add(a.mul(ru)?)?
        .add(b.mul(rv)?)?
        .add(c.mul(ru)?.mul(rv)?)?;
    let receiver_residual = rp[1].sub(rh)?;
    if receiver_residual.lo > 0.0 || receiver_residual.hi < 0.0 {
        return Err(Uncertifiable::FirstHit);
    }
    Ok(())
}

fn srgb_f64(c: f64) -> f64 {
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.max(1e-8).powf(1.0 / 2.4) - 0.055
    }
}

fn pixel_loss_f64(
    linear: [f32; 3],
    target: [f32; 3],
    exposure: f32,
    struct_weight: f32,
) -> Result<f64, Uncertifiable> {
    let mut diff = [0.0; 3];
    for k in 0..3 {
        let x = f64::from(linear[k]) * f64::from(exposure);
        let tm = x / (1.0 + x);
        diff[k] = srgb_f64(tm) - srgb_f64(f64::from(target[k]));
    }
    let ld = 0.2126 * diff[0] + 0.7152 * diff[1] + 0.0722 * diff[2];
    let value = diff.iter().map(|v| v * v).sum::<f64>() / 3.0 + f64::from(struct_weight) * ld * ld;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Uncertifiable::UnresolvedRoot)
    }
}

fn outward_abs_difference(a: f64, b: f64) -> Result<f64, Uncertifiable> {
    let value = (a - b).abs();
    if value.is_finite() {
        Ok(up(value))
    } else {
        Err(Uncertifiable::UnresolvedRoot)
    }
}

/// A conservative bound on the difference between any values admitted by two
/// finite intervals. This records f32 input movement without pretending that
/// it is a rounding bound on the separately uploaded edge weight.
fn outward_interval_difference(a: Iv, b: Iv) -> Result<f64, Uncertifiable> {
    if !a.lo.is_finite() || !a.hi.is_finite() || !b.lo.is_finite() || !b.hi.is_finite() {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let difference = [
        (a.lo - b.lo).abs(),
        (a.lo - b.hi).abs(),
        (a.hi - b.lo).abs(),
        (a.hi - b.hi).abs(),
    ]
    .into_iter()
    .fold(0.0_f64, f64::max);
    if difference.is_finite() {
        Ok(up(difference))
    } else {
        Err(Uncertifiable::UnresolvedRoot)
    }
}

fn iv_contains(iv: Iv, value: f32) -> bool {
    let value = f64::from(value);
    value >= iv.lo && value <= iv.hi
}

fn f32_ulp(v: f32) -> Result<f64, Uncertifiable> {
    if !ordinary_f32(v) {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let center = f64::from(v);
    Ok((center - f64::from(f32_next_down(v)))
        .abs()
        .max((f64::from(f32_next_up(v)) - center).abs()))
}

/// Enclose a correctly-rounded f32 result when WGSL leaves the rounding
/// direction unspecified. The adjacent representable value on each side is
/// included; subnormal/non-finite endpoints reject the audit.
fn wgsl_correct_result(exact: Iv) -> Result<Iv, Uncertifiable> {
    if !exact.lo.is_finite() || !exact.hi.is_finite() || exact.lo > exact.hi {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    // A singleton real result that is itself an ordinary f32 is exact under
    // every permitted rounding direction. Preserve that fact rather than
    // widening an identity such as `x * 1.0` into adjacent values, which can
    // create a synthetic cross-zero subtraction in the audited mean-image
    // reconstruction.
    if exact.lo == exact.hi {
        let rounded = exact.lo as f32;
        if ordinary_f32(rounded) && f64::from(rounded).to_bits() == exact.lo.to_bits() {
            return Ok(Iv::p(exact.lo));
        }
    }
    let lo = exact.lo as f32;
    let hi = exact.hi as f32;
    if !lo.is_finite() || !hi.is_finite() {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    if exact.lo == 0.0 && exact.hi == 0.0 {
        return Ok(Iv::p(0.0));
    }
    let min_normal = f64::from(f32::MIN_POSITIVE);
    // Intermediate source expressions may legitimately cancel through zero
    // (the Frisvad basis has exactly that shape for an upright normal).  A
    // real interval can represent zero and the surrounding normal outcomes;
    // rejecting it would turn a documented rounding alternative into a false
    // proof failure.  Subnormal endpoints are widened only to the adjacent
    // normal/zero cell.  Final audited values still require `ordinary_f32`.
    let lower = f64::from(f32_next_down(lo));
    let upper = f64::from(f32_next_up(hi));
    let result = Iv {
        lo: if lower != 0.0 && lower.abs() < min_normal {
            if lower < 0.0 {
                -min_normal
            } else {
                0.0
            }
        } else {
            lower
        },
        hi: if upper != 0.0 && upper.abs() < min_normal {
            if upper > 0.0 {
                min_normal
            } else {
                0.0
            }
        } else {
            upper
        },
    };
    if result.lo > result.hi || !result.lo.is_finite() || !result.hi.is_finite() {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    Ok(result)
}

fn wgsl_add(a: Iv, b: Iv) -> Result<Iv, Uncertifiable> {
    // Every finite f32 pair has an exactly representable f64 sum (including
    // cancellation), so a point-pair needs no host outward-rounding halo
    // before the WGSL rounding enclosure is applied.
    let exact = if a.lo == a.hi && b.lo == b.hi {
        Iv::p(a.lo + b.lo)
    } else {
        a.add(b)?
    };
    wgsl_correct_result(exact)
}
fn wgsl_sub(a: Iv, b: Iv) -> Result<Iv, Uncertifiable> {
    let exact = if a.lo == a.hi && b.lo == b.hi {
        Iv::p(a.lo - b.lo)
    } else {
        a.sub(b)?
    };
    wgsl_correct_result(exact)
}
fn wgsl_mul(a: Iv, b: Iv) -> Result<Iv, Uncertifiable> {
    // A binary32 product has at most 48 significand bits, all of which fit
    // in binary64; preserve the exact point product before modelling its
    // subsequent f32 rounding.
    let exact = if a.lo == a.hi && b.lo == b.hi {
        Iv::p(a.lo * b.lo)
    } else {
        a.mul(b)?
    };
    wgsl_correct_result(exact)
}
fn wgsl_div(a: Iv, b: Iv) -> Result<Iv, Uncertifiable> {
    // WGSL specifies the 2.5-ULP division bound only for a finite,
    // sign-stable divisor whose magnitude is in [2^-126, 2^126].  A finite
    // source input outside that domain is not a numerical certificate: reject
    // the event rather than treating the host quotient as a bounded replay.
    let min_divisor = f64::from(f32::MIN_POSITIVE);
    let max_divisor = 2.0_f64.powi(126);
    let negative_domain = b.hi <= -min_divisor && b.lo >= -max_divisor;
    let positive_domain = b.lo >= min_divisor && b.hi <= max_divisor;
    if !(b.lo.is_finite() && b.hi.is_finite() && (negative_domain || positive_domain)) {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let rounded = wgsl_correct_result(a.div(b)?)?;
    // IEEE-754 division of an exact zero numerator by a finite nonzero
    // denominator is exactly signed zero.  The interval represents real
    // values, so its sign is immaterial; do not manufacture a subnormal
    // cross-zero range by applying the generic operation allowance below.
    if rounded.lo == 0.0 && rounded.hi == 0.0 {
        return Ok(Iv::p(0.0));
    }
    let lo = rounded.lo as f32;
    let hi = rounded.hi as f32;
    let radius = 2.5 * f32_ulp(lo)?.max(f32_ulp(hi)?);
    Ok(Iv {
        lo: down(rounded.lo - radius),
        hi: up(rounded.hi + radius),
    })
}

/// Enclose WGSL inverseSqrt.  The correctly-rounded neighborhood accounts
/// for the unspecified rounding direction of the reference result, then the
/// documented 2-ULP built-in allowance is applied at the finite normal
/// endpoint scale.  This intentionally over-encloses a half-ULP overlap; it
/// never assumes a host transcendental result is the device result.
fn wgsl_inverse_sqrt(x: Iv) -> Result<Iv, Uncertifiable> {
    if x.lo <= 0.0 {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let exact = Iv::p(1.0).div(x.sqrt()?)?;
    let rounded = wgsl_correct_result(exact)?;
    let lo = rounded.lo as f32;
    let hi = rounded.hi as f32;
    if !ordinary_f32(lo) || !ordinary_f32(hi) || lo <= 0.0 || hi <= 0.0 {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let radius = 2.0 * f32_ulp(lo)?.max(f32_ulp(hi)?);
    let result = Iv {
        lo: down(rounded.lo - radius),
        hi: up(rounded.hi + radius),
    };
    if result.lo <= 0.0 || !result.lo.is_finite() || !result.hi.is_finite() {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    Ok(result)
}

/// `sqrt` is specified in WGSL through `1 / inverseSqrt`.  Reusing that
/// documented 2-ULP allowance, instead of treating the host's sqrt as the
/// device result, keeps the source-chain replay conservative.
fn wgsl_sqrt(x: Iv) -> Result<Iv, Uncertifiable> {
    wgsl_div(Iv::p(1.0), wgsl_inverse_sqrt(x)?)
}

fn wgsl_min(a: Iv, b: Iv) -> Result<Iv, Uncertifiable> {
    if !a.lo.is_finite() || !a.hi.is_finite() || !b.lo.is_finite() || !b.hi.is_finite() {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    Ok(Iv {
        lo: a.lo.min(b.lo),
        hi: a.hi.min(b.hi),
    })
}

fn wgsl_max(a: Iv, b: Iv) -> Result<Iv, Uncertifiable> {
    if !a.lo.is_finite() || !a.hi.is_finite() || !b.lo.is_finite() || !b.hi.is_finite() {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    Ok(Iv {
        lo: a.lo.max(b.lo),
        hi: a.hi.max(b.hi),
    })
}

/// The WGSL `mix` definition is `x * (1 - a) + y * a`.  Arithmetic may be
/// reassociated or contracted, so retain both parenthesizations and a single
/// fused real expression.  The inputs are finite f32-source enclosures.
fn wgsl_mix(x: Iv, y: Iv, a: Iv) -> Result<Iv, Uncertifiable> {
    let one_minus_a = wgsl_sub(Iv::p(1.0), a)?;
    let defined = wgsl_add(wgsl_mul(x, one_minus_a)?, wgsl_mul(y, a)?)?;
    let reassociated = wgsl_add(x, wgsl_mul(a, wgsl_sub(y, x)?)?)?;
    let fused = wgsl_correct_result(x.mul(Iv::p(1.0).sub(a)?)?.add(y.mul(a)?)?)?;
    Ok(iv_hull(iv_hull(defined, reassociated), fused))
}

/// A representation-derived alternating-series enclosure for atan(x) on
/// [0, 1/5].  Machin's identity below uses only this range.  There is no
/// empirical iteration cap: termination requires the proven alternating
/// remainder to be no wider than accumulated outward f64 rounding.
fn atan_small_interval(x: f64) -> Result<Iv, Uncertifiable> {
    if !x.is_finite() || !(0.0..=0.2).contains(&x) {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    if x == 0.0 {
        return Ok(Iv::p(0.0));
    }
    let x2 = Iv::p(x).mul(Iv::p(x))?;
    let mut degree = 1_u64;
    let mut term = Iv::p(x);
    let mut sum = term;
    let mut subtract = true;
    loop {
        let next_degree = degree.checked_add(2).ok_or(Uncertifiable::UnresolvedRoot)?;
        let next = term
            .mul(x2)?
            .mul(Iv::p(degree as f64))?
            .div(Iv::p(next_degree as f64))?;
        let candidate = if subtract {
            sum.sub(next)?
        } else {
            sum.add(next)?
        };
        let omitted_degree = next_degree
            .checked_add(2)
            .ok_or(Uncertifiable::UnresolvedRoot)?;
        let omitted = next
            .mul(x2)?
            .mul(Iv::p(next_degree as f64))?
            .div(Iv::p(omitted_degree as f64))?;
        let tail = omitted.abs().hi;
        let width = up(candidate.hi - candidate.lo);
        if tail <= width {
            return Ok(Iv {
                lo: down(candidate.lo - tail),
                hi: up(candidate.hi + tail),
            });
        }
        if next == term || candidate == sum {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        degree = next_degree;
        term = next;
        sum = candidate;
        subtract = !subtract;
    }
}

/// A rigorous PI enclosure from Machin's formula.  The WGSL constant is
/// separately rounded to binary32 at each use; this reference only bounds
/// the mathematical trigonometric series below.
fn pi_interval() -> Result<Iv, Uncertifiable> {
    Iv::p(16.0)
        .mul(atan_small_interval(0.2)?)?
        .sub(Iv::p(4.0).mul(atan_small_interval(1.0 / 239.0)?)?)
}

/// Positive-magnitude Taylor enclosure for sin/cos.  The alternating-tail
/// bound is used only once term magnitudes have begun to decrease; before
/// then it simply advances.  Inputs are restricted to [-pi, pi], exactly the
/// documented WGSL sin/cos accuracy domain used by `inv_edge_cosine_dir`.
fn trig_point_interval(x: f64, cosine: bool) -> Result<Iv, Uncertifiable> {
    let pi = pi_interval()?;
    if !x.is_finite() || x < -pi.lo || x > pi.hi {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let magnitude = x.abs();
    let x2 = Iv::p(magnitude).mul(Iv::p(magnitude))?;
    let mut degree = if cosine { 0_u64 } else { 1_u64 };
    let mut term = if cosine { Iv::p(1.0) } else { Iv::p(magnitude) };
    let mut sum = term;
    let mut subtract = true;
    let mut decreasing = false;
    loop {
        let d1 = degree.checked_add(1).ok_or(Uncertifiable::UnresolvedRoot)?;
        let d2 = degree.checked_add(2).ok_or(Uncertifiable::UnresolvedRoot)?;
        let next = term.mul(x2)?.div(Iv::p(d1 as f64).mul(Iv::p(d2 as f64))?)?;
        decreasing |= next.hi <= term.lo;
        let candidate = if subtract {
            sum.sub(next)?
        } else {
            sum.add(next)?
        };
        let d3 = d2.checked_add(1).ok_or(Uncertifiable::UnresolvedRoot)?;
        let d4 = d2.checked_add(2).ok_or(Uncertifiable::UnresolvedRoot)?;
        let omitted = next.mul(x2)?.div(Iv::p(d3 as f64).mul(Iv::p(d4 as f64))?)?;
        let tail = omitted.abs().hi;
        let width = up(candidate.hi - candidate.lo);
        if decreasing && tail <= width {
            let result = Iv {
                lo: down(candidate.lo - tail),
                hi: up(candidate.hi + tail),
            };
            return Ok(if cosine || x >= 0.0 {
                result
            } else {
                result.neg()
            });
        }
        if next == term || candidate == sum {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        degree = d2;
        term = next;
        sum = candidate;
        subtract = !subtract;
    }
}

fn trig_interval(x: Iv, cosine: bool) -> Result<Iv, Uncertifiable> {
    if !x.lo.is_finite() || !x.hi.is_finite() || x.lo > x.hi {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let endpoints = iv_hull(
        trig_point_interval(x.lo, cosine)?,
        trig_point_interval(x.hi, cosine)?,
    );
    // Both sin and cos are globally 1-Lipschitz.  This enclosure remains
    // valid even if the tiny operation interval happens to straddle an
    // extremum, without using host libm as a mathematical oracle.
    let radius = up(x.hi - x.lo);
    Ok(Iv {
        lo: down(endpoints.lo - radius),
        hi: up(endpoints.hi + radius),
    })
}

/// WGSL §15.7 gives sin/cos absolute error <= 2^-11 for [-pi, pi].
fn wgsl_trig(x: Iv, cosine: bool) -> Result<Iv, Uncertifiable> {
    let raw = trig_interval(x, cosine)?;
    let error = 0.000_488_281_25_f64; // exactly 2^-11
    Ok(Iv {
        lo: down(raw.lo - error),
        hi: up(raw.hi + error),
    })
}

fn wgsl_sin(x: Iv) -> Result<Iv, Uncertifiable> {
    wgsl_trig(x, false)
}

fn wgsl_cos(x: Iv) -> Result<Iv, Uncertifiable> {
    wgsl_trig(x, true)
}

/// Alternating Taylor enclosure for atan on [0, 2/3].  Its term magnitudes
/// decrease from the first term, so the first omitted term bounds the
/// remainder.  This is used by the equirectangular environment-map proof,
/// not by host libm.
fn atan_series_interval(x: f64) -> Result<Iv, Uncertifiable> {
    if !x.is_finite() || !(0.0..=(2.0 / 3.0)).contains(&x) {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    if x == 0.0 {
        return Ok(Iv::p(0.0));
    }
    let x2 = Iv::p(x).mul(Iv::p(x))?;
    let mut degree = 1_u64;
    let mut term = Iv::p(x);
    let mut sum = term;
    let mut subtract = true;
    loop {
        let next_degree = degree.checked_add(2).ok_or(Uncertifiable::UnresolvedRoot)?;
        let next = term
            .mul(x2)?
            .mul(Iv::p(degree as f64))?
            .div(Iv::p(next_degree as f64))?;
        let candidate = if subtract {
            sum.sub(next)?
        } else {
            sum.add(next)?
        };
        let omitted_degree = next_degree
            .checked_add(2)
            .ok_or(Uncertifiable::UnresolvedRoot)?;
        let omitted = next
            .mul(x2)?
            .mul(Iv::p(next_degree as f64))?
            .div(Iv::p(omitted_degree as f64))?;
        let tail = omitted.abs().hi;
        let width = up(candidate.hi - candidate.lo);
        if tail <= width {
            return Ok(Iv {
                lo: down(candidate.lo - tail),
                hi: up(candidate.hi + tail),
            });
        }
        if next == term || candidate == sum {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        degree = next_degree;
        term = next;
        sum = candidate;
        subtract = !subtract;
    }
}

fn atan_point_interval(x: f64) -> Result<Iv, Uncertifiable> {
    if !x.is_finite() || x.abs() > 1.0 {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let magnitude = x.abs();
    let positive = if magnitude <= 0.2 {
        atan_series_interval(magnitude)?
    } else {
        // atan(r) = pi/4 + atan((r-1)/(r+1)), with the latter magnitude
        // <= 2/3 for r in [1/5, 1].
        let reduced = (magnitude - 1.0) / (magnitude + 1.0);
        let pi_quarter = pi_interval()?.div(Iv::p(4.0))?;
        pi_quarter.add(atan_series_interval(reduced.abs())?.neg())?
    };
    Ok(if sign < 0.0 { positive.neg() } else { positive })
}

fn atan_interval(x: Iv) -> Result<Iv, Uncertifiable> {
    if !x.lo.is_finite() || !x.hi.is_finite() || x.lo < -1.0 || x.hi > 1.0 {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let endpoints = iv_hull(atan_point_interval(x.lo)?, atan_point_interval(x.hi)?);
    let radius = up(x.hi - x.lo);
    Ok(Iv {
        lo: down(endpoints.lo - radius),
        hi: up(endpoints.hi + radius),
    })
}

/// Mathematical atan2 enclosure with explicit quadrant and ratio separation.
/// A component/ratio boundary that cannot be proven chooses no texel; the
/// audit rejects rather than guessing an equirectangular seam side.
fn rigorous_atan2(y: Iv, x: Iv) -> Result<Iv, Uncertifiable> {
    let sx = x.sign().ok_or(Uncertifiable::UnresolvedRoot)?;
    let sy = y.sign().ok_or(Uncertifiable::UnresolvedRoot)?;
    let ax = x.abs();
    let ay = y.abs();
    let pi = pi_interval()?;
    if ay.hi <= ax.lo {
        let base = atan_interval(y.div(x)?)?;
        if sx > 0 {
            Ok(base)
        } else if sy > 0 {
            pi.add(base)
        } else {
            pi.neg().add(base)
        }
    } else if ay.lo > ax.hi {
        let base = atan_interval(x.div(y)?)?;
        let half_pi = pi.div(Iv::p(2.0))?;
        if sy > 0 {
            half_pi.sub(base)
        } else {
            half_pi.neg().sub(base)
        }
    } else {
        Err(Uncertifiable::UnresolvedRoot)
    }
}

fn wgsl_atan2(y: Iv, x: Iv) -> Result<Iv, Uncertifiable> {
    // WGSL specifies the 4096-ULP atan2 bound only for a normal finite y and
    // an x magnitude in its normal range.  Enforce that domain explicitly.
    let min_normal = f64::from(f32::MIN_POSITIVE);
    if y.lo.abs().min(y.hi.abs()) < min_normal
        || x.lo.abs().min(x.hi.abs()) < min_normal
        || x.abs().hi > f64::from(f32::MAX)
    {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let raw = rigorous_atan2(y, x)?;
    let lo = raw.lo as f32;
    let hi = raw.hi as f32;
    if !ordinary_f32(lo) || !ordinary_f32(hi) {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let error = 4096.0 * f32_ulp(lo)?.max(f32_ulp(hi)?);
    Ok(Iv {
        lo: down(raw.lo - error),
        hi: up(raw.hi + error),
    })
}

fn wgsl_acos(x: Iv) -> Result<Iv, Uncertifiable> {
    let clamped = wgsl_clamp(x, -1.0, 1.0)?;
    let one_minus_square = Iv::p(1.0).sub(clamped.mul(clamped)?)?;
    if one_minus_square.lo <= 0.0 {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let raw = rigorous_atan2(one_minus_square.sqrt()?, clamped)?;
    let lo = raw.lo as f32;
    let hi = raw.hi as f32;
    if !ordinary_f32(lo) || !ordinary_f32(hi) {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    // WGSL §15.7 gives acos a 6.77e-5 absolute allowance or the inherited
    // atan2/sqrt path allowance, whichever is larger.
    let inherited = 4096.0 * f32_ulp(lo)?.max(f32_ulp(hi)?);
    let error = 6.77e-5_f64.max(inherited);
    Ok(Iv {
        lo: down(raw.lo - error),
        hi: up(raw.hi + error),
    })
}

fn wgsl_dot3(a: [Iv; 3], b: [Iv; 3]) -> Result<Iv, Uncertifiable> {
    let p = [
        wgsl_mul(a[0], b[0])?,
        wgsl_mul(a[1], b[1])?,
        wgsl_mul(a[2], b[2])?,
    ];
    let left = wgsl_add(wgsl_add(p[0], p[1])?, p[2])?;
    let right = wgsl_add(p[0], wgsl_add(p[1], p[2])?)?;
    let contracted =
        wgsl_correct_result(a[0].mul(b[0])?.add(a[1].mul(b[1])?)?.add(a[2].mul(b[2])?)?)?;
    Ok(iv_hull(iv_hull(left, right), contracted))
}

fn wgsl_cross3(a: [Iv; 3], b: [Iv; 3]) -> Result<[Iv; 3], Uncertifiable> {
    Ok([
        wgsl_sub(wgsl_mul(a[1], b[2])?, wgsl_mul(a[2], b[1])?)?,
        wgsl_sub(wgsl_mul(a[2], b[0])?, wgsl_mul(a[0], b[2])?)?,
        wgsl_sub(wgsl_mul(a[0], b[1])?, wgsl_mul(a[1], b[0])?)?,
    ])
}

fn wgsl_normalize3(v: [Iv; 3]) -> Result<[Iv; 3], Uncertifiable> {
    let dot = wgsl_dot3(v, v)?;
    if dot.lo <= 0.0 {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let inverse_length = wgsl_inverse_sqrt(dot)?;
    let length = wgsl_div(Iv::p(1.0), inverse_length)?;
    if length.lo <= 0.0 {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    Ok([
        wgsl_div(v[0], length)?,
        wgsl_div(v[1], length)?,
        wgsl_div(v[2], length)?,
    ])
}

fn exact_u32_f32(v: u32) -> Result<Iv, Uncertifiable> {
    let f = v as f32;
    if f as u32 != v {
        return Err(Uncertifiable::Camera);
    }
    Ok(Iv::p(f64::from(f)))
}

/// Re-evaluate the shader's captured receiver-to-pixel projection and the
/// corresponding `inv_cam_ray` entirely with WGSL operation enclosures.
fn validate_inverse_camera_mapping(
    event: &ProjectedEvent,
    audit: &GpuAuditRecord,
) -> Result<[f64; 2], Uncertifiable> {
    macro_rules! mapping_step {
        ($stage:literal, $result:expr) => {{
            let _ = $stage;
            $result?
        }};
    }
    let p = event.record.receiver.map(|v| Iv::p(f64::from(v)));
    let o = [0, 1, 2].map(|k| Iv::p(f64::from(audit.camera_origin[k])));
    let right = [0, 1, 2].map(|k| Iv::p(f64::from(audit.camera_right[k])));
    let up = [0, 1, 2].map(|k| Iv::p(f64::from(audit.camera_up[k])));
    let forward = [0, 1, 2].map(|k| Iv::p(f64::from(audit.camera_forward[k])));
    let delta = [
        wgsl_sub(p[0], o[0])?,
        wgsl_sub(p[1], o[1])?,
        wgsl_sub(p[2], o[2])?,
    ];
    let up_cross_forward = mapping_step!("up-cross-forward", wgsl_cross3(up, forward));
    let determinant = mapping_step!("basis determinant", wgsl_dot3(right, up_cross_forward));
    if determinant.sign().is_none() {
        return Err(Uncertifiable::Camera);
    }
    let x = mapping_step!("projected x", wgsl_dot3(delta, up_cross_forward));
    let forward_cross_right = mapping_step!("forward-cross-right", wgsl_cross3(forward, right));
    let y = mapping_step!("projected y", wgsl_dot3(delta, forward_cross_right));
    let right_cross_up = mapping_step!("right-cross-up", wgsl_cross3(right, up));
    let z = mapping_step!("projected z", wgsl_dot3(delta, right_cross_up));
    let depth = mapping_step!("camera depth", wgsl_div(z, determinant));
    if depth.lo <= 0.0 {
        return Err(Uncertifiable::Camera);
    }
    let half_h = Iv::p(f64::from(audit.camera_projection[2]));
    let half_w = Iv::p(f64::from(audit.camera_projection[3]));
    let ndc_x = mapping_step!(
        "ndc x",
        wgsl_div(mapping_step!("x/z", wgsl_div(x, z)), half_w)
    );
    let ndc_y = mapping_step!(
        "ndc y",
        wgsl_div(mapping_step!("y/z", wgsl_div(y, z)), half_h)
    );
    let width = exact_u32_f32(audit.camera_dimensions[0])?;
    let height = exact_u32_f32(audit.camera_dimensions[1])?;
    let pixel_x = exact_u32_f32(audit.camera_dimensions[2])?;
    let pixel_y = exact_u32_f32(audit.camera_dimensions[3])?;
    let jx = mapping_step!(
        "jitter x",
        wgsl_sub(
            mapping_step!(
                "jitter x scale",
                wgsl_mul(
                    mapping_step!(
                        "jitter x half",
                        wgsl_mul(
                            mapping_step!("ndc x + 1", wgsl_add(ndc_x, Iv::p(1.0))),
                            Iv::p(0.5)
                        )
                    ),
                    width,
                )
            ),
            mapping_step!("pixel x + half", wgsl_add(pixel_x, Iv::p(0.5))),
        )
    );
    let jy = mapping_step!(
        "jitter y",
        wgsl_sub(
            mapping_step!(
                "jitter y scale",
                wgsl_mul(
                    mapping_step!(
                        "jitter y half",
                        wgsl_mul(
                            mapping_step!("1 - ndc y", wgsl_sub(Iv::p(1.0), ndc_y)),
                            Iv::p(0.5)
                        )
                    ),
                    height,
                )
            ),
            mapping_step!("pixel y + half", wgsl_add(pixel_y, Iv::p(0.5))),
        )
    );
    let actual_jx = audit.camera_projection[0];
    let actual_jy = audit.camera_projection[1];
    let ideal_jx = event.camera_proposal.jitter_x;
    let ideal_jy = event.camera_proposal.jitter_y;
    if !iv_contains(jx, actual_jx)
        || !iv_contains(jy, actual_jy)
        || jx.lo <= -0.5
        || jx.hi >= 0.5
        || jy.lo <= -0.5
        || jy.hi >= 0.5
        || jx.sign().is_none()
        || jy.sign().is_none()
        || ideal_jx.sign().is_none()
        || ideal_jy.sign().is_none()
        || jx.sign() != ideal_jx.sign()
        || jy.sign() != ideal_jy.sign()
    {
        return Err(Uncertifiable::Density);
    }

    let actual_x = Iv::p(f64::from(actual_jx));
    let actual_y = Iv::p(f64::from(actual_jy));
    let abs_x = if actual_jx > 0.0 {
        actual_x
    } else if actual_jx < 0.0 {
        actual_x.neg()
    } else {
        return Err(Uncertifiable::Density);
    };
    let abs_y = if actual_jy > 0.0 {
        actual_y
    } else if actual_jy < 0.0 {
        actual_y.neg()
    } else {
        return Err(Uncertifiable::Density);
    };
    let pdf_x = mapping_step!(
        "tent pdf x",
        wgsl_mul(
            Iv::p(2.0),
            mapping_step!(
                "tent x interior",
                wgsl_sub(
                    Iv::p(1.0),
                    mapping_step!("tent x double", wgsl_mul(Iv::p(2.0), abs_x))
                )
            ),
        )
    );
    let pdf_y = mapping_step!(
        "tent pdf y",
        wgsl_mul(
            Iv::p(2.0),
            mapping_step!(
                "tent y interior",
                wgsl_sub(
                    Iv::p(1.0),
                    mapping_step!("tent y double", wgsl_mul(Iv::p(2.0), abs_y))
                )
            ),
        )
    );
    if pdf_x.lo <= 0.0
        || pdf_y.lo <= 0.0
        || event.camera_proposal.tent_pdf_x.lo <= 0.0
        || event.camera_proposal.tent_pdf_y.lo <= 0.0
        || event.camera_proposal.jacobian.lo <= 0.0
        || event.camera_proposal.density.lo <= 0.0
    {
        return Err(Uncertifiable::Density);
    }

    // Replay `inv_cam_ray` with the captured jitter and projection operands.
    let jx = Iv::p(f64::from(actual_jx));
    let jy = Iv::p(f64::from(actual_jy));
    let ndc_x = mapping_step!(
        "forward ndc x",
        wgsl_sub(
            mapping_step!(
                "forward ndc x double",
                wgsl_mul(
                    mapping_step!(
                        "forward ndc x divide",
                        wgsl_div(
                            mapping_step!(
                                "forward pixel x",
                                wgsl_add(
                                    mapping_step!(
                                        "forward pixel x half",
                                        wgsl_add(pixel_x, Iv::p(0.5))
                                    ),
                                    jx
                                )
                            ),
                            width,
                        )
                    ),
                    Iv::p(2.0),
                )
            ),
            Iv::p(1.0),
        )
    );
    let ndc_y = mapping_step!(
        "forward ndc y",
        wgsl_sub(
            mapping_step!(
                "forward ndc y double",
                wgsl_mul(
                    mapping_step!(
                        "forward ndc y complement",
                        wgsl_sub(
                            Iv::p(1.0),
                            mapping_step!(
                                "forward ndc y divide",
                                wgsl_div(
                                    mapping_step!(
                                        "forward pixel y",
                                        wgsl_add(
                                            mapping_step!(
                                                "forward pixel y half",
                                                wgsl_add(pixel_y, Iv::p(0.5))
                                            ),
                                            jy
                                        )
                                    ),
                                    height,
                                )
                            ),
                        )
                    ),
                    Iv::p(2.0),
                )
            ),
            Iv::p(1.0),
        )
    );
    let camera_ray = mapping_step!(
        "camera-space normalization",
        wgsl_normalize3([
            mapping_step!("camera ray x", wgsl_mul(ndc_x, half_w)),
            mapping_step!("camera ray y", wgsl_mul(ndc_y, half_h)),
            Iv::p(-1.0),
        ])
    );
    let world = [0, 1, 2].map(|k| {
        wgsl_add(
            wgsl_add(
                wgsl_mul(camera_ray[0], right[k])?,
                wgsl_mul(camera_ray[1], up[k])?,
            )?,
            wgsl_mul(camera_ray[2], forward[k].neg())?,
        )
    });
    let world = mapping_step!(
        "world normalization",
        wgsl_normalize3([world[0]?, world[1]?, world[2]?])
    );
    if !(0..3).all(|k| iv_contains(world[k], audit.camera_ray[k])) {
        return Err(Uncertifiable::Camera);
    }
    Ok([
        outward_interval_difference(jx, ideal_jx)?,
        outward_interval_difference(jy, ideal_jy)?,
    ])
}

/// Rigorous enclosure of ln(m), 1 <= m <= 2, using
/// ln(m)=2*sum(y^(2k+1)/(2k+1)), y=(m-1)/(m+1). After degree d the positive
/// remainder is bounded by 2*y^(d+2)/((d+2)*(1-y^2)). Iteration stops only
/// when that proven remainder is no wider than accumulated f64 interval
/// rounding; there is no empirical tolerance or iteration cap.
fn ln_unit_interval(m: f64) -> Result<Iv, Uncertifiable> {
    if !m.is_finite() || !(1.0..=2.0).contains(&m) {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    if m == 1.0 {
        return Ok(Iv::p(0.0));
    }
    let y = Iv::p(m).sub(Iv::p(1.0))?.div(Iv::p(m).add(Iv::p(1.0))?)?;
    let y2 = y.mul(y)?;
    if y.lo < 0.0 || y2.hi >= 1.0 {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let mut power = y;
    let mut sum = y;
    let mut degree = 1_u64;
    loop {
        let next_power = power.mul(y2)?;
        let next_degree = degree.checked_add(2).ok_or(Uncertifiable::UnresolvedRoot)?;
        let next_term = next_power.div(Iv::p(next_degree as f64))?;
        let next_sum = sum.add(next_term)?;
        let after_power = next_power.mul(y2)?;
        let after_degree = next_degree
            .checked_add(2)
            .ok_or(Uncertifiable::UnresolvedRoot)?;
        let tail = Iv::p(2.0)
            .mul(after_power)?
            .div(Iv::p(after_degree as f64).mul(Iv::p(1.0).sub(y2)?)?)?;
        let width = up(next_sum.hi - next_sum.lo);
        if tail.hi <= width {
            return Ok(Iv {
                lo: down(2.0 * next_sum.lo),
                hi: up(2.0 * next_sum.hi + tail.hi),
            });
        }
        if next_power == power || next_sum == sum {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        power = next_power;
        sum = next_sum;
        degree = next_degree;
    }
}

fn ln_two_interval() -> Result<Iv, Uncertifiable> {
    ln_unit_interval(2.0)
}

/// Exact bit decomposition x=m*2^e followed by the rigorous ln series above.
fn log2_point_interval(x: f64) -> Result<Iv, Uncertifiable> {
    if !x.is_normal() || x <= 0.0 {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let bits = x.to_bits();
    let raw_exp = ((bits >> 52) & 0x7ff) as i32;
    let exponent = raw_exp - 1023;
    let mantissa = f64::from_bits((bits & ((1_u64 << 52) - 1)) | (1023_u64 << 52));
    Iv::p(exponent as f64).add(ln_unit_interval(mantissa)?.div(ln_two_interval()?)?)
}

fn rigorous_log2(x: Iv) -> Result<Iv, Uncertifiable> {
    if x.lo <= 0.0 || x.lo > x.hi {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let lo = log2_point_interval(x.lo)?;
    let hi = log2_point_interval(x.hi)?;
    Ok(Iv {
        lo: lo.lo,
        hi: hi.hi,
    })
}

/// Rigorous enclosure of exp(z), 0 <= z < ln(2), using its positive Taylor
/// series. After term k, the remaining term ratios are bounded by
/// z/(k+2), yielding next_term/(1-z/(k+2)). Termination is representation-
/// derived exactly as for `ln_unit_interval`.
fn exp_unit_interval(z: Iv) -> Result<Iv, Uncertifiable> {
    if z.lo < 0.0 || z.hi >= 1.0 {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let mut sum = Iv::p(1.0);
    let mut term = Iv::p(1.0);
    let mut degree = 0_u64;
    loop {
        let next_degree = degree.checked_add(1).ok_or(Uncertifiable::UnresolvedRoot)?;
        let next_term = term.mul(z)?.div(Iv::p(next_degree as f64))?;
        let next_sum = sum.add(next_term)?;
        let ratio_den = next_degree
            .checked_add(1)
            .ok_or(Uncertifiable::UnresolvedRoot)?;
        let ratio = z.div(Iv::p(ratio_den as f64))?;
        let following = next_term.mul(z)?.div(Iv::p(ratio_den as f64))?;
        let tail = following.div(Iv::p(1.0).sub(ratio)?)?;
        let width = up(next_sum.hi - next_sum.lo);
        if tail.hi <= width {
            return Ok(Iv {
                lo: next_sum.lo,
                hi: up(next_sum.hi + tail.hi),
            });
        }
        if next_term == term || next_sum == sum {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        term = next_term;
        sum = next_sum;
        degree = next_degree;
    }
}

fn exact_power_of_two(exponent: i32) -> Result<f64, Uncertifiable> {
    if !(-1022..=1023).contains(&exponent) {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    Ok(f64::from_bits(((exponent + 1023) as u64) << 52))
}

fn exp2_point_interval(x: f64) -> Result<Iv, Uncertifiable> {
    if !x.is_finite() {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let n = x.floor();
    if n < f64::from(i32::MIN) || n > f64::from(i32::MAX) {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let exponent = n as i32;
    let remainder = Iv::p(x).sub(Iv::p(n))?;
    let z = remainder.mul(ln_two_interval()?)?;
    exp_unit_interval(z)?.mul(Iv::p(exact_power_of_two(exponent)?))
}

fn rigorous_exp2(x: Iv) -> Result<Iv, Uncertifiable> {
    let lo = exp2_point_interval(x.lo)?;
    let hi = exp2_point_interval(x.hi)?;
    Ok(Iv {
        lo: lo.lo,
        hi: hi.hi,
    })
}

/// WGSL §15.7 allowances used here: log2 has absolute error 2^-21 on
/// [0.5,2] and otherwise 3 ULP; exp2 has (3+2*|x|) ULP. The mathematical
/// endpoint references above use no host transcendental functions.
fn wgsl_log2(x: Iv) -> Result<Iv, Uncertifiable> {
    let raw = rigorous_log2(x)?;
    let error = if x.lo >= 0.5 && x.hi <= 2.0 {
        4.768_371_582_031_25e-7_f64 // exactly 2^-21
    } else {
        let lo = raw.lo as f32;
        let hi = raw.hi as f32;
        if !ordinary_f32(lo) || !ordinary_f32(hi) {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        3.0 * f32_ulp(lo)?.max(f32_ulp(hi)?)
    };
    Ok(Iv {
        lo: down(raw.lo - error),
        hi: up(raw.hi + error),
    })
}

fn wgsl_exp2(x: Iv) -> Result<Iv, Uncertifiable> {
    let raw = rigorous_exp2(x)?;
    let lo = raw.lo as f32;
    let hi = raw.hi as f32;
    if !ordinary_f32(lo) || !ordinary_f32(hi) {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let ulp = f32_ulp(lo)?.max(f32_ulp(hi)?);
    let error = (3.0 + 2.0 * x.lo.abs().max(x.hi.abs())) * ulp;
    Ok(Iv {
        lo: down(raw.lo - error),
        hi: up(raw.hi + error),
    })
}

/// WGSL §15.7 gives `exp` the same `(3 + 2*abs(x))` ULP form used for
/// `exp2`.  Convert the mathematical reference through the rigorous ln(2)
/// and exp2 enclosures; never use host libm as the claimed device result.
fn wgsl_exp(x: Iv) -> Result<Iv, Uncertifiable> {
    let raw = rigorous_exp2(x.div(ln_two_interval()?)?)?;
    let lo = raw.lo as f32;
    let hi = raw.hi as f32;
    if !ordinary_f32(lo) || !ordinary_f32(hi) || lo <= 0.0 || hi <= 0.0 {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let ulp = f32_ulp(lo)?.max(f32_ulp(hi)?);
    let error = (3.0 + 2.0 * x.lo.abs().max(x.hi.abs())) * ulp;
    Ok(Iv {
        lo: down(raw.lo - error),
        hi: up(raw.hi + error),
    })
}

fn wgsl_pow_srgb(x: Iv) -> Result<Iv, Uncertifiable> {
    let exponent = wgsl_div(Iv::p(1.0), Iv::p(f64::from(2.4_f32)))?;
    wgsl_exp2(wgsl_mul(exponent, wgsl_log2(x)?)?)
}

fn wgsl_srgb(x: Iv) -> Result<Iv, Uncertifiable> {
    let threshold = f64::from(0.0031308_f32);
    if x.hi <= threshold {
        wgsl_mul(Iv::p(f64::from(12.92_f32)), x)
    } else if x.lo > threshold {
        let floor = f64::from(1e-8_f32);
        let base = Iv {
            lo: x.lo.max(floor),
            hi: x.hi.max(floor),
        };
        wgsl_sub(
            wgsl_mul(Iv::p(f64::from(1.055_f32)), wgsl_pow_srgb(base)?)?,
            Iv::p(f64::from(0.055_f32)),
        )
    } else {
        Err(Uncertifiable::UnresolvedRoot)
    }
}

fn wgsl_loss_interval(
    linear: [Iv; 3],
    target: [f32; 3],
    exposure: f32,
    struct_weight: f32,
) -> Result<Iv, Uncertifiable> {
    let mut diff = [Iv::p(0.0); 3];
    for k in 0..3 {
        let exposed = wgsl_mul(linear[k], Iv::p(f64::from(exposure)))?;
        // `tm = exposed / (1 + exposed)`; spell it directly after proving the
        // denominator interval excludes zero.
        let tm = wgsl_div(exposed, wgsl_add(Iv::p(1.0), exposed)?)?;
        diff[k] = wgsl_sub(wgsl_srgb(tm)?, wgsl_srgb(Iv::p(f64::from(target[k])))?)?;
    }
    let square = [
        wgsl_mul(diff[0], diff[0])?,
        wgsl_mul(diff[1], diff[1])?,
        wgsl_mul(diff[2], diff[2])?,
    ];
    let rgb_sum = iv_hull(
        wgsl_add(wgsl_add(square[0], square[1])?, square[2])?,
        wgsl_add(square[0], wgsl_add(square[1], square[2])?)?,
    );
    let mse = wgsl_div(rgb_sum, Iv::p(3.0))?;
    let lum_term = [
        wgsl_mul(Iv::p(f64::from(0.2126_f32)), diff[0])?,
        wgsl_mul(Iv::p(f64::from(0.7152_f32)), diff[1])?,
        wgsl_mul(Iv::p(f64::from(0.0722_f32)), diff[2])?,
    ];
    let lum = iv_hull(
        wgsl_add(wgsl_add(lum_term[0], lum_term[1])?, lum_term[2])?,
        wgsl_add(lum_term[0], wgsl_add(lum_term[1], lum_term[2])?)?,
    );
    let structural = wgsl_mul(wgsl_mul(Iv::p(f64::from(struct_weight)), lum)?, lum)?;
    wgsl_add(mse, structural)
}

fn iv_width(iv: Iv) -> Result<f64, Uncertifiable> {
    let width = up(iv.hi - iv.lo);
    if width.is_finite() && width >= 0.0 {
        Ok(width)
    } else {
        Err(Uncertifiable::UnresolvedRoot)
    }
}

fn exact_u32_as_f32(value: u32) -> Result<f32, Uncertifiable> {
    let projected = value as f32;
    if !ordinary_f32(projected) || projected as u32 != value {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    Ok(projected)
}

fn wgsl_u32_to_f32(value: u32) -> Result<Iv, Uncertifiable> {
    wgsl_correct_result(Iv::p(f64::from(value)))
}

fn terrain_seed_hash_u32(value: u32) -> u32 {
    let mut x = value;
    x = (x ^ (x >> 16)).wrapping_mul(0x7feb_352d);
    x = (x ^ (x >> 15)).wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

fn inv_seed_u32(seed: [u32; 2], pixel: u32, frame: u32) -> u32 {
    let result = terrain_seed_hash_u32(
        seed[0]
            ^ seed[1]
            ^ terrain_seed_hash_u32(pixel)
            ^ terrain_seed_hash_u32(frame.wrapping_add(1)),
    );
    if result == 0 {
        0x6d2b_79f5
    } else {
        result
    }
}

fn xorshift_u32(state: &mut u32) -> u32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    x
}

/// Bind every final audit to the values actually uploaded for this selected
/// replica.  The matching preflight token prevents a later replay from
/// swapping any source UBO field while retaining a plausible downstream
/// loss/gradient tuple.
fn validate_source_snapshot(
    ctx: &GpuAuditContext<'_>,
    event: &ProjectedEvent,
    audit: &GpuAuditRecord,
) -> Result<(), Uncertifiable> {
    let env_height = exact_u32_as_f32(ctx.env_dimensions[1])?;
    if audit.source_seed
        != [
            ctx.seed[0],
            ctx.seed[1],
            event.record.ids[1],
            event.record.ids[2],
        ]
        || audit.source_terrain
            != [
                ctx.terrain_width as u32,
                ctx.terrain_height as u32,
                ctx.terrain_flags,
                ctx.env_dimensions[0],
            ]
        || !ctx
            .light_direction
            .iter()
            .zip(audit.source_light_direction[..3].iter())
            .all(|(expected, observed)| expected.to_bits() == observed.to_bits())
        || audit.source_light_direction[3].to_bits() != 0.0_f32.to_bits()
        || !ctx
            .light_color
            .iter()
            .zip(audit.source_lighting[..3].iter())
            .all(|(expected, observed)| expected.to_bits() == observed.to_bits())
        || audit.source_lighting[3].to_bits() != ctx.turbidity_excess.to_bits()
        || audit.source_environment[0].to_bits() != ctx.env_intensity.to_bits()
        || audit.source_environment[1].to_bits() != env_height.to_bits()
        || audit.source_environment[2].to_bits() != 0.0_f32.to_bits()
        || audit.source_environment[3].to_bits() != 0.0_f32.to_bits()
    {
        return Err(Uncertifiable::InvalidInput);
    }
    Ok(())
}

fn validate_source_omega(
    ctx: &GpuAuditContext<'_>,
    audit: &GpuAuditRecord,
) -> Result<[Iv; 3], Uncertifiable> {
    let raw = ctx.light_direction.map(|v| Iv::p(f64::from(v)));
    let mut omega = wgsl_normalize3(raw)?;
    if audit.reservoir_provenance[1] < 3.0 {
        omega = wgsl_normalize3(omega)?;
    }
    if !(0..3).all(|k| iv_contains(omega[k], audit.sun_provenance[k])) {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    Ok(omega)
}

/// Replay the independent keyed random stream and the complete local cosine
/// construction used only by `main_inverse_edge_certified`.  The host has no
/// latitude to substitute a reported IBL ray: unknown phase branch, intrinsic
/// domain, or f32 result rejects before terrain visibility is classified.
fn validate_ibl_direction_source(
    ctx: &GpuAuditContext<'_>,
    event: &ProjectedEvent,
    audit: &GpuAuditRecord,
) -> Result<[Iv; 3], Uncertifiable> {
    macro_rules! direction_step {
        ($stage:literal, $result:expr) => {{
            let _ = $stage;
            $result?
        }};
    }
    let pixel = event.record.ids[0];
    let frame = event.record.ids[1];
    let sample = event.record.ids[2];
    let mut state = terrain_seed_hash_u32(
        inv_seed_u32(ctx.seed, pixel, frame) ^ terrain_seed_hash_u32(sample) ^ 0xa511_e9b3,
    );
    if state == 0 {
        state = 0x6d2b_79f5;
    }
    let u1 = direction_step!(
        "first xorshift conversion",
        wgsl_div(
            direction_step!(
                "first u32 conversion",
                wgsl_u32_to_f32(xorshift_u32(&mut state))
            ),
            Iv::p(4_294_967_296.0),
        )
    );
    let u2 = direction_step!(
        "second xorshift conversion",
        wgsl_div(
            direction_step!(
                "second u32 conversion",
                wgsl_u32_to_f32(xorshift_u32(&mut state))
            ),
            Iv::p(4_294_967_296.0),
        )
    );
    if !iv_contains(u1, audit.reservoir_provenance[2])
        || !iv_contains(u2, audit.reservoir_provenance[3])
    {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let n = [
        event.record.normal[0],
        event.record.normal[1],
        event.record.normal[2],
    ]
    .map(|v| Iv::p(f64::from(v)));
    let sign = Iv::p(if event.record.normal[2] < 0.0 {
        -1.0
    } else {
        1.0
    });
    let a = direction_step!(
        "Frisvad reciprocal",
        wgsl_div(
            Iv::p(-1.0),
            direction_step!("Frisvad denominator", wgsl_add(sign, n[2]))
        )
    );
    let b = direction_step!(
        "Frisvad b",
        wgsl_mul(direction_step!("Frisvad xy", wgsl_mul(n[0], n[1])), a)
    );
    let t = [
        direction_step!(
            "Frisvad tangent x",
            wgsl_add(
                Iv::p(1.0),
                direction_step!(
                    "Frisvad tangent scale",
                    wgsl_mul(
                        direction_step!(
                            "Frisvad tangent xx",
                            wgsl_mul(
                                direction_step!("Frisvad tangent sx", wgsl_mul(sign, n[0])),
                                n[0]
                            )
                        ),
                        a,
                    )
                ),
            )
        ),
        direction_step!("Frisvad tangent y", wgsl_mul(sign, b)),
        direction_step!("Frisvad tangent z", wgsl_mul(sign.neg(), n[0])),
    ];
    let bt = [
        b,
        direction_step!(
            "Frisvad bitangent y",
            wgsl_add(
                sign,
                direction_step!(
                    "Frisvad bitangent scale",
                    wgsl_mul(
                        direction_step!("Frisvad bitangent yy", wgsl_mul(n[1], n[1])),
                        a
                    )
                ),
            )
        ),
        n[1].neg(),
    ];
    let pi = Iv::p(f64::from(std::f32::consts::PI));
    let two_pi = direction_step!("two pi", wgsl_mul(Iv::p(2.0), pi));
    let raw_phi = direction_step!("phase", wgsl_mul(two_pi, u2));
    let phi = if raw_phi.hi <= pi.lo {
        raw_phi
    } else if raw_phi.lo > pi.hi {
        direction_step!("phase reduction", wgsl_sub(raw_phi, two_pi))
    } else {
        return Err(Uncertifiable::UnresolvedRoot);
    };
    let radial = direction_step!("cosine radial sqrt", wgsl_sqrt(u1));
    let local = [
        direction_step!(
            "cosine local x",
            wgsl_mul(radial, direction_step!("cosine", wgsl_cos(phi)))
        ),
        direction_step!(
            "cosine local y",
            wgsl_mul(radial, direction_step!("sine", wgsl_sin(phi)))
        ),
        direction_step!(
            "cosine local z",
            wgsl_sqrt(direction_step!(
                "cosine z max",
                wgsl_max(
                    Iv::p(0.0),
                    direction_step!("cosine z subtract", wgsl_sub(Iv::p(1.0), u1)),
                )
            ))
        ),
    ];
    let world = [0, 1, 2].map(|k| {
        wgsl_add(
            wgsl_add(wgsl_mul(local[0], t[k])?, wgsl_mul(local[1], bt[k])?)?,
            wgsl_mul(local[2], n[k])?,
        )
    });
    let direction = direction_step!(
        "cosine world normalization",
        wgsl_normalize3([
            direction_step!("cosine world x", world[0]),
            direction_step!("cosine world y", world[1]),
            direction_step!("cosine world z", world[2]),
        ])
    );
    if !(0..3).all(|k| iv_contains(direction[k], audit.ibl_ray[k])) {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    Ok(direction)
}

fn wgsl_clamp(x: Iv, lo: f64, hi: f64) -> Result<Iv, Uncertifiable> {
    if !lo.is_finite() || !hi.is_finite() || lo > hi {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    if x.hi <= lo {
        Ok(Iv::p(lo))
    } else if x.lo >= hi {
        Ok(Iv::p(hi))
    } else {
        Ok(Iv {
            lo: x.lo.max(lo),
            hi: x.hi.min(hi),
        })
    }
}

fn stable_floor_fraction(coordinate: Iv, base: u32, extent: u32) -> Result<Iv, Uncertifiable> {
    if base >= extent {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let base_f = wgsl_u32_to_f32(base)?;
    if base + 1 < extent {
        let next_f = wgsl_u32_to_f32(base + 1)?;
        if coordinate.lo < base_f.lo || coordinate.hi >= next_f.hi {
            return Err(Uncertifiable::UnresolvedRoot);
        }
    } else if coordinate.lo < base_f.lo || coordinate.hi > base_f.hi {
        // The clamp makes the final texel exactly integral.  A wider result
        // cannot prove which side of the f32 clamp/floor boundary executed.
        return Err(Uncertifiable::UnresolvedRoot);
    }
    wgsl_sub(coordinate, base_f)
}

/// Validate the live mapped-albedo texture load, bilinear weights, and two
/// nested WGSL `mix` operations.  An unstable floor/clamp result rejects; no
/// host-side re-sampling is substituted for the shader's texels.
fn validate_surface_albedo_source(
    ctx: &GpuAuditContext<'_>,
    event: &ProjectedEvent,
    audit: &GpuAuditRecord,
) -> Result<[Iv; 3], Uncertifiable> {
    if ctx.terrain_flags & 2 == 0 {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let width = u32::try_from(ctx.terrain_width).map_err(|_| Uncertifiable::UnresolvedRoot)?;
    let height = u32::try_from(ctx.terrain_height).map_err(|_| Uncertifiable::UnresolvedRoot)?;
    let texel_count = width
        .checked_mul(height)
        .ok_or(Uncertifiable::UnresolvedRoot)?;
    if audit.surface_texels[0] >= texel_count {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let p = event.record.receiver;
    let tx = wgsl_clamp(
        wgsl_div(
            wgsl_sub(
                Iv::p(f64::from(p[0])),
                Iv::p(f64::from(ctx.terrain_origin[0])),
            )?,
            Iv::p(f64::from(ctx.terrain_spacing[0])),
        )?,
        0.0,
        f64::from(exact_u32_as_f32(width)? - 1.0),
    )?;
    let tz = wgsl_clamp(
        wgsl_div(
            wgsl_sub(
                Iv::p(f64::from(p[2])),
                Iv::p(f64::from(ctx.terrain_origin[1])),
            )?,
            Iv::p(f64::from(ctx.terrain_spacing[1])),
        )?,
        0.0,
        f64::from(exact_u32_as_f32(height)? - 1.0),
    )?;
    let x0 = audit.surface_texels[0] % width;
    let z0 = audit.surface_texels[0] / width;
    let x1 = (x0 + 1).min(width - 1);
    let z1 = (z0 + 1).min(height - 1);
    let expected_texels = [
        z0 * width + x0,
        z0 * width + x1,
        z1 * width + x0,
        z1 * width + x1,
    ];
    if audit.surface_texels != expected_texels {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let fx = stable_floor_fraction(tx, x0, width)?;
    let fz = stable_floor_fraction(tz, z0, height)?;
    let weights = [
        wgsl_mul(wgsl_sub(Iv::p(1.0), fx)?, wgsl_sub(Iv::p(1.0), fz)?)?,
        wgsl_mul(fx, wgsl_sub(Iv::p(1.0), fz)?)?,
        wgsl_mul(wgsl_sub(Iv::p(1.0), fx)?, fz)?,
        wgsl_mul(fx, fz)?,
    ];
    if !(0..4).all(|k| iv_contains(weights[k], audit.surface_weights[k])) {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let mut albedo = [Iv::p(0.0); 3];
    for channel in 0..3 {
        let sample = |texel: u32| -> Result<Iv, Uncertifiable> {
            let offset = usize::try_from(texel)
                .ok()
                .and_then(|v| v.checked_mul(3))
                .and_then(|v| v.checked_add(channel))
                .ok_or(Uncertifiable::UnresolvedRoot)?;
            ctx.albedo
                .get(offset)
                .copied()
                .filter(|value| ordinary_f32(*value) && *value >= 0.0)
                .map(|value| Iv::p(f64::from(value)))
                .ok_or(Uncertifiable::UnresolvedRoot)
        };
        let row0 = wgsl_mix(sample(expected_texels[0])?, sample(expected_texels[1])?, fx)?;
        let row1 = wgsl_mix(sample(expected_texels[2])?, sample(expected_texels[3])?, fx)?;
        albedo[channel] = wgsl_mix(row0, row1, fz)?;
        if !iv_contains(albedo[channel], audit.surface_albedo[channel]) {
            return Err(Uncertifiable::UnresolvedRoot);
        }
    }
    Ok(albedo)
}

fn atmosphere_terms(
    ctx: &GpuAuditContext<'_>,
    omega_y: Iv,
) -> Result<(Iv, [Iv; 3]), Uncertifiable> {
    let turbidity = wgsl_add(
        wgsl_max(Iv::p(f64::from(ctx.turbidity_excess)), Iv::p(0.0))?,
        Iv::p(1.0),
    )?;
    let excess = wgsl_sub(turbidity, Iv::p(1.0))?;
    let haze = wgsl_add(Iv::p(1.0), wgsl_mul(Iv::p(f64::from(0.12_f32)), excess)?)?;
    let air_mass = wgsl_div(Iv::p(1.0), wgsl_max(omega_y, Iv::p(f64::from(0.05_f32)))?)?;
    let beta = [0.10_f32, 0.16_f32, 0.26_f32];
    let mut transmit = [Iv::p(0.0); 3];
    for k in 0..3 {
        transmit[k] = wgsl_exp(wgsl_mul(
            Iv::p(-f64::from(beta[k])),
            wgsl_mul(excess, air_mass)?,
        )?)?;
    }
    Ok((haze, transmit))
}

fn stable_environment_texel(coordinate: Iv, extent: u32) -> Result<u32, Uncertifiable> {
    if extent == 0 || coordinate.lo < 0.0 || coordinate.hi > 1.0 {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let scaled = wgsl_mul(coordinate, wgsl_u32_to_f32(extent)?)?;
    if scaled.lo < 0.0 || scaled.hi > f64::from(extent) {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let lo = scaled.lo.floor();
    let hi = scaled.hi.floor();
    if lo < 0.0 || hi < 0.0 || lo > f64::from(u32::MAX) || hi > f64::from(u32::MAX) {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let lower = (lo as u32).min(extent - 1);
    let upper = (hi as u32).min(extent - 1);
    if lower != upper {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    Ok(lower)
}

fn validate_environment_source(
    ctx: &GpuAuditContext<'_>,
    audit: &GpuAuditRecord,
    omega_y: Iv,
) -> Result<[Iv; 3], Uncertifiable> {
    let (haze, _) = atmosphere_terms(ctx, omega_y)?;
    if audit.environment_effective[3].to_bits() != 0.0_f32.to_bits() {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let mut environment = [Iv::p(0.0); 3];
    match ctx.env_map {
        None => {
            if audit.environment_lookup != [0, 0, 0, 0] {
                return Err(Uncertifiable::UnresolvedRoot);
            }
            for k in 0..3 {
                environment[k] = wgsl_mul(Iv::p(f64::from(ctx.env_intensity)), haze)?;
            }
        }
        Some((texels, width, height)) => {
            if audit.environment_lookup[2] != width || audit.environment_lookup[3] != height {
                return Err(Uncertifiable::UnresolvedRoot);
            }
            // `terrain_env_radiance` normalizes the actual f32 edge ray a
            // second time.  That ray was already source-validated above; we
            // now replay the map lookup from those exact shader operands.
            let d = wgsl_normalize3([
                Iv::p(f64::from(audit.ibl_ray[0])),
                Iv::p(f64::from(audit.ibl_ray[1])),
                Iv::p(f64::from(audit.ibl_ray[2])),
            ])?;
            let pi = Iv::p(f64::from(std::f32::consts::PI));
            let uu = wgsl_add(
                wgsl_div(wgsl_atan2(d[2], d[0])?, wgsl_mul(Iv::p(2.0), pi)?)?,
                Iv::p(0.5),
            )?;
            let vv = wgsl_div(wgsl_acos(d[1])?, pi)?;
            let px = stable_environment_texel(uu, width)?;
            let py = stable_environment_texel(vv, height)?;
            if audit.environment_lookup[0] != px || audit.environment_lookup[1] != py {
                return Err(Uncertifiable::UnresolvedRoot);
            }
            let texel = usize::try_from(py)
                .ok()
                .and_then(|row| row.checked_mul(width as usize))
                .and_then(|row| row.checked_add(px as usize))
                .and_then(|index| index.checked_mul(3))
                .ok_or(Uncertifiable::UnresolvedRoot)?;
            for k in 0..3 {
                let sample = *texels.get(texel + k).ok_or(Uncertifiable::UnresolvedRoot)?;
                if !ordinary_f32(sample) {
                    return Err(Uncertifiable::UnresolvedRoot);
                }
                environment[k] = wgsl_mul(
                    wgsl_mul(
                        Iv::p(f64::from(sample)),
                        Iv::p(f64::from(ctx.env_intensity)),
                    )?,
                    haze,
                )?;
            }
        }
    }
    for k in 0..3 {
        if !iv_contains(environment[k], audit.environment_effective[k]) {
            return Err(Uncertifiable::UnresolvedRoot);
        }
    }
    Ok(environment)
}

fn validate_source_radiances(
    ctx: &GpuAuditContext<'_>,
    event: &ProjectedEvent,
    audit: &GpuAuditRecord,
    omega: [Iv; 3],
    ibl_branch: AuditRayBranch,
) -> Result<([Iv; 3], [Iv; 3]), Uncertifiable> {
    if audit.sampled_spectrum[3].to_bits() != 0.0_f32.to_bits()
        || audit.surface_albedo[3] < 0.0
        || audit.radiance_sun[3].to_bits() != 0.0_f32.to_bits()
    {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let albedo = validate_surface_albedo_source(ctx, event, audit)?;
    let (haze, transmit) = atmosphere_terms(ctx, omega[1])?;
    let spectrum = [0, 1, 2].map(|k| wgsl_mul(Iv::p(f64::from(ctx.light_color[k])), transmit[k]));
    let spectrum = [spectrum[0]?, spectrum[1]?, spectrum[2]?];
    let baseline = wgsl_min(wgsl_min(spectrum[0], spectrum[1])?, spectrum[2])?;
    let residual = [0, 1, 2].map(|k| wgsl_max(wgsl_sub(spectrum[k], baseline)?, Iv::p(0.0)));
    let residual = [residual[0]?, residual[1]?, residual[2]?];
    let sampled = if audit.reservoir_provenance[1] < 3.0 {
        let channel = audit.reservoir_provenance[1] as usize;
        [0, 1, 2].map(|k| {
            let mask = Iv::p(if channel == k { 1.0 } else { 0.0 });
            wgsl_add(
                baseline,
                wgsl_mul(
                    wgsl_mul(mask, residual[k])?,
                    Iv::p(f64::from(audit.reservoir_provenance[0])),
                )?,
            )
        })
    } else {
        spectrum.map(Ok)
    };
    let sampled = [sampled[0]?, sampled[1]?, sampled[2]?];
    if !(0..3).all(|k| iv_contains(sampled[k], audit.sampled_spectrum[k])) {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let n = [
        event.record.normal[0],
        event.record.normal[1],
        event.record.normal[2],
    ]
    .map(|v| Iv::p(f64::from(v)));
    let nd = wgsl_max(wgsl_dot3(n, omega)?, Iv::p(0.0))?;
    if !iv_contains(nd, audit.surface_albedo[3]) {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let environment = validate_environment_source(ctx, audit, omega[1])?;
    // `haze` is retained here to make the exact environment/sun shared
    // atmosphere dependency explicit; `validate_environment_source` repeats
    // the same source arithmetic before comparing the audited value.
    let _ = haze;
    let mut sun = [Iv::p(0.0); 3];
    let mut ibl = [Iv::p(0.0); 3];
    for k in 0..3 {
        sun[k] = wgsl_mul(wgsl_mul(albedo[k], sampled[k])?, nd)?;
        ibl[k] = if ibl_branch == AuditRayBranch::Clear {
            wgsl_mul(albedo[k], environment[k])?
        } else {
            Iv::p(0.0)
        };
        if !iv_contains(sun[k], audit.radiance_sun[k])
            || !iv_contains(ibl[k], audit.radiance_ibl[k])
        {
            return Err(Uncertifiable::UnresolvedRoot);
        }
    }
    Ok((sun, ibl))
}

/// Validate every captured GPU event against its retained ideal-real event.
/// Unknown terrain branches, grazing camera roots, malformed flags, and
/// unresolved arithmetic are rejected rather than counted as validated.
pub(crate) fn validate_gpu_audits(
    ctx: &GpuAuditContext<'_>,
    events: &[ProjectedEvent],
    audits: &[GpuAuditRecord],
    atomic_edge_sum: [f32; 3],
) -> Result<GpuAuditNumericalBound, Uncertifiable> {
    macro_rules! audit_step {
        ($stage:literal, $result:expr) => {{
            let _ = $stage;
            $result?
        }};
    }
    if events.len() != audits.len() || !audit_context_valid(ctx, events.len()) {
        return Err(Uncertifiable::InvalidInput);
    }
    if !atomic_edge_sum.into_iter().all(ordinary_f32) {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let mut summary = GpuAuditNumericalBound {
        event_count: events.len(),
        atomic_sum_observed: atomic_edge_sum,
        ..GpuAuditNumericalBound::default()
    };
    let mut total_gradient = [Iv::p(0.0); 3];
    let mut sum_abs_gradient = [0.0_f64; 3];
    for (event, audit) in events.iter().zip(audits) {
        let all = [
            audit.camera_ray,
            audit.camera_hit,
            audit.ibl_ray,
            audit.ibl_origin,
            audit.lit_linear,
            audit.shadow_linear,
            audit.loss_jump,
            audit.gradient,
            audit.radiance_base,
            audit.radiance_sun,
            audit.radiance_ibl,
            audit.target_raw,
            audit.mean_linear,
            audit.original_linear,
            audit.camera_origin,
            audit.camera_right,
            audit.camera_up,
            audit.camera_forward,
            audit.camera_projection,
            audit.sun_provenance,
            audit.reservoir_provenance,
            audit.source_light_direction,
            audit.source_lighting,
            audit.source_environment,
            audit.surface_albedo,
            audit.surface_weights,
            audit.environment_effective,
            audit.sampled_spectrum,
        ];
        if !all.iter().flatten().copied().all(ordinary_f32)
            || audit.ibl_ray[3].to_bits() != 0.0_f32.to_bits()
                && audit.ibl_ray[3].to_bits() != 1.0_f32.to_bits()
        {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        let actual_provenance = gpu_edge_provenance(audit)?;
        if actual_provenance != ctx.provenance
            || !audit_matches_provenance(audit, &ctx.provenance)
            || event.camera != ctx.provenance.camera
            || !(0..3).all(|k| {
                (event.sun_direction[k] as f32).to_bits()
                    == ctx.provenance.sun_direction[k].to_bits()
                    && f64::from(ctx.provenance.sun_direction[k]).to_bits()
                        == event.sun_direction[k].to_bits()
            })
        {
            return Err(Uncertifiable::InvalidInput);
        }
        audit_step!(
            "source UBO snapshot",
            validate_source_snapshot(ctx, event, audit)
        );
        let source_omega = audit_step!("source sun direction", validate_source_omega(ctx, audit));
        let camera_projection_error = audit_step!(
            "inverse camera mapping",
            validate_inverse_camera_mapping(event, audit)
        );
        summary.max_camera_jitter_projection_error = summary
            .max_camera_jitter_projection_error
            .max(camera_projection_error[0])
            .max(camera_projection_error[1]);
        audit_step!(
            "camera first hit",
            validate_camera_first_hit(ctx, event, audit)
        );
        summary.accepted_camera_witness_count += 1;
        let p = event.record.receiver;
        let n = event.record.normal;
        for k in 0..3 {
            // The shader forms `p + n * 1e-3`.  A backend may retain those
            // two basic f32 operations or contract them, so admit only the
            // operation-defined cells for both forms.  This binds the
            // branch proof below to the reported, actual offset origin;
            // ambiguity in its terrain traversal still rejects there.
            let separate = f32_separate_mad_cell(n[k], ctx.normal_offset, p[k])?;
            let fused = f32_rounding_cell(n[k].mul_add(ctx.normal_offset, p[k]))?;
            let possible = iv_hull(separate, fused);
            if !iv_contains(possible, audit.ibl_origin[k]) {
                return Err(Uncertifiable::UnresolvedRoot);
            }
        }
        let source_ibl_direction = audit_step!(
            "keyed IBL direction",
            validate_ibl_direction_source(ctx, event, audit)
        );
        for component in source_ibl_direction {
            summary.max_ibl_direction_source_width = summary
                .max_ibl_direction_source_width
                .max(iv_width(component)?);
        }
        let ibl_branch = audit_step!(
            "IBL visibility",
            classify_audit_ray(
                ctx,
                [
                    audit.ibl_origin[0],
                    audit.ibl_origin[1],
                    audit.ibl_origin[2],
                ],
                [audit.ibl_ray[0], audit.ibl_ray[1], audit.ibl_ray[2]],
                f64::from(ctx.ray_tmin),
                f64::from(ctx.ray_tmax),
                None,
            )
        );
        let reported_occluded = audit.ibl_ray[3] == 1.0;
        if reported_occluded != (ibl_branch == AuditRayBranch::Occluded) {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        match ibl_branch {
            AuditRayBranch::Clear => summary.ibl_clear_count += 1,
            AuditRayBranch::Occluded => summary.ibl_occluded_count += 1,
        }
        let (source_sun, source_ibl) = audit_step!(
            "event radiance sources",
            validate_source_radiances(ctx, event, audit, source_omega, ibl_branch)
        );
        for component in source_sun.into_iter().chain(source_ibl) {
            let width = iv_width(component)?;
            summary.max_source_radiance_width = summary.max_source_radiance_width.max(width);
            summary.max_radiance_reconstruction_width =
                summary.max_radiance_reconstruction_width.max(width);
        }
        let ibl_visible = audit.radiance_ibl[3];
        if (ibl_visible.to_bits() != 0.0_f32.to_bits()
            && ibl_visible.to_bits() != 1.0_f32.to_bits())
            || (ibl_visible == 1.0) != (ibl_branch == AuditRayBranch::Clear)
        {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        let target = [
            audit.target_raw[0],
            audit.target_raw[1],
            audit.target_raw[2],
        ];
        if audit.target_raw[3].to_bits() != 0.0_f32.to_bits()
            || audit.ibl_origin[3].to_bits() != 0.0_f32.to_bits()
            || audit.mean_linear[3].to_bits() != 0.0_f32.to_bits()
            || audit.original_linear[3].to_bits() != 0.0_f32.to_bits()
            || audit.radiance_sun[3].to_bits() != 0.0_f32.to_bits()
            || audit.lit_linear[3].to_bits() != 0.0_f32.to_bits()
            || audit.shadow_linear[3].to_bits() != 0.0_f32.to_bits()
            || audit.loss_jump[3].to_bits() != 0.0_f32.to_bits()
            || audit.gradient[3].to_bits() != 0.0_f32.to_bits()
            || target
                .into_iter()
                .any(|v| !ordinary_f32(v) || !(0.0..=1.0).contains(&v))
        {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        let scale = audit.radiance_base[3];
        if scale <= 0.0 {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        let scale_interval = wgsl_mul(
            Iv::p(f64::from(ctx.mean_image_weight)),
            Iv::p(f64::from(ctx.replay_sample_weight)),
        )?;
        if !iv_contains(scale_interval, scale) {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        summary.max_radiance_reconstruction_width = summary
            .max_radiance_reconstruction_width
            .max(iv_width(scale_interval)?);
        let scale = Iv::p(f64::from(scale));
        let mut lit_reconstructed = [Iv::p(0.0); 3];
        let mut shadow_reconstructed = [Iv::p(0.0); 3];
        for k in 0..3 {
            // The audited source values are the exact f32 operands read by
            // the replay: the running mean and the selected primal sample.
            // Reconstruct `mean - original * inv_rfs` in the shader's order
            // before using it in either nonlinear-loss branch.
            let base = wgsl_sub(
                Iv::p(f64::from(audit.mean_linear[k])),
                wgsl_mul(Iv::p(f64::from(audit.original_linear[k])), scale)?,
            )?;
            if !iv_contains(base, audit.radiance_base[k]) {
                return Err(Uncertifiable::UnresolvedRoot);
            }
            let sun = source_sun[k];
            let ibl = source_ibl[k];
            lit_reconstructed[k] = wgsl_add(base, wgsl_mul(wgsl_add(sun, ibl)?, scale)?)?;
            shadow_reconstructed[k] = wgsl_add(base, wgsl_mul(ibl, scale)?)?;
            if !iv_contains(lit_reconstructed[k], audit.lit_linear[k])
                || !iv_contains(shadow_reconstructed[k], audit.shadow_linear[k])
            {
                return Err(Uncertifiable::UnresolvedRoot);
            }
            summary.max_radiance_reconstruction_width = summary
                .max_radiance_reconstruction_width
                .max(iv_width(base)?)
                .max(iv_width(lit_reconstructed[k])?)
                .max(iv_width(shadow_reconstructed[k])?);
        }
        let lit_rgb = [
            audit.lit_linear[0],
            audit.lit_linear[1],
            audit.lit_linear[2],
        ];
        let shadow_rgb = [
            audit.shadow_linear[0],
            audit.shadow_linear[1],
            audit.shadow_linear[2],
        ];
        if lit_rgb.into_iter().chain(shadow_rgb).any(|v| v < 0.0) {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        let lit_interval = wgsl_loss_interval(
            lit_reconstructed,
            target,
            ctx.provenance.camera.camera_exposure as f32,
            ctx.struct_weight,
        )?;
        let shadow_interval = wgsl_loss_interval(
            shadow_reconstructed,
            target,
            ctx.provenance.camera.camera_exposure as f32,
            ctx.struct_weight,
        )?;
        if !iv_contains(lit_interval, audit.loss_jump[0])
            || !iv_contains(shadow_interval, audit.loss_jump[1])
        {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        summary.max_loss_interval_width = summary
            .max_loss_interval_width
            .max(iv_width(lit_interval)?)
            .max(iv_width(shadow_interval)?);
        let lit_ref = pixel_loss_f64(
            lit_rgb,
            target,
            ctx.provenance.camera.camera_exposure as f32,
            ctx.struct_weight,
        )?;
        let shadow_ref = pixel_loss_f64(
            shadow_rgb,
            target,
            ctx.provenance.camera.camera_exposure as f32,
            ctx.struct_weight,
        )?;
        let reported_lit = f64::from(audit.loss_jump[0]);
        let reported_shadow = f64::from(audit.loss_jump[1]);
        let reported_jump = f64::from(audit.loss_jump[2]);
        let jump_ref = (lit_ref - shadow_ref) * f64::from(ctx.jump_numerator)
            / f64::from(ctx.jump_denominator);
        let jump_interval = wgsl_div(
            wgsl_mul(
                wgsl_sub(lit_interval, shadow_interval)?,
                Iv::p(f64::from(ctx.jump_numerator)),
            )?,
            Iv::p(f64::from(ctx.jump_denominator)),
        )?;
        if !iv_contains(jump_interval, audit.loss_jump[2]) {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        let loss_difference = outward_abs_difference(lit_ref, reported_lit)?
            .max(outward_abs_difference(shadow_ref, reported_shadow)?);
        let jump_difference = outward_abs_difference(jump_ref, reported_jump)?;
        summary.max_loss_reference_difference =
            summary.max_loss_reference_difference.max(loss_difference);
        summary.max_jump_reference_difference =
            summary.max_jump_reference_difference.max(jump_difference);
        for k in 0..3 {
            let weight = f64::from(event.record.grad_weight[k]);
            let observed = f64::from(audit.gradient[k]);
            let multiply_difference = outward_abs_difference(observed, reported_jump * weight)?;
            let upload_error = event.absolute_error[6 + k];
            let actual_gradient = wgsl_mul(jump_interval, Iv::p(weight))?;
            if !iv_contains(actual_gradient, audit.gradient[k]) {
                return Err(Uncertifiable::UnresolvedRoot);
            }
            // Preserve an exactly-zero uploaded weight. Applying f64 outward
            // rounding to `0 ± 0` would manufacture a subnormal sign range
            // even though WGSL multiplies by an exact zero on this channel.
            let ideal_weight = if weight == 0.0 && upload_error == 0.0 {
                Iv::p(0.0)
            } else {
                Iv {
                    lo: down(weight - upload_error),
                    hi: up(weight + upload_error),
                }
            };
            total_gradient[k] = total_gradient[k].add(jump_interval.mul(ideal_weight)?)?;
            summary.max_upload_weight_error = summary.max_upload_weight_error.max(upload_error);
            summary.max_gradient_multiply_difference = summary
                .max_gradient_multiply_difference
                .max(multiply_difference);
            summary.gradient_sum_f64[k] += observed;
            summary.replay_order_sum_f32[k] += audit.gradient[k];
            if !ordinary_f32(summary.replay_order_sum_f32[k]) {
                return Err(Uncertifiable::UnresolvedRoot);
            }
            sum_abs_gradient[k] = up(sum_abs_gradient[k] + observed.abs());
        }
    }
    for k in 0..3 {
        // Every event executes `old + value`, including the first addition to
        // zero. Any partial sum is bounded by the sum of input magnitudes.
        let additions = events.len() as f64;
        let rounding = if sum_abs_gradient[k] == 0.0 {
            0.0
        } else {
            let magnitude = sum_abs_gradient[k] as f32;
            if !ordinary_f32(magnitude) {
                return Err(Uncertifiable::UnresolvedRoot);
            }
            up(additions * f32_ulp(magnitude)?)
        };
        summary.accumulation_rounding_bound[k] = rounding;
        let exact_observed_sum = summary.gradient_sum_f64[k];
        let accumulation = Iv {
            lo: down(exact_observed_sum - rounding),
            hi: up(exact_observed_sum + rounding),
        };
        if !iv_contains(accumulation, atomic_edge_sum[k]) {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        total_gradient[k] = Iv {
            lo: down(total_gradient[k].lo - rounding),
            hi: up(total_gradient[k].hi + rounding),
        };
        if !iv_contains(total_gradient[k], atomic_edge_sum[k]) {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        summary.gradient_total_interval[k] = [total_gradient[k].lo, total_gradient[k].hi];
        summary.atomic_sum_difference[k] =
            up((f64::from(atomic_edge_sum[k]) - f64::from(summary.replay_order_sum_f32[k])).abs());
    }
    Ok(summary)
}

/// Source-provenance regression seam.  Each mutation leaves the reported
/// branch/loss/gradient/atomic witnesses untouched, so passing it would mean
/// a downstream-consistent fabrication could bypass the source replay.
#[cfg(test)]
pub(crate) fn assert_gpu_audit_source_corruptions_reject(
    ctx: &GpuAuditContext<'_>,
    events: &[ProjectedEvent],
    audits: &[GpuAuditRecord],
    atomic_edge_sum: [f32; 3],
) {
    let Some(first) = audits.first() else {
        return;
    };
    let mut altered_ray = audits.to_vec();
    altered_ray[0].ibl_ray[0] = if first.ibl_ray[0].abs() < 0.5 {
        0.75
    } else {
        -0.75
    };
    assert!(
        validate_gpu_audits(ctx, events, &altered_ray, atomic_edge_sum).is_err(),
        "a changed IBL ray with unchanged visibility/loss/gradient must reject"
    );

    let mut altered_material = audits.to_vec();
    altered_material[0].surface_albedo[0] = if first.surface_albedo[0] < 0.5 {
        0.75
    } else {
        0.25
    };
    assert!(
        validate_gpu_audits(ctx, events, &altered_material, atomic_edge_sum).is_err(),
        "a changed source albedo with unchanged radiance/loss/gradient must reject"
    );

    let mut altered_environment = audits.to_vec();
    altered_environment[0].environment_effective[0] = if first.environment_effective[0] < 0.5 {
        0.75
    } else {
        0.25
    };
    assert!(
        validate_gpu_audits(ctx, events, &altered_environment, atomic_edge_sum).is_err(),
        "a changed source environment with unchanged radiance/loss/gradient must reject"
    );

    if let Some((_, width, _)) = ctx.env_map {
        if width > 1 {
            let mut altered_lookup = audits.to_vec();
            altered_lookup[0].environment_lookup[0] = (first.environment_lookup[0] + 1) % width;
            assert!(
                validate_gpu_audits(ctx, events, &altered_lookup, atomic_edge_sum).is_err(),
                "a changed angular environment texel with unchanged radiance/loss/gradient must reject"
            );
        }
    }
}

fn receiver_point_normal(
    input: &SliceInput<'_>,
    dep: Iv,
) -> Result<([Iv; 3], [Iv; 3]), Uncertifiable> {
    let cell = input
        .dem
        .cell(input.receiver_cell.0, input.receiver_cell.1)?;
    let (x, z) = coords(input, dep);
    let h = cell.h(x, z)?;
    let hx = cell.hx(z)?;
    let hz = cell.hz(x)?;
    let norm = Ad::c(Iv::p(1.0))
        .add(hx.mul(hx)?)?
        .add(hz.mul(hz)?)?
        .sqrt()?;
    let nx = hx.neg().div(norm)?.v;
    let ny = Ad::c(Iv::p(1.0)).div(norm)?.v;
    let nz = hz.neg().div(norm)?.v;
    Ok(([x.v, h.v, z.v], [nx, ny, nz]))
}

struct Context<'a> {
    input: &'a SliceInput<'a>,
    receiver: Cell,
    blocker: Cell,
}
fn coords(input: &SliceInput<'_>, d: Iv) -> (Ad, Ad) {
    let a = Ad::var(d, 0);
    let b = Ad::var(Iv::p(input.free), 1);
    match input.chart {
        Chart::DependentX => (a, b),
        Chart::DependentZ => (b, a),
    }
}
fn origin(ctx: &Context<'_>, d: Iv) -> Result<[Ad; 3], Uncertifiable> {
    let (x, z) = coords(ctx.input, d);
    let h = ctx.receiver.h(x, z)?;
    let hx = ctx.receiver.hx(z)?;
    let hz = ctx.receiver.hz(x)?;
    let one = Ad::c(Iv::p(1.0));
    let norm = one.add(hx.mul(hx)?)?.add(hz.mul(hz)?)?.sqrt()?;
    let eps = Ad::c(Iv::p(ctx.input.normal_offset));
    Ok([
        x.sub(eps.mul(hx.div(norm)?)?)?,
        h.add(eps.div(norm)?)?,
        z.sub(eps.mul(hz.div(norm)?)?)?,
    ])
}
fn omega(input: &SliceInput<'_>) -> [Ad; 3] {
    std::array::from_fn(|k| Ad::var(Iv::p(input.omega[k]), 3 + k))
}
fn ray(ctx: &Context<'_>, d: Iv, t: Ad) -> Result<[Ad; 3], Uncertifiable> {
    let o = origin(ctx, d)?;
    let w = omega(ctx.input);
    Ok([
        o[0].add(t.mul(w[0])?)?,
        o[1].add(t.mul(w[1])?)?,
        o[2].add(t.mul(w[2])?)?,
    ])
}
fn gap(ctx: &Context<'_>, d: Iv, t: Ad) -> Result<Ad, Uncertifiable> {
    let p = ray(ctx, d, t)?;
    p[1].sub(ctx.blocker.h(p[0], p[2])?)
}
fn gap_t(ctx: &Context<'_>, d: Iv, t: Ad) -> Result<Ad, Uncertifiable> {
    let p = ray(ctx, d, t)?;
    let w = omega(ctx.input);
    w[1].sub(ctx.blocker.hx(p[2])?.mul(w[0])?)?
        .sub(ctx.blocker.hz(p[0])?.mul(w[2])?)
}

fn midpoint(a: f64, b: f64) -> Option<f64> {
    let m = if a.signum() != b.signum() {
        a * 0.5 + b * 0.5
    } else {
        a + (b - a) * 0.5
    };
    (m > a && m < b).then_some(m)
}

// A completed queue is a covering certificate: every rejected box has a
// strict range or monotonic-face exclusion, and every accepted box has one
// interior regular root. No finite scan count or iteration cap is assumed.
fn isolate_1d<F>(domain: Iv, f: F) -> Result<TrackedVec<Iv>, Uncertifiable>
where
    F: Fn(Iv) -> Result<Ad, Uncertifiable>,
{
    let mut todo = TrackedVec::new("inverse-cert-scalar-queue");
    todo.push(domain)?;
    let mut roots = TrackedVec::new("inverse-cert-scalar-roots");
    while let Some(b) = todo.pop() {
        let y = f(b)?;
        if y.v.sign().is_some() {
            continue;
        }
        if y.d[0].sign().is_some() {
            let left = f(Iv::p(b.lo))?.v.sign();
            let right = f(Iv::p(b.hi))?.v.sign();
            match (left, right) {
                (Some(a), Some(c)) if a == c => continue,
                (Some(a), Some(c)) if a != c => {
                    roots.push(refine_scalar(b, &f)?)?;
                    continue;
                }
                _ => {}
            }
        }
        let m = midpoint(b.lo, b.hi).ok_or(Uncertifiable::UnresolvedRoot)?;
        todo.push(Iv { lo: m, hi: b.hi })?;
        todo.push(Iv { lo: b.lo, hi: m })?;
    }
    Ok(roots)
}

fn refine_scalar<F>(mut box_: Iv, f: &F) -> Result<Iv, Uncertifiable>
where
    F: Fn(Iv) -> Result<Ad, Uncertifiable>,
{
    loop {
        let m = match midpoint(box_.lo, box_.hi) {
            Some(v) => v,
            None => return Ok(box_),
        };
        let derivative = f(box_)?.d[0];
        if derivative.sign().is_none() {
            return Ok(box_);
        }
        let n = Iv::p(m).sub(f(Iv::p(m))?.v.div(derivative)?)?;
        let smaller = Iv {
            lo: box_.lo.max(n.lo),
            hi: box_.hi.min(n.hi),
        };
        if smaller.lo > smaller.hi {
            return Err(Uncertifiable::UnresolvedRoot);
        }
        if smaller.lo <= box_.lo && smaller.hi >= box_.hi {
            return Ok(box_);
        }
        box_ = smaller;
    }
}

fn isolate_2d<F>(dep: Iv, time: Iv, f: F) -> Result<TrackedVec<(Iv, Iv)>, Uncertifiable>
where
    F: Fn(Iv, Iv) -> Result<(Ad, Ad), Uncertifiable>,
{
    let mut todo = TrackedVec::new("inverse-cert-smooth-queue");
    todo.push((dep, time))?;
    let mut roots = TrackedVec::new("inverse-cert-smooth-roots");
    while let Some((x, t)) = todo.pop() {
        let (y0, y1) = f(x, t)?;
        if y0.v.sign().is_some() || y1.v.sign().is_some() {
            continue;
        }
        let mx = midpoint(x.lo, x.hi).ok_or(Uncertifiable::UnresolvedRoot)?;
        let mt = midpoint(t.lo, t.hi).ok_or(Uncertifiable::UnresolvedRoot)?;
        let (c0, c1) = f(Iv::p(mx), Iv::p(mt))?;
        let j = [[y0.d[0], y0.d[2]], [y1.d[0], y1.d[2]]];
        // An approximate inverse is only a preconditioner. All Krawczyk
        // enclosure and contraction inequalities use outward intervals.
        let a = j[0][0].lo * 0.5 + j[0][0].hi * 0.5;
        let b = j[0][1].lo * 0.5 + j[0][1].hi * 0.5;
        let c = j[1][0].lo * 0.5 + j[1][0].hi * 0.5;
        let d = j[1][1].lo * 0.5 + j[1][1].hi * 0.5;
        let det = a * d - b * c;
        if det.is_finite() && det != 0.0 {
            let q = [[d / det, -b / det], [-c / det, a / det]];
            if q.iter().flatten().all(|v| v.is_finite()) {
                let mut m = [[Iv::p(0.0); 2]; 2];
                for i in 0..2 {
                    for k in 0..2 {
                        m[i][k] = Iv::p(if i == k { 1.0 } else { 0.0 }).sub(
                            Iv::p(q[i][0])
                                .mul(j[0][k])?
                                .add(Iv::p(q[i][1]).mul(j[1][k])?)?,
                        )?;
                    }
                }
                let contraction =
                    (0..2).all(|i| m[i][0].abs().add(m[i][1].abs()).is_ok_and(|v| v.hi < 1.0));
                if contraction {
                    let delta = [x.sub(Iv::p(mx))?, t.sub(Iv::p(mt))?];
                    let fc = [c0.v, c1.v];
                    let mut k = [Iv::p(0.0); 2];
                    for i in 0..2 {
                        k[i] = Iv::p(if i == 0 { mx } else { mt })
                            .sub(Iv::p(q[i][0]).mul(fc[0])?.add(Iv::p(q[i][1]).mul(fc[1])?)?)?
                            .add(m[i][0].mul(delta[0])?.add(m[i][1].mul(delta[1])?)?)?;
                    }
                    if x.interior_contains(k[0]) && t.interior_contains(k[1]) {
                        // A unique ideal-real root is proved, but the replay
                        // uploads f32. Keep contracting its enclosure until
                        // each coordinate lies in one f32 rounding cell.
                        // This criterion comes from the upload format, not
                        // an iteration or geometric tolerance. If f64 can
                        // no longer contract enough, reject the slice.
                        if (k[0].lo as f32) == (k[0].hi as f32)
                            && (k[1].lo as f32) == (k[1].hi as f32)
                        {
                            roots.push((k[0], k[1]))?;
                        } else if k[0].lo > x.lo
                            || k[0].hi < x.hi
                            || k[1].lo > t.lo
                            || k[1].hi < t.hi
                        {
                            todo.push((k[0], k[1]))?;
                        } else {
                            return Err(Uncertifiable::UnresolvedRoot);
                        }
                        continue;
                    }
                }
            }
        }
        let wx = x.hi - x.lo;
        let wt = t.hi - t.lo;
        if wx >= wt {
            todo.push((Iv { lo: mx, hi: x.hi }, t))?;
            todo.push((Iv { lo: x.lo, hi: mx }, t))?;
        } else {
            todo.push((x, Iv { lo: mt, hi: t.hi }))?;
            todo.push((x, Iv { lo: t.lo, hi: mt }))?;
        }
    }
    Ok(roots)
}

#[derive(Clone, Copy)]
enum Kind {
    Smooth,
    Cutoff,
    EdgeX(Iv),
    EdgeZ(Iv),
    EntryX(Iv),
    EntryZ(Iv),
}
#[derive(Clone, Copy)]
struct Raw {
    kind: Kind,
    dep: Iv,
    t: Iv,
    cell: Cell,
    class: EventClass,
}

fn edge_time(ctx: &Context<'_>, dep: Iv, kind: Kind) -> Result<Ad, Uncertifiable> {
    let o = origin(ctx, dep)?;
    let w = omega(ctx.input);
    match kind {
        Kind::EdgeX(edge) => Ad::c(edge).sub(o[0])?.div(w[0]),
        Kind::EdgeZ(edge) => Ad::c(edge).sub(o[2])?.div(w[2]),
        _ => Err(Uncertifiable::InvalidInput),
    }
}
fn isolate_edge(
    ctx: &Context<'_>,
    domain: Iv,
    kind: Kind,
) -> Result<TrackedVec<Iv>, Uncertifiable> {
    let (axis, edge, component) = match kind {
        Kind::EdgeX(e) => (0, e, ctx.input.omega[0]),
        Kind::EdgeZ(e) => (2, e, ctx.input.omega[2]),
        _ => return Err(Uncertifiable::InvalidInput),
    };
    if component == 0.0 {
        let o = origin(ctx, domain)?[axis].v;
        if o.hi < edge.lo || o.lo > edge.hi {
            return Ok(TrackedVec::new("inverse-cert-edge-roots"));
        }
        return Err(Uncertifiable::CoincidentEvent);
    }
    isolate_1d(domain, |d| Ok(scalar_event(ctx, d, kind)?.0))
}
fn scalar_event(ctx: &Context<'_>, dep: Iv, kind: Kind) -> Result<(Ad, Iv), Uncertifiable> {
    let t = match kind {
        Kind::Cutoff => Ad::c(Iv::p(ctx.input.tmin)),
        Kind::EdgeX(_) | Kind::EdgeZ(_) => edge_time(ctx, dep, kind)?,
        Kind::EntryX(edge) => {
            let t = Ad::c(Iv::p(ctx.input.tmin));
            let p = ray(ctx, dep, t)?;
            return Ok((p[0].sub(Ad::c(edge))?, t.v));
        }
        Kind::EntryZ(edge) => {
            let t = Ad::c(Iv::p(ctx.input.tmin));
            let p = ray(ctx, dep, t)?;
            return Ok((p[2].sub(Ad::c(edge))?, t.v));
        }
        Kind::Smooth => return Err(Uncertifiable::InvalidInput),
    };
    let f = gap(ctx, dep, t)?;
    Ok((f, t.v))
}
fn event_position(ctx: &Context<'_>, dep: Iv, t: Iv) -> Result<[Iv; 3], Uncertifiable> {
    let p = ray(ctx, dep, Ad::c(t))?;
    Ok([p[0].v, p[1].v, p[2].v])
}
fn root_in_cell(raw: Raw, input: &SliceInput<'_>) -> Result<bool, Uncertifiable> {
    let ctx = Context {
        input,
        receiver: input
            .dem
            .cell(input.receiver_cell.0, input.receiver_cell.1)?,
        blocker: raw.cell,
    };
    let p = event_position(&ctx, raw.dep, raw.t)?;
    let start = Iv::p(input.tmin);
    let end = Iv::p(input.tmax);
    if raw.t.hi <= start.lo || raw.t.lo >= end.hi {
        return Ok(false);
    }
    if raw.t.lo <= start.hi || raw.t.hi >= end.lo {
        return Err(Uncertifiable::UnresolvedRoot);
    }
    let c = raw.cell;
    match raw.kind {
        Kind::Smooth | Kind::Cutoff => {
            if c.strict_contains(p[0], p[2])? {
                Ok(true)
            } else if c.may_contain(p[0], p[2])? {
                Err(Uncertifiable::CoincidentEvent)
            } else {
                Ok(false)
            }
        }
        Kind::EdgeX(_) | Kind::EntryX(_) => {
            let z1 = c.z0.add(Iv::p(c.sz))?;
            if p[2].lo > c.z0.hi && p[2].hi < z1.lo {
                Ok(true)
            } else if p[2].hi < c.z0.lo || p[2].lo > z1.hi {
                Ok(false)
            } else {
                Err(Uncertifiable::CoincidentEvent)
            }
        }
        Kind::EdgeZ(_) | Kind::EntryZ(_) => {
            let x1 = c.x0.add(Iv::p(c.sx))?;
            if p[0].lo > c.x0.hi && p[0].hi < x1.lo {
                Ok(true)
            } else if p[0].hi < c.x0.lo || p[0].lo > x1.hi {
                Ok(false)
            } else {
                Err(Uncertifiable::CoincidentEvent)
            }
        }
    }
}

fn enumerate(input: &SliceInput<'_>) -> Result<TrackedVec<Raw>, Uncertifiable> {
    let dem = &input.dem;
    let receiver = dem.cell(input.receiver_cell.0, input.receiver_cell.1)?;
    let o = origin(
        &Context {
            input,
            receiver,
            blocker: receiver,
        },
        input.dependent,
    )?;
    let ymax = Iv::p(dem.max_h())
        .sub(Iv::p(o[1].v.lo))?
        .div(Iv::p(input.omega[1]))?;
    let t_bound = up(ymax.hi).min(input.tmax);
    let mut out = TrackedVec::new("inverse-cert-candidates");
    for cz in 0..dem.height - 1 {
        for cx in 0..dem.width - 1 {
            let cell = dem.cell(cx, cz)?;
            let ctx = Context {
                input,
                receiver,
                blocker: cell,
            };
            for d in isolate_1d(input.dependent, |d| {
                Ok(scalar_event(&ctx, d, Kind::Cutoff)?.0)
            })?
            .iter()
            .copied()
            {
                let raw = Raw {
                    kind: Kind::Cutoff,
                    dep: d,
                    t: Iv::p(input.tmin),
                    cell,
                    class: EventClass::Cutoff,
                };
                // tmin is a boundary event, not an admissible hit itself.
                if cutoff_in_cell(raw, input)? {
                    out.push(raw)?;
                }
            }
            if t_bound > input.tmin {
                for (d, t) in isolate_2d(
                    input.dependent,
                    Iv {
                        lo: input.tmin,
                        hi: t_bound,
                    },
                    |d, t| {
                        let tv = Ad::var(t, 2);
                        Ok((gap(&ctx, d, tv)?, gap_t(&ctx, d, tv)?))
                    },
                )?
                .iter()
                .copied()
                {
                    let raw = Raw {
                        kind: Kind::Smooth,
                        dep: d,
                        t,
                        cell,
                        class: EventClass::SmoothGrazing,
                    };
                    if root_in_cell(raw, input)? {
                        out.push(raw)?;
                    }
                }
            }
            // Each grid edge is enumerated once: left/bottom edges of every cell,
            // plus the right/top DEM perimeter of the final row/column.
            let ex = cell.x0;
            let kind = Kind::EdgeX(ex);
            for d in isolate_edge(&ctx, input.dependent, kind)?.iter().copied() {
                let (_, t) = scalar_event(&ctx, d, kind)?;
                let class = if cx == 0 {
                    EventClass::PerimeterX
                } else {
                    EventClass::CreaseX
                };
                let raw = Raw {
                    kind,
                    dep: d,
                    t,
                    cell,
                    class,
                };
                if root_in_cell(raw, input)? {
                    out.push(raw)?;
                }
            }
            let ez = cell.z0;
            let kind = Kind::EdgeZ(ez);
            for d in isolate_edge(&ctx, input.dependent, kind)?.iter().copied() {
                let (_, t) = scalar_event(&ctx, d, kind)?;
                let class = if cz == 0 {
                    EventClass::PerimeterZ
                } else {
                    EventClass::CreaseZ
                };
                let raw = Raw {
                    kind,
                    dep: d,
                    t,
                    cell,
                    class,
                };
                if root_in_cell(raw, input)? {
                    out.push(raw)?;
                }
            }
            if cx == dem.width - 2 {
                let kind = Kind::EdgeX(cell.x0.add(Iv::p(cell.sx))?);
                for d in isolate_edge(&ctx, input.dependent, kind)?.iter().copied() {
                    let (_, t) = scalar_event(&ctx, d, kind)?;
                    let raw = Raw {
                        kind,
                        dep: d,
                        t,
                        cell,
                        class: EventClass::PerimeterX,
                    };
                    if root_in_cell(raw, input)? {
                        out.push(raw)?;
                    }
                }
            }
            if cz == dem.height - 2 {
                let kind = Kind::EdgeZ(cell.z0.add(Iv::p(cell.sz))?);
                for d in isolate_edge(&ctx, input.dependent, kind)?.iter().copied() {
                    let (_, t) = scalar_event(&ctx, d, kind)?;
                    let raw = Raw {
                        kind,
                        dep: d,
                        t,
                        cell,
                        class: EventClass::PerimeterZ,
                    };
                    if root_in_cell(raw, input)? {
                        out.push(raw)?;
                    }
                }
            }
            // The strict tmin test on the conservative below-surface entry
            // creates a separate boundary where an edge's entry time equals
            // tmin while the ray is below that edge. F=0 edge roots alone do
            // not cover it.
            let entry_edges = [
                Some(Kind::EntryX(cell.x0)),
                Some(Kind::EntryZ(cell.z0)),
                if cx == dem.width - 2 {
                    Some(Kind::EntryX(cell.x0.add(Iv::p(cell.sx))?))
                } else {
                    None
                },
                if cz == dem.height - 2 {
                    Some(Kind::EntryZ(cell.z0.add(Iv::p(cell.sz))?))
                } else {
                    None
                },
            ];
            for kind in entry_edges.into_iter().flatten() {
                for d in isolate_1d(input.dependent, |d| Ok(scalar_event(&ctx, d, kind)?.0))?
                    .iter()
                    .copied()
                {
                    let (_, t) = scalar_event(&ctx, d, kind)?;
                    let class = match kind {
                        Kind::EntryX(_) => EventClass::CellEntryX,
                        Kind::EntryZ(_) => EventClass::CellEntryZ,
                        _ => unreachable!(),
                    };
                    let raw = Raw {
                        kind,
                        dep: d,
                        t,
                        cell,
                        class,
                    };
                    if entry_in_cell(raw, input)? {
                        out.push(raw)?;
                    }
                }
            }
        }
    }
    let mut inside = TrackedVec::with_capacity(out.len(), "inverse-cert-inside-roots")?;
    for raw in out.iter().copied() {
        if receiver_root_in_cell(raw, input)? {
            inside.push(raw)?;
        }
    }
    let mut out = inside;
    out.sort_unstable_by(|a, b| a.dep.lo.total_cmp(&b.dep.lo));
    for pair in out.windows(2) {
        if pair[0].dep.hi >= pair[1].dep.lo {
            return Err(Uncertifiable::CoincidentEvent);
        }
    }
    Ok(out)
}

fn receiver_root_in_cell(raw: Raw, input: &SliceInput<'_>) -> Result<bool, Uncertifiable> {
    let cell = input
        .dem
        .cell(input.receiver_cell.0, input.receiver_cell.1)?;
    let (low, high) = match input.chart {
        Chart::DependentX => (cell.x0, cell.x0.add(Iv::p(cell.sx))?),
        Chart::DependentZ => (cell.z0, cell.z0.add(Iv::p(cell.sz))?),
    };
    if raw.dep.lo > low.hi && raw.dep.hi < high.lo {
        Ok(true)
    } else if raw.dep.hi < low.lo || raw.dep.lo > high.hi {
        Ok(false)
    } else {
        Err(Uncertifiable::CoincidentEvent)
    }
}

fn cutoff_in_cell(raw: Raw, input: &SliceInput<'_>) -> Result<bool, Uncertifiable> {
    let ctx = Context {
        input,
        receiver: input
            .dem
            .cell(input.receiver_cell.0, input.receiver_cell.1)?,
        blocker: raw.cell,
    };
    let p = event_position(&ctx, raw.dep, raw.t)?;
    if raw.cell.strict_contains(p[0], p[2])? {
        Ok(true)
    } else if raw.cell.may_contain(p[0], p[2])? {
        Err(Uncertifiable::CoincidentEvent)
    } else {
        Ok(false)
    }
}

fn entry_in_cell(raw: Raw, input: &SliceInput<'_>) -> Result<bool, Uncertifiable> {
    let ctx = Context {
        input,
        receiver: input
            .dem
            .cell(input.receiver_cell.0, input.receiver_cell.1)?,
        blocker: raw.cell,
    };
    let p = event_position(&ctx, raw.dep, raw.t)?;
    let c = raw.cell;
    let segment = match raw.kind {
        Kind::EntryX(_) => {
            let end = c.z0.add(Iv::p(c.sz))?;
            if p[2].hi < c.z0.lo || p[2].lo > end.hi {
                return Ok(false);
            }
            p[2].lo > c.z0.hi && p[2].hi < end.lo
        }
        Kind::EntryZ(_) => {
            let end = c.x0.add(Iv::p(c.sx))?;
            if p[0].hi < c.x0.lo || p[0].lo > end.hi {
                return Ok(false);
            }
            p[0].lo > c.x0.hi && p[0].hi < end.lo
        }
        _ => return Err(Uncertifiable::InvalidInput),
    };
    if !segment {
        return Err(Uncertifiable::CoincidentEvent);
    }
    let f = gap(&ctx, raw.dep, Ad::c(raw.t))?.v;
    if f.hi < 0.0 {
        Ok(true)
    } else if f.lo > 0.0 {
        Ok(false)
    } else {
        Err(Uncertifiable::CoincidentEvent)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Visibility {
    Lit,
    Shadow,
}

fn axis_span(
    o: Iv,
    a: Iv,
    b: Iv,
    w: f64,
    tmin: f64,
    tmax: f64,
) -> Result<Option<(Iv, Iv)>, Uncertifiable> {
    if w == 0.0 {
        if o.lo > a.hi && o.hi < b.lo {
            return Ok(Some((Iv::p(tmin), Iv::p(tmax))));
        }
        if o.hi < a.lo || o.lo > b.hi {
            return Ok(None);
        }
        return Err(Uncertifiable::FirstHit);
    }
    let t0 = a.sub(o)?.div(Iv::p(w))?;
    let t1 = b.sub(o)?.div(Iv::p(w))?;
    Ok(Some(if w > 0.0 { (t0, t1) } else { (t1, t0) }))
}

fn classify(input: &SliceInput<'_>, dep: f64) -> Result<Visibility, Uncertifiable> {
    let receiver = input
        .dem
        .cell(input.receiver_cell.0, input.receiver_cell.1)?;
    let base = Context {
        input,
        receiver,
        blocker: receiver,
    };
    let o = origin(&base, Iv::p(dep))?;
    for cz in 0..input.dem.height - 1 {
        for cx in 0..input.dem.width - 1 {
            let blocker = input.dem.cell(cx, cz)?;
            let x1 = blocker.x0.add(Iv::p(blocker.sx))?;
            let z1 = blocker.z0.add(Iv::p(blocker.sz))?;
            let Some((xe, xx)) = axis_span(
                o[0].v,
                blocker.x0,
                x1,
                input.omega[0],
                input.tmin,
                input.tmax,
            )?
            else {
                continue;
            };
            let Some((ze, zx)) = axis_span(
                o[2].v,
                blocker.z0,
                z1,
                input.omega[2],
                input.tmin,
                input.tmax,
            )?
            else {
                continue;
            };
            let enter = Iv {
                lo: xe.lo.max(ze.lo).max(input.tmin),
                hi: xe.hi.max(ze.hi).max(input.tmin),
            };
            let exit = Iv {
                lo: xx.lo.min(zx.lo).min(input.tmax),
                hi: xx.hi.min(zx.hi).min(input.tmax),
            };
            if enter.lo > exit.hi {
                continue;
            }
            let ctx = Context {
                input,
                receiver,
                blocker,
            };
            let full = Iv {
                lo: enter.lo,
                hi: exit.hi,
            };
            if gap(&ctx, Iv::p(dep), Ad::c(full))?.v.lo > 0.0 {
                continue;
            }
            if enter.hi >= exit.lo {
                return Err(Uncertifiable::FirstHit);
            }
            let admissible_entry = if enter.lo > input.tmin && enter.hi < input.tmax {
                true
            } else if enter.hi == input.tmin {
                false
            } else {
                return Err(Uncertifiable::FirstHit);
            };
            let entry_gap = gap(&ctx, Iv::p(dep), Ad::c(enter))?.v;
            if admissible_entry && entry_gap.hi < 0.0 {
                return Ok(Visibility::Shadow);
            }
            let mut todo = TrackedVec::new("inverse-cert-first-hit-queue");
            todo.push(full)?;
            let mut interior_sign = None;
            while let Some(t) = todo.pop() {
                let g = gap(&ctx, Iv::p(dep), Ad::c(t))?.v;
                if let Some(s) = g.sign() {
                    let a = t.lo.max(enter.hi);
                    let b = t.hi.min(exit.lo);
                    if a < b {
                        if s < 0 && admissible_entry {
                            return Ok(Visibility::Shadow);
                        }
                        if interior_sign.is_some_and(|prev| prev != s) {
                            return Ok(Visibility::Shadow);
                        }
                        interior_sign = Some(s);
                    }
                    continue;
                }
                let a = t.lo.max(enter.hi);
                let b = t.hi.min(exit.lo);
                if let Some(m) = midpoint(a, b) {
                    if let Some(s) = gap(&ctx, Iv::p(dep), Ad::c(Iv::p(m)))?.v.sign() {
                        if interior_sign.is_some_and(|prev| prev != s) {
                            // Two strict interior samples with opposite signs
                            // force an admissible root inside this bilinear cell.
                            return Ok(Visibility::Shadow);
                        }
                        interior_sign = Some(s);
                    }
                }
                let m = midpoint(t.lo, t.hi).ok_or(Uncertifiable::FirstHit)?;
                todo.push(Iv { lo: m, hi: t.hi })?;
                todo.push(Iv { lo: t.lo, hi: m })?;
            }
            if admissible_entry && entry_gap.sign().is_none() {
                return Err(Uncertifiable::FirstHit);
            }
        }
    }
    Ok(Visibility::Lit)
}

fn cross_iv(a: [Iv; 3], b: [Iv; 3]) -> Result<[Iv; 3], Uncertifiable> {
    Ok([
        a[1].mul(b[2])?.sub(a[2].mul(b[1])?)?,
        a[2].mul(b[0])?.sub(a[0].mul(b[2])?)?,
        a[0].mul(b[1])?.sub(a[1].mul(b[0])?)?,
    ])
}
fn dot_iv(a: [Ad; 3], b: [Iv; 3]) -> Result<Ad, Uncertifiable> {
    a[0].mul(Ad::c(b[0]))?
        .add(a[1].mul(Ad::c(b[1]))?)?
        .add(a[2].mul(Ad::c(b[2]))?)
}

fn ad_abs(a: Ad) -> Result<Ad, Uncertifiable> {
    match a.v.sign() {
        Some(1) => Ok(a),
        Some(-1) => Ok(a.neg()),
        _ => Err(Uncertifiable::Density),
    }
}

fn camera_density(
    input: &SliceInput<'_>,
    dep: Iv,
) -> Result<Option<CameraProposal>, Uncertifiable> {
    let cam = &input.camera;
    if cam.width == 0
        || cam.height == 0
        || cam.pixel_x >= cam.width
        || cam.pixel_y >= cam.height
        || cam.fov_y <= 0.0
        || cam.half_h <= 0.0
        || cam.half_w <= 0.0
        || cam.aspect <= 0.0
        || cam.camera_exposure <= 0.0
        || ![
            cam.fov_y,
            cam.half_h,
            cam.half_w,
            cam.aspect,
            cam.camera_exposure,
        ]
        .into_iter()
        .chain(cam.origin)
        .chain(cam.right)
        .chain(cam.up)
        .chain(cam.forward)
        .all(f64::is_finite)
    {
        return Err(Uncertifiable::InvalidInput);
    }
    let receiver = input
        .dem
        .cell(input.receiver_cell.0, input.receiver_cell.1)?;
    let (x, z) = coords(input, dep);
    let h = receiver.h(x, z)?;
    let p = [x, h, z];
    let delta = [
        p[0].sub(Ad::c(Iv::p(cam.origin[0])))?,
        p[1].sub(Ad::c(Iv::p(cam.origin[1])))?,
        p[2].sub(Ad::c(Iv::p(cam.origin[2])))?,
    ];
    if cam.origin[1] <= input.dem.max_h() {
        return Err(Uncertifiable::Camera);
    }
    // Along camera-to-receiver t in [0,1], the gap is strictly decreasing
    // across every DEM cell. With the origin above the global height maximum,
    // the receiver is its unique first terrain intersection.
    for cz in 0..input.dem.height - 1 {
        for cx in 0..input.dem.width - 1 {
            let c = input.dem.cell(cx, cz)?;
            let z1 = c.z0.add(Iv::p(c.sz))?;
            let x1 = c.x0.add(Iv::p(c.sx))?;
            let hx = c
                .hx(Ad::c(Iv {
                    lo: c.z0.lo,
                    hi: z1.hi,
                }))?
                .v;
            let hz = c
                .hz(Ad::c(Iv {
                    lo: c.x0.lo,
                    hi: x1.hi,
                }))?
                .v;
            let slope = delta[1]
                .v
                .sub(hx.mul(delta[0].v)?)?
                .sub(hz.mul(delta[2].v)?)?;
            if slope.hi >= 0.0 {
                return Err(Uncertifiable::Camera);
            }
        }
    }
    let r = cam.right.map(Iv::p);
    let u = cam.up.map(Iv::p);
    let f = cam.forward.map(Iv::p);
    let nr = dot_iv(delta, cross_iv(u, f)?)?;
    let nu = dot_iv(delta, cross_iv(f, r)?)?;
    let nf = dot_iv(delta, cross_iv(r, u)?)?;
    let determinant = dot_iv([Ad::c(r[0]), Ad::c(r[1]), Ad::c(r[2])], cross_iv(u, f)?)?.v;
    if determinant.sign().is_none() || nf.v.div(determinant)?.lo <= 0.0 {
        return Err(Uncertifiable::Camera);
    }
    let ndcx = nr.div(nf)?.div(Ad::c(Iv::p(cam.half_w)))?;
    let ndcy = nu.div(nf)?.div(Ad::c(Iv::p(cam.half_h)))?;
    let two = Ad::c(Iv::p(2.0));
    let one = Ad::c(Iv::p(1.0));
    let jx = ndcx
        .add(one)?
        .div(two)?
        .mul(Ad::c(Iv::p(cam.width as f64)))?
        .sub(Ad::c(Iv::p(cam.pixel_x as f64 + 0.5)))?;
    let jy = one
        .sub(ndcy)?
        .div(two)?
        .mul(Ad::c(Iv::p(cam.height as f64)))?
        .sub(Ad::c(Iv::p(cam.pixel_y as f64 + 0.5)))?;
    if jx.v.hi <= -0.5 || jx.v.lo >= 0.5 || jy.v.hi <= -0.5 || jy.v.lo >= 0.5 {
        return Ok(None);
    }
    if jx.v.lo <= -0.5 || jx.v.hi >= 0.5 || jy.v.lo <= -0.5 || jy.v.hi >= 0.5 {
        return Err(Uncertifiable::Density);
    }
    // The tent proposal has a kink at jitter zero; a straddling interval is
    // rejected instead of hiding an undefined derivative branch.
    let ax = ad_abs(jx)?;
    let ay = ad_abs(jy)?;
    let px = Iv::p(2.0).mul(Iv::p(1.0).sub(Iv::p(2.0).mul(ax.v)?)?)?;
    let py = Iv::p(2.0).mul(Iv::p(1.0).sub(Iv::p(2.0).mul(ay.v)?)?)?;
    if px.lo <= 0.0 || py.lo <= 0.0 {
        return Err(Uncertifiable::Density);
    }
    let (ix, iz) = match input.chart {
        Chart::DependentX => (0, 1),
        Chart::DependentZ => (1, 0),
    };
    let jac = jx.d[ix].mul(jy.d[iz])?.sub(jx.d[iz].mul(jy.d[ix])?)?;
    if jac.sign().is_none() {
        return Err(Uncertifiable::Density);
    }
    let jacobian = jac.abs();
    Ok(Some(CameraProposal {
        jitter_x: jx.v,
        jitter_y: jy.v,
        tent_pdf_x: px,
        tent_pdf_y: py,
        jacobian,
        density: px.mul(py)?.mul(jacobian)?,
    }))
}

fn validate_slice(input: &SliceInput<'_>) -> Result<(), Uncertifiable> {
    input.dem.validate()?;
    if input.receiver_cell.0 >= input.dem.width - 1
        || input.receiver_cell.1 >= input.dem.height - 1
        || ![input.free, input.tmin, input.tmax, input.normal_offset]
            .into_iter()
            .chain(input.omega)
            .all(f64::is_finite)
        || input.omega[1] <= 0.0
        || input.tmin != 0.001
        || input.tmax != 1e30
        || input.normal_offset != 0.001
        || !input.dependent.lo.is_finite()
        || !input.dependent.hi.is_finite()
        || input.dependent.lo >= input.dependent.hi
    {
        return Err(Uncertifiable::InvalidInput);
    }
    let cell = input
        .dem
        .cell(input.receiver_cell.0, input.receiver_cell.1)?;
    let x1 = cell.x0.add(Iv::p(cell.sx))?;
    let z1 = cell.z0.add(Iv::p(cell.sz))?;
    match input.chart {
        Chart::DependentX
            if input.dependent.lo < cell.x0.lo
                || input.dependent.hi > x1.hi
                || input.free <= cell.z0.hi
                || input.free >= z1.lo =>
        {
            return Err(Uncertifiable::InvalidInput)
        }
        Chart::DependentZ
            if input.dependent.lo < cell.z0.lo
                || input.dependent.hi > z1.hi
                || input.free <= cell.x0.hi
                || input.free >= x1.lo =>
        {
            return Err(Uncertifiable::InvalidInput)
        }
        _ => {}
    }
    // The finite ray cutoff is not otherwise an enumerated event family.
    // Prove that every sun ray on this receiver tile is strictly above the
    // entire DEM before tmax, so F(tmax)=0 is impossible. This also bounds
    // every smooth-grazing time used by the isolator.
    let ctx = Context {
        input,
        receiver: cell,
        blocker: cell,
    };
    let oy = origin(&ctx, input.dependent)?[1].v;
    let escape = Iv::p(input.dem.max_h())
        .sub(Iv::p(oy.lo))?
        .div(Iv::p(input.omega[1]))?;
    if escape.hi >= input.tmax {
        return Err(Uncertifiable::InvalidInput);
    }
    Ok(())
}

/// Exhaustively enumerate and certify the events of one sampled receiver
/// slice. An empty successful result is a valid zero-event result. Any unresolved root,
/// first-hit ordering, camera support, or density rejects the whole slice.
pub(crate) fn certify_slice(
    input: &SliceInput<'_>,
) -> Result<TrackedVec<CertifiedEvent>, Uncertifiable> {
    validate_slice(input)?;
    // Pixel jitter has compact support. An entire receiver cell projecting
    // outside this pixel cannot contribute an event, even if its geometry has
    // unrelated degeneracies. This range exclusion is certified before the
    // expensive root search and cannot hide an in-support root.
    match camera_density(input, input.dependent) {
        Ok(None) => return Ok(TrackedVec::new("inverse-cert-slice-events")),
        Ok(Some(_)) | Err(Uncertifiable::Density) => {
            // A whole receiver cell can straddle a pixel-support edge or a
            // tent-density kink even when each event is regular. Only the
            // strict outside proof may cull it; certify density at each root.
        }
        Err(e) => return Err(e),
    }
    let roots = enumerate(input)?;
    let mut output = TrackedVec::new("inverse-cert-slice-events");
    let free_span = match input.chart {
        Chart::DependentX => Iv::p(input.dem.sz).mul(Iv::p((input.dem.height - 1) as f64))?,
        Chart::DependentZ => Iv::p(input.dem.sx).mul(Iv::p((input.dem.width - 1) as f64))?,
    };
    let q = Iv::p(0.5).div(free_span)?;
    if q.lo <= 0.0 {
        return Err(Uncertifiable::Density);
    }
    for (i, raw) in roots.iter().copied().enumerate() {
        let left_bound = if i == 0 {
            input.dependent.lo
        } else {
            roots[i - 1].dep.hi
        };
        let right_bound = if i + 1 == roots.len() {
            input.dependent.hi
        } else {
            roots[i + 1].dep.lo
        };
        let left = midpoint(left_bound, raw.dep.lo).ok_or(Uncertifiable::FirstHit)?;
        let right = midpoint(raw.dep.hi, right_bound).ok_or(Uncertifiable::FirstHit)?;
        let receiver = input
            .dem
            .cell(input.receiver_cell.0, input.receiver_cell.1)?;
        let (edge0, edge1) = match input.chart {
            Chart::DependentX => (receiver.x0, receiver.x0.add(Iv::p(receiver.sx))?),
            Chart::DependentZ => (receiver.z0, receiver.z0.add(Iv::p(receiver.sz))?),
        };
        if left <= edge0.hi || right >= edge1.lo {
            return Err(Uncertifiable::FirstHit);
        }
        let lower = classify(input, left)?;
        let upper = classify(input, right)?;
        if lower == upper {
            continue;
        }
        let ctx = Context {
            input,
            receiver: input
                .dem
                .cell(input.receiver_cell.0, input.receiver_cell.1)?,
            blocker: raw.cell,
        };
        let equation = match raw.kind {
            Kind::Smooth => gap(&ctx, raw.dep, Ad::var(raw.t, 2))?,
            _ => scalar_event(&ctx, raw.dep, raw.kind)?.0,
        };
        let du = equation.d[0];
        let ds = equation.d[1];
        if du.sign().is_none() {
            return Err(Uncertifiable::Density);
        }
        let au = du.abs();
        let as_ = ds.abs();
        let selected = match input.chart {
            Chart::DependentX if au.lo >= as_.hi => true,
            Chart::DependentX if au.hi < as_.lo => false,
            Chart::DependentZ if au.lo > as_.hi => true,
            Chart::DependentZ if au.hi <= as_.lo => false,
            _ => return Err(Uncertifiable::Density),
        };
        if !selected {
            continue;
        }
        let Some(camera_proposal) = camera_density(input, raw.dep)? else {
            continue;
        };
        let density = camera_proposal.density;
        if density.lo <= 0.0 {
            return Err(Uncertifiable::Density);
        }
        // The consumer multiplies this by (loss_lit - loss_shadow). Growing
        // the lower-coordinate side contributes (loss_lower-loss_upper)*du/dθ,
        // where du/dθ=-Gθ/G_u. Keep G_u signed and orient by the proven side
        // classification; using |G_u| here would reverse some events.
        let orientation = if lower == Visibility::Lit {
            Iv::p(1.0)
        } else {
            Iv::p(-1.0)
        };
        let factor = density.div(q)?.div(du)?.mul(orientation)?;
        let mut weight = [Iv::p(0.0); 3];
        for (k, v) in weight.iter_mut().enumerate() {
            *v = equation.d[3 + k].neg().mul(factor)?
        }
        let (receiver_xyz, normal_xyz) = receiver_point_normal(input, raw.dep)?;
        let receiver_cell = input
            .dem
            .cell(input.receiver_cell.0, input.receiver_cell.1)?;
        output.push(CertifiedEvent {
            class: raw.class,
            receiver: raw.dep,
            receiver_xyz,
            receiver_cell: input.receiver_cell,
            receiver_cell_world: [
                receiver_cell.x0,
                receiver_cell.x0.add(Iv::p(receiver_cell.sx))?,
                receiver_cell.z0,
                receiver_cell.z0.add(Iv::p(receiver_cell.sz))?,
            ],
            normal_xyz,
            pixel: [input.camera.pixel_x, input.camera.pixel_y],
            signed_sun_weight: weight,
            camera_proposal,
            sun_direction: input.omega,
            #[cfg(test)]
            lit_on_lower_side: lower == Visibility::Lit,
            camera: input.camera,
        })?;
    }
    Ok(output)
}

/// Scene-level producer for one globally drawn free coordinate. Every
/// receiver cell intersected by the sampled line is visited exactly once;
/// event boxes near an uncertain receiver-cell boundary reject the draw.
pub(crate) fn certify_scene_slice(
    input: &SceneSliceInput<'_>,
) -> Result<TrackedVec<CertifiedEvent>, Uncertifiable> {
    input.dem.validate()?;
    if !input.free.is_finite() {
        return Err(Uncertifiable::InvalidInput);
    }
    let (global0, global1) = match input.chart {
        Chart::DependentX => (
            Iv::p(input.dem.z0),
            Iv::p(input.dem.z0)
                .add(Iv::p(input.dem.sz).mul(Iv::p((input.dem.height - 1) as f64))?)?,
        ),
        Chart::DependentZ => (
            Iv::p(input.dem.x0),
            Iv::p(input.dem.x0)
                .add(Iv::p(input.dem.sx).mul(Iv::p((input.dem.width - 1) as f64))?)?,
        ),
    };
    if input.free <= global0.hi || input.free >= global1.lo {
        return Err(Uncertifiable::InvalidInput);
    }
    let mut events = TrackedVec::new("inverse-cert-scene-events");
    for cz in 0..input.dem.height - 1 {
        for cx in 0..input.dem.width - 1 {
            let cell = input.dem.cell(cx, cz)?;
            let (lo, hi) = match input.chart {
                Chart::DependentX => (cell.z0, cell.z0.add(Iv::p(cell.sz))?),
                Chart::DependentZ => (cell.x0, cell.x0.add(Iv::p(cell.sx))?),
            };
            if input.free <= lo.lo || input.free >= hi.hi {
                continue;
            }
            if input.free <= lo.hi || input.free >= hi.lo {
                return Err(Uncertifiable::CoincidentEvent);
            }
            let (d0, d1) = match input.chart {
                Chart::DependentX => (cell.x0, cell.x0.add(Iv::p(cell.sx))?),
                Chart::DependentZ => (cell.z0, cell.z0.add(Iv::p(cell.sz))?),
            };
            let slice = SliceInput {
                dem: input.dem,
                receiver_cell: (cx, cz),
                chart: input.chart,
                free: input.free,
                dependent: Iv {
                    lo: d0.lo,
                    hi: d1.hi,
                },
                omega: input.omega,
                tmin: input.tmin,
                tmax: input.tmax,
                normal_offset: input.normal_offset,
                camera: input.camera,
            };
            let slice_events = certify_slice(&slice)?;
            for event in slice_events.iter().copied() {
                events.push(event)?;
            }
        }
    }
    events.sort_unstable_by(|a, b| a.receiver.lo.total_cmp(&b.receiver.lo));
    for pair in events.windows(2) {
        if pair[0].receiver.hi >= pair[1].receiver.lo {
            return Err(Uncertifiable::CoincidentEvent);
        }
    }
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nadir_camera(pixel_x: u32, pixel_y: u32, width: u32, height: u32) -> Camera {
        let fov_y = 38.0_f64.to_radians();
        let half_h = (fov_y * 0.5).tan();
        let aspect = width as f64 / height as f64;
        Camera {
            origin: [0.0, 42.0, 0.001],
            right: [1.0, 0.0, 0.0],
            up: [0.0, 0.0, -1.0],
            forward: [0.0, -1.0, 0.0],
            fov_y,
            half_h,
            half_w: half_h * aspect,
            aspect,
            camera_exposure: 1.0,
            width,
            height,
            pixel_x,
            pixel_y,
        }
    }

    fn gaussian_heights() -> Vec<f64> {
        let mut heights = Vec::with_capacity(16 * 16);
        for z in 0..16 {
            for x in 0..16 {
                let xx = -1.0 + 2.0 * x as f64 / 15.0;
                let zz = -1.0 + 2.0 * z as f64 / 15.0;
                let h = 8.0 * (-((xx - 0.35).powi(2) + (zz + 0.2).powi(2)) / 0.08).exp()
                    + 6.0 * (-((xx + 0.45).powi(2) + (zz - 0.4).powi(2)) / 0.06).exp()
                    + 4.0 * (-((xx + 0.1).powi(2) + (zz + 0.55).powi(2)) / 0.05).exp()
                    + 1.2 * (0.5 + 0.5 * xx);
                heights.push(f64::from(h as f32));
            }
        }
        heights
    }

    #[test]
    fn gaussian_scene_certifies_smooth_and_crease_without_planar_cells() {
        let heights = gaussian_heights();
        let spacing = f64::from(32.0_f32 / 15.0);
        let origin = f64::from(-(15.0_f32) * spacing as f32 * 0.5);
        let dem = Dem {
            width: 16,
            height: 16,
            x0: origin,
            z0: origin,
            sx: spacing,
            sz: spacing,
            heights: &heights,
        };
        let az = 215.0_f64.to_radians();
        let el = 30.0_f64.to_radians();
        let omega = [az.cos() * el.cos(), el.sin(), az.sin() * el.cos()];
        for (free, camera, expected) in [
            (
                -2.5,
                nadir_camera(39, 68, 96, 96),
                EventClass::SmoothGrazing,
            ),
            (10.0, nadir_camera(82, 61, 96, 96), EventClass::CreaseX),
        ] {
            let input = SceneSliceInput {
                dem,
                chart: Chart::DependentZ,
                free,
                omega,
                camera,
                tmin: 0.001,
                tmax: 1e30,
                normal_offset: 0.001,
            };
            let events = certify_scene_slice(&input).expect("Gaussian slice must certify");
            assert!(events.iter().any(|e| e.class == expected));
            let projected = project_gpu_events(
                &events,
                &camera,
                DiscreteDraw {
                    replicate_count: 2,
                    frame_count: 4,
                    samples_per_frame: 4,
                    frame: 1,
                    sample: 2,
                },
            )
            .expect("accepted interval events must project");
            assert!(projected
                .iter()
                .all(|e| e.absolute_error.iter().all(|v| v.is_finite())));
            // An ideal interior receiver can round onto a cell edge in f32.
            // Reject that projection before the GPU can take another branch.
            let mut edge_rounded = events[0];
            edge_rounded.receiver_cell = (0, 0);
            edge_rounded.receiver_cell_world = [Iv::p(0.0), Iv::p(1.0), Iv::p(0.0), Iv::p(1.0)];
            edge_rounded.receiver_xyz[0] = Iv::p(1.0 - 2.0_f64.powi(-25));
            edge_rounded.receiver_xyz[2] = Iv::p(0.5);
            assert_eq!(
                project_gpu_events(
                    &[edge_rounded],
                    &camera,
                    DiscreteDraw {
                        replicate_count: 2,
                        frame_count: 4,
                        samples_per_frame: 4,
                        frame: 1,
                        sample: 2,
                    }
                )
                .unwrap_err(),
                Uncertifiable::Density
            );
            let wrong_camera = Camera {
                origin: [0.0, 43.0, 0.001],
                ..camera
            };
            assert_eq!(
                project_gpu_events(
                    &events,
                    &wrong_camera,
                    DiscreteDraw {
                        replicate_count: 2,
                        frame_count: 4,
                        samples_per_frame: 4,
                        frame: 1,
                        sample: 2,
                    }
                )
                .unwrap_err(),
                Uncertifiable::InvalidInput
            );
        }
    }

    #[test]
    fn bilinear_dem_perimeter_events_are_certified_in_both_charts() {
        let x_heights = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let z_heights = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let camera = Camera {
            origin: [0.0, 10.0, 0.0],
            right: [1.0, 0.0, 0.0],
            up: [0.0, 0.0, -1.0],
            forward: [0.0, -1.0, 0.0],
            fov_y: 2.0 * 0.1_f64.atan(),
            half_h: 0.1,
            half_w: 0.1,
            aspect: 1.0,
            camera_exposure: 1.0,
            width: 1,
            height: 1,
            pixel_x: 0,
            pixel_y: 0,
        };
        for (width, height, heights, chart, omega, expected) in [
            (
                3,
                2,
                &x_heights[..],
                Chart::DependentX,
                [3.0_f64.sqrt() / 2.0, 0.5, 0.0],
                EventClass::PerimeterX,
            ),
            (
                2,
                3,
                &z_heights[..],
                Chart::DependentZ,
                [0.0, 0.5, 3.0_f64.sqrt() / 2.0],
                EventClass::PerimeterZ,
            ),
        ] {
            let input = SliceInput {
                dem: Dem {
                    width,
                    height,
                    x0: 0.0,
                    z0: 0.0,
                    sx: 1.0,
                    sz: 1.0,
                    heights,
                },
                receiver_cell: (0, 0),
                chart,
                free: 0.5,
                dependent: Iv { lo: 0.0, hi: 1.0 },
                omega,
                tmin: 0.001,
                tmax: 1e30,
                normal_offset: 0.001,
                camera,
            };
            let events = certify_slice(&input).expect("perimeter fixture must certify");
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].class, expected);
            assert!(events[0]
                .signed_sun_weight
                .iter()
                .all(|v| v.lo.is_finite() && v.hi.is_finite()));
        }
    }

    #[test]
    fn strict_tmin_cutoff_and_below_entry_events_are_certified() {
        let cutoff = [
            0.0, 0.0, 0.0, 0.0, 0.004, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004, 0.002, 0.0,
        ];
        let entry_x = [0.0, 0.0, 0.004, 0.004, 0.0, 0.0, 0.004, 0.004];
        let entry_z = [0.0, 0.0, 0.0, 0.0, 0.004, 0.004, 0.004, 0.004];
        let camera = Camera {
            origin: [0.0, 10.0, 0.0],
            right: [1.0, 0.0, 0.0],
            up: [0.0, 0.0, -1.0],
            forward: [0.0, -1.0, 0.0],
            fov_y: 2.0 * 0.0005_f64.atan(),
            half_h: 0.0005,
            half_w: 0.0005,
            aspect: 1.0,
            camera_exposure: 1.0,
            width: 1,
            height: 1,
            pixel_x: 0,
            pixel_y: 0,
        };
        let c = 3.0_f64.sqrt() / 2.0;
        for (width, height, spacing, heights, chart, free, omega, expected) in [
            (
                7,
                2,
                0.0002,
                &cutoff[..],
                Chart::DependentX,
                0.0001,
                [c, 0.5, 0.0],
                EventClass::Cutoff,
            ),
            (
                4,
                2,
                0.0006,
                &entry_x[..],
                Chart::DependentX,
                0.0003,
                [c, 0.5, 0.0],
                EventClass::CellEntryX,
            ),
            (
                2,
                4,
                0.0006,
                &entry_z[..],
                Chart::DependentZ,
                0.0003,
                [0.0, 0.5, c],
                EventClass::CellEntryZ,
            ),
        ] {
            let events = certify_slice(&SliceInput {
                dem: Dem {
                    width,
                    height,
                    x0: 0.0,
                    z0: 0.0,
                    sx: spacing,
                    sz: spacing,
                    heights,
                },
                receiver_cell: (0, 0),
                chart,
                free,
                dependent: Iv {
                    lo: 0.0,
                    hi: spacing,
                },
                omega,
                tmin: 0.001,
                tmax: 1e30,
                normal_offset: 0.001,
                camera,
            })
            .expect("strict ray interval fixture must certify");
            assert_eq!(events.len(), 1, "{expected:?}");
            assert_eq!(events[0].class, expected);
            assert!(!events[0].lit_on_lower_side);
            assert!(events[0].signed_sun_weight.iter().any(|v| v.lo > 0.0));
        }
    }

    #[test]
    fn environment_lookup_rejects_f32_texel_boundary_ambiguity() {
        assert_eq!(
            stable_environment_texel(Iv::p(0.25), 2).expect("interior texel"),
            0
        );
        assert_eq!(
            stable_environment_texel(
                Iv {
                    lo: 0.499_999_9,
                    hi: 0.500_000_1,
                },
                2,
            )
            .unwrap_err(),
            Uncertifiable::UnresolvedRoot,
        );
    }

    #[test]
    fn high_exposure_reinhard_loss_rejects_outside_wgsl_division_accuracy_domain() {
        // This exposure is finite in f32, but `1 + linear * exposure` is
        // larger than WGSL's documented 2^126 division-divisor ceiling.  A
        // loss certificate must fail closed instead of assigning it a 2.5-ULP
        // error budget.
        assert_eq!(
            wgsl_loss_interval([Iv::p(1.0); 3], [0.0; 3], 1.0e38_f32, 0.0).unwrap_err(),
            Uncertifiable::UnresolvedRoot,
        );
    }
}
