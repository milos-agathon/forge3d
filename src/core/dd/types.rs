/// Unit roundoff for IEEE-754 binary32.
pub const DD_U: f64 = 5.960_464_477_539_063e-8;
/// Joldeș-Muller-Popescu (2017) addition bound, in u².
pub const DD_ADD_BOUND_U2: f64 = 3.0;
/// Joldeș-Muller-Popescu (2017) multiplication bound, in u².
pub const DD_MUL_BOUND_U2: f64 = 7.0;
/// Joldeș-Muller-Popescu (2017) division bound, in u².
pub const DD_DIV_BOUND_U2: f64 = 15.0;
/// DUPLA square-root bound, in u².
pub const DD_SQRT_BOUND_U2: f64 = 15.0;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DD {
    pub hi: f32,
    pub lo: f32,
}

impl DD {
    pub const ZERO: Self = Self { hi: 0.0, lo: 0.0 };

    pub fn to_f64(self) -> f64 {
        self.hi as f64 + self.lo as f64
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DDVec3 {
    pub x: DD,
    pub y: DD,
    pub z: DD,
}

impl DDVec3 {
    pub fn from_dvec3(value: glam::DVec3) -> Self {
        Self {
            x: DD::from_f64(value.x),
            y: DD::from_f64(value.y),
            z: DD::from_f64(value.z),
        }
    }

    pub fn to_dvec3(self) -> glam::DVec3 {
        glam::DVec3::new(self.x.to_f64(), self.y.to_f64(), self.z.to_f64())
    }
}
