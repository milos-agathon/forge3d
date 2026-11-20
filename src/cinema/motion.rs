//! Derivatives of the consumer's continuous cubic with its stored native keys.
//! Float32 evaluation roundoff is measured separately by the dense native probe.
use super::{binding, validate::Keyframe};
use glam::DVec3;

#[derive(Clone, Copy)]
struct Jet([f64; 4]);
impl Jet {
    fn mul(self, other: Self) -> Self {
        Self(std::array::from_fn(|n| {
            (0..=n).map(|i| self.0[i] * other.0[n - i]).sum()
        }))
    }
    fn sin(self) -> Self {
        let [a, b, c, d] = self.0;
        let (s, co) = a.sin_cos();
        Self([
            s,
            co * b,
            co * c - s * b * b / 2.0,
            co * d - s * b * c - co * b.powi(3) / 6.0,
        ])
    }
    fn cos(self) -> Self {
        let [a, b, c, d] = self.0;
        let (s, co) = a.sin_cos();
        Self([
            co,
            -s * b,
            -s * c - co * b * b / 2.0,
            -s * d - co * b * c + s * b.powi(3) / 6.0,
        ])
    }
}

#[derive(Clone, Copy)]
struct Cubic([f64; 4]);
impl Cubic {
    fn new([a, b, c, d]: [f64; 4]) -> Self {
        Self([
            b,
            (c - a) / 2.0,
            (2.0 * a - 5.0 * b + 4.0 * c - d) / 2.0,
            (-a + 3.0 * b - 3.0 * c + d) / 2.0,
        ])
    }
    fn sample(self, u: f64, h: f64) -> Jet {
        let [a, b, c, d] = self.0;
        Jet([
            a + u * (b + u * (c + u * d)),
            (b + u * (2.0 * c + u * 3.0 * d)) / h,
            (c + 3.0 * d * u) / h.powi(2),
            d / h.powi(3),
        ])
    }
    fn extrema(self, derivative: usize) -> Vec<f64> {
        let [_, b, c, d] = self.0;
        let mut candidates = vec![0.0, 1.0];
        if derivative == 0 {
            let discriminant = 4.0 * c * c - 12.0 * d * b;
            if d != 0.0 && discriminant >= 0.0 {
                candidates.extend([
                    (-2.0 * c + discriminant.sqrt()) / (6.0 * d),
                    (-2.0 * c - discriminant.sqrt()) / (6.0 * d),
                ]);
            } else if d == 0.0 && c != 0.0 {
                candidates.push(-b / (2.0 * c));
            }
        } else if derivative == 1 && d != 0.0 {
            candidates.push(-c / (3.0 * d));
        }
        candidates
            .into_iter()
            .filter(|u| *u >= 0.0 && *u <= 1.0)
            .collect()
    }
    // Exact polynomial extrema, expressed as ordinary derivatives.
    fn bound(self, derivative: usize, h: f64) -> f64 {
        let factorial = [1.0, 1.0, 2.0, 6.0][derivative];
        self.extrema(derivative)
            .into_iter()
            .map(|u| self.sample(u, h).0[derivative].abs() * factorial)
            .fold(0.0, f64::max)
    }
    fn trig_bound(self, cosine: bool) -> f64 {
        let values: Vec<_> = self
            .extrema(0)
            .into_iter()
            .map(|u| self.sample(u, 1.0).0[0])
            .collect();
        let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let peak = if cosine {
            0.0
        } else {
            std::f64::consts::FRAC_PI_2
        };
        if ((lo - peak) / std::f64::consts::PI).ceil()
            <= ((hi - peak) / std::f64::consts::PI).floor()
        {
            1.0
        } else if cosine {
            lo.cos().abs().max(hi.cos().abs())
        } else {
            lo.sin().abs().max(hi.sin().abs())
        }
    }
}
struct Curve {
    t0: f64,
    t1: f64,
    phi: Cubic,
    theta: Cubic,
    radius: Cubic,
    target: [Cubic; 3],
}
impl Curve {
    fn sample(&self, time: f64) -> ([DVec3; 4], [DVec3; 4]) {
        let h = self.t1 - self.t0;
        let u = ((time - self.t0) / h).clamp(0.0, 1.0);
        let p = self.phi.sample(u, h);
        let t = self.theta.sample(u, h);
        let r = self.radius.sample(u, h);
        let direction = [t.sin().mul(p.cos()), t.cos(), t.sin().mul(p.sin())];
        let target = self.target.map(|c| c.sample(u, h));
        let eye = std::array::from_fn::<_, 3, _>(|i| r.mul(direction[i]));
        let factorials = [1.0, 1.0, 2.0, 6.0];
        let aims = std::array::from_fn(|i| {
            DVec3::new(target[0].0[i], target[1].0[i], target[2].0[i]) * factorials[i]
        });
        let eyes = std::array::from_fn(|i| {
            aims[i] + DVec3::new(eye[0].0[i], eye[1].0[i], eye[2].0[i]) * factorials[i]
        });
        (eyes, aims)
    }
    fn jerk_bound(&self) -> f64 {
        let h = self.t1 - self.t0;
        let [vp, ap, jp] = std::array::from_fn(|i| self.phi.bound(i + 1, h));
        let [vt, at, jt] = std::array::from_fn(|i| self.theta.bound(i + 1, h));
        let s = self.theta.trig_bound(false);
        let c = self.theta.trig_bound(true);
        // Exact norms of spherical partials: azimuth derivatives carry sin(theta),
        // and mixed elevation/azimuth derivatives alternate sin/cos(theta).
        // Retain these factors to bound the entire interval near a camera pole.
        let d1 = vt + s * vp;
        let d2 = at + s * ap + vt * vt + 2.0 * c * vt * vp + s * vp * vp;
        let d3 = jt
            + s * jp
            + 3.0 * (vt * at + c * (vp * at + vt * ap) + s * vp * ap)
            + vt.powi(3)
            + 3.0 * s * vt * vt * vp
            + 3.0 * c * vt * vp * vp
            + s * vp.powi(3);
        let target = DVec3::new(
            self.target[0].bound(3, h),
            self.target[1].bound(3, h),
            self.target[2].bound(3, h),
        )
        .length();
        target
            + self.radius.bound(3, h)
            + 3.0 * self.radius.bound(2, h) * d1
            + 3.0 * self.radius.bound(1, h) * d2
            + self.radius.bound(0, h) * d3
    }
}

pub struct Consumer {
    curves: Vec<Curve>,
}
impl Consumer {
    pub fn new(keys: &[Keyframe]) -> Result<Self, String> {
        let native: Vec<_> = keys.iter().map(Keyframe::native).collect();
        if native.iter().any(|k| {
            !k.time.is_finite()
                || !k.phi_deg.is_finite()
                || !k.theta_deg.is_finite()
                || !k.fov_deg.is_finite()
                || !k.radius.is_finite()
                || k.target.is_some_and(|t| t.iter().any(|v| !v.is_finite()))
        }) {
            return Err(binding(
                "consumer_precision",
                "keyframes exceed native finite precision",
            ));
        }
        let mut curves = Vec::new();
        for i in 0..native.len().saturating_sub(1) {
            let controls = [
                native[i.saturating_sub(1)],
                native[i],
                native[i + 1],
                native[(i + 2).min(native.len() - 1)],
            ];
            let target: Option<Vec<_>> = controls.iter().map(|k| k.target).collect();
            let target = target.ok_or_else(|| binding("consumer", "animation lost target"))?;
            let t0 = f64::from(native[i].time);
            let t1 = f64::from(native[i + 1].time);
            if t1 <= t0 {
                return Err(binding(
                    "consumer_precision",
                    "native keyframe times coalesced",
                ));
            }
            curves.push(Curve {
                t0,
                t1,
                phi: Cubic::new(controls.map(|k| f64::from(k.phi_deg).to_radians())),
                theta: Cubic::new(controls.map(|k| f64::from(k.theta_deg).to_radians())),
                radius: Cubic::new(controls.map(|k| k.radius)),
                target: std::array::from_fn(|axis| {
                    Cubic::new(std::array::from_fn(|j| target[j][axis]))
                }),
            });
        }
        if curves.is_empty() {
            return Err(binding("consumer", "animation has no intervals"));
        }
        Ok(Self { curves })
    }
    pub fn sample(&self, time: f64) -> ([DVec3; 4], [DVec3; 4]) {
        let i = self
            .curves
            .partition_point(|c| c.t1 <= time)
            .min(self.curves.len() - 1);
        self.curves[i].sample(time)
    }
    pub fn sample_before(&self, time: f64) -> ([DVec3; 4], [DVec3; 4]) {
        let i = self
            .curves
            .partition_point(|c| c.t1 < time)
            .min(self.curves.len() - 1);
        self.curves[i].sample(time)
    }
    pub fn jerk_bound(&self, start: f64, end: f64) -> f64 {
        self.curves
            .iter()
            .filter(|c| c.t1 > start && c.t0 < end)
            .map(Curve::jerk_bound)
            .fold(0.0, f64::max)
    }
    pub fn knot_jerk(&self, start: f64, end: f64) -> f64 {
        self.curves
            .iter()
            .filter(|c| c.t1 > start && c.t0 < end)
            .flat_map(|c| {
                [c.t0, c.t1].map(|t| {
                    let (eye, target) = c.sample(t);
                    eye[3].length().max(target[3].length())
                })
            })
            .fold(0.0, f64::max)
    }
    pub fn acceleration_jump(&self, start: f64, end: f64) -> f64 {
        self.curves
            .windows(2)
            .filter(|pair| pair[0].t1 > start && pair[0].t1 < end)
            .map(|pair| {
                let (left_eye, left_target) = pair[0].sample(pair[0].t1);
                let (right_eye, right_target) = pair[1].sample(pair[1].t0);
                left_eye[2]
                    .distance(right_eye[2])
                    .max(left_target[2].distance(right_target[2]))
            })
            .fold(0.0, f64::max)
    }
}
