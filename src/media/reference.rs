//! Unbiased heterogeneous-media reference kernel shared by terrain reference callers.

use super::model::normalize;
use super::{
    delta_track_counted, power_heuristic, ratio_track, russian_roulette, DirectionalSun,
    EnvironmentDistribution, MediaError, Ray, Rgb, SampleIdentity, TrackingContext,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceSurfaceHit {
    pub distance: f32,
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub albedo: Rgb,
}

/// The portion of a geometry ray occupied by the canonical medium. Geometry
/// reach is deliberately separate: a bounded cloud can end before terrain or
/// the environment without turning that boundary into a surface or miss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceMediumInterval {
    pub start: f32,
    pub end: f32,
}

/// Geometry seam implemented by the hybrid terrain reference with its
/// authoritative `terrain_trace` traversal. Media never owns a second terrain
/// marcher; every camera, sun, environment, and phase ray returns through this
/// interface.
pub trait ReferenceScene {
    fn intersect(
        &self,
        ray: Ray,
        maximum_distance: f32,
    ) -> Result<Option<ReferenceSurfaceHit>, MediaError>;
    fn occluded(&self, ray: Ray, maximum_distance: f32) -> Result<bool, MediaError>;
    fn geometry_reach(&self, ray: Ray) -> Result<f32, MediaError>;
    fn medium_interval(
        &self,
        ray: Ray,
        maximum_distance: f32,
    ) -> Result<Option<ReferenceMediumInterval>, MediaError>;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceTransportConfig {
    pub roulette_start_bounce: u32,
    pub roulette_minimum_probability: f32,
}

impl ReferenceTransportConfig {
    fn validate(self) -> Result<(), MediaError> {
        if !self.roulette_minimum_probability.is_finite()
            || !(0.0..=1.0).contains(&self.roulette_minimum_probability)
        {
            return Err(MediaError::InvalidTransport(
                "roulette minimum must be a finite probability".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceTransportSample {
    pub radiance: Rgb,
    pub spectral_channel: u32,
    pub collision_count: u32,
    pub surface_count: u32,
    pub tracking_step_count: u64,
}

/// Trace one deterministic hero-wavelength path. RGB channels are stratified
/// by sample index (each consecutive triple covers all three) and weighted
/// by three. A deterministic per-triple rotation keeps incomplete terminal
/// triples unbiased without sacrificing exact triple coverage.
/// Surface environment NEE and cosine-BSDF continuation use power-heuristic
/// MIS with both PDFs measured per unit solid angle; the sun remains delta.
pub fn trace_reference_sample<S: ReferenceScene>(
    context: &TrackingContext,
    scene: &S,
    environment: &EnvironmentDistribution,
    sun: DirectionalSun,
    camera_ray: Ray,
    identity: SampleIdentity,
    config: ReferenceTransportConfig,
) -> Result<ReferenceTransportSample, MediaError> {
    config.validate()?;
    const SPECTRAL_CHANNEL_DIMENSION: u64 = 0x4e45_5048_454c_4500;
    let triple_identity = SampleIdentity {
        sample: identity.sample / 3,
        ..identity
    };
    let mut attempt = 0u64;
    let rotation = exact_ternary(|| {
        let bits = triple_identity.random_bits(
            SPECTRAL_CHANNEL_DIMENSION.wrapping_add(attempt.wrapping_mul(0x9e37_79b9_7f4a_7c15)),
        );
        attempt = attempt.wrapping_add(1);
        bits
    });
    let channel = ((identity.sample % 3) as usize + rotation) % 3;
    let sample = trace_reference_core(
        context,
        scene,
        environment,
        sun,
        camera_ray,
        identity,
        (config, Some(channel)),
    )?;
    Ok(ReferenceTransportSample {
        radiance: Rgb::new(
            sample.radiance.components().map(|v| v * 3.0),
            "reference radiance",
        )?,
        spectral_channel: channel as u32,
        collision_count: sample.collision_count,
        surface_count: sample.surface_count,
        tracking_step_count: sample.tracking_step_count,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ReferenceRgbTransportSample {
    pub radiance: Rgb,
    pub collision_count: u32,
    pub surface_count: u32,
    pub tracking_step_count: u64,
}

/// Shared RGB paths with real/null-event likelihood ratios. Unlike hero-channel
/// sampling, a grey scene produces identical RGB components in every sample.
/// Environment/surface MIS and the delta sun use the same strategies as the
/// public single-channel reference; only the spectral proposal differs.
pub(crate) fn trace_reference_rgb_sample<S: ReferenceScene>(
    context: &TrackingContext,
    scene: &S,
    environment: &EnvironmentDistribution,
    sun: DirectionalSun,
    camera_ray: Ray,
    identity: SampleIdentity,
    config: ReferenceTransportConfig,
) -> Result<ReferenceRgbTransportSample, MediaError> {
    trace_reference_core(
        context,
        scene,
        environment,
        sun,
        camera_ray,
        identity,
        (config, None),
    )
}

fn trace_reference_core<S: ReferenceScene>(
    context: &TrackingContext,
    scene: &S,
    environment: &EnvironmentDistribution,
    sun: DirectionalSun,
    camera_ray: Ray,
    identity: SampleIdentity,
    sampling: (ReferenceTransportConfig, Option<usize>),
) -> Result<ReferenceRgbTransportSample, MediaError> {
    let (config, channel) = sampling;
    config.validate()?;
    let mut ray = Ray {
        origin: camera_ray.origin,
        direction: normalize(camera_ray.direction).ok_or_else(|| {
            MediaError::InvalidTransport("camera direction must be finite and nonzero".into())
        })?,
    };
    let mut throughput = std::array::from_fn::<_, 3, _>(|c| {
        if channel.is_none_or(|selected| selected == c) {
            1.0f32
        } else {
            0.0
        }
    });
    let mut radiance = [0.0f32; 3];
    let mut collision_count = 0;
    let mut surface_count = 0;
    let mut tracking_step_count = 0;
    let mut previous_continuation_pdf = None;

    let mut bounce = 0u32;
    loop {
        let bounce_id = SampleIdentity { bounce, ..identity };
        let geometry_reach = scene.geometry_reach(ray)?;
        validate_distance(geometry_reach)?;
        let surface = scene.intersect(ray, geometry_reach)?;
        if let Some(hit) = surface {
            validate_surface_hit(hit, geometry_reach)?;
        }
        let geometry_distance = surface.map_or(geometry_reach, |hit| hit.distance);
        let interval = validated_medium_interval(scene, ray, geometry_distance)?;
        let (collision, weights, steps) = if let Some(interval) = interval {
            let segment = shifted_ray(ray, interval.start);
            let length = interval.end - interval.start;
            if let Some(channel) = channel {
                let (collision, steps) =
                    delta_track_counted(context, segment, length, channel, bounce_id)?;
                (collision, Rgb::ONE, steps)
            } else {
                super::tracking::spectral_delta_track(context, segment, length, bounce_id)?
            }
        } else {
            (None, Rgb::ONE, 0)
        };
        tracking_step_count += steps;
        for (value, weight) in throughput.iter_mut().zip(weights.components()) {
            *value *= weight;
        }

        if let Some(collision) = collision {
            collision_count += 1;
            let sigma_t = collision.extinction.components();
            let density = context
                .medium()
                .density()
                .physical_density(collision.position);
            let sigma_s = context.medium().sigma_s().components();
            for c in 0..3 {
                if sigma_t[c] > 0.0 {
                    throughput[c] *= (sigma_s[c] * density) / sigma_t[c];
                } else {
                    // A channel with no extinction has zero real-event weight.
                    throughput[c] = 0.0;
                }
            }
            if throughput == [0.0; 3] {
                break;
            }

            let to_sun = sun.direction_to_sun;
            let sun_ray = Ray {
                origin: collision.position,
                direction: to_sun,
            };
            let sun_distance = scene.geometry_reach(sun_ray)?;
            validate_distance(sun_distance)?;
            if !scene.occluded(sun_ray, sun_distance)? {
                let (transmittance, steps) = reference_transmittance(
                    context,
                    scene,
                    sun_ray,
                    sun_distance,
                    SampleIdentity {
                        sample: identity.sample.wrapping_add(0x1000),
                        ..bounce_id
                    },
                )?;
                tracking_step_count += steps;
                for c in 0..3 {
                    radiance[c] += throughput[c]
                        * context
                            .medium()
                            .phase()
                            .evaluate(dot(ray.direction, to_sun))?
                        * sun.radiance.components()[c]
                        * transmittance.components()[c];
                }
            }

            let env = environment.sample([
                bounce_id.uniform(10),
                bounce_id.uniform(11),
                bounce_id.uniform(12),
                bounce_id.uniform(13),
            ])?;
            let env_ray = Ray {
                origin: collision.position,
                direction: env.direction,
            };
            let env_distance = scene.geometry_reach(env_ray)?;
            validate_distance(env_distance)?;
            if !scene.occluded(env_ray, env_distance)? {
                let phase_pdf = context
                    .medium()
                    .phase()
                    .evaluate(dot(ray.direction, env.direction))?;
                let (light_weight, _) = power_heuristic(env.pdf_solid_angle, phase_pdf)?;
                let (transmittance, steps) = reference_transmittance(
                    context,
                    scene,
                    env_ray,
                    env_distance,
                    SampleIdentity {
                        sample: identity.sample.wrapping_add(0x2000),
                        ..bounce_id
                    },
                )?;
                tracking_step_count += steps;
                for c in 0..3 {
                    radiance[c] += throughput[c]
                        * phase_pdf
                        * env.radiance.components()[c]
                        * transmittance.components()[c]
                        * light_weight
                        / env.pdf_solid_angle;
                }
            }

            let phase = context.medium().phase().sample(
                ray.direction,
                [bounce_id.uniform(20), bounce_id.uniform(21)],
            )?;
            for value in &mut throughput {
                *value *= phase.value / phase.pdf;
            }
            previous_continuation_pdf = Some(phase.pdf);
            ray = Ray {
                origin: collision.position,
                direction: phase.direction,
            };
        } else if let Some(hit) = surface {
            surface_count += 1;
            let albedo = hit.albedo.components();
            let normal = normalize(hit.normal).ok_or_else(|| {
                MediaError::InvalidTransport("surface normal must be finite and nonzero".into())
            })?;
            let to_sun = sun.direction_to_sun;
            let cosine = dot(normal, to_sun).max(0.0);
            if cosine > 0.0 {
                let sun_ray = Ray {
                    origin: hit.position,
                    direction: to_sun,
                };
                let sun_distance = scene.geometry_reach(sun_ray)?;
                validate_distance(sun_distance)?;
                if !scene.occluded(sun_ray, sun_distance)? {
                    let (transmittance, steps) = reference_transmittance(
                        context,
                        scene,
                        sun_ray,
                        sun_distance,
                        SampleIdentity {
                            sample: identity.sample.wrapping_add(0x3000),
                            ..bounce_id
                        },
                    )?;
                    tracking_step_count += steps;
                    for c in 0..3 {
                        radiance[c] += throughput[c]
                            * albedo[c]
                            * cosine
                            * sun.radiance.components()[c]
                            * transmittance.components()[c]
                            / std::f32::consts::PI;
                    }
                }
            }

            let env = environment.sample([
                bounce_id.uniform(32),
                bounce_id.uniform(33),
                bounce_id.uniform(34),
                bounce_id.uniform(35),
            ])?;
            let cos_l = dot(normal, env.direction).max(0.0);
            if cos_l > 0.0 {
                let env_ray = Ray {
                    origin: hit.position,
                    direction: env.direction,
                };
                let reach = scene.geometry_reach(env_ray)?;
                validate_distance(reach)?;
                if !scene.occluded(env_ray, reach)? {
                    let (transmittance, steps) = reference_transmittance(
                        context,
                        scene,
                        env_ray,
                        reach,
                        SampleIdentity {
                            sample: identity.sample.wrapping_add(0x6000),
                            ..bounce_id
                        },
                    )?;
                    tracking_step_count += steps;
                    let (light_weight, _) =
                        power_heuristic(env.pdf_solid_angle, cos_l / std::f32::consts::PI)?;
                    for c in 0..3 {
                        radiance[c] += throughput[c]
                            * albedo[c]
                            * cos_l
                            * env.radiance.components()[c]
                            * transmittance.components()[c]
                            * light_weight
                            / (std::f32::consts::PI * env.pdf_solid_angle);
                    }
                }
            }
            let direction = cosine_hemisphere(normal, bounce_id.uniform(30), bounce_id.uniform(31));
            for (value, albedo) in throughput.iter_mut().zip(albedo) {
                *value *= albedo;
            }
            previous_continuation_pdf =
                Some(dot(normal, direction).max(0.0) / std::f32::consts::PI);
            ray = Ray {
                origin: hit.position,
                direction,
            };
        } else {
            let env = environment.radiance(ray.direction).components();
            let weight = if let Some(phase_pdf) = previous_continuation_pdf {
                power_heuristic(phase_pdf, environment.pdf(ray.direction))?.0
            } else {
                1.0
            };
            for c in 0..3 {
                radiance[c] += throughput[c] * env[c] * weight;
            }
            break;
        }

        if bounce >= config.roulette_start_bounce {
            let rgb = Rgb::new(throughput, "reference throughput")?;
            let Some(scale) = russian_roulette(
                rgb,
                bounce_id.uniform(40),
                config.roulette_minimum_probability,
            )?
            else {
                break;
            };
            for value in &mut throughput {
                *value *= scale;
            }
        }
        if throughput
            .iter()
            .chain(radiance.iter())
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(MediaError::InvalidTransport(
                "reference path produced non-finite or negative transport".into(),
            ));
        }
        bounce = bounce.checked_add(1).ok_or_else(|| {
            MediaError::InvalidTransport("reference bounce identity overflowed".into())
        })?;
    }

    Ok(ReferenceRgbTransportSample {
        radiance: Rgb::new(radiance, "reference radiance")?,
        collision_count,
        surface_count,
        tracking_step_count,
    })
}

fn exact_ternary(mut random_bits: impl FnMut() -> u64) -> usize {
    loop {
        let bits = random_bits();
        // 0..u64::MAX contains u64::MAX values, exactly divisible by three.
        // Rejecting the sole remaining value makes modulo reduction unbiased.
        if bits != u64::MAX {
            return (bits % 3) as usize;
        }
    }
}

fn validate_distance(distance: f32) -> Result<(), MediaError> {
    if distance.is_finite() && distance >= 0.0 {
        Ok(())
    } else {
        Err(MediaError::InvalidTransport(
            "scene segment distances must be finite and non-negative".into(),
        ))
    }
}

fn validated_medium_interval<S: ReferenceScene>(
    scene: &S,
    ray: Ray,
    maximum_distance: f32,
) -> Result<Option<ReferenceMediumInterval>, MediaError> {
    let Some(interval) = scene.medium_interval(ray, maximum_distance)? else {
        return Ok(None);
    };
    if !interval.start.is_finite()
        || !interval.end.is_finite()
        || interval.start < 0.0
        || interval.end < interval.start
        || interval.end > maximum_distance
    {
        return Err(MediaError::InvalidTransport(
            "scene returned an invalid medium interval".into(),
        ));
    }
    Ok((interval.end > interval.start).then_some(interval))
}

fn shifted_ray(ray: Ray, distance: f32) -> Ray {
    Ray {
        origin: [
            ray.origin[0] + ray.direction[0] * distance,
            ray.origin[1] + ray.direction[1] * distance,
            ray.origin[2] + ray.direction[2] * distance,
        ],
        direction: ray.direction,
    }
}

fn reference_transmittance<S: ReferenceScene>(
    context: &TrackingContext,
    scene: &S,
    ray: Ray,
    maximum_distance: f32,
    identity: SampleIdentity,
) -> Result<(Rgb, u64), MediaError> {
    let Some(interval) = validated_medium_interval(scene, ray, maximum_distance)? else {
        return Ok((Rgb::ONE, 0));
    };
    ratio_track(
        context,
        shifted_ray(ray, interval.start),
        interval.end - interval.start,
        identity,
    )
}

fn validate_surface_hit(hit: ReferenceSurfaceHit, maximum_distance: f32) -> Result<(), MediaError> {
    if !hit.distance.is_finite()
        || hit.distance < 0.0
        || hit.distance > maximum_distance
        || hit
            .position
            .iter()
            .chain(&hit.normal)
            .any(|value| !value.is_finite())
        || normalize(hit.normal).is_none()
    {
        return Err(MediaError::InvalidTransport(
            "reference scene returned an invalid surface hit".into(),
        ));
    }
    Ok(())
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cosine_hemisphere(normal: [f32; 3], u0: f32, u1: f32) -> [f32; 3] {
    let sign = if normal[2] < 0.0 { -1.0 } else { 1.0 };
    let a = -1.0 / (sign + normal[2]);
    let b = normal[0] * normal[1] * a;
    let tangent = [
        1.0 + sign * normal[0] * normal[0] * a,
        sign * b,
        -sign * normal[0],
    ];
    let bitangent = [b, sign + normal[1] * normal[1] * a, -normal[1]];
    let radius = u0.sqrt();
    let phi = std::f32::consts::TAU * u1;
    let local = [radius * phi.cos(), radius * phi.sin(), (1.0 - u0).sqrt()];
    [
        local[0] * tangent[0] + local[1] * bitangent[0] + local[2] * normal[0],
        local[0] * tangent[1] + local[1] * bitangent[1] + local[2] * normal[1],
        local[0] * tangent[2] + local[1] * bitangent[2] + local[2] * normal[2],
    ]
}

#[cfg(test)]
mod exact_ternary_tests {
    use super::exact_ternary;

    #[test]
    fn accepted_range_has_equal_ternary_buckets() {
        assert_eq!(u128::from(u64::MAX) % 3, 0);
        for (bits, expected) in [
            (0, 0),
            (1, 1),
            (2, 2),
            (u64::MAX - 3, 0),
            (u64::MAX - 2, 1),
            (u64::MAX - 1, 2),
        ] {
            assert_eq!(exact_ternary(|| bits), expected);
        }
    }

    #[test]
    fn rejected_tail_value_consumes_the_next_draw() {
        let mut draws = [u64::MAX, 5].into_iter();
        assert_eq!(exact_ternary(|| draws.next().unwrap()), 2);
        assert_eq!(draws.next(), None);
    }
}
