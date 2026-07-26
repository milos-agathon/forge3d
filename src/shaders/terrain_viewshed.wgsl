struct ViewshedUniforms {
    dimensions: vec4<u32>,
    observer: vec4<f32>,
    metric: vec4<f32>,
    physics: vec4<f32>,
    geodetic: vec4<f32>,
}

struct ViewshedCell {
    visible: u32,
    curvature_drop_m: f32,
    refraction_gain_m: f32,
    horizon_distance_m: f32,
}

@group(0) @binding(0) var<uniform> uniforms: ViewshedUniforms;
@group(0) @binding(1) var<storage, read> heights: array<f32>;
@group(0) @binding(2) var<storage, read> geodesic_positions_m: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> result: array<ViewshedCell>;

fn height_at(pixel: vec2<f32>) -> f32 {
    let limit = vec2<f32>(
        f32(uniforms.dimensions.x - 1u),
        f32(uniforms.dimensions.y - 1u),
    );
    let p = clamp(pixel, vec2<f32>(0.0), limit);
    let lo = vec2<u32>(floor(p));
    let hi = min(lo + vec2<u32>(1u), uniforms.dimensions.xy - vec2<u32>(1u));
    let fraction = p - vec2<f32>(lo);
    let width = uniforms.dimensions.x;
    let h00 = heights[lo.y * width + lo.x];
    let h10 = heights[lo.y * width + hi.x];
    let h01 = heights[hi.y * width + lo.x];
    let h11 = heights[hi.y * width + hi.x];
    return mix(mix(h00, h10, fraction.x), mix(h01, h11, fraction.x), fraction.y);
}

fn inverse_radius(direction_m: vec2<f32>) -> f32 {
    let distance_squared = dot(direction_m, direction_m);
    if distance_squared == 0.0 {
        return 0.0;
    }
    let east_fraction_squared = direction_m.x * direction_m.x / distance_squared;
    let north_fraction_squared = direction_m.y * direction_m.y / distance_squared;
    return north_fraction_squared * uniforms.physics.x
        + east_fraction_squared * uniforms.physics.y;
}

fn latlon_to_pixel(latitude: f32, longitude: f32) -> vec2<f32> {
    var longitude_deg = degrees(longitude);
    if longitude_deg < uniforms.geodetic.z {
        longitude_deg += 360.0;
    }
    if longitude_deg > uniforms.geodetic.z + 180.0 {
        longitude_deg -= 360.0;
    }
    return vec2<f32>(
        (longitude_deg - uniforms.geodetic.z) / uniforms.metric.y - 0.5,
        (uniforms.geodetic.w - degrees(latitude)) / uniforms.metric.z - 0.5,
    );
}

fn geodesic_sample_pixel(azimuth: f32, distance_m: f32) -> vec2<f32> {
    if uniforms.metric.w > 0.0 {
        let angular_distance = distance_m / uniforms.metric.w;
        let sin_latitude = sin(uniforms.geodetic.x) * cos(angular_distance)
            + cos(uniforms.geodetic.x) * sin(angular_distance) * cos(azimuth);
        let latitude = asin(clamp(sin_latitude, -1.0, 1.0));
        let longitude = uniforms.geodetic.y + atan2(
            sin(azimuth) * sin(angular_distance) * cos(uniforms.geodetic.x),
            cos(angular_distance) - sin(uniforms.geodetic.x) * sin(latitude),
        );
        return latlon_to_pixel(latitude, longitude);
    }
    let flattening = 1.0 / 298.257223563;
    let semi_major = 6378137.0;
    let semi_minor = semi_major * (1.0 - flattening);
    let reduced_latitude = atan((1.0 - flattening) * tan(uniforms.geodetic.x));
    let sin_u1 = sin(reduced_latitude);
    let cos_u1 = cos(reduced_latitude);
    let sin_azimuth = sin(azimuth);
    let cos_azimuth = cos(azimuth);
    let sigma1 = atan2(tan(reduced_latitude), cos_azimuth);
    let sin_alpha = cos_u1 * sin_azimuth;
    let cos_sq_alpha = 1.0 - sin_alpha * sin_alpha;
    let u_sq = cos_sq_alpha
        * (semi_major * semi_major - semi_minor * semi_minor)
        / (semi_minor * semi_minor);
    let coefficient_a = 1.0 + u_sq / 16384.0
        * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
    let coefficient_b = u_sq / 1024.0
        * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));
    var sigma = distance_m / (semi_minor * coefficient_a);
    for (var iteration = 0u; iteration < 4u; iteration += 1u) {
        let two_sigma_m = 2.0 * sigma1 + sigma;
        let sin_sigma = sin(sigma);
        let cos_sigma = cos(sigma);
        let cos_two_sigma_m = cos(two_sigma_m);
        let delta_sigma = coefficient_b * sin_sigma
            * (cos_two_sigma_m + coefficient_b / 4.0
                * (cos_sigma * (-1.0 + 2.0 * cos_two_sigma_m * cos_two_sigma_m)
                    - coefficient_b / 6.0 * cos_two_sigma_m
                    * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                    * (-3.0 + 4.0 * cos_two_sigma_m * cos_two_sigma_m)));
        sigma = distance_m / (semi_minor * coefficient_a) + delta_sigma;
    }
    let sin_sigma = sin(sigma);
    let cos_sigma = cos(sigma);
    let two_sigma_m = 2.0 * sigma1 + sigma;
    let temporary = sin_u1 * sin_sigma - cos_u1 * cos_sigma * cos_azimuth;
    let latitude = atan2(
        sin_u1 * cos_sigma + cos_u1 * sin_sigma * cos_azimuth,
        (1.0 - flattening) * sqrt(sin_alpha * sin_alpha + temporary * temporary),
    );
    let lambda = atan2(
        sin_sigma * sin_azimuth,
        cos_u1 * cos_sigma - sin_u1 * sin_sigma * cos_azimuth,
    );
    let coefficient_c = flattening / 16.0 * cos_sq_alpha
        * (4.0 + flattening * (4.0 - 3.0 * cos_sq_alpha));
    let cos_two_sigma_m = cos(two_sigma_m);
    let longitude_delta = lambda - (1.0 - coefficient_c) * flattening * sin_alpha
        * (sigma + coefficient_c * sin_sigma
            * (cos_two_sigma_m + coefficient_c * cos_sigma
                * (-1.0 + 2.0 * cos_two_sigma_m * cos_two_sigma_m)));
    return latlon_to_pixel(latitude, uniforms.geodetic.y + longitude_delta);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= uniforms.dimensions.x || id.y >= uniforms.dimensions.y {
        return;
    }
    let index = id.y * uniforms.dimensions.x + id.x;
    let target_pixel = vec2<f32>(id.xy);
    let delta_pixel = target_pixel - uniforms.observer.xy;
    let direction_m = geodesic_positions_m[index];
    let distance_m = length(direction_m);
    let azimuth = atan2(direction_m.x, direction_m.y);
    let inv_radius = inverse_radius(direction_m);
    let vacuum_drop = 0.5 * inv_radius * distance_m * distance_m;
    let effective_drop = vacuum_drop * uniforms.physics.z;
    let refraction_gain = vacuum_drop - effective_drop;
    let observer_elevation = height_at(uniforms.observer.xy) + uniforms.observer.z;
    let target_absolute_elevation = heights[index] + uniforms.observer.w;
    var horizon_distance = uniforms.metric.x;
    if inv_radius > 0.0 {
        let effective_inv_radius = inv_radius * uniforms.physics.z;
        horizon_distance = sqrt(2.0 * max(observer_elevation, 0.0) / effective_inv_radius)
            + sqrt(2.0 * max(target_absolute_elevation, 0.0) / effective_inv_radius);
    }

    if distance_m == 0.0 {
        result[index] = ViewshedCell(1u, 0.0, 0.0, horizon_distance);
        return;
    }
    if distance_m > uniforms.metric.x {
        result[index] = ViewshedCell(0u, vacuum_drop, refraction_gain, horizon_distance);
        return;
    }

    let target_elevation = target_absolute_elevation - effective_drop;
    let half_cell_steps = u32(ceil(2.0 * max(abs(delta_pixel.x), abs(delta_pixel.y))));
    let steps = max(half_cell_steps, 1u);
    var visible = 1u;
    for (var step = 1u; step < steps; step += 1u) {
        let fraction = f32(step) / f32(steps);
        let sample_distance = distance_m * fraction;
        let sample_pixel = geodesic_sample_pixel(azimuth, sample_distance);
        let outer_min = vec2<f32>(-0.5);
        let outer_max = vec2<f32>(uniforms.dimensions.xy) - vec2<f32>(0.5);
        if any(sample_pixel < outer_min) || any(sample_pixel > outer_max) {
            visible = 2u;
            break;
        }
        // Camera-relative distance avoids subtracting large geodetic coordinates;
        // h_eff is evaluated directly so f32 retains sub-metre precision at 100 km.
        let sample_drop = 0.5 * inv_radius * uniforms.physics.z
            * sample_distance * sample_distance;
        let terrain_effective = height_at(sample_pixel) - sample_drop;
        let sightline = mix(observer_elevation, target_elevation, fraction);
        if terrain_effective > sightline + 0.001 {
            visible = 0u;
            break;
        }
    }
    result[index] = ViewshedCell(
        visible,
        vacuum_drop,
        refraction_gain,
        horizon_distance,
    );
}
