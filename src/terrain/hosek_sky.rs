use super::hosek_rgb_data::{DATASETS_RGB, DATASETS_RGB_RAD};

#[derive(Clone, Copy)]
pub(crate) struct HosekRgbSky {
    configs: [[f32; 9]; 3],
    radiances: [f32; 3],
}

#[cfg_attr(not(feature = "extension-module"), allow(dead_code))]
impl HosekRgbSky {
    pub(crate) fn uniform_a_d(self) -> [[f32; 4]; 3] {
        [
            [
                self.configs[0][0],
                self.configs[0][1],
                self.configs[0][2],
                self.configs[0][3],
            ],
            [
                self.configs[1][0],
                self.configs[1][1],
                self.configs[1][2],
                self.configs[1][3],
            ],
            [
                self.configs[2][0],
                self.configs[2][1],
                self.configs[2][2],
                self.configs[2][3],
            ],
        ]
    }

    pub(crate) fn uniform_e_h(self) -> [[f32; 4]; 3] {
        [
            [
                self.configs[0][4],
                self.configs[0][5],
                self.configs[0][6],
                self.configs[0][7],
            ],
            [
                self.configs[1][4],
                self.configs[1][5],
                self.configs[1][6],
                self.configs[1][7],
            ],
            [
                self.configs[2][4],
                self.configs[2][5],
                self.configs[2][6],
                self.configs[2][7],
            ],
        ]
    }

    pub(crate) fn uniform_i(self) -> [f32; 4] {
        [
            self.configs[0][8],
            self.configs[1][8],
            self.configs[2][8],
            0.0,
        ]
    }

    pub(crate) fn uniform_radiance(self) -> [f32; 4] {
        [self.radiances[0], self.radiances[1], self.radiances[2], 0.0]
    }
}

fn elevation_basis(solar_elevation: f32) -> [f32; 6] {
    let normalized = (solar_elevation / std::f32::consts::FRAC_PI_2).clamp(0.0, 1.0);
    let s = normalized.powf(1.0 / 3.0);
    let one_minus = 1.0 - s;
    [
        one_minus.powi(5),
        5.0 * one_minus.powi(4) * s,
        10.0 * one_minus.powi(3) * s.powi(2),
        10.0 * one_minus.powi(2) * s.powi(3),
        5.0 * one_minus * s.powi(4),
        s.powi(5),
    ]
}

fn mix_config_block(
    dataset: &[f32; 1080],
    offset: usize,
    coeff_index: usize,
    basis: [f32; 6],
) -> f32 {
    let mut value = 0.0;
    for (elevation_index, weight) in basis.into_iter().enumerate() {
        value += weight * dataset[offset + coeff_index + elevation_index * 9];
    }
    value
}

fn mix_radiance_block(dataset: &[f32; 120], offset: usize, basis: [f32; 6]) -> f32 {
    let mut value = 0.0;
    for (elevation_index, weight) in basis.into_iter().enumerate() {
        value += weight * dataset[offset + elevation_index];
    }
    value
}

fn cook_configuration(
    dataset: &[f32; 1080],
    turbidity: f32,
    albedo: f32,
    solar_elevation: f32,
) -> [f32; 9] {
    let turbidity = turbidity.clamp(1.0, 10.0);
    let int_turbidity = turbidity.floor().clamp(1.0, 10.0) as usize;
    let turbidity_rem = if int_turbidity == 10 {
        0.0
    } else {
        turbidity - int_turbidity as f32
    };
    let albedo = albedo.clamp(0.0, 1.0);
    let basis = elevation_basis(solar_elevation);
    let low_turbidity_offset = 9 * 6 * (int_turbidity - 1);
    let high_turbidity_offset = 9 * 6 * int_turbidity;
    let albedo_one_offset = 9 * 6 * 10;
    let mut config = [0.0; 9];

    for (coeff_index, value) in config.iter_mut().enumerate() {
        *value += (1.0 - albedo)
            * (1.0 - turbidity_rem)
            * mix_config_block(dataset, low_turbidity_offset, coeff_index, basis);
        *value += albedo
            * (1.0 - turbidity_rem)
            * mix_config_block(
                dataset,
                albedo_one_offset + low_turbidity_offset,
                coeff_index,
                basis,
            );
        if int_turbidity != 10 {
            *value += (1.0 - albedo)
                * turbidity_rem
                * mix_config_block(dataset, high_turbidity_offset, coeff_index, basis);
            *value += albedo
                * turbidity_rem
                * mix_config_block(
                    dataset,
                    albedo_one_offset + high_turbidity_offset,
                    coeff_index,
                    basis,
                );
        }
    }
    config
}

fn cook_radiance(dataset: &[f32; 120], turbidity: f32, albedo: f32, solar_elevation: f32) -> f32 {
    let turbidity = turbidity.clamp(1.0, 10.0);
    let int_turbidity = turbidity.floor().clamp(1.0, 10.0) as usize;
    let turbidity_rem = if int_turbidity == 10 {
        0.0
    } else {
        turbidity - int_turbidity as f32
    };
    let albedo = albedo.clamp(0.0, 1.0);
    let basis = elevation_basis(solar_elevation);
    let low_turbidity_offset = 6 * (int_turbidity - 1);
    let high_turbidity_offset = 6 * int_turbidity;
    let albedo_one_offset = 6 * 10;

    let mut value = (1.0 - albedo)
        * (1.0 - turbidity_rem)
        * mix_radiance_block(dataset, low_turbidity_offset, basis);
    value += albedo
        * (1.0 - turbidity_rem)
        * mix_radiance_block(dataset, albedo_one_offset + low_turbidity_offset, basis);
    if int_turbidity != 10 {
        value += (1.0 - albedo)
            * turbidity_rem
            * mix_radiance_block(dataset, high_turbidity_offset, basis);
        value += albedo
            * turbidity_rem
            * mix_radiance_block(dataset, albedo_one_offset + high_turbidity_offset, basis);
    }
    value
}

pub(crate) fn hosek_rgb_sky(turbidity: f32, albedo: f32, solar_elevation: f32) -> HosekRgbSky {
    let mut configs = [[0.0; 9]; 3];
    let mut radiances = [0.0; 3];
    for channel in 0..3 {
        configs[channel] =
            cook_configuration(DATASETS_RGB[channel], turbidity, albedo, solar_elevation);
        radiances[channel] = cook_radiance(
            DATASETS_RGB_RAD[channel],
            turbidity,
            albedo,
            solar_elevation,
        );
    }
    HosekRgbSky { configs, radiances }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval_channel(config: [f32; 9], theta: f32, gamma: f32) -> f32 {
        let cos_gamma = gamma.cos();
        let cos_theta = theta.cos();
        let exp_m = (config[4] * gamma).exp();
        let ray_m = cos_gamma * cos_gamma;
        let mie_denom = (1.0 + config[8] * config[8] - 2.0 * config[8] * cos_gamma).max(1.0e-4);
        let mie_m = (1.0 + ray_m) / mie_denom.powf(1.5);
        let zenith = cos_theta.max(0.0).sqrt();
        (1.0 + config[0] * (config[1] / (cos_theta + 0.01)).exp())
            * (config[2]
                + config[3] * exp_m
                + config[5] * ray_m
                + config[6] * mie_m
                + config[7] * zenith)
    }

    #[test]
    fn rgb_radiance_matches_reference_values() {
        let cases = [
            (5.0_f32, [4.7384243, 4.948955, 4.363799]),
            (30.0_f32, [7.7493615, 11.054964, 15.104219]),
            (75.0_f32, [7.530953, 11.732584, 18.974882]),
        ];
        for (elevation_deg, expected) in cases {
            let sky = hosek_rgb_sky(3.0, 0.3, elevation_deg.to_radians());
            for (channel, expected_value) in expected.into_iter().enumerate() {
                assert!(
                    (sky.radiances[channel] - expected_value).abs() < 1.0e-4,
                    "channel {channel} elevation {elevation_deg}"
                );
            }
        }
    }

    #[test]
    fn green_horizon_to_zenith_ratios_match_reference_values() {
        let cases = [
            (5.0_f32, 3.759987),
            (30.0_f32, 4.636012),
            (75.0_f32, 1.903856),
        ];
        for (elevation_deg, expected_ratio) in cases {
            let solar_elevation = elevation_deg.to_radians();
            let sun_zenith = std::f32::consts::FRAC_PI_2 - solar_elevation;
            let sky = hosek_rgb_sky(3.0, 0.3, solar_elevation);
            let zenith = eval_channel(sky.configs[1], 0.0, sun_zenith) * sky.radiances[1];
            let horizon = eval_channel(
                sky.configs[1],
                89.0_f32.to_radians(),
                std::f32::consts::FRAC_PI_2,
            ) * sky.radiances[1];
            assert!(
                (horizon / zenith - expected_ratio).abs() < 1.0e-4,
                "elevation {elevation_deg}"
            );
        }
    }
}
