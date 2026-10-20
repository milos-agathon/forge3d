//! One owner for manual/observed lighting and the celestial draw state.
//!
//! The viewer uploads these facts; it does not decide direction conventions,
//! moonlight scaling, observation revisions, or daylight restoration.

use crate::astro::{night, observation::ObservationSnapshot};

pub(crate) struct SkyLightingState {
    light_direction: [f32; 3],
    solar_direction: [f32; 3],
    daylight_intensity: f32,
    directional_scale: f32,
    observation_revision: Option<u64>,
    night_instances: Vec<night::NightInstance>,
    terrain_source: String,
}

impl Default for SkyLightingState {
    fn default() -> Self {
        Self {
            light_direction: [0.3, 0.6, -1.0],
            solar_direction: [0.3, 0.6, -1.0],
            daylight_intensity: 1.0,
            directional_scale: 1.0,
            observation_revision: None,
            night_instances: Vec::new(),
            terrain_source: "manual_angles".into(),
        }
    }
}

impl SkyLightingState {
    pub(crate) fn set_manual_sun(
        &mut self,
        azimuth_deg: f32,
        elevation_deg: f32,
        intensity: Option<f32>,
        source: Option<&str>,
    ) {
        self.light_direction = manual_sun_direction(azimuth_deg, elevation_deg);
        self.solar_direction = self.light_direction;
        if let Some(intensity) = intensity {
            self.set_intensity(intensity);
        }
        self.directional_scale = 1.0;
        self.observation_revision = None;
        self.night_instances.clear();
        self.terrain_source = source.unwrap_or("manual_angles").into();
    }

    /// Publish one complete observation. Re-entering after a manual edit must
    /// rebuild instances even when the supplied observation has the same ID.
    pub(crate) fn observe(&mut self, observation: &ObservationSnapshot) -> bool {
        if self.observation_revision == Some(observation.revision) {
            return false;
        }
        let sun = observation.bodies[0].1;
        let moon = observation.bodies[1].1;
        let solar_direction = night::horizontal_direction(
            // Preserve the viewer's existing f32 solar-angle reduction.
            f64::from(sun.azimuth.value() as f32),
            f64::from(sun.altitude.value() as f32),
        )
        .as_vec3()
        .to_array();
        let (light_direction, directional_scale) = if sun.altitude.value() > 0.0 {
            (solar_direction, 1.0)
        } else {
            (
                night::horizontal_direction(moon.azimuth.value(), moon.altitude.value())
                    .as_vec3()
                    .to_array(),
                night::normalized_moonlight(
                    observation.moon_phase.phase_angle.value(),
                    moon.altitude.value(),
                ),
            )
        };
        let night_instances = night::instances(observation);
        self.solar_direction = solar_direction;
        self.light_direction = light_direction;
        // Observations remain authoritative even after a manual intensity 0.
        self.daylight_intensity = 1.0;
        self.directional_scale = directional_scale;
        self.night_instances = night_instances;
        self.terrain_source = "solar_time".into();
        self.observation_revision = Some(observation.revision);
        true
    }

    pub(crate) fn is_observed(&self) -> bool {
        self.observation_revision.is_some()
    }

    pub(crate) fn light_direction(&self) -> [f32; 3] {
        self.light_direction
    }

    /// Airless, true Sun direction, independent of the active night light.
    pub(crate) fn solar_direction(&self) -> [f32; 3] {
        self.solar_direction
    }

    pub(crate) fn set_intensity(&mut self, intensity: f32) {
        self.daylight_intensity = intensity.max(0.0);
        self.reset_terrain_source();
    }

    /// Match the legacy source label when lighting is resynchronized after
    /// an intensity edit or terrain attachment. Explicit sun commands may
    /// supply a custom label until that next synchronization.
    pub(crate) fn reset_terrain_source(&mut self) {
        self.terrain_source = if self.is_observed() {
            "solar_time"
        } else {
            "manual_angles"
        }
        .into();
    }

    pub(crate) fn effective_intensity(&self) -> f32 {
        if self.light_direction[1] > 0.0 {
            self.daylight_intensity * self.directional_scale
        } else {
            0.0
        }
    }

    pub(crate) fn daylight_intensity(&self) -> f32 {
        self.daylight_intensity
    }

    pub(crate) fn night_instances(&self) -> &[night::NightInstance] {
        &self.night_instances
    }

    pub(crate) fn night_instance_count(&self) -> u32 {
        self.night_instances.len() as u32
    }

    pub(crate) fn terrain_sun(&self) -> (f32, f32, f32, &str) {
        let (azimuth, elevation) = terrain_sun_angles(self.light_direction);
        (
            azimuth,
            elevation,
            self.effective_intensity(),
            &self.terrain_source,
        )
    }

    /// Resolve a partial edit before the terrain command commits any changes.
    /// Use the unscaled daylight value, never a terrain copy of moonlight.
    pub(crate) fn requested_manual_sun(
        &self,
        azimuth: Option<f32>,
        elevation: Option<f32>,
        intensity: Option<f32>,
    ) -> Option<(f32, f32, f32)> {
        if azimuth.is_none() && elevation.is_none() && intensity.is_none() {
            return None;
        }
        let (current_azimuth, current_elevation) = terrain_sun_angles(self.light_direction);
        Some((
            azimuth.unwrap_or(current_azimuth),
            elevation.unwrap_or(current_elevation).clamp(-90.0, 90.0),
            intensity.unwrap_or(self.daylight_intensity).max(0.0),
        ))
    }
}

fn manual_sun_direction(azimuth_deg: f32, elevation_deg: f32) -> [f32; 3] {
    let azimuth = azimuth_deg.to_radians();
    let elevation = elevation_deg.to_radians();
    glam::Vec3::new(
        elevation.cos() * azimuth.sin(),
        elevation.sin(),
        elevation.cos() * azimuth.cos(),
    )
    .normalize()
    .to_array()
}

fn terrain_sun_angles(direction: [f32; 3]) -> (f32, f32) {
    let direction = glam::Vec3::from_array(direction).normalize();
    (
        direction
            .x
            .atan2(direction.z)
            .to_degrees()
            .rem_euclid(360.0),
        direction.y.clamp(-1.0, 1.0).asin().to_degrees(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::{observation::set_observation, time::UtcDateTime, Observer};
    use crate::geo::units::Angle;

    fn observation(hour: u8) -> std::sync::Arc<ObservationSnapshot> {
        set_observation(
            UtcDateTime::new(2026, 7, 26, hour, 0, 0.0).unwrap(),
            Observer::new(Angle::new(52.3676), Angle::new(4.9041), 0.0).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn manual_sun_direction_matches_the_viewer_azimuth_convention() {
        let direction = manual_sun_direction(90.0, 30.0);
        assert!((direction[0] - 30_f32.to_radians().cos()).abs() < 1e-6);
        assert!((direction[1] - 0.5).abs() < 1e-6);
        assert!(direction[2].abs() < 1e-6);
        assert!(manual_sun_direction(0.0, 0.0)[2] > 0.0);
    }

    #[test]
    fn geographic_observation_basis_maps_to_terrain_renderer_azimuth() {
        for (geographic, expected_renderer) in [(0.0, 180.0), (90.0, 90.0), (180.0, 0.0)] {
            let direction = night::horizontal_direction(geographic, 0.0).as_vec3();
            let (azimuth, elevation) = terrain_sun_angles(direction.to_array());
            assert!((azimuth - expected_renderer).abs() < 1.0e-5);
            assert!(elevation.abs() < 1.0e-5);
        }
    }

    #[test]
    fn manual_night_manual_restores_daylight_and_reentering_rebuilds_instances() {
        let night = observation(22);
        assert!(night.bodies[0].1.altitude.value() < 0.0);
        assert!(night.bodies[1].1.altitude.value() > 0.0);
        let mut state = SkyLightingState::default();
        state.set_manual_sun(90.0, 30.0, Some(0.0), None);
        assert!(state.observe(&night));
        assert_ne!(state.solar_direction(), state.light_direction());
        let sun = night.bodies[0].1;
        let moon = night.bodies[1].1;
        assert_eq!(
            state.solar_direction(),
            night::horizontal_direction(
                f64::from(sun.azimuth.value() as f32),
                f64::from(sun.altitude.value() as f32),
            )
            .as_vec3()
            .to_array()
        );
        assert_eq!(
            state.light_direction(),
            night::horizontal_direction(moon.azimuth.value(), moon.altitude.value())
                .as_vec3()
                .to_array()
        );
        assert_eq!(
            state.terrain_sun().2,
            night::normalized_moonlight(
                night.moon_phase.phase_angle.value(),
                moon.altitude.value()
            )
        );
        assert!(state.solar_direction()[1] < 0.0);
        assert!(state.effective_intensity() > 0.0 && state.effective_intensity() < 1.0);
        let instances = bytemuck::cast_slice::<_, u8>(state.night_instances()).to_vec();
        assert!(!instances.is_empty());
        assert!(!state.observe(&night));
        state.set_manual_sun(0.0, 30.0, None, None);
        assert!(!state.is_observed());
        assert_eq!(state.night_instance_count(), 0);
        assert_eq!(state.solar_direction(), state.light_direction());
        assert_eq!(state.effective_intensity(), 1.0);
        assert_eq!(state.terrain_sun().3, "manual_angles");
        assert!(state.observe(&night));
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(state.night_instances()),
            instances
        );
    }

    #[test]
    fn night_day_observations_restore_authoritative_sun_and_intensity() {
        let mut state = SkyLightingState::default();
        assert!(state.observe(&observation(22)));
        state.set_intensity(0.0);
        assert_eq!(state.effective_intensity(), 0.0);
        let day = observation(12);
        assert!(day.bodies[0].1.altitude.value() > 0.0);
        assert!(state.observe(&day));
        assert_eq!(state.light_direction(), state.solar_direction());
        assert_eq!(state.effective_intensity(), 1.0);
        assert_eq!(state.terrain_sun().3, "solar_time");
        state.set_manual_sun(90.0, -30.0, Some(2.0), Some("custom_source"));
        assert_eq!(state.effective_intensity(), 0.0);
        assert_eq!(state.daylight_intensity(), 2.0);
        assert_eq!(state.terrain_sun().3, "custom_source");
        state.set_manual_sun(90.0, 30.0, None, None);
        assert_eq!(state.effective_intensity(), 2.0);
    }

    #[test]
    fn partial_manual_edit_uses_daylight_intensity_after_moonlight() {
        let mut state = SkyLightingState::default();
        state.observe(&observation(22));
        assert_eq!(
            state
                .requested_manual_sun(None, Some(30.0), None)
                .unwrap()
                .2,
            1.0
        );
    }

    #[test]
    fn observed_intensity_edit_preserves_non_unit_daylight_for_partial_sun() {
        let night = observation(22);
        let moon = night.bodies[1].1;
        assert!(night.bodies[0].1.altitude.value() < 0.0);
        assert!(moon.altitude.value() > 0.0);
        let moonlight = night::normalized_moonlight(
            night.moon_phase.phase_angle.value(),
            moon.altitude.value(),
        );
        let mut state = SkyLightingState::default();
        state.observe(&night);
        state.set_intensity(2.0);
        assert!(state.is_observed());
        assert_eq!(state.daylight_intensity(), 2.0);
        assert_eq!(state.terrain_sun().2, 2.0 * moonlight);
        assert_eq!(
            state.requested_manual_sun(None, Some(30.0), None).unwrap().2,
            2.0
        );
    }

    #[test]
    fn legacy_source_labels_follow_intensity_edits_and_terrain_attachment() {
        let mut state = SkyLightingState::default();
        state.set_manual_sun(90.0, 30.0, Some(1.0), Some("custom"));
        assert_eq!(state.terrain_sun().3, "custom");
        state.set_intensity(2.0);
        assert_eq!(state.terrain_sun().3, "manual_angles");
        assert_eq!(state.effective_intensity(), 2.0);
        state.set_manual_sun(90.0, 30.0, None, Some("custom"));
        state.reset_terrain_source();
        assert_eq!(state.terrain_sun().3, "manual_angles");
        state.observe(&observation(22));
        state.set_intensity(2.0);
        assert!(state.is_observed());
        assert_eq!(state.terrain_sun().3, "solar_time");
        state.reset_terrain_source();
        assert_eq!(state.terrain_sun().3, "solar_time");
    }
}
