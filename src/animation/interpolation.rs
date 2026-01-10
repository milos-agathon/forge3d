//! Cubic Hermite (Catmull-Rom) interpolation for smooth camera paths
//!
//! Provides smooth interpolation between keyframes with automatic tangent computation.

/// Cubic Hermite interpolation (Catmull-Rom spline)
///
/// Given 4 control points (p0, p1, p2, p3) and parameter t in [0, 1],
/// interpolates between p1 and p2 with tangents derived from neighbors.
///
/// # Arguments
/// * `p0` - Control point before segment start
/// * `p1` - Segment start value
/// * `p2` - Segment end value  
/// * `p3` - Control point after segment end
/// * `t` - Interpolation parameter [0, 1]
pub fn cubic_hermite(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;

    // Catmull-Rom basis functions
    let h1 = -0.5 * t3 + t2 - 0.5 * t;
    let h2 = 1.5 * t3 - 2.5 * t2 + 1.0;
    let h3 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
    let h4 = 0.5 * t3 - 0.5 * t2;

    h1 * p0 + h2 * p1 + h3 * p2 + h4 * p3
}

/// Linear interpolation between two values
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Smooth step interpolation (ease-in-out)
pub fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Smoother step interpolation (even smoother ease-in-out)
pub fn smootherstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cubic_hermite_endpoints() {
        // At t=0, should return p1
        let result = cubic_hermite(0.0, 1.0, 2.0, 3.0, 0.0);
        assert!((result - 1.0).abs() < 1e-6);

        // At t=1, should return p2
        let result = cubic_hermite(0.0, 1.0, 2.0, 3.0, 1.0);
        assert!((result - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_cubic_hermite_midpoint() {
        // For linear sequence, midpoint should be ~1.5
        let result = cubic_hermite(0.0, 1.0, 2.0, 3.0, 0.5);
        assert!((result - 1.5).abs() < 0.1);
    }

    #[test]
    fn test_lerp() {
        assert!((lerp(0.0, 10.0, 0.0) - 0.0).abs() < 1e-6);
        assert!((lerp(0.0, 10.0, 1.0) - 10.0).abs() < 1e-6);
        assert!((lerp(0.0, 10.0, 0.5) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_smoothstep() {
        assert!((smoothstep(0.0) - 0.0).abs() < 1e-6);
        assert!((smoothstep(1.0) - 1.0).abs() < 1e-6);
        assert!((smoothstep(0.5) - 0.5).abs() < 1e-6);
    }
}
