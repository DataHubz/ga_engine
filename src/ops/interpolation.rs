//! src/ops/interpolation.rs
//! Slerp-style interpolation between two `Rotor3`

use crate::bivector::Bivector3;
use crate::rotor::Rotor3;

/// Spherical linear interpolation of two rotors `r1` → `r2` by fraction `t` in [0,1].
///
/// Returns a new `Rotor3` at fraction `t` along the shortest path.
pub fn slerp(r1: &Rotor3, r2: &Rotor3, t: f64) -> Rotor3 {
    // Compute cosine of half-angle between the rotors:
    let dot = {
        let b1 = r1.bivector();
        let b2 = r2.bivector();
        r1.scalar() * r2.scalar()
            + b1.xy * b2.xy
            + b1.yz * b2.yz
            + b1.zx * b2.zx
    }
    // Clamp into [-1,1]
    .clamp(-1.0, 1.0);

    // actual half-angle
    let theta = dot.acos();

    if theta.abs() < 1e-8 {
        // nearly the same, just return the first rotor
        return r1.clone();
    }

    let sintheta = theta.sin();
    let a = ((1.0 - t) * theta).sin() / sintheta;
    let b = (t * theta).sin() / sintheta;

    // interpolate scalar and bivector parts
    let mut m = crate::multivector::Multivector3::zero();
    m.scalar = a * r1.scalar() + b * r2.scalar();

    let b1 = r1.bivector();
    let b2 = r2.bivector();
    m.bivector = Bivector3::new(
        a * b1.xy + b * b2.xy,
        a * b1.yz + b * b2.yz,
        a * b1.zx + b * b2.zx,
    );

    // rebuild a Rotor3 from this even multivector
    Rotor3::from_multivector(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::Vec3;

    const EPS: f64 = 1e-12;

    #[test]
    fn slerp_identity_to_90() {
        let r0 = Rotor3::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.0);
        let r1 = Rotor3::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f64::consts::FRAC_PI_2);

        // halfway should be 45°
        let rm = slerp(&r0, &r1, 0.5);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let v_rot = rm.rotate_fast(v);

        let expected = Vec3::new(
            (std::f64::consts::FRAC_PI_4).cos(),
            (std::f64::consts::FRAC_PI_4).sin(),
            0.0,
        );
        assert!((v_rot.x - expected.x).abs() < EPS);
        assert!((v_rot.y - expected.y).abs() < EPS);
        assert!((v_rot.z - expected.z).abs() < EPS);
    }
}
