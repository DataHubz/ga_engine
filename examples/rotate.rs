// examples/rotate.rs
use ga_engine::prelude::*;

fn main() {
    // original point
    let p = Vec3::new(1.0, 0.0, 0.0);
    // classical 90° about Z
    let m = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let p1 = apply_matrix3(&m, p);

    // GA rotor for +90° about Z
    let r = Rotor3::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f64::consts::FRAC_PI_2);
    let p2 = r.rotate(p);

    // allow a tiny epsilon
    const EPS: f64 = 1e-12;
    assert!(
        (p1.x - p2.x).abs() < EPS,
        "x mismatch: {} vs {}",
        p1.x,
        p2.x
    );
    assert!(
        (p1.y - p2.y).abs() < EPS,
        "y mismatch: {} vs {}",
        p1.y,
        p2.y
    );
    assert!(
        (p1.z - p2.z).abs() < EPS,
        "z mismatch: {} vs {}",
        p1.z,
        p2.z
    );

    // pretty‐print p2 rounded to 6 decimal places
    println!("✔ p1 ≈ p2 = {}", Rounded::new(&p2, 6));
}
