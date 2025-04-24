use ga_engine::prelude::*;

fn main() {
    // Make a pure bivector B = e23 * 0.1  (i.e. a rotation by 0.2 rad about x)
    let biv = Bivector3::new(0.1, 0.0, 0.0);

    // Build the rotor R = exp(B)
    let r = Rotor3::from_bivector(biv);

    // Apply to a Vec3
    let v = Vec3::new(1.0, 0.0, 0.0);
    let v_rot = r.rotate_fast(v);
    println!("rotated v = {:?}", v_rot);

    // Project v onto the Y axis
    let proj = v.project_onto(&Vec3::new(0.0, 1.0, 0.0));
    println!("projection of v onto Y = {:?}", proj);
}
