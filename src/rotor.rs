// src/rotor.rs

use crate::{
  vector::Vec3,
  bivector::Bivector3,
  multivector::Multivector3,
};
use wide::f64x4;

/// A 3-D rotor (unit even multivector) for rotations.
#[derive(Clone, Debug, PartialEq)]
pub struct Rotor3 {
  inner: Multivector3,
  axis:  Vec3,
  w:     f64,
  s:     f64,
}

impl Rotor3 {
  /// Build a rotor from `axis` and `angle` (radians).
  pub fn from_axis_angle(axis: Vec3, angle: f64) -> Self {
      let half = angle * 0.5;
      let w = half.cos();
      let s = half.sin();
      // normalize axis
      let axis_norm = axis.scale(1.0 / axis.norm());

      // build the full multivector
      let mut m = Multivector3::zero();
      m.scalar = w;
      m.bivector = Bivector3::new(
          -axis_norm.x * s,
          -axis_norm.y * s,
          -axis_norm.z * s,
      );

      Rotor3 { inner: m, axis: axis_norm, w, s }
  }

  /// Rotate a vector via sandwich product: r * v * r⁻¹.
  pub fn rotate(&self, v: Vec3) -> Vec3 {
      let mv = Multivector3::from_vector(v);
      let r = &self.inner;
      let r_inv = r.reverse();
      r.gp(&mv).gp(&r_inv).vector
  }

  /// Fast quaternion‐style rotate (~20 flops), fully inlined.
  #[inline(always)]
  pub fn rotate_fast(&self, v: Vec3) -> Vec3 {
      let ax = self.axis.x;
      let ay = self.axis.y;
      let az = self.axis.z;
      let vx = v.x;
      let vy = v.y;
      let vz = v.z;

      // t = axis × v
      let tx = ay * vz - az * vy;
      let ty = az * vx - ax * vz;
      let tz = ax * vy - ay * vx;

      // u = axis × t
      let ux = ay * tz - az * ty;
      let uy = az * tx - ax * tz;
      let uz = ax * ty - ay * tx;

      let k1 = 2.0 * self.w * self.s;
      let k2 = 2.0 * self.s * self.s;

      // chain two FMA per component
      let x = k2.mul_add(ux, k1.mul_add(tx, vx));
      let y = k2.mul_add(uy, k1.mul_add(ty, vy));
      let z = k2.mul_add(uz, k1.mul_add(tz, vz));

      Vec3::new(x, y, z)
  }

  /// SIMD‐4× rotate using `wide::f64x4`.
  #[inline(always)]
  pub fn rotate_simd(&self, vs: [Vec3; 4]) -> [Vec3; 4] {
      let ax = f64x4::splat(self.axis.x);
      let ay = f64x4::splat(self.axis.y);
      let az = f64x4::splat(self.axis.z);

      let vx = f64x4::from([vs[0].x, vs[1].x, vs[2].x, vs[3].x]);
      let vy = f64x4::from([vs[0].y, vs[1].y, vs[2].y, vs[3].y]);
      let vz = f64x4::from([vs[0].z, vs[1].z, vs[2].z, vs[3].z]);

      let tx = ay * vz - az * vy;
      let ty = az * vx - ax * vz;
      let tz = ax * vy - ay * vx;

      let ux = ay * tz - az * ty;
      let uy = az * tx - ax * tz;
      let uz = ax * ty - ay * tx;

      let k1 = f64x4::splat(2.0 * self.w * self.s);
      let k2 = f64x4::splat(2.0 * self.s * self.s);

      let rx = k2.mul_add(ux, k1.mul_add(tx, vx));
      let ry = k2.mul_add(uy, k1.mul_add(ty, vy));
      let rz = k2.mul_add(uz, k1.mul_add(tz, vz));

      let ax = rx.to_array();
      let ay = ry.to_array();
      let az = rz.to_array();

      [
          Vec3::new(ax[0], ay[0], az[0]),
          Vec3::new(ax[1], ay[1], az[1]),
          Vec3::new(ax[2], ay[2], az[2]),
          Vec3::new(ax[3], ay[3], az[3]),
      ]
  }

  /// SIMD‐8× rotate by running two 4× lanes.
  #[inline(always)]
  pub fn rotate_simd8(&self, vs: [Vec3; 8]) -> [Vec3; 8] {
      // first half
      let r0 = self.rotate_simd([vs[0], vs[1], vs[2], vs[3]]);
      // second half
      let r1 = self.rotate_simd([vs[4], vs[5], vs[6], vs[7]]);
      // combine
      [r0[0], r0[1], r0[2], r0[3],
       r1[0], r1[1], r1[2], r1[3]]
  }
}
