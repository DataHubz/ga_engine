//! Semantic adapters and canonical use-cases: 3D point rotation via classical matrix vs. GA rotor

/// 3D vector type
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    /// Create a new Vec3
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Vector norm (length)
    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Return a normalized copy of this vector
    pub fn normalized(&self) -> Self {
        let n = self.norm();
        Self::new(self.x / n, self.y / n, self.z / n)
    }

    /// Cross product
    pub fn cross(&self, other: &Vec3) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Multiply by scalar
    pub fn mul_scalar(&self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::Add<&Vec3> for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: &Vec3) -> Vec3 {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

/// Apply a 3×3 matrix (row-major) to a Vec3
pub fn apply_matrix3(m: &[f64; 9], v: &Vec3) -> Vec3 {
    Vec3::new(
        m[0] * v.x + m[1] * v.y + m[2] * v.z,
        m[3] * v.x + m[4] * v.y + m[5] * v.z,
        m[6] * v.x + m[7] * v.y + m[8] * v.z,
    )
}

use crate::ga::geometric_product_full;

/// GA rotor for 3D rotations
pub struct Rotor3 {
    mv: [f64; 8],   // multivector components
    axis: Vec3,     // unit rotation axis
    w: f64,         // scalar part (cos half-angle)
    s: f64,         // sin half-angle
}

impl Rotor3 {
    /// Construct a rotor from an axis (Vec3) and angle (radians).
    pub fn from_axis_angle(axis: &Vec3, angle: f64) -> Self {
        let half = angle * 0.5;
        let w = half.cos();
        let s = half.sin();
        let axis_norm = axis.normalized();
        // Bivector parts (with GA sign convention)
        let b23 = -axis_norm.x * s;
        let b31 = -axis_norm.y * s;
        let b12 = -axis_norm.z * s;
        let mut mv = [0.0; 8];
        mv[0] = w;
        mv[4] = b23;
        mv[5] = b31;
        mv[6] = b12;
        Self { mv, axis: axis_norm, w, s }
    }

    /// Reverse (conjugate) of the rotor: invert bivector parts.
    pub fn conjugate(&self) -> Self {
        let mut mv_conj = self.mv;
        mv_conj[4] = -mv_conj[4];
        mv_conj[5] = -mv_conj[5];
        mv_conj[6] = -mv_conj[6];
        Self {
            mv: mv_conj,
            axis: self.axis,
            w: self.w,
            s: self.s,
        }
    }

    /// Rotate a Vec3 using the GA sandwich product: r * v * r⁻¹.
    pub fn rotate(&self, v: &Vec3) -> Vec3 {
        // Promote v to a pure-vector multivector
        let mut v_mv = [0.0; 8];
        v_mv[1] = v.x;
        v_mv[2] = v.y;
        v_mv[3] = v.z;
        // r * v
        let mut tmp = [0.0; 8];
        geometric_product_full(&self.mv, &v_mv, &mut tmp);
        // (r * v) * r⁻¹
        let rot_conj = self.conjugate();
        let mut res_mv = [0.0; 8];
        geometric_product_full(&tmp, &rot_conj.mv, &mut res_mv);
        Vec3::new(res_mv[1], res_mv[2], res_mv[3])
    }

    /// Fast rotate using quaternion-style formula (~20 flops), fully inlined and unrolled.
    #[inline(always)]
    pub fn rotate_fast(&self, v: &Vec3) -> Vec3 {
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

        // uxt = axis × t
        let ux = ay * tz - az * ty;
        let uy = az * tx - ax * tz;
        let uz = ax * ty - ay * tx;

        let k1 = 2.0 * self.w * self.s;        // scalar
        let k2 = 2.0 * self.s * self.s;

        // chain two mul_adds per component:
        let x = k1.mul_add(tx, vx);            // vx + k1*tx
        let x = k2.mul_add(ux, x);             // (vx + k1*tx) + k2*ux
        let y = k1.mul_add(ty, vy);
        let y = k2.mul_add(uy, y);
        let z = k1.mul_add(tz, vz);
        let z = k2.mul_add(uz, z);

        Vec3::new(x, y, z)
    }
}
