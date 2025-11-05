/// Medical Imaging Examples for Clifford FHE
///
/// This module demonstrates privacy-preserving 3D medical imaging classification
/// using batched Clifford FHE on GPU backends (Metal + CUDA).

pub mod point_cloud;
pub mod clifford_encoding;
pub mod synthetic_data;

// Re-export commonly used types
pub use point_cloud::{Point3D, PointCloud};
pub use clifford_encoding::{Multivector3D, encode_point_cloud, encode_batch};
pub use synthetic_data::{ShapeType, generate_dataset, train_test_split};
