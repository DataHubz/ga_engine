fn main() {
    // Tell the linker to link against the system Accelerate framework
    println!("cargo:rustc-link-lib=framework=Accelerate");
}
