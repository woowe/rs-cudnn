extern crate rs_cudnn;

use rs_cudnn::Cudnn;

fn main() {
    let v = unsafe { Cudnn::get_version() };
    println!("CUDNN VERSION {}", v);
}