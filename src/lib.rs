extern crate libc;

mod ffi;

pub struct Cudnn {
}

impl Cudnn {
    pub fn get_version() -> usize {
        return unsafe { ffi::cudnnGetVersion() };
    }
}