//! CUDA Bridge — Safe wrappers for GPU memory transfer
//!
//! This module provides a safe Rust interface to CUDA operations.
//! The actual unsafe FFI calls are isolated to this module.
//!
//! # Safety Contract
//!
//! All `#[repr(C)]` structs in `goldsnnail` are guaranteed to have:
//! - No internal pointers
//! - 4-byte alignment (f32/u32)
//! - Contiguous heap allocation via `Vec<T>`
//!
//! This allows zero-copy transfer via `cudaMemcpy` without pointer patching.

#![allow(unsafe_code)]

use crate::substrate::{SpikeBuffer, StateArena, WeightMatrix, NeuronIdx};
use crate::geometry::Quaternion;

/// CUDA device pointer (opaque handle).
#[derive(Debug, Clone, Copy)]
pub struct DevicePtr<T>(pub *mut T);

impl<T> DevicePtr<T> {
    /// Creates a device pointer from a raw address.
    ///
    /// # Safety
    ///
    /// The caller must ensure the pointer was allocated by `cuda_malloc`
    /// and has not been freed.
    pub unsafe fn from_raw(ptr: *mut T) -> Self {
        Self(ptr)
    }

    /// Returns the raw device pointer.
    pub fn as_raw(&self) -> *mut T {
        self.0
    }
}

/// Safe wrapper around `cudaMalloc`.
///
/// Returns `None` if allocation fails (e.g., out of GPU memory).
pub fn cuda_malloc<T>(count: usize) -> Option<DevicePtr<T>> {
    // Stub: in a real project this would call the CUDA driver API.
    // For now we return a null pointer to document the interface.
    let ptr: *mut T = std::ptr::null_mut();
    if ptr.is_null() {
        None
    } else {
        Some(DevicePtr(ptr))
    }
}

/// Safe wrapper around `cudaMemcpy Host → Device`.
pub fn cuda_memcpy_host_to_device<T>(
    dst: DevicePtr<T>,
    src: &[T],
) -> Result<(), &'static str> {
    if src.is_empty() {
        return Ok(());
    }
    let bytes = src.len() * std::mem::size_of::<T>();
    // Stub: real implementation would call `cudaMemcpy(dst, src, bytes, HtoD)`.
    let _ = (dst, src, bytes);
    Ok(())
}

/// Safe wrapper around `cudaMemcpy Device → Host`.
pub fn cuda_memcpy_device_to_host<T>(
    dst: &mut [T],
    src: DevicePtr<T>,
) -> Result<(), &'static str> {
    if dst.is_empty() {
        return Ok(());
    }
    let bytes = dst.len() * std::mem::size_of::<T>();
    // Stub: real implementation would call `cudaMemcpy(dst, src, bytes, DtoH)`.
    let _ = (dst, src, bytes);
    Ok(())
}

/// Uploads a `WeightMatrix` to GPU memory.
///
/// Returns a device pointer to the weight data, or an error string.
pub fn upload_weights(weights: &WeightMatrix) -> Result<DevicePtr<f32>, &'static str> {
    let count = weights.data.len();
    let dev = cuda_malloc::<f32>(count).ok_or("cudaMalloc failed")?;
    cuda_memcpy_host_to_device(dev, &weights.data)?;
    Ok(dev)
}

/// Downloads spike indices from GPU to host.
pub fn download_spikes(dev: DevicePtr<u32>, count: usize) -> Result<SpikeBuffer, &'static str> {
    let mut host = vec![0u32; count];
    cuda_memcpy_device_to_host(&mut host, dev)?;
    let mut buf = SpikeBuffer::new(count);
    for &idx in &host {
        let _ = buf.push(idx);
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_malloc_returns_none_in_stub() {
        let ptr: Option<DevicePtr<f32>> = cuda_malloc(1024);
        assert!(ptr.is_none(), "Stub cuda_malloc should return None");
    }

    #[test]
    fn cuda_memcpy_stub_succeeds() {
        let data = [1.0f32, 2.0, 3.0];
        let dev = unsafe { DevicePtr::from_raw(std::ptr::null_mut()) };
        assert!(cuda_memcpy_host_to_device(dev, &data).is_ok());
    }

    #[test]
    fn upload_weights_returns_error_in_stub() {
        let wm = WeightMatrix::new(10, 10);
        assert!(upload_weights(&wm).is_err());
    }
}
