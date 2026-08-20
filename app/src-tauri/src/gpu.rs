use std::process::Command;

pub fn detect_gpu() -> Result<String, String> {
    #[cfg(target_os = "windows")]
    {
        let vulkan_dlls = [
            r"C:\Windows\System32\vulkan-1.dll",
            r"C:\Windows\SysWOW64\vulkan-1.dll",
        ];
        for dll in &vulkan_dlls {
            if std::path::Path::new(dll).exists() {
                return Ok("Vulkan detected (vulkan-1.dll found)".to_string());
            }
        }

        if Command::new("nvidia-smi").output().is_ok() {
            return Ok("CUDA detected (nvidia-smi found)".to_string());
        }

        let cuda_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        ];
        for path in &cuda_paths {
            if std::path::Path::new(path).exists() {
                return Ok("CUDA detected (toolkit found)".to_string());
            }
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        let vulkan_paths = [
            "/usr/lib/libvulkan.so.1",
            "/usr/local/lib/libvulkan.so.1",
        ];
        for path in &vulkan_paths {
            if std::path::Path::new(path).exists() {
                return Ok("Vulkan detected".to_string());
            }
        }

        if Command::new("nvidia-smi").output().is_ok() {
            return Ok("CUDA detected (nvidia-smi found)".to_string());
        }
    }

    Ok("CPU only (no GPU backend detected)".to_string())
}
