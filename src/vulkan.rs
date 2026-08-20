//! Vulkan Compute Bridge für AMD Radeon.
//! Direktes Mapping von DOD-Strukturen auf SSBOs.
#![cfg(feature = "vulkan")]

use ash::vk;
use gpu_allocator::vulkan::*;
use gpu_allocator::MemoryLocation;
use std::sync::Arc;

/// Vulkan Compute Kontext — ein Device, eine Queue, ein CommandPool
pub struct VulkanCompute {
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub command_pool: vk::CommandPool,
    pub allocator: Arc<std::sync::Mutex<Allocator>>,
}

/// GPU-Buffer mit CPU-seitigem Staging
pub struct GpuBuffer<T: Copy> {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    pub len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl VulkanCompute {
    /// Erstelle Kontext auf erster Radeon-GPU mit Compute-Queue
    pub fn new() -> Result<Self, String> {
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| format!("Vulkan Entry failed: {}", e))?;

        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::API_VERSION_1_2);

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info);

        let instance = unsafe {
            entry.create_instance(&instance_create_info, None)
                .map_err(|e| format!("Instance creation: {:?}", e))?
        };

        // Finde AMD/Physical Device
        let pdevices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| format!("No physical devices: {:?}", e))?;
        
        let (pdevice, queue_family_index) = pdevices.iter()
            .find_map(|&pd| {
                let props = unsafe { instance.get_physical_device_queue_family_properties(pd) };
                props.iter().enumerate().find_map(|(i, q)| {
                    if q.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                        Some((pd, i as u32))
                    } else {
                        None
                    }
                })
            })
            .ok_or("No compute queue found")?;

        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&[1.0])];

        let device = unsafe {
            instance.create_device(
                pdevice,
                &vk::DeviceCreateInfo::default().queue_create_infos(&queue_info),
                None,
            )
        }.map_err(|e| format!("Device creation: {:?}", e))?;

        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }.map_err(|e| format!("Command pool: {:?}", e))?;

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: pdevice,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        }).map_err(|e| format!("Allocator: {:?}", e))?;

        Ok(Self {
            instance,
            device,
            queue,
            queue_family_index,
            command_pool,
            allocator: Arc::new(std::sync::Mutex::new(allocator)),
        })
    }

    /// Erstelle Storage Buffer aus flachem Slice (DOD-kompatibel)
    pub fn create_buffer<T: Copy>(&self, data: &[T]) -> Result<GpuBuffer<T>, String> {
        let size = std::mem::size_of_val(data) as u64;
        let len = data.len();

        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            self.device.create_buffer(&buffer_info, None)
        }.map_err(|e| format!("Buffer creation: {:?}", e))?;

        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let allocation = self.allocator.lock().unwrap()
            .allocate(&AllocationCreateDesc {
                name: "dod_buffer",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| format!("Allocation: {:?}", e))?;

        unsafe {
            self.device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .map_err(|e| format!("Bind memory: {:?}", e))?;
        }

        // Staging upload
        self.upload_to_buffer(&buffer, data)?;

        Ok(GpuBuffer {
            buffer,
            allocation,
            len,
            _phantom: std::marker::PhantomData,
        })
    }

    fn upload_to_buffer<T: Copy>(&self, dst: &vk::Buffer, data: &[T]) -> Result<(), String> {
        let size = std::mem::size_of_val(data) as u64;
        
        // Staging buffer (CPU-visible)
        let staging_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC);

        let staging = unsafe { self.device.create_buffer(&staging_info, None) }
            .map_err(|e| format!("Staging buffer: {:?}", e))?;
        
        let req = unsafe { self.device.get_buffer_memory_requirements(staging) };
        let mut alloc = self.allocator.lock().unwrap();
        let staging_alloc = alloc.allocate(&AllocationCreateDesc {
            name: "staging",
            requirements: req,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }).map_err(|e| format!("Staging alloc: {:?}", e))?;
        drop(alloc);

        unsafe {
            self.device.bind_buffer_memory(staging, staging_alloc.memory(), staging_alloc.offset())
                .map_err(|e| format!("Staging bind: {:?}", e))?;
        }

        // Map & copy
        let ptr = staging_alloc.mapped_ptr().unwrap().as_ptr() as *mut T;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }

        // Command buffer für copy
        let cmd = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }.map_err(|e| format!("Alloc cmd: {:?}", e))?[0];

        unsafe {
            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            ).unwrap();

            let region = vk::BufferCopy::default().size(size);
            self.device.cmd_copy_buffer(cmd, staging, *dst, &[region]);

            self.device.end_command_buffer(cmd).unwrap();

            let cmd_slice = [cmd];
            let submit = [vk::SubmitInfo::default().command_buffers(&cmd_slice)];
            self.device.queue_submit(self.queue, &submit, vk::Fence::null()).unwrap();
            self.device.queue_wait_idle(self.queue).unwrap();

            self.device.free_command_buffers(self.command_pool, &[cmd]);
            self.device.destroy_buffer(staging, None);
        }

        let mut alloc = self.allocator.lock().unwrap();
        alloc.free(staging_alloc).unwrap();

        Ok(())
    }

    /// Lade Buffer zurück zur CPU
    pub fn download_buffer<T: Copy>(&self, gpu: &GpuBuffer<T>, dst: &mut [T]) -> Result<(), String> {
        assert_eq!(dst.len(), gpu.len);
        let size = std::mem::size_of_val(dst) as u64;

        // Staging für Download
        let staging_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST);

        let staging = unsafe { self.device.create_buffer(&staging_info, None) }
            .map_err(|e| format!("Download staging: {:?}", e))?;
        
        let req = unsafe { self.device.get_buffer_memory_requirements(staging) };
        let mut alloc = self.allocator.lock().unwrap();
        let staging_alloc = alloc.allocate(&AllocationCreateDesc {
            name: "download_staging",
            requirements: req,
            location: MemoryLocation::GpuToCpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }).map_err(|e| format!("Download alloc: {:?}", e))?;
        drop(alloc);

        unsafe {
            self.device.bind_buffer_memory(staging, staging_alloc.memory(), staging_alloc.offset())
                .map_err(|e| format!("Download bind: {:?}", e))?;
        }

        let cmd = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }.map_err(|e| format!("Download cmd: {:?}", e))?[0];

        unsafe {
            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            ).unwrap();

            let region = vk::BufferCopy::default().size(size);
            self.device.cmd_copy_buffer(cmd, gpu.buffer, staging, &[region]);

            self.device.end_command_buffer(cmd).unwrap();

            let cmd_slice = [cmd];
            let submit = [vk::SubmitInfo::default().command_buffers(&cmd_slice)];
            self.device.queue_submit(self.queue, &submit, vk::Fence::null()).unwrap();
            self.device.queue_wait_idle(self.queue).unwrap();

            self.device.free_command_buffers(self.command_pool, &[cmd]);
        }

        let ptr = staging_alloc.mapped_ptr().unwrap().as_ptr() as *const T;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), dst.len());
        }

        unsafe { self.device.destroy_buffer(staging, None); }
        let mut alloc = self.allocator.lock().unwrap();
        alloc.free(staging_alloc).unwrap();

        Ok(())
    }

    /// Erstelle Compute Pipeline aus SPIR-V Bytes
    pub fn create_compute_pipeline(&self, spirv: &[u32]) -> Result<(vk::Pipeline, vk::PipelineLayout, vk::DescriptorSetLayout), String> {
        let shader_module = unsafe {
            self.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(spirv),
                None,
            )
        }.map_err(|e| format!("Shader module: {:?}", e))?;

        let entry = std::ffi::CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry);

        let dsl_bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];

        let dsl = unsafe {
            self.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&dsl_bindings),
                None,
            )
        }.map_err(|e| format!("DSL: {:?}", e))?;

        let pipeline_layout = unsafe {
            self.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[dsl]),
                None,
            )
        }.map_err(|e| format!("Pipeline layout: {:?}", e))?;

        let create_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            self.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[create_info],
                None,
            )
        }.map_err(|e| format!("Pipeline: {:?}", e))?[0];

        unsafe { self.device.destroy_shader_module(shader_module, None); }

        Ok((pipeline, pipeline_layout, dsl))
    }

    /// Dispatch Compute Shader
    pub fn dispatch(
        &self,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
        descriptor_set: vk::DescriptorSet,
        group_count_x: u32,
    ) -> Result<(), String> {
        let cmd = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }.map_err(|e| format!("Dispatch cmd: {:?}", e))?[0];

        unsafe {
            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            ).unwrap();

            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            self.device.cmd_dispatch(cmd, group_count_x, 1, 1);

            // Memory barrier für SSBO
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );

            self.device.end_command_buffer(cmd).unwrap();

            let cmd_slice = [cmd];
            let submit = [vk::SubmitInfo::default().command_buffers(&cmd_slice)];
            self.device.queue_submit(self.queue, &submit, vk::Fence::null()).unwrap();
            self.device.queue_wait_idle(self.queue).unwrap();

            self.device.free_command_buffers(self.command_pool, &[cmd]);
        }

        Ok(())
    }
}

impl Drop for VulkanCompute {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
