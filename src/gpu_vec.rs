use mem_align::MemAlign;
use wgpu::*;

pub struct GPUVec<T: Copy> {
    buffer: wgpu::Buffer,
    mem_align: MemAlign<T>,
}

impl<T> GPUVec<T>
where
    T: Copy,
{
    fn new(device: &wgpu::Device, capacity: usize, label: &str) -> Self {
        let mem_align = MemAlign::new(capacity);

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: mem_align.byte_size() as _,
            usage: BufferUsages::MAP_WRITE,
            mapped_at_creation: true,
        });

        Self { buffer, mem_align }
    }

    fn capacity(&self) -> usize {
        self.mem_align.capacity()
    }
}

impl<T: Copy> std::ops::Index<usize> for GPUVec<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        let view = self.buffer.slice(..).get_mapped_range();
        let slice =
            unsafe { std::slice::from_raw_parts(view.as_ptr() as *const T, self.capacity()) };
        &slice[index]
    }
}