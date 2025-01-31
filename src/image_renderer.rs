use std::{fmt, sync::Arc};
use wgpu::util::DeviceExt;

use crate::*;

pub(crate) struct RendererInactive;

pub enum RenderImageError {
    NoImages,
    FailedToRetrieveLastMut,
    ImageNotFound { index: usize },
    IndexOutOfBounds { index: usize, total_images: usize },
    RendererInactive,
}

impl fmt::Debug for RenderImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RenderImageError::NoImages => {
                write!(f, "No images are currently stored in the renderer. This should never happen and the fact that it did is extremely concerning.")
            }
            RenderImageError::FailedToRetrieveLastMut => {
                write!(f, "Failed to retrieve mutable reference to last image")
            }
            RenderImageError::ImageNotFound { index } => {
                write!(
                    f,
                    "Image at index {} could not be found in the stored images",
                    index
                )
            }
            RenderImageError::IndexOutOfBounds {
                index,
                total_images,
            } => {
                write!(
                    f,
                    "Image index {} is out of bounds (total images: {})",
                    index, total_images
                )
            }
            RenderImageError::RendererInactive => {
                write!(f, "Cannot render image - renderer is currently inactive")
            }
        }
    }
}

#[derive(Copy, Clone)]
pub struct ImageIndex {
    pub index: usize,
}

#[derive(Copy, Clone, PartialEq)]
pub enum AxisAlignEnum {
    Start,
    Center,
    End,
}

impl AxisAlign for AxisAlignEnum {
    fn value(&self) -> f32 {
        match self {
            AxisAlignEnum::Start => 1.0,
            AxisAlignEnum::Center => 0.0,
            AxisAlignEnum::End => -1.0,
        }
    }
}

impl AxisAlign for f32 {
    fn value(&self) -> f32 {
        *self
    }
}

pub trait AxisAlign {
    fn value(&self) -> f32;
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Fit {
    Fill,
    Contain,
    Cover,
}

// Updated RenderImage struct
pub struct RenderImage {
    offset_pixels: LocalVector,
    size_pixels: LocalSize,
    fit: Option<Fit>,
    horizontal_align: Box<dyn AxisAlign>,
    vertical_align: Box<dyn AxisAlign>,
    index: ImageIndex,
    image_uniform_buffer: wgpu::Buffer,
    aspect_bind_group: wgpu::BindGroup,
    clip_overflow: bool,
}

impl RenderImage {
    pub fn scale(&mut self, scale: f32) -> &mut Self {
        self.size_pixels = self.size_pixels * scale;
        self
    }

    pub fn width(&mut self, width: f32) -> &mut Self {
        self.size_pixels.width = width;
        self
    }

    pub fn height(&mut self, height: f32) -> &mut Self {
        self.size_pixels.height = height;
        self
    }

    pub fn size(&mut self, size: [f32; 2]) -> &mut Self {
        self.size_pixels = LocalSize::new(size[0], size[1]);
        self
    }

    pub fn offset(&mut self, offset: [f32; 2]) -> &mut Self {
        self.offset_pixels = LocalVector::new(offset[0], offset[1]);
        self
    }

    pub fn fit(&mut self, fit: Fit) -> &mut Self {
        self.fit = Some(fit);
        self
    }

    pub fn h_align<T: AxisAlign + 'static>(&mut self, align: T) -> &mut Self {
        self.horizontal_align = Box::new(align);
        self
    }

    pub fn v_align<T: AxisAlign + 'static>(&mut self, align: T) -> &mut Self {
        self.vertical_align = Box::new(align);
        self
    }

    pub fn overflow_hidden(&mut self) -> &mut Self {
        self.clip_overflow = true;
        self
    }
}

pub(crate) struct ImageRenderer {
    stored_images: Vec<StoredImage>,
    render_images: Vec<RenderImage>,
    bind_group_layout: wgpu::BindGroupLayout,
    aspect_bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    queue: Arc<wgpu::Queue>,
    sampler: wgpu::Sampler,
    device: Arc<wgpu::Device>,
    window_size: ScreenSize,
    device_px_ratio: f32,
}

impl ImageRenderer {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        texture_format: wgpu::TextureFormat,
    ) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Vertex data for quad
        let vertices = [
            Vertex {
                position: [-1.0, -1.0],
                tex_coords: [0.0, 1.0],
            }, // Bottom-left
            Vertex {
                position: [1.0, -1.0],
                tex_coords: [1.0, 1.0],
            }, // Bottom-right
            Vertex {
                position: [1.0, 1.0],
                tex_coords: [1.0, 0.0],
            }, // Top-right
            Vertex {
                position: [-1.0, 1.0],
                tex_coords: [0.0, 0.0],
            }, // Top-left
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Index data
        let indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Create bind group layout for aspect uniform
        let aspect_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Aspect Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let shader_source = r#"
            struct VertexInput {
                @location(0) position: vec2<f32>,
                @location(1) tex_coords: vec2<f32>,
            };

            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
                @location(0) tex_coords: vec2<f32>,
            };

            struct ImageUniform {
                offset: vec2<f32>,
                scale: vec2<f32>,
            };

            @group(1) @binding(0) var<uniform> uniform: ImageUniform;

            @vertex
            fn vs_main(input: VertexInput) -> VertexOutput {
                var output: VertexOutput;
                output.position = vec4(input.position * uniform.scale + uniform.offset, 0.0, 1.0);
                output.tex_coords = input.tex_coords;
                return output;
            }

            @group(0) @binding(0) var texture: texture_2d<f32>;
            @group(0) @binding(1) var texture_sampler: sampler;

            @fragment
            fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
                // let sample = textureSample(texture, texture_sampler, input.tex_coords);
                // if (sample.a == 0.0) {
                //     return vec4<f32>(0.3, 0.0, 0.0, 1.0);
                // } else {
                //     return sample; 
                // }

                return textureSample(texture, texture_sampler, input.tex_coords);
            }
        "#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        };

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout, &aspect_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[vertex_buffer_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: texture_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        ImageRenderer {
            stored_images: Vec::new(),
            render_images: Vec::new(),
            bind_group_layout,
            pipeline: render_pipeline,
            vertex_buffer,
            index_buffer,
            queue,
            sampler,
            aspect_bind_group_layout,
            device,
            window_size: ScreenSize::zero(),
            device_px_ratio: 1.0,
        }
    }

    pub fn encode(&self, rpass: &mut wgpu::RenderPass) {
        // Set pipeline and buffers
        rpass.set_pipeline(&self.pipeline);
        rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        rpass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

        let window_size = ScreenSize::new(
            self.window_size.width * self.device_px_ratio,
            self.window_size.height * self.device_px_ratio,
        );

        // Update uniforms (if needed, ensure these are done before the render pass)
        self.render_images.iter().for_each(|image| {
            let stored_image = &self.stored_images[image.index.index];
            // Calculate aspect_uniform and write to buffer
            let container_width = image.size_pixels.width * self.device_px_ratio;
            let container_height = image.size_pixels.height * self.device_px_ratio;
            let offset_x = image.offset_pixels.x * self.device_px_ratio;
            let offset_y = image.offset_pixels.y * self.device_px_ratio;

            let image_width = stored_image.width as f32;
            let image_height = stored_image.height as f32;
            let fit = image.fit.unwrap_or(Fit::Fill);

            let (image_scale, image_offset) = Self::calculate_fit(
                fit,
                image_width,
                image_height,
                container_width,
                container_height,
                &*image.horizontal_align,
                &*image.vertical_align,
            );

            let container_scale_x = container_width / window_size.width;
            let container_scale_y = container_height / window_size.height;

            let container_center_x =
                (offset_x + container_width / 2.0) / window_size.width * 2.0 - 1.0;
            let container_center_y =
                1.0 - (offset_y + container_height / 2.0) / window_size.height * 2.0;

            // Calculate the quad's scale in NDC
            let quad_scale_x = container_scale_x * image_scale[0];
            let quad_scale_y = container_scale_y * image_scale[1];

            // Calculate the image's offset within the container in NDC
            let image_offset_x_ndc = image_offset[0] * container_scale_x * 2.0;
            let image_offset_y_ndc = image_offset[1] * container_scale_y * 2.0;

            // Calculate the quad's offset in NDC
            let quad_offset_x =
                (container_center_x - container_scale_x) + image_offset_x_ndc + quad_scale_x;
            let quad_offset_y =
                (container_center_y + container_scale_y) - image_offset_y_ndc - quad_scale_y;

            let aspect_uniform = ImageUniform {
                offset: [quad_offset_x, quad_offset_y],
                scale: [quad_scale_x, quad_scale_y],
            };

            self.queue.write_buffer(
                &image.image_uniform_buffer,
                0,
                bytemuck::cast_slice(&[aspect_uniform]),
            );

            let stored_image = &self.stored_images[image.index.index];
            let mut skip = false;

            if image.clip_overflow {
                // Calculate clamped scissor rect
                let x_start = offset_x.max(0.0);
                let y_start = offset_y.max(0.0);
                let x_end = (offset_x + container_width).min(window_size.width);
                let y_end = (offset_y + container_height).min(window_size.height);

                let scissor_x = x_start as u32;
                let scissor_y = y_start as u32;
                let scissor_width = (x_end - x_start) as u32;
                let scissor_height = (y_end - y_start) as u32;

                if scissor_width == 0 || scissor_height == 0 {
                    skip = true;
                } else {
                    rpass.set_scissor_rect(scissor_x, scissor_y, scissor_width, scissor_height);
                }
            } else {
                // Reset scissor to the entire window when overflow is visible
                rpass.set_scissor_rect(0, 0, window_size.width as u32, window_size.height as u32);
            }

            if !skip {
                rpass.set_bind_group(0, &stored_image.bind_group, &[]);
                rpass.set_bind_group(1, &image.aspect_bind_group, &[]);
                rpass.draw_indexed(0..6, 0, 0..1);
            }
        });
    }

    fn calculate_fit(
        fit: Fit,
        image_width: f32,
        image_height: f32,
        container_width: f32,
        container_height: f32,
        horizontal_align: &dyn AxisAlign,
        vertical_align: &dyn AxisAlign,
    ) -> ([f32; 2], [f32; 2]) {
        let (scaled_width, scaled_height, offset_x, offset_y) = match fit {
            Fit::Fill => (container_width, container_height, 0.0, 0.0),
            Fit::Contain => {
                let scale_x = container_width / image_width;
                let scale_y = container_height / image_height;
                let scale = f32::min(scale_x, scale_y);
                let scaled_width = image_width * scale;
                let scaled_height = image_height * scale;

                // Use alignment values to compute offsets
                let offset_x =
                    (container_width - scaled_width) * (1.0 - horizontal_align.value()) / 2.0;
                let offset_y =
                    (container_height - scaled_height) * (1.0 - vertical_align.value()) / 2.0;

                (scaled_width, scaled_height, offset_x, offset_y)
            }
            Fit::Cover => {
                let scale_x = container_width / image_width;
                let scale_y = container_height / image_height;
                let scale = f32::max(scale_x, scale_y);
                let scaled_width = image_width * scale;
                let scaled_height = image_height * scale;

                // Use alignment values to compute offsets
                let offset_x =
                    (container_width - scaled_width) * (1.0 - horizontal_align.value()) / 2.0;
                let offset_y =
                    (container_height - scaled_height) * (1.0 - vertical_align.value()) / 2.0;

                (scaled_width, scaled_height, offset_x, offset_y)
            }
        };

        let image_scale_x = scaled_width / container_width;
        let image_scale_y = scaled_height / container_height;

        let offset_x_normalized = offset_x / container_width;
        let offset_y_normalized = offset_y / container_height;

        (
            [image_scale_x, image_scale_y],
            [offset_x_normalized, offset_y_normalized],
        )
    }

    pub fn store_image(&mut self, path: &str) -> ImageIndex {
        // Load image
        let img = image::open(path).expect("Failed to load image").to_rgba8();
        let (width, height) = img.dimensions();
        let rgba = img.into_raw();

        // Create texture
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let image_index = self.stored_images.len();

        // Create texture/sampler bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        // Add to images vector
        self.stored_images.push(StoredImage {
            width,
            height,
            bind_group,
        });

        ImageIndex { index: image_index }
    }

    pub fn begin(&mut self, window_width: f32, window_height: f32, device_px_ratio: f32) {
        self.render_images.clear();
        self.window_size = ScreenSize::new(window_width, window_height);
        self.device_px_ratio = device_px_ratio;
    }

    pub fn image(&mut self, index: ImageIndex) -> Result<&mut RenderImage, RenderImageError> {
        // Check if any images are stored
        if self.stored_images.is_empty() {
            return Err(RenderImageError::NoImages);
        }

        // Check if index is valid
        if index.index >= self.stored_images.len() {
            return Err(RenderImageError::IndexOutOfBounds {
                index: index.index,
                total_images: self.stored_images.len(),
            });
        }

        let stored_image = match self.stored_images.get(index.index) {
            Some(image) => image,
            None => return Err(RenderImageError::ImageNotFound { index: index.index }),
        };

        // Create uniform buffer for image transform
        let image_uniform_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Image Uniform Buffer"),
                    contents: bytemuck::cast_slice(&[ImageUniform {
                        offset: [0.0, 0.0],
                        scale: [1.0, 1.0],
                    }]),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        // Create bind group for aspect ratio uniforms
        let aspect_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Aspect Bind Group"),
            layout: &self.aspect_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &image_uniform_buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        });

        // Create new render image with default settings
        let render_image = RenderImage {
            offset_pixels: LocalVector::zero(),
            size_pixels: LocalSize::new(
                (stored_image.width as f32) / self.device_px_ratio,
                (stored_image.height as f32) / self.device_px_ratio,
            ),
            fit: Some(Fit::Fill),
            horizontal_align: Box::new(AxisAlignEnum::Center),
            vertical_align: Box::new(AxisAlignEnum::Center),
            index,
            image_uniform_buffer,
            aspect_bind_group,
            clip_overflow: false,
        };

        // Add the render image to the vector
        self.render_images.push(render_image);

        // Return mutable reference to the newly added render image
        match self.render_images.last_mut() {
            Some(image) => Ok(image),
            None => Err(RenderImageError::FailedToRetrieveLastMut), // This should never happen but handle it anyway
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ImageUniform {
    offset: [f32; 2],
    scale: [f32; 2],
}

struct StoredImage {
    width: u32,
    height: u32,
    bind_group: wgpu::BindGroup,
}
