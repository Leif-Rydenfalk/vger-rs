use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

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

#[derive(Copy, Clone)]
pub struct ImageIndex {
    index: usize,
}

pub struct RenderImage {
    offset: [f32; 2],
    aspect_ratio_scale: [f32; 2], // scale to remove stretching from different aspect ratios
    size: [f32; 2],               // the size of the image in pixels
    index: ImageIndex,
    image_uniform_buffer: wgpu::Buffer,
    aspect_bind_group: wgpu::BindGroup,
}

impl RenderImage {
    pub fn scale(&mut self, scale: f32) -> &mut Self {
        self.size = [self.size[0] * scale, self.size[1] * scale];
        self
    }

    pub fn size(&mut self, size: [f32; 2]) -> &mut Self {
        self.size = size;
        self
    }

    pub fn offset(&mut self, offset: [f32; 2]) -> &mut Self {
        self.offset = offset;
        self
    }
}

struct ImageRenderer {
    stored_images: Vec<StoredImage>,
    render_images: Vec<RenderImage>,
    bind_group_layout: wgpu::BindGroupLayout, // For texture and sampler
    aspect_bind_group_layout: wgpu::BindGroupLayout, // For scaling uniform
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    queue: Arc<wgpu::Queue>,
    sampler: wgpu::Sampler,
    device: Arc<wgpu::Device>,
    window_size: winit::dpi::PhysicalSize<u32>,
}

impl ImageRenderer {
    fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: &wgpu::SurfaceConfiguration,
        window_size: winit::dpi::PhysicalSize<u32>,
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

        // Update shader source to include aspect ratio scaling
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
                    format: config.format,
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
            window_size,
        }
    }

    // In the window_resized method, use the correct offset and scale:
    fn window_resized(&mut self, window_size: winit::dpi::PhysicalSize<u32>) {
        self.window_size = window_size;
        for image in &mut self.render_images {
            let stored_image = &self.stored_images[image.index.index];
            let (scale_x, scale_y) = Self::get_image_scale(
                window_size.width as f32,
                window_size.height as f32,
                stored_image.width as f32,
                stored_image.height as f32,
                None,
                None,
            );
            image.aspect_ratio_scale = [scale_x, scale_y]; // Update the scale in RenderImage
        }
    }

    fn get_image_scale(
        window_width: f32,
        window_height: f32,
        image_width: f32,
        image_height: f32,
        max_rendered_width: Option<f32>,  // In pixels
        max_rendered_height: Option<f32>, // In pixels
    ) -> (f32, f32) {
        let window_aspect_ratio = window_width / window_height;
        let image_aspect_ratio = image_width / image_height;

        // let scale_x = window_aspect_ratio / image_aspect_ratio;
        // let scale_y = image_aspect_ratio / window_aspect_ratio;
        // (scale_x, scale_y)

        let mut scale_x = 1.0;
        let mut scale_y = 1.0;

        if window_aspect_ratio > image_aspect_ratio {
            scale_x = image_aspect_ratio / window_aspect_ratio;
        } else {
            scale_y = window_aspect_ratio / image_aspect_ratio;
        }

        if let Some(max_width) = max_rendered_width {
            scale_x = scale_x.min(max_width / image_width);
        }

        if let Some(max_height) = max_rendered_height {
            scale_y = scale_y.min(max_height / image_height);
        }

        (scale_x, scale_y)
    }

    fn render(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        // Update all uniforms on the GPU
        for image in &self.render_images {
            let scale = [
                image.aspect_ratio_scale[0] * image.size[0],
                image.aspect_ratio_scale[1] * image.size[1],
            ];
            let aspect_uniform = ImageUniform {
                offset: image.offset,
                scale,
            };
            self.queue.write_buffer(
                &image.image_uniform_buffer,
                0,
                bytemuck::cast_slice(&[aspect_uniform]),
            );
        }

        // Render all the images in the render_images vector
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

        for image in &self.render_images {
            let stored_image = &self.stored_images[image.index.index];
            render_pass.set_bind_group(0, &stored_image.bind_group, &[]);
            render_pass.set_bind_group(1, &image.aspect_bind_group, &[]);
            render_pass.draw_indexed(0..6, 0, 0..1);
        }
    }

    fn store_image(&mut self, path: &str) -> ImageIndex {
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

    fn begin_frame(&mut self) {
        self.render_images.clear();
    }

    fn image(&mut self, index: ImageIndex) -> &mut RenderImage {
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

        let aspect_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Aspect Bind Group"),
            layout: &self.aspect_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &image_uniform_buffer, // Use the created uniform buffer
                    offset: 0,
                    size: None,
                }),
            }],
        });

        let stored_image = &self.stored_images[index.index];
        let (scale_x, scale_y) = Self::get_image_scale(
            self.window_size.width as f32,
            self.window_size.height as f32,
            stored_image.width as f32,
            stored_image.height as f32,
            None,
            None,
        );
        self.render_images.push(RenderImage {
            offset: [0.0, 0.0],
            aspect_ratio_scale: [scale_x, scale_y], // Initialize scale
            size: [1.0, 1.0],
            index,
            image_uniform_buffer,
            aspect_bind_group,
        });
        self.render_images.last_mut().unwrap()
    }
}

struct DrawContext {
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: wgpu::SurfaceConfiguration,
    image_renderer: Option<ImageRenderer>,
    images: Vec<ImageIndex>,
}

async fn setup(window: Arc<Window>) -> DrawContext {
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        console_log::init_with_level(log::Level::Warn).expect("Could't initialize logger");
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas().unwrap()))
                    .ok()
            })
            .expect("couldn't append canvas to document body");
    }

    let instance = wgpu::Instance::default();
    let size = window.inner_size();
    let surface = instance.create_surface(window.clone()).unwrap();

    let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, Some(&surface))
        .await
        .expect("No suitable GPU adapters found!");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
            },
            None,
        )
        .await
        .unwrap();

    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let surface_caps = surface.get_capabilities(&adapter);
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_caps.formats[0],
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
        desired_maximum_frame_latency: 2,
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
    };
    surface.configure(&device, &config);
    // Load image
    let img = image::open("assets/images/rust.png")
        .expect("Failed to load image")
        .to_rgba8();
    let (width, height) = img.dimensions();
    let rgba = img.into_raw();

    // Create texture
    let texture_size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Texture"),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
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
        texture_size,
    );

    let mut image_renderer =
        ImageRenderer::new(device.clone(), queue.clone(), &config, window.inner_size());

    let mut images = Vec::new();
    images.push(image_renderer.store_image("assets/images/rust.png"));
    images.push(image_renderer.store_image("assets/images/image.png"));

    DrawContext {
        surface,
        device,
        queue,
        config,
        image_renderer: Some(image_renderer),
        images,
    }
}

#[derive(Default)]
pub struct App {
    window: Option<Arc<Window>>,
    context: Option<DrawContext>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(Window::default_attributes().with_title("Triangle Example"))
                    .unwrap(),
            );
            self.window = Some(window.clone());
            self.context = Some(futures::executor::block_on(setup(window.clone())));
            // Request the first redraw
            window.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(context) = &mut self.context {
                    context.config.width = size.width.max(1);
                    context.config.height = size.height.max(1);
                    context.surface.configure(&context.device, &context.config);

                    if let Some(image_renderer) = &mut context.image_renderer {
                        image_renderer.window_resized(size);
                    }

                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(context) = &mut self.context {
                    let frame = match context.surface.get_current_texture() {
                        Ok(frame) => frame,
                        Err(_) => {
                            context.surface.configure(&context.device, &context.config);
                            context
                                .surface
                                .get_current_texture()
                                .expect("Failed to acquire next texture")
                        }
                    };

                    let view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    let mut encoder =
                        context
                            .device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("Render Encoder"),
                            });

                    if let Some(image_renderer) = &mut context.image_renderer {
                        image_renderer.begin_frame();
                        image_renderer.image(context.images[0]).scale(0.5);
                        image_renderer
                            .image(context.images[1])
                            .scale(0.5)
                            .offset([0.0, 0.5]);
                        image_renderer.render(&mut encoder, &view);
                    }

                    context.queue.submit(std::iter::once(encoder.finish()));
                    frame.present();
                }
            }
            _ => (),
        }
    }
}
