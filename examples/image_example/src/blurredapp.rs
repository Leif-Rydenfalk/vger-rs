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

struct DrawContext {
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
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

    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
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

    // Create uniform buffer for texture size
    let texture_size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Texture Size Buffer"),
        contents: bytemuck::cast_slice(&[width as f32, height as f32]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create bind group layout
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            // Existing texture entry...
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
            // Existing sampler entry...
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // New uniform buffer entry...
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(8), // Two f32s
                },
                count: None,
            },
        ],
    });

    // Create bind group
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: texture_size_buffer.as_entire_binding(),
            },
        ],
    });

    // Vertex data for quad
    let vertices = [
        Vertex {
            position: [-0.5, -0.5],
            tex_coords: [0.0, 1.0],
        }, // Bottom-left
        Vertex {
            position: [0.5, -0.5],
            tex_coords: [1.0, 1.0],
        }, // Bottom-right
        Vertex {
            position: [0.5, 0.5],
            tex_coords: [1.0, 0.0],
        }, // Top-right
        Vertex {
            position: [-0.5, 0.5],
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

    // Updated shader code
    let shader_source = r#"
        struct VertexInput {
            @location(0) position: vec2<f32>,
            @location(1) tex_coords: vec2<f32>,
        };

        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) tex_coords: vec2<f32>,
        };

        @vertex
        fn vs_main(input: VertexInput) -> VertexOutput {
            var output: VertexOutput;
            output.position = vec4(input.position, 0.0, 1.0);
            output.tex_coords = input.tex_coords;
            return output;
        }

        @group(0) @binding(0) var texture: texture_2d<f32>;
        @group(0) @binding(1) var texture_sampler: sampler;
        @group(0) @binding(2) var<uniform> texture_size: vec2<f32>;

        @fragment
        fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
            let texel_size = vec2(1.0) / texture_size;
            let sample_offsets = array(
                vec2(-1.0, -1.0),
                vec2(-1.0,  0.0),
                vec2(-1.0,  1.0),
                vec2( 0.0, -1.0),
                vec2( 0.0,  0.0),
                vec2( 0.0,  1.0),
                vec2( 1.0, -1.0),
                vec2( 1.0,  0.0),
                vec2( 1.0,  1.0),
            );
            let blur_radius = 1.0;
            var color = vec4(0.0);
            
            // Manually unroll the loop using constant indices
            color += textureSample(texture, texture_sampler, input.tex_coords + sample_offsets[0] * texel_size * blur_radius);
            color += textureSample(texture, texture_sampler, input.tex_coords + sample_offsets[1] * texel_size * blur_radius);
            color += textureSample(texture, texture_sampler, input.tex_coords + sample_offsets[2] * texel_size * blur_radius);
            color += textureSample(texture, texture_sampler, input.tex_coords + sample_offsets[3] * texel_size * blur_radius);
            color += textureSample(texture, texture_sampler, input.tex_coords + sample_offsets[4] * texel_size * blur_radius);
            color += textureSample(texture, texture_sampler, input.tex_coords + sample_offsets[5] * texel_size * blur_radius);
            color += textureSample(texture, texture_sampler, input.tex_coords + sample_offsets[6] * texel_size * blur_radius);
            color += textureSample(texture, texture_sampler, input.tex_coords + sample_offsets[7] * texel_size * blur_radius);
            color += textureSample(texture, texture_sampler, input.tex_coords + sample_offsets[8] * texel_size * blur_radius);
            
            color /= 9.0;
            return color;
        }
    "#;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
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
                blend: Some(wgpu::BlendState::REPLACE),
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

    DrawContext {
        surface,
        device,
        queue,
        config,
        render_pipeline,
        vertex_buffer,
        index_buffer,
        bind_group,
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
                    // Request redraw to update with the new size immediately
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

                    {
                        let mut render_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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

                        render_pass.set_pipeline(&context.render_pipeline);
                        render_pass.set_vertex_buffer(0, context.vertex_buffer.slice(..));
                        render_pass.set_index_buffer(
                            context.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint16,
                        );
                        render_pass.set_bind_group(0, &context.bind_group, &[]);
                        render_pass.draw_indexed(0..6, 0, 0..1);
                    }

                    context.queue.submit(std::iter::once(encoder.finish()));
                    frame.present();
                }
            }
            _ => (),
        }
    }
}
