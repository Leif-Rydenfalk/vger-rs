use std::sync::Arc;
use wgpu::MemoryHints::Performance;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

use vger::*;

struct DrawContext {
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: wgpu::SurfaceConfiguration,
    vger: Vger,
}

async fn setup(window: Arc<Window>) -> DrawContext {
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        let query_string = web_sys::window().unwrap().location().search().unwrap();
        let level: log::Level = parse_url_query_string(&query_string, "RUST_LOG")
            .map(|x| x.parse().ok())
            .flatten()
            .unwrap_or(log::Level::Error);
        console_log::init_with_level(level).expect("could not initialize logger");
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()?))
                    .ok()
            })
            .expect("couldn't append canvas to document body");
    }

    let instance_desc = wgpu::InstanceDescriptor::default();

    let instance = wgpu::Instance::new(instance_desc);
    let size = window.inner_size();
    let surface = instance
        .create_surface(window)
        .expect("Failed to create surface!");
    let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, Some(&surface))
        .await
        .expect("No suitable GPU adapters found on the system!");

    let trace_dir = std::env::var("WGPU_TRACE");
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                memory_hints: Performance,
                label: None,
                required_features: wgpu::Features::default(),
                required_limits: wgpu::Limits::default(),
            },
            trace_dir.ok().as_ref().map(std::path::Path::new),
        )
        .await
        .expect("Unable to find a suitable GPU adapter!");
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface.get_capabilities(&adapter).formats[0],
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
        desired_maximum_frame_latency: 2,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
    };
    surface.configure(&device, &config);

    let vger = Vger::new(device.clone(), queue.clone(), config.format);

    DrawContext {
        surface,
        device,
        queue,
        config,
        vger,
    }
}

/// Renders everything using vger.
fn render(vger: &mut Vger, window_size: [f32; 2]) {
    vger.save();

    vger.translate([window_size[0] / 2.0, window_size[1] / 2.0]);

    let paint = vger.color_paint(Color {
        r: 1.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    });

    vger.fill_circle([0.0, 0.0], 10.0, paint);

    vger.restore();
}

#[derive(Default)]
pub struct App {
    window: Option<Arc<Window>>,
    context: Option<DrawContext>,
}

impl<'window> ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let win_attr = Window::default_attributes().with_title("wgpu winit example");
            // use Arc.
            let window = Arc::new(
                event_loop
                    .create_window(win_attr)
                    .expect("create window err."),
            );
            self.window = Some(window.clone());
            let window = window.clone();
            self.context = Some(futures::executor::block_on(setup(window)));
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let (Some(window), Some(context)) = (&self.window, &mut self.context) {
                    context.config.width = size.width.max(1);
                    context.config.height = size.height.max(1);
                    context.surface.configure(&context.device, &context.config);
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                // Redraw the application.
                //
                // It's preferable for applications that do not render continuously to render in
                // this event rather than in MainEventsCleared, since rendering in here allows
                // the program to gracefully handle redraws requested by the OS.

                if let (Some(window), Some(context)) = (&self.window, &mut self.context) {
                    let window_size = window.inner_size();
                    let scale = window.scale_factor() as f32;
                    let width = window_size.width as f32 / scale;
                    let height = window_size.height as f32 / scale;

                    let surface = &context.surface;
                    let device = context.device.clone();
                    let config = context.config.clone();
                    let frame = match surface.get_current_texture() {
                        Ok(frame) => frame,
                        Err(_) => {
                            surface.configure(&device, &config);
                            surface
                                .get_current_texture()
                                .expect("Failed to acquire next surface texture!")
                        }
                    };

                    let vger = &mut context.vger;
                    vger.begin(width, height, scale);
                    render(vger, [width, height]);

                    let texture_view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    let desc = wgpu::RenderPassDescriptor {
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &texture_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        ..<_>::default()
                    };

                    vger.encode(&desc);

                    frame.present();
                }
            }
            _ => (),
        }
    }
}
