use std::sync::Arc;
use wgpu::MemoryHints::Performance;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

use vger::defs::*;
use vger::*;

/// Represents the state required for rendering and interacting with the GPU.
struct DrawContext {
    surface: wgpu::Surface<'static>, // The surface (window) to render to.
    device: Arc<wgpu::Device>,       // The GPU device used for rendering operations.
    config: wgpu::SurfaceConfiguration, // Configuration for the surface (window), including size and format.
    vger: Vger,                         // The Vger instance used to perform high-level 2D drawing.
}

/// Sets up the `wgpu` and `vger` context for rendering.
async fn setup(window: Arc<Window>) -> DrawContext {
    // WASM-specific setup: configures logging and appends the canvas to the document body
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
        // On WASM, append the canvas to the HTML body.
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()?))
                    .ok()
            })
            .expect("couldn't append canvas to document body");
    }

    // Create a `wgpu` instance and select a GPU adapter
    let instance_desc = wgpu::InstanceDescriptor::default();
    let instance = wgpu::Instance::new(instance_desc);
    let size = window.inner_size();
    let surface = instance
        .create_surface(window)
        .expect("Failed to create surface!");

    // Choose a GPU adapter suitable for rendering
    let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, Some(&surface))
        .await
        .expect("No suitable GPU adapters found on the system!");

    // Retrieve the GPU device and queue
    let trace_dir = std::env::var("WGPU_TRACE");
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                memory_hints: Performance, // Optimizing for performance.
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

    // Set up the surface configuration with parameters such as window size and format
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT, // This is a render target
        format: surface.get_capabilities(&adapter).formats[0], // Choose the first compatible format
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo, // Wait for vertical sync to avoid tearing
        desired_maximum_frame_latency: 2,      // Max frame latency for smooth rendering
        alpha_mode: wgpu::CompositeAlphaMode::Auto, // Handle transparency automatically
        view_formats: vec![],                  // No additional formats needed for the view
    };

    surface.configure(&device, &config);

    // Create the Vger instance for 2D rendering with the selected device and queue
    // let vger = Vger::new(device.clone(), queue.clone(), config.format);
    let vger = Vger::new_with_font(
        device.clone(),
        queue.clone(),
        config.format,
        include_bytes!("../../../assets/fonts/Sniglet/Sniglet-Regular.ttf") as &[u8],
    );

    // Return the initialized rendering context
    DrawContext {
        surface,
        device,
        config,
        vger,
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Dot,
    Images,
}

impl Default for Tab {
    fn default() -> Self {
        Tab::Dot
    }
}

/// Selects the tab based on the mouse position
/// If the mouse is hovering over a rectangle, it will select the images tab
fn tab_selector(vger: &mut Vger, mouse_pos: Option<LocalPoint>, tab: Tab) -> Tab {
    let font_size: u32 = 20;

    let new_tab_if_hover = match tab {
        Tab::Dot => Tab::Images,
        Tab::Images => Tab::Dot,
    };

    let (rect, text_offset) = {
        let padding = 10.0;
        let width = 200.0;
        let height = 50.0;
        let (origin, size) = match tab {
            Tab::Dot => (
                LocalPoint::new(padding, padding),
                LocalSize::new(width, height),
            ),
            Tab::Images => (
                LocalPoint::new(width + padding * 2.0, padding),
                LocalSize::new(width, height),
            ),
        };

        let half_text_width = match new_tab_if_hover {
            Tab::Dot => 20.0,
            Tab::Images => 30.0,
        };

        (
            LocalRect::new(origin, size),
            [
                origin.x + size.width / 2.0 - half_text_width,
                origin.y + size.height / 2.0 + font_size as f32 / 2.0,
            ],
        )
    };

    let mut hover = false;

    // Check if the mouse is hovering over the rectangle
    if let Some(mouse_pos) = mouse_pos {
        if rect.contains(mouse_pos) {
            hover = true;
        }
    }

    let radius = 15.0;
    let paint_index = vger.color_paint(Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    });
    vger.fill_rect(rect, radius, paint_index);
    vger.save();
    vger.translate(text_offset);
    let text = match tab {
        Tab::Dot => "Images",
        Tab::Images => "Dot",
    };
    vger.text(
        text,
        font_size,
        Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        },
        None,
    );
    vger.restore();

    if hover {
        new_tab_if_hover
    } else {
        tab
    }
}

/// Renders the scene to the window using `vger`
///
/// This function draws a red circle in the center of the window and renders sample text.
fn dot(vger: &mut Vger, window_size: [f32; 2]) {
    // Save the current drawing state (useful for transformations)
    vger.save();

    // Translate the drawing context to the center of the window
    vger.translate([window_size[0] / 2.0, window_size[1] / 2.0]);

    // Create a red paint (color) for the circle
    let paint = vger.color_paint(Color {
        r: 1.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    });

    // Draw a circle at the center with radius 10
    vger.fill_circle([0.0, 0.0], 10.0, paint);

    // Restore the previous drawing state (undo transformations)
    vger.restore();
    vger.save();

    // Translate the drawing context to the top-left corner
    vger.translate([10.0, window_size[1] - 20.0]);

    vger.text(
        "I am not restricted by any bounds and will stretch towards infinity if needed.",
        12,
        Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        },
        None,
    );

    vger.restore();
    vger.save();

    // Translate the drawing context to the top-left corner
    vger.translate([10.0, window_size[1] - 50.0]);

    vger.text(
        "I am restricted by a width of window width - 20 pixels to give me some padding which looks nice.",
        15,
        Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        },
        Some(window_size[0] - 20.0),
    );
    vger.restore();
}

fn images(vger: &mut Vger, images: &Vec<ImageIndex>) {
    // Save the current drawing state (useful for transformations)
    vger.save();

    fn row(
        image_index: ImageIndex,
        vger: &mut Vger,
        y_offset: f32,
        width_height: f32,
    ) -> (f32, f32) {
        let size = [width_height, width_height];
        let padding = 20.0;
        let padding_size = width_height + padding;

        // First row
        vger.image(image_index)
            .fit(Fit::Contain)
            .v_align(AxisAlignEnum::Start)
            .size(size)
            .overflow_hidden()
            .offset([0.0, y_offset]);
        vger.image(image_index)
            .fit(Fit::Contain)
            .v_align(AxisAlignEnum::Center)
            .size(size)
            .overflow_hidden()
            .offset([padding_size, y_offset]);

        vger.image(image_index)
            .v_align(AxisAlignEnum::End)
            .fit(Fit::Contain)
            .size(size)
            .overflow_hidden()
            .offset([padding_size * 2.0, y_offset]);

        // Custom vertical alignment
        vger.image(image_index)
            .v_align(0.618033989)
            .fit(Fit::Contain)
            .size(size)
            .overflow_hidden()
            .offset([padding_size * 3.0, y_offset]);

        // Second row
        // No fit which causes the image to be stretched
        vger.image(image_index)
            .size(size)
            .overflow_hidden()
            .offset([0.0, padding_size + y_offset]);

        // Hidden overflow
        vger.image(image_index)
            .fit(Fit::Cover)
            .h_align(AxisAlignEnum::Start)
            .v_align(AxisAlignEnum::Start)
            .overflow_hidden()
            .size(size)
            .offset([padding_size, padding_size + y_offset]);

        // Visible overflow
        vger.image(image_index)
            .fit(Fit::Cover)
            .h_align(AxisAlignEnum::Start)
            .v_align(AxisAlignEnum::Start)
            .size(size)
            .offset([padding_size * 2.0, padding_size + y_offset]);

        // return the y offset and size for the next row
        (padding_size * 2.0 + y_offset, width_height * 2.0)
    }

    // let mut y_offset = 0.0;
    // let mut width_height = 2.0;
    // (0..8).for_each(|_| {
    //     (y_offset, width_height) = row(images[0], vger, y_offset, width_height);
    // });

    let y_offset = 50.0;
    let width_height = 200.0;

    row(images[0], vger, y_offset, width_height);

    vger.restore();
}

/// Represents the application state, holding the window and drawing context.
#[derive(Default)]
pub struct App {
    window: Option<Arc<Window>>, // The application window, wrapped in an `Option` for initialization
    context: Option<DrawContext>, // The drawing context, wrapped in an `Option` for initialization
    images: Vec<ImageIndex>,
    mouse_pos: Option<[f32; 2]>, // The current mouse position
    tab: Tab,
}

impl<'window> ApplicationHandler for App {
    /// This function is called when the application is resumed (after initialization).
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // If the window is not already initialized, create it and set up the rendering context.
        if self.window.is_none() {
            let win_attr = Window::default_attributes().with_title("vger basic example");

            // Create the window using `winit`
            let window = Arc::new(
                event_loop
                    .create_window(win_attr)
                    .expect("create window err."),
            );
            self.window = Some(window.clone());

            // Initialize the rendering context asynchronously.
            let window = window.clone();
            (self.context, self.images) = {
                let mut context = futures::executor::block_on(setup(window));
                let vger = &mut context.vger;
                let mut images = Vec::new();
                images.push(vger.store_image("assets/images/rust.png"));
                (Some(context), images)
            };
        }
    }

    /// This function handles window events such as resizing, closing, and redraw requests.
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_pos = Some([position.x as f32, position.y as f32]);
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::CursorLeft { .. } => {
                self.mouse_pos = None;
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            // Close the application when the window close event is triggered.
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            // Handle window resizing: update surface configuration and trigger a redraw.
            WindowEvent::Resized(size) => {
                if let (Some(window), Some(context)) = (&self.window, &mut self.context) {
                    context.config.width = size.width.max(1); // Ensure at least 1px width
                    context.config.height = size.height.max(1); // Ensure at least 1px height
                    context.surface.configure(&context.device, &context.config);
                    window.request_redraw(); // Request a redraw after resize
                }
            }
            // Trigger rendering when the window requests a redraw.
            WindowEvent::RedrawRequested => {
                if let (Some(window), Some(context)) = (&self.window, &mut self.context) {
                    let window_size = window.inner_size();
                    let scale = window.scale_factor() as f32;
                    let width = window_size.width as f32 / scale;
                    let height = window_size.height as f32 / scale;

                    // Acquire the current frame for rendering
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

                    // Begin rendering with `vger` and set the window size and scale factor
                    let vger = &mut context.vger;
                    vger.begin(width, height, scale);

                    let mouse_pos = match self.mouse_pos {
                        Some([x, y]) => {
                            // Convert the mouse position to local coordinates
                            let scale = window.scale_factor() as f32;
                            let x = x / scale;
                            let y = y / scale;
                            let y = height - y; // Invert height to match Vger's coordinate system
                            Some(LocalPoint::new(x, y))
                        }
                        None => None,
                    };
                    self.tab = tab_selector(vger, mouse_pos, self.tab);
                    match self.tab {
                        Tab::Dot => dot(vger, [width, height]),
                        Tab::Images => images(vger, &self.images),
                    }

                    // Create a texture view for the frame
                    let texture_view = frame
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    // Set up the render pass descriptor
                    let desc = wgpu::RenderPassDescriptor {
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &texture_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), // Clear to black before rendering
                                store: wgpu::StoreOp::Store,                   // Store the result
                            },
                        })],
                        ..<_>::default() // Default for other fields
                    };

                    // Encode the render commands for `vger`
                    vger.encode(&desc);

                    // Present the frame to the window
                    frame.present();
                }
            }
            _ => (), // Ignore other events for now
        }
    }
}
