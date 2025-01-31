// use futures::executor::block_on;
// mod buffer;
// mod context;
// mod gaussian_blur;
// mod grayscale;
// mod threshold;
// mod utils;

// pub use crate::buffer::*;
// pub use crate::context::*;
// pub use crate::gaussian_blur::*;
// pub use crate::grayscale::*;
// pub use crate::threshold::*;
// pub use crate::utils::*;

// fn main() {
//     let context = WgContext::new();
//     let context = block_on(context);
//     let image = image::open("assets/images/rust.png").unwrap().to_rgba8();
//     let width = image.width();
//     let height = image.height();
//     let image_buffer = WgImageBuffer::from_host_image(&context, image);
//     let mut grayscale = GrayScale::new(&context, width, height);
//     grayscale.run(&image_buffer);
//     let grayscale_image = grayscale.output_image.to_host_image(&context);
//     grayscale_image
//         .unwrap()
//         .save("assets/images/rust_grayscale.png")
//         .unwrap();
// }

use crate::app::App;
use winit::error::EventLoopError;
use winit::event_loop::{ControlFlow, EventLoop};

mod app;

fn main() -> Result<(), EventLoopError> {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app)
}
