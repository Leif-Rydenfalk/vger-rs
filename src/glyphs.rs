use crate::atlas::Atlas;
use rect_packer::Rect;
use std::collections::HashMap;

/// Represents information about a single glyph (character) in a font.
///
/// A `GlyphInfo` contains two pieces of data:
/// - A rectangle (`rect`) that indicates where the glyph's image is stored in the texture atlas.
/// - The metrics of the glyph (`metrics`), which describe its dimensions, spacing, and other layout-related properties.
#[derive(Copy, Clone, Debug)]
pub struct GlyphInfo {
    /// The rectangle in the texture atlas where the glyph's image is stored.
    ///
    /// This is an `Option` because not all glyphs may have a valid texture rectangle. Some glyphs may be missing
    /// from the atlas or not require one (e.g., whitespace characters or special handling for certain cases).
    pub rect: Option<Rect>,

    /// The metrics of the glyph, such as its width, height, and other characteristics related to layout.
    ///
    /// This is a `fontdue::Metrics` object from the `fontdue` crate, which provides detailed measurements for the glyph.
    pub metrics: fontdue::Metrics,
}

/// A cache for storing and retrieving glyph information for efficient text rendering.
///
/// The `GlyphCache` is responsible for managing a texture atlas that contains the images of all the glyphs used in a font,
/// as well as storing the metrics (dimensions, spacing, etc.) for each glyph. This allows for efficient lookups and
/// rendering of characters, especially when handling dynamic or repeated text rendering.
///
/// It also helps reduce the cost of recalculating or reloading glyph data for each character and size combination.
///
/// # Fields
/// - `atlas`: The texture atlas that contains the images of all the glyphs.
/// - `font`: The font object used to generate and manage the glyphs.
/// - `info`: A cache (hash map) that stores information about each glyph, keyed by the character and font size.
pub struct GlyphCache {
    /// The texture atlas where all the glyph images are stored.
    ///
    /// An atlas is a large texture that contains multiple smaller glyph images, which are used for efficient rendering.
    /// This allows the GPU to use a single texture for multiple glyphs, reducing texture switching during rendering.
    pub atlas: Atlas,

    /// The font object that contains the font data and methods for generating glyphs.
    ///
    /// The font is typically loaded from a font file and used to generate the glyph images and metrics.
    pub font: fontdue::Font,

    /// A cache of glyph information, indexed by a tuple of (character, size).
    ///
    /// This hash map stores the `GlyphInfo` for each unique character and font size combination. The cache allows
    /// for fast lookup of previously generated glyphs, saving the cost of re-generating them every time they are needed.
    ///
    /// The tuple key consists of:
    /// - A `char`: the character being rendered (e.g., `'a'`, `'1'`, `'@'`).
    /// - A `u32`: the font size used when generating the glyph.
    ///
    /// This cache helps improve performance by reusing already computed glyph data.
    info: HashMap<(char, u32), GlyphInfo>,
}

impl GlyphCache {
    pub fn new(device: &wgpu::Device, font_bytes: &[u8]) -> Self {
        let mut settings = fontdue::FontSettings::default();
        settings.collection_index = 0;
        settings.scale = 100.0;

        Self {
            atlas: Atlas::new(device),
            font: fontdue::Font::from_bytes(font_bytes, settings).unwrap(),
            info: HashMap::new(),
        }
    }

    pub fn get_glyph(&mut self, c: char, size: f32) -> GlyphInfo {
        let factor = 65536.0;

        // Convert size to fixed point so we can hash it.
        let size_fixed_point = (size * factor) as u32;

        // Do we already have a glyph?
        match self.info.get(&(c, size_fixed_point)) {
            Some(info) => *info,
            None => {
                let (metrics, data) = self.font.rasterize(c, size_fixed_point as f32 / factor);

                /*
                let mut i = 0;
                for _ in 0..metrics.height {
                    for _ in 0..metrics.width {
                        print!("{} ", if data[i] != 0 { '*' } else { ' ' });
                        i += 1;
                    }
                    print!("\n");
                }
                */

                let rect =
                    self.atlas
                        .add_region(&data, metrics.width as u32, metrics.height as u32);

                let info = GlyphInfo { rect, metrics };

                self.info.insert((c, size_fixed_point), info);
                info
            }
        }
    }

    pub fn update(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        self.atlas.update(device, encoder);
    }

    pub fn create_view(&self) -> wgpu::TextureView {
        self.atlas.create_view()
    }

    pub fn usage(&self) -> f32 {
        self.atlas.usage()
    }

    pub fn clear(&mut self) {
        self.info.clear();
        self.atlas.clear();
    }
}
