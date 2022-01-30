#[derive(Copy, Clone)]
pub struct Prim {
    /// Type of primitive.
    prim_type: u32,

    /// Stroke width.
    width: f32,

    /// Radius of circles. Corner radius for rounded rectangles.
    radius: f32,

    /// Control vertices.
    cvs: [f32; 6],

    /// Start of the control vertices, if they're in a separate buffer.
    start: u32,

    /// Number of control vertices (vgerCurve and vgerPathFill)
    count: u32,

    /// Index of paint applied to drawing region.
    paint: u32,

    /// Glyph region index. (used internally)
    glyph: u32,

    /// Index of transform applied to drawing region. (used internally)
    xform: u32,

    /// Min and max coordinates of the quad we're rendering. (used internally)
    quad_bounds: [f32; 4],

    /// Min and max coordinates in texture space. (used internally)
    tex_bounds: [f32; 4],
}
