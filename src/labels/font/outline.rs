use lyon_path::{math, path::Builder};
use ttf_parser::OutlineBuilder;

pub(crate) struct PathSink<'a> {
    builder: &'a mut Builder,
    scale: f32,
    offset: glam::Vec2,
}

impl<'a> PathSink<'a> {
    pub(crate) fn new(builder: &'a mut Builder, scale: f32, offset: glam::Vec2) -> Self {
        Self {
            builder,
            scale,
            offset,
        }
    }
}

impl OutlineBuilder for PathSink<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        self.builder.begin(math::point(
            self.offset.x + x * self.scale,
            self.offset.y - y * self.scale,
        ));
    }

    fn line_to(&mut self, x: f32, y: f32) {
        self.builder.line_to(math::point(
            self.offset.x + x * self.scale,
            self.offset.y - y * self.scale,
        ));
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        self.builder.quadratic_bezier_to(
            math::point(
                self.offset.x + x1 * self.scale,
                self.offset.y - y1 * self.scale,
            ),
            math::point(
                self.offset.x + x * self.scale,
                self.offset.y - y * self.scale,
            ),
        );
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        self.builder.cubic_bezier_to(
            math::point(
                self.offset.x + x1 * self.scale,
                self.offset.y - y1 * self.scale,
            ),
            math::point(
                self.offset.x + x2 * self.scale,
                self.offset.y - y2 * self.scale,
            ),
            math::point(
                self.offset.x + x * self.scale,
                self.offset.y - y * self.scale,
            ),
        );
    }

    fn close(&mut self) {
        self.builder.end(true);
    }
}
