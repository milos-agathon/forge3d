//! Matrix stack utility for hierarchical transformations
//!
//! Provides push/pop semantics for managing transformation hierarchies,
//! commonly used in scene graph rendering and nested coordinate systems.

use crate::error::{RenderError, RenderResult};
use glam::Mat4;

/// Matrix stack for hierarchical transformation management
#[derive(Debug, Clone)]
pub struct MatrixStack {
    /// Stack of transformation matrices  
    stack: Vec<Mat4>,
    /// Maximum allowed stack depth to prevent overflow
    max_depth: usize,
}

impl MatrixStack {
    /// Create a new matrix stack with identity as the initial matrix
    pub fn new() -> Self {
        Self {
            stack: vec![Mat4::IDENTITY],
            max_depth: 64, // Reasonable default for most use cases
        }
    }

    /// Create a new matrix stack with a custom initial matrix
    pub fn with_initial(initial: Mat4) -> Self {
        Self {
            stack: vec![initial],
            max_depth: 64,
        }
    }

    /// Create a new matrix stack with a custom maximum depth
    pub fn with_max_depth(max_depth: usize) -> Self {
        Self {
            stack: vec![Mat4::IDENTITY],
            max_depth: max_depth.max(1), // Ensure at least depth of 1
        }
    }

    /// Get the current top matrix (without modifying the stack)
    pub fn top(&self) -> Mat4 {
        // Safe unwrap: stack is never empty (initialized with at least one element)
        *self.stack.last().unwrap()
    }

    /// Push the current matrix onto the stack
    ///
    /// This duplicates the current matrix so transformations can be applied
    /// and later restored with pop().
    pub fn push(&mut self) -> RenderResult<()> {
        if self.stack.len() >= self.max_depth {
            return Err(RenderError::render(&format!(
                "Matrix stack overflow: maximum depth {} exceeded",
                self.max_depth
            )));
        }

        let current = self.top();
        self.stack.push(current);
        Ok(())
    }

    /// Pop the top matrix from the stack, restoring the previous state
    pub fn pop(&mut self) -> RenderResult<()> {
        if self.stack.len() <= 1 {
            return Err(RenderError::render(
                "Matrix stack underflow: cannot pop the last matrix",
            ));
        }

        self.stack.pop();
        Ok(())
    }

    /// Replace the current top matrix
    pub fn load(&mut self, matrix: Mat4) {
        if let Some(top) = self.stack.last_mut() {
            *top = matrix;
        }
    }

    /// Multiply the current matrix by the given matrix
    ///
    /// This applies the transformation: current = current * matrix
    pub fn mult(&mut self, matrix: Mat4) {
        if let Some(top) = self.stack.last_mut() {
            *top = *top * matrix;
        }
    }

    /// Load an identity matrix as the current matrix
    pub fn load_identity(&mut self) {
        self.load(Mat4::IDENTITY);
    }

    /// Apply a translation to the current matrix
    pub fn translate(&mut self, translation: glam::Vec3) {
        self.mult(Mat4::from_translation(translation));
    }

    /// Apply a rotation to the current matrix
    pub fn rotate(&mut self, rotation: glam::Quat) {
        self.mult(Mat4::from_quat(rotation));
    }

    /// Apply a rotation around an axis to the current matrix
    pub fn rotate_axis(&mut self, axis: glam::Vec3, angle_radians: f32) {
        let rotation = glam::Quat::from_axis_angle(axis.normalize(), angle_radians);
        self.rotate(rotation);
    }

    /// Apply a scale to the current matrix
    pub fn scale(&mut self, scale: glam::Vec3) {
        self.mult(Mat4::from_scale(scale));
    }

    /// Apply uniform scale to the current matrix
    pub fn scale_uniform(&mut self, scale: f32) {
        self.scale(glam::Vec3::splat(scale));
    }

    /// Apply a transformation matrix (translation, rotation, scale)
    pub fn transform(&mut self, translation: glam::Vec3, rotation: glam::Quat, scale: glam::Vec3) {
        self.mult(Mat4::from_scale_rotation_translation(
            scale,
            rotation,
            translation,
        ));
    }

    /// Get the current stack depth
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Check if the stack is at its maximum depth
    pub fn is_full(&self) -> bool {
        self.stack.len() >= self.max_depth
    }

    /// Get the maximum allowed stack depth
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Set a new maximum stack depth
    ///
    /// If the current stack exceeds the new max depth, this will return an error.
    pub fn set_max_depth(&mut self, max_depth: usize) -> RenderResult<()> {
        let new_max = max_depth.max(1);
        if self.stack.len() > new_max {
            return Err(RenderError::render(&format!(
                "Cannot set max depth to {}: current stack depth is {}",
                new_max,
                self.stack.len()
            )));
        }
        self.max_depth = new_max;
        Ok(())
    }

    /// Reset the stack to contain only the identity matrix
    pub fn reset(&mut self) {
        self.stack.clear();
        self.stack.push(Mat4::IDENTITY);
    }

    /// Reset the stack to contain only the given initial matrix
    pub fn reset_with(&mut self, initial: Mat4) {
        self.stack.clear();
        self.stack.push(initial);
    }

    /// Create a scoped transformation that automatically pops when dropped
    pub fn scoped(&mut self) -> RenderResult<ScopedTransform<'_>> {
        self.push()?;
        Ok(ScopedTransform { stack: self })
    }
}

impl Default for MatrixStack {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII helper for scoped transformations
///
/// When this goes out of scope, it automatically pops the matrix stack,
/// ensuring balanced push/pop operations.
pub struct ScopedTransform<'a> {
    stack: &'a mut MatrixStack,
}

impl<'a> ScopedTransform<'a> {
    /// Get the current top matrix
    pub fn top(&self) -> Mat4 {
        self.stack.top()
    }

    /// Replace the current matrix
    pub fn load(&mut self, matrix: Mat4) {
        self.stack.load(matrix);
    }

    /// Multiply by the given matrix
    pub fn mult(&mut self, matrix: Mat4) {
        self.stack.mult(matrix);
    }

    /// Load identity matrix
    pub fn load_identity(&mut self) {
        self.stack.load_identity();
    }

    /// Apply translation
    pub fn translate(&mut self, translation: glam::Vec3) {
        self.stack.translate(translation);
    }

    /// Apply rotation
    pub fn rotate(&mut self, rotation: glam::Quat) {
        self.stack.rotate(rotation);
    }

    /// Apply rotation around axis
    pub fn rotate_axis(&mut self, axis: glam::Vec3, angle_radians: f32) {
        self.stack.rotate_axis(axis, angle_radians);
    }

    /// Apply scale
    pub fn scale(&mut self, scale: glam::Vec3) {
        self.stack.scale(scale);
    }

    /// Apply uniform scale
    pub fn scale_uniform(&mut self, scale: f32) {
        self.stack.scale_uniform(scale);
    }

    /// Apply transformation
    pub fn transform(&mut self, translation: glam::Vec3, rotation: glam::Quat, scale: glam::Vec3) {
        self.stack.transform(translation, rotation, scale);
    }
}

impl<'a> Drop for ScopedTransform<'a> {
    fn drop(&mut self) {
        // Ignore errors during drop - we can't handle them anyway
        let _ = self.stack.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    #[test]
    fn test_matrix_stack_basic_operations() {
        let mut stack = MatrixStack::new();

        // Initial state should be identity
        assert_eq!(stack.top(), Mat4::IDENTITY);
        assert_eq!(stack.depth(), 1);

        // Test push/pop
        stack.push().unwrap();
        assert_eq!(stack.depth(), 2);
        assert_eq!(stack.top(), Mat4::IDENTITY);

        stack.pop().unwrap();
        assert_eq!(stack.depth(), 1);
    }

    #[test]
    fn test_matrix_stack_transformations() {
        let mut stack = MatrixStack::new();

        // Apply translation
        stack.translate(Vec3::new(1.0, 2.0, 3.0));
        let translated = stack.top();
        assert_ne!(translated, Mat4::IDENTITY);

        // Push and apply rotation
        stack.push().unwrap();
        stack.rotate(Quat::from_rotation_y(std::f32::consts::PI / 2.0));
        let rotated = stack.top();
        assert_ne!(rotated, translated);

        // Pop should restore translation-only state
        stack.pop().unwrap();
        assert_eq!(stack.top(), translated);
    }

    #[test]
    fn test_matrix_stack_overflow_protection() {
        let mut stack = MatrixStack::with_max_depth(2);

        stack.push().unwrap(); // Depth 2 (at max)

        // Should fail on overflow
        assert!(stack.push().is_err());
        assert_eq!(stack.depth(), 2);
    }

    #[test]
    fn test_matrix_stack_underflow_protection() {
        let mut stack = MatrixStack::new();

        // Should fail to pop below depth 1
        assert!(stack.pop().is_err());
        assert_eq!(stack.depth(), 1);
    }

    #[test]
    fn test_scoped_transform() {
        let mut stack = MatrixStack::new();
        let initial = stack.top();

        let transformed = {
            let mut scoped = stack.scoped().unwrap();
            scoped.translate(Vec3::new(5.0, 0.0, 0.0));
            scoped.top()
        }; // scoped drops here, should auto-pop

        // Verify transformation was different
        assert_ne!(transformed, initial);

        // Stack should be restored to initial state
        assert_eq!(stack.top(), initial);
        assert_eq!(stack.depth(), 1);
    }

    #[test]
    fn test_matrix_stack_reset() {
        let mut stack = MatrixStack::new();

        stack.push().unwrap();
        stack.translate(Vec3::new(1.0, 0.0, 0.0));
        stack.push().unwrap();

        assert_eq!(stack.depth(), 3);

        stack.reset();
        assert_eq!(stack.depth(), 1);
        assert_eq!(stack.top(), Mat4::IDENTITY);
    }
}
