//! Grid-based collision detection for labels.

/// Grid-based collision detection to prevent label overlap.
/// Uses a simple 2D grid where each cell tracks occupied rectangles.
pub struct CollisionGrid {
    width: u32,
    height: u32,
    cell_size: u32,
    cols: u32,
    rows: u32,
    /// Each cell contains a list of occupied rectangles [x0, y0, x1, y1].
    cells: Vec<Vec<[f32; 4]>>,
}

impl CollisionGrid {
    /// Create a new collision grid.
    ///
    /// # Arguments
    /// * `width` - Screen width in pixels
    /// * `height` - Screen height in pixels
    /// * `cell_size` - Size of each grid cell in pixels (e.g., 10)
    pub fn new(width: u32, height: u32, cell_size: u32) -> Self {
        let cell_size = cell_size.max(1);
        let cols = (width + cell_size - 1) / cell_size;
        let rows = (height + cell_size - 1) / cell_size;
        let cell_count = (cols * rows) as usize;

        Self {
            width,
            height,
            cell_size,
            cols,
            rows,
            cells: vec![Vec::new(); cell_count],
        }
    }

    /// Clear all occupied rectangles.
    pub fn clear(&mut self) {
        for cell in &mut self.cells {
            cell.clear();
        }
    }

    /// Try to insert a rectangle. Returns true if successful (no collision),
    /// false if the rectangle would overlap with an existing one.
    ///
    /// # Arguments
    /// * `bounds` - Rectangle bounds [x0, y0, x1, y1] in screen pixels
    pub fn try_insert(&mut self, bounds: [f32; 4]) -> bool {
        let [x0, y0, x1, y1] = bounds;

        // Clamp to screen bounds
        let x0 = x0.max(0.0);
        let y0 = y0.max(0.0);
        let x1 = x1.min(self.width as f32);
        let y1 = y1.min(self.height as f32);

        // Check if completely off-screen
        if x0 >= x1 || y0 >= y1 {
            return false;
        }

        // Get cell range
        let col0 = (x0 / self.cell_size as f32).floor() as u32;
        let col1 = (x1 / self.cell_size as f32).ceil() as u32;
        let row0 = (y0 / self.cell_size as f32).floor() as u32;
        let row1 = (y1 / self.cell_size as f32).ceil() as u32;

        let col0 = col0.min(self.cols - 1);
        let col1 = col1.min(self.cols);
        let row0 = row0.min(self.rows - 1);
        let row1 = row1.min(self.rows);

        // Check for collisions
        for row in row0..row1 {
            for col in col0..col1 {
                let idx = (row * self.cols + col) as usize;
                if idx < self.cells.len() {
                    for existing in &self.cells[idx] {
                        if rects_overlap([x0, y0, x1, y1], *existing) {
                            return false;
                        }
                    }
                }
            }
        }

        // No collision, insert into all overlapping cells
        for row in row0..row1 {
            for col in col0..col1 {
                let idx = (row * self.cols + col) as usize;
                if idx < self.cells.len() {
                    self.cells[idx].push([x0, y0, x1, y1]);
                }
            }
        }

        true
    }

    /// Check if a rectangle would collide without inserting.
    pub fn check_collision(&self, bounds: [f32; 4]) -> bool {
        let [x0, y0, x1, y1] = bounds;

        let x0 = x0.max(0.0);
        let y0 = y0.max(0.0);
        let x1 = x1.min(self.width as f32);
        let y1 = y1.min(self.height as f32);

        if x0 >= x1 || y0 >= y1 {
            return true; // Off-screen counts as collision
        }

        let col0 = (x0 / self.cell_size as f32).floor() as u32;
        let col1 = (x1 / self.cell_size as f32).ceil() as u32;
        let row0 = (y0 / self.cell_size as f32).floor() as u32;
        let row1 = (y1 / self.cell_size as f32).ceil() as u32;

        let col0 = col0.min(self.cols - 1);
        let col1 = col1.min(self.cols);
        let row0 = row0.min(self.rows - 1);
        let row1 = row1.min(self.rows);

        for row in row0..row1 {
            for col in col0..col1 {
                let idx = (row * self.cols + col) as usize;
                if idx < self.cells.len() {
                    for existing in &self.cells[idx] {
                        if rects_overlap([x0, y0, x1, y1], *existing) {
                            return true;
                        }
                    }
                }
            }
        }

        false
    }
}

/// Check if two axis-aligned rectangles overlap.
#[inline]
fn rects_overlap(a: [f32; 4], b: [f32; 4]) -> bool {
    a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_collision() {
        let mut grid = CollisionGrid::new(100, 100, 10);
        assert!(grid.try_insert([0.0, 0.0, 10.0, 10.0]));
        assert!(grid.try_insert([50.0, 50.0, 60.0, 60.0]));
    }

    #[test]
    fn test_collision() {
        let mut grid = CollisionGrid::new(100, 100, 10);
        assert!(grid.try_insert([0.0, 0.0, 20.0, 20.0]));
        assert!(!grid.try_insert([10.0, 10.0, 30.0, 30.0])); // Overlaps
    }

    #[test]
    fn test_clear() {
        let mut grid = CollisionGrid::new(100, 100, 10);
        assert!(grid.try_insert([0.0, 0.0, 50.0, 50.0]));
        grid.clear();
        assert!(grid.try_insert([0.0, 0.0, 50.0, 50.0])); // Should work after clear
    }
}
