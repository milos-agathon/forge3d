//! Typography controls for label rendering.
//!
//! Provides tracking (letter-spacing), kerning, and other
//! typographic adjustments for atlas-style text rendering.

/// Typography settings for text rendering.
#[derive(Debug, Clone, Copy)]
pub struct TypographySettings {
    /// Letter-spacing adjustment as a multiple of font size (0.0 = normal).
    pub tracking: f32,
    /// Enable kerning adjustments from font metrics.
    pub kerning: bool,
    /// Line height as a multiple of font size (1.0 = single space).
    pub line_height: f32,
    /// Word spacing adjustment as a multiple of space width.
    pub word_spacing: f32,
    /// Baseline shift in pixels (positive = up).
    pub baseline_shift: f32,
}

impl Default for TypographySettings {
    fn default() -> Self {
        Self {
            tracking: 0.0,
            kerning: true,
            line_height: 1.2,
            word_spacing: 1.0,
            baseline_shift: 0.0,
        }
    }
}

impl TypographySettings {
    /// Create settings with custom tracking.
    pub fn with_tracking(mut self, tracking: f32) -> Self {
        self.tracking = tracking;
        self
    }

    /// Create settings with kerning disabled.
    pub fn without_kerning(mut self) -> Self {
        self.kerning = false;
        self
    }

    /// Create settings with custom line height.
    pub fn with_line_height(mut self, height: f32) -> Self {
        self.line_height = height;
        self
    }

    /// Create settings with custom word spacing.
    pub fn with_word_spacing(mut self, spacing: f32) -> Self {
        self.word_spacing = spacing;
        self
    }

    /// Create settings with baseline shift.
    pub fn with_baseline_shift(mut self, shift: f32) -> Self {
        self.baseline_shift = shift;
        self
    }
}

/// Kerning pair lookup table.
#[derive(Debug, Clone, Default)]
pub struct KerningTable {
    /// Kerning pairs as (char1, char2) -> adjustment.
    pairs: std::collections::HashMap<(char, char), f32>,
}

impl KerningTable {
    /// Create a new empty kerning table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a kerning pair.
    pub fn add_pair(&mut self, left: char, right: char, adjustment: f32) {
        self.pairs.insert((left, right), adjustment);
    }

    /// Get kerning adjustment for a pair.
    pub fn get_kerning(&self, left: char, right: char) -> f32 {
        self.pairs.get(&(left, right)).copied().unwrap_or(0.0)
    }

    /// Load common kerning pairs for Latin text.
    pub fn load_common_latin_pairs(&mut self) {
        // Common kerning pairs for readable text
        let pairs = [
            ('A', 'V', -0.08),
            ('A', 'W', -0.06),
            ('A', 'Y', -0.08),
            ('A', 'v', -0.04),
            ('A', 'w', -0.03),
            ('A', 'y', -0.04),
            ('F', 'a', -0.03),
            ('F', 'o', -0.03),
            ('L', 'T', -0.08),
            ('L', 'V', -0.08),
            ('L', 'W', -0.06),
            ('L', 'Y', -0.08),
            ('P', 'a', -0.03),
            ('P', 'o', -0.03),
            ('T', 'a', -0.08),
            ('T', 'e', -0.08),
            ('T', 'o', -0.08),
            ('T', 'r', -0.04),
            ('V', 'a', -0.06),
            ('V', 'e', -0.06),
            ('V', 'o', -0.06),
            ('W', 'a', -0.04),
            ('W', 'e', -0.04),
            ('W', 'o', -0.04),
            ('Y', 'a', -0.08),
            ('Y', 'e', -0.08),
            ('Y', 'o', -0.08),
            ('f', 'f', -0.02),
            ('r', 'a', -0.02),
            ('r', 'e', -0.02),
            ('r', 'o', -0.02),
        ];

        for (left, right, adj) in pairs {
            self.add_pair(left, right, adj);
        }
    }
}

/// Compute glyph advances with typography settings applied.
pub fn compute_advances_with_typography(
    text: &str,
    base_advances: &[f32],
    font_size: f32,
    settings: &TypographySettings,
    kerning_table: Option<&KerningTable>,
) -> Vec<f32> {
    let chars: Vec<char> = text.chars().collect();
    let mut advances = Vec::with_capacity(chars.len());

    for (i, &ch) in chars.iter().enumerate() {
        let base = if i < base_advances.len() {
            base_advances[i]
        } else {
            0.5 // Fallback
        };

        let mut advance = base * font_size;

        // Apply word spacing for spaces
        if ch == ' ' {
            advance *= settings.word_spacing;
        }

        // Apply tracking
        advance += settings.tracking * font_size;

        // Apply kerning
        if settings.kerning && i < chars.len() - 1 {
            if let Some(table) = kerning_table {
                let kern = table.get_kerning(ch, chars[i + 1]);
                advance += kern * font_size;
            }
        }

        advances.push(advance);
    }

    advances
}

/// Text case transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextCase {
    /// No transformation.
    #[default]
    None,
    /// ALL UPPERCASE.
    Upper,
    /// all lowercase.
    Lower,
    /// Title Case (First Letter Of Each Word).
    Title,
    /// SMALL CAPS (uppercase with reduced size for lowercase).
    SmallCaps,
}

/// Apply text case transformation.
pub fn apply_text_case(text: &str, case: TextCase) -> String {
    match case {
        TextCase::None => text.to_string(),
        TextCase::Upper => text.to_uppercase(),
        TextCase::Lower => text.to_lowercase(),
        TextCase::Title => {
            let mut result = String::with_capacity(text.len());
            let mut capitalize_next = true;
            for ch in text.chars() {
                if ch.is_whitespace() {
                    capitalize_next = true;
                    result.push(ch);
                } else if capitalize_next {
                    result.extend(ch.to_uppercase());
                    capitalize_next = false;
                } else {
                    result.push(ch);
                }
            }
            result
        }
        TextCase::SmallCaps => {
            // For small caps, we return uppercase but the renderer
            // should scale lowercase letters
            text.to_uppercase()
        }
    }
}

/// Check if a character should be rendered at reduced size for small caps.
pub fn is_small_cap_char(original: char, transformed: char) -> bool {
    original.is_lowercase() && transformed.is_uppercase()
}

/// Small caps scale factor for lowercase letters rendered as uppercase.
pub const SMALL_CAPS_SCALE: f32 = 0.8;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typography_defaults() {
        let settings = TypographySettings::default();
        assert_eq!(settings.tracking, 0.0);
        assert!(settings.kerning);
    }

    #[test]
    fn test_text_case_upper() {
        assert_eq!(apply_text_case("Hello World", TextCase::Upper), "HELLO WORLD");
    }

    #[test]
    fn test_text_case_title() {
        assert_eq!(apply_text_case("hello world", TextCase::Title), "Hello World");
    }

    #[test]
    fn test_kerning_table() {
        let mut table = KerningTable::new();
        table.add_pair('A', 'V', -0.1);
        assert!((table.get_kerning('A', 'V') - (-0.1)).abs() < 0.001);
        assert_eq!(table.get_kerning('X', 'Y'), 0.0);
    }
}
