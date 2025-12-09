//! HUD rendering helpers for the interactive viewer
//!
//! Provides seven-segment numeric displays and block text rendering
//! for on-screen statistics and debug overlays.

use crate::core::text_overlay::TextInstance;

/// Push a solid colored rectangle to the HUD instance buffer
pub fn push_rect(
    inst: &mut Vec<TextInstance>,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    color: [f32; 4],
) {
    inst.push(TextInstance {
        rect_min: [x0, y0],
        rect_max: [x1, y1],
        uv_min: [0.0, 0.0],
        uv_max: [1.0, 1.0],
        color,
    });
}

/// Push a single 3x5 block character (A-Z subset)
pub fn push_char_3x5(
    inst: &mut Vec<TextInstance>,
    x: f32,
    y: f32,
    scale: f32,
    ch: char,
    color: [f32; 4],
) -> f32 {
    let cell = 2.0 * scale; // pixel size
    let spacing = 1.0 * scale; // inter-char spacing
    let pat: Option<[&str; 5]> = match ch.to_ascii_uppercase() {
        'A' => Some([" X ", "X X", "XXX", "X X", "X X"]),
        'B' => Some(["XX ", "X X", "XX ", "X X", "XX "]),
        'C' => Some([" XX", "X  ", "X  ", "X  ", " XX"]),
        'D' => Some(["XX ", "X X", "X X", "X X", "XX "]),
        'E' => Some(["XXX", "X  ", "XX ", "X  ", "XXX"]),
        'F' => Some(["XXX", "X  ", "XX ", "X  ", "X  "]),
        'G' => Some([" XX", "X  ", "X X", "X X", " XX"]),
        'H' => Some(["X X", "X X", "XXX", "X X", "X X"]),
        'K' => Some(["X X", "XX ", "X  ", "XX ", "X X"]),
        'L' => Some(["X  ", "X  ", "X  ", "X  ", "XXX"]),
        'N' => Some(["X X", "XX ", "X X", "X X", "X X"]),
        'O' => Some(["XXX", "X X", "X X", "X X", "XXX"]),
        'P' => Some(["XX ", "X X", "XX ", "X  ", "X  "]),
        'R' => Some(["XX ", "X X", "XX ", "X X", "X X"]),
        'S' => Some([" XX", "X  ", " XX", "  X", "XX "]),
        'T' => Some(["XXX", " X ", " X ", " X ", " X "]),
        'U' => Some(["X X", "X X", "X X", "X X", "XXX"]),
        'Y' => Some(["X X", "X X", " X ", " X ", " X "]),
        _ => None,
    };
    if let Some(rows) = pat {
        for (r, row) in rows.iter().enumerate() {
            for (c, ch2) in row.chars().enumerate() {
                if ch2 == 'X' {
                    let x0 = x + c as f32 * cell;
                    let y0 = y + r as f32 * cell;
                    push_rect(inst, x0, y0, x0 + cell, y0 + cell, color);
                }
            }
        }
    }
    3.0 * cell + spacing
}

/// Push a string of 3x5 block characters
pub fn push_text_3x5(
    inst: &mut Vec<TextInstance>,
    mut x: f32,
    y: f32,
    scale: f32,
    text: &str,
    color: [f32; 4],
) -> f32 {
    for ch in text.chars() {
        if ch == ' ' {
            x += 2.0 * scale;
            continue;
        }
        x += push_char_3x5(inst, x, y, scale, ch, color);
    }
    x
}

/// Push a seven-segment style digit (0-9, '-', '.')
pub fn push_digit(
    inst: &mut Vec<TextInstance>,
    x: f32,
    y: f32,
    scale: f32,
    ch: char,
    color: [f32; 4],
) -> f32 {
    // 7-segment layout (a..g), plus dot segment 'dp'
    //  --a--
    // |     |
    // f     b
    // |     |
    //  --g--
    // |     |
    // e     c
    // |     |
    //  --d--   . dp
    let thick = 2.0 * scale;
    let w = 10.0 * scale; // char width
    let h = 18.0 * scale; // char height
    let mut seg = |a: bool, b: bool, c: bool, d: bool, e: bool, f: bool, g: bool, dp: bool| {
        if a {
            push_rect(inst, x + thick, y, x + w - thick, y + thick, color);
        }
        if b {
            push_rect(
                inst,
                x + w - thick,
                y + thick,
                x + w,
                y + h / 2.0 - thick,
                color,
            );
        }
        if c {
            push_rect(
                inst,
                x + w - thick,
                y + h / 2.0 + thick,
                x + w,
                y + h - thick,
                color,
            );
        }
        if d {
            push_rect(inst, x + thick, y + h - thick, x + w - thick, y + h, color);
        }
        if e {
            push_rect(
                inst,
                x,
                y + h / 2.0 + thick,
                x + thick,
                y + h - thick,
                color,
            );
        }
        if f {
            push_rect(inst, x, y + thick, x + thick, y + h / 2.0 - thick, color);
        }
        if g {
            push_rect(
                inst,
                x + thick,
                y + h / 2.0 - thick / 2.0,
                x + w - thick,
                y + h / 2.0 + thick / 2.0,
                color,
            );
        }
        if dp {
            push_rect(
                inst,
                x + w + thick * 0.5,
                y + h - thick * 1.5,
                x + w + thick * 1.5,
                y + h - thick * 0.5,
                color,
            );
        }
    };
    match ch {
        '0' => seg(true, true, true, true, true, true, false, false),
        '1' => seg(false, true, true, false, false, false, false, false),
        '2' => seg(true, true, false, true, true, false, true, false),
        '3' => seg(true, true, true, true, false, false, true, false),
        '4' => seg(false, true, true, false, false, true, true, false),
        '5' => seg(true, false, true, true, false, true, true, false),
        '6' => seg(true, false, true, true, true, true, true, false),
        '7' => seg(true, true, true, false, false, false, false, false),
        '8' => seg(true, true, true, true, true, true, true, false),
        '9' => seg(true, true, true, true, false, true, true, false),
        '-' => {
            // center segment only
            seg(false, false, false, false, false, false, true, false);
        }
        '.' => {
            seg(false, false, false, false, false, false, false, true);
        }
        _ => {}
    }
    w + 4.0 * scale // advance including small spacing
}

/// Push a formatted number using seven-segment digits
pub fn push_number(
    inst: &mut Vec<TextInstance>,
    mut x: f32,
    y: f32,
    scale: f32,
    value: f32,
    digits: usize,
    frac: usize,
    color: [f32; 4],
) -> f32 {
    let s = format!("{val:.prec$}", val = value, prec = frac);
    // Optionally truncate/limit total characters
    let mut count = 0usize;
    for ch in s.chars() {
        if count >= digits + 1 {
            break;
        }
        x += push_digit(inst, x, y, scale, ch, color);
        count += 1;
    }
    x
}
