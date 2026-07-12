use super::value::device_delta;
use super::PositioningGlyph;
use ttf_parser::gpos::Anchor;
use ttf_parser::Face;

pub(super) fn attach(
    face: &Face<'_>,
    buffer: &mut [PositioningGlyph],
    mark: usize,
    parent: usize,
    mark_anchor: Anchor<'_>,
    parent_anchor: Anchor<'_>,
    component: Option<u16>,
) -> Result<(), super::TextError> {
    let overflow = || super::TextError::MalformedOpenType("GPOS positioning overflow");
    let origin = |end: usize, offset: i32, vertical: bool| {
        buffer[..end].iter().try_fold(offset, |sum, glyph| {
            sum.checked_add(if vertical {
                glyph.y_advance
            } else {
                glyph.x_advance
            })
            .ok_or_else(overflow)
        })
    };
    let parent_x = origin(parent, buffer[parent].x_offset, false)?;
    let parent_y = origin(parent, buffer[parent].y_offset, true)?;
    let mark_x = origin(mark, buffer[mark].x_offset, false)?;
    let mark_y = origin(mark, buffer[mark].y_offset, true)?;
    let parent_anchor_x = i32::from(parent_anchor.x)
        .checked_add(device_delta(face, parent_anchor.x_device)?)
        .ok_or_else(overflow)?;
    let parent_anchor_y = i32::from(parent_anchor.y)
        .checked_add(device_delta(face, parent_anchor.y_device)?)
        .ok_or_else(overflow)?;
    let mark_anchor_x = i32::from(mark_anchor.x)
        .checked_add(device_delta(face, mark_anchor.x_device)?)
        .ok_or_else(overflow)?;
    let mark_anchor_y = i32::from(mark_anchor.y)
        .checked_add(device_delta(face, mark_anchor.y_device)?)
        .ok_or_else(overflow)?;
    let x_delta = parent_x
        .checked_add(parent_anchor_x)
        .and_then(|value| value.checked_sub(mark_x))
        .and_then(|value| value.checked_sub(mark_anchor_x))
        .ok_or_else(overflow)?;
    let y_delta = parent_y
        .checked_add(parent_anchor_y)
        .and_then(|value| value.checked_sub(mark_y))
        .and_then(|value| value.checked_sub(mark_anchor_y))
        .ok_or_else(overflow)?;
    buffer[mark].x_offset = buffer[mark]
        .x_offset
        .checked_add(x_delta)
        .ok_or_else(overflow)?;
    buffer[mark].y_offset = buffer[mark]
        .y_offset
        .checked_add(y_delta)
        .ok_or_else(overflow)?;
    buffer[mark].x_advance = 0;
    buffer[mark].y_advance = 0;
    buffer[mark].attached_to = Some(parent);
    buffer[mark].ligature_component = component;
    Ok(())
}
