// Unicode 17.0 emoji-data.txt Extended_Pictographic ranges whose code points are
// reserved (General_Category=Cn). SHA-256:
// 2cb2bb9455cda83e8481541ecf5b6dfda66a3bb89efa3fa7c5297eccf607b72b

pub(super) fn is_unassigned_extended_pictographic(character: char) -> bool {
    matches!(
        character as u32,
        0x1f02c..=0x1f02f
            | 0x1f094..=0x1f09f
            | 0x1f0af..=0x1f0b0
            | 0x1f0c0
            | 0x1f0d0
            | 0x1f0f6..=0x1f0ff
            | 0x1f1ae..=0x1f1e5
            | 0x1f203..=0x1f20f
            | 0x1f23c..=0x1f23f
            | 0x1f249..=0x1f24f
            | 0x1f252..=0x1f25f
            | 0x1f266..=0x1f2ff
            | 0x1f6d9..=0x1f6db
            | 0x1f6ed..=0x1f6ef
            | 0x1f6fd..=0x1f6ff
            | 0x1f7da..=0x1f7df
            | 0x1f7ec..=0x1f7ef
            | 0x1f7f1..=0x1f7ff
            | 0x1f80c..=0x1f80f
            | 0x1f848..=0x1f84f
            | 0x1f85a..=0x1f85f
            | 0x1f888..=0x1f88f
            | 0x1f8ae..=0x1f8af
            | 0x1f8bc..=0x1f8bf
            | 0x1f8c2..=0x1f8cf
            | 0x1f8d9..=0x1f8ff
            | 0x1fa58..=0x1fa5f
            | 0x1fa6e..=0x1fa6f
            | 0x1fa7d..=0x1fa7f
            | 0x1fa8b..=0x1fa8d
            | 0x1fac7
            | 0x1fac9..=0x1facc
            | 0x1fadd..=0x1fade
            | 0x1faeb..=0x1faee
            | 0x1faf9..=0x1faff
            | 0x1fc00..=0x1fffd
    )
}
