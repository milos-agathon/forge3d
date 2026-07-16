//! Allocation-free temporal-history validity shared by viewer effects.

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct TemporalHistoryState {
    valid: bool,
}

impl TemporalHistoryState {
    pub(crate) const fn invalid() -> Self {
        Self { valid: false }
    }

    pub(crate) const fn is_valid(self) -> bool {
        self.valid
    }

    pub(crate) fn invalidate(&mut self) {
        self.valid = false;
    }

    pub(crate) fn mark_populated(&mut self) {
        self.valid = true;
    }

    pub(crate) fn blend_alpha(self, configured: f32) -> f32 {
        if self.valid {
            configured
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TemporalHistoryState;

    #[test]
    fn invalidation_bypasses_one_frame_then_reuses_new_history() {
        let mut state = TemporalHistoryState::invalid();
        for _ in 0..10 {
            assert_eq!(state.blend_alpha(0.85), 0.0);
            state.mark_populated();
            assert_eq!(state.blend_alpha(0.85), 0.85);
            state.invalidate();
        }
    }
}
