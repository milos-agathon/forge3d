// src/picking/selection.rs
// Multi-select and selection set management
// Part of Plan 2: Standard — GPU Ray Picking + Hover Support

use std::collections::{HashMap, HashSet};

/// Highlight style for a selection set
#[derive(Debug, Clone)]
pub struct SelectionStyle {
    /// Highlight color (RGBA)
    pub color: [f32; 4],
    /// Whether to show outline
    pub outline: bool,
    /// Outline width in pixels
    pub outline_width: f32,
    /// Whether to show glow effect
    pub glow: bool,
    /// Glow intensity (0.0 - 1.0)
    pub glow_intensity: f32,
}

impl Default for SelectionStyle {
    fn default() -> Self {
        Self {
            color: [1.0, 0.8, 0.0, 0.5], // Yellow semi-transparent
            outline: false,
            outline_width: 2.0,
            glow: false,
            glow_intensity: 0.5,
        }
    }
}

impl SelectionStyle {
    /// Create a new selection style with the given color
    pub fn with_color(color: [f32; 4]) -> Self {
        Self {
            color,
            ..Default::default()
        }
    }

    /// Enable outline rendering
    pub fn with_outline(mut self, width: f32) -> Self {
        self.outline = true;
        self.outline_width = width;
        self
    }

    /// Enable glow effect
    pub fn with_glow(mut self, intensity: f32) -> Self {
        self.glow = true;
        self.glow_intensity = intensity;
        self
    }
}

/// A named selection set containing feature IDs
#[derive(Debug, Clone)]
pub struct SelectionSet {
    /// Name of the selection set
    pub name: String,
    /// Selected feature IDs
    pub features: HashSet<u32>,
    /// Visual style for this selection
    pub style: SelectionStyle,
    /// Whether this selection is visible
    pub visible: bool,
}

impl SelectionSet {
    /// Create a new empty selection set
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            features: HashSet::new(),
            style: SelectionStyle::default(),
            visible: true,
        }
    }

    /// Create with a specific style
    pub fn with_style(name: impl Into<String>, style: SelectionStyle) -> Self {
        Self {
            name: name.into(),
            features: HashSet::new(),
            style,
            visible: true,
        }
    }

    /// Add a feature to the selection
    pub fn add(&mut self, feature_id: u32) {
        self.features.insert(feature_id);
    }

    /// Add multiple features
    pub fn add_many(&mut self, feature_ids: impl IntoIterator<Item = u32>) {
        self.features.extend(feature_ids);
    }

    /// Remove a feature from the selection
    pub fn remove(&mut self, feature_id: u32) {
        self.features.remove(&feature_id);
    }

    /// Toggle a feature in the selection
    pub fn toggle(&mut self, feature_id: u32) {
        if self.features.contains(&feature_id) {
            self.features.remove(&feature_id);
        } else {
            self.features.insert(feature_id);
        }
    }

    /// Check if a feature is selected
    pub fn contains(&self, feature_id: u32) -> bool {
        self.features.contains(&feature_id)
    }

    /// Clear all selections
    pub fn clear(&mut self) {
        self.features.clear();
    }

    /// Get number of selected features
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Check if selection is empty
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Get all selected feature IDs
    pub fn feature_ids(&self) -> Vec<u32> {
        self.features.iter().copied().collect()
    }
}

/// Manager for multiple selection sets
#[derive(Debug, Default)]
pub struct SelectionManager {
    /// Named selection sets
    sets: HashMap<String, SelectionSet>,
    /// Primary selection set name
    primary_set: String,
    /// Whether multi-select mode is enabled
    multi_select: bool,
    /// Hover feature ID (not part of any set, just for highlighting)
    hover_feature: Option<u32>,
}

impl SelectionManager {
    /// Create a new selection manager
    pub fn new() -> Self {
        let mut manager = Self {
            sets: HashMap::new(),
            primary_set: "primary".to_string(),
            multi_select: false,
            hover_feature: None,
        };
        
        // Create default primary selection set
        manager.create_set("primary");
        manager
    }

    /// Create a new selection set
    pub fn create_set(&mut self, name: impl Into<String>) -> &mut SelectionSet {
        let name = name.into();
        self.sets.entry(name.clone()).or_insert_with(|| SelectionSet::new(name.clone()));
        self.sets.get_mut(&name).unwrap()
    }

    /// Create a selection set with a specific style
    pub fn create_set_with_style(
        &mut self,
        name: impl Into<String>,
        style: SelectionStyle,
    ) -> &mut SelectionSet {
        let name = name.into();
        self.sets.insert(name.clone(), SelectionSet::with_style(name.clone(), style));
        self.sets.get_mut(&name).unwrap()
    }

    /// Get a selection set by name
    pub fn get_set(&self, name: &str) -> Option<&SelectionSet> {
        self.sets.get(name)
    }

    /// Get a mutable selection set by name
    pub fn get_set_mut(&mut self, name: &str) -> Option<&mut SelectionSet> {
        self.sets.get_mut(name)
    }

    /// Remove a selection set
    pub fn remove_set(&mut self, name: &str) {
        if name != "primary" {
            self.sets.remove(name);
        }
    }

    /// Get the primary selection set
    pub fn primary(&self) -> Option<&SelectionSet> {
        self.sets.get(&self.primary_set)
    }

    /// Get the primary selection set mutably
    pub fn primary_mut(&mut self) -> Option<&mut SelectionSet> {
        self.sets.get_mut(&self.primary_set)
    }

    /// Set the primary selection set
    pub fn set_primary(&mut self, name: impl Into<String>) {
        self.primary_set = name.into();
    }

    /// Enable or disable multi-select mode
    pub fn set_multi_select(&mut self, enabled: bool) {
        self.multi_select = enabled;
    }

    /// Check if multi-select is enabled
    pub fn is_multi_select(&self) -> bool {
        self.multi_select
    }

    /// Handle a pick event (single click)
    /// If multi-select is disabled, clears existing selection first
    /// If multi-select is enabled with shift, toggles the feature
    pub fn handle_pick(&mut self, feature_id: u32, shift_held: bool) {
        if let Some(set) = self.sets.get_mut(&self.primary_set) {
            if self.multi_select && shift_held {
                set.toggle(feature_id);
            } else {
                set.clear();
                if feature_id != 0 {
                    set.add(feature_id);
                }
            }
        }
    }

    /// Add feature to a specific selection set
    pub fn add_to_set(&mut self, set_name: &str, feature_id: u32) {
        if let Some(set) = self.sets.get_mut(set_name) {
            set.add(feature_id);
        }
    }

    /// Add multiple features to a specific selection set
    pub fn add_many_to_set(&mut self, set_name: &str, feature_ids: impl IntoIterator<Item = u32>) {
        if let Some(set) = self.sets.get_mut(set_name) {
            set.add_many(feature_ids);
        }
    }

    /// Remove feature from a specific selection set
    pub fn remove_from_set(&mut self, set_name: &str, feature_id: u32) {
        if let Some(set) = self.sets.get_mut(set_name) {
            set.remove(feature_id);
        }
    }

    /// Clear a specific selection set
    pub fn clear_set(&mut self, set_name: &str) {
        if let Some(set) = self.sets.get_mut(set_name) {
            set.clear();
        }
    }

    /// Clear all selection sets
    pub fn clear_all(&mut self) {
        for set in self.sets.values_mut() {
            set.clear();
        }
    }

    /// Get all selected feature IDs from the primary set
    pub fn get_selection(&self) -> Vec<u32> {
        self.sets.get(&self.primary_set)
            .map(|s| s.feature_ids())
            .unwrap_or_default()
    }

    /// Get all selected feature IDs from a specific set
    pub fn get_selection_from(&self, set_name: &str) -> Vec<u32> {
        self.sets.get(set_name)
            .map(|s| s.feature_ids())
            .unwrap_or_default()
    }

    /// Check if a feature is selected in any visible set
    pub fn is_selected(&self, feature_id: u32) -> bool {
        self.sets.values()
            .filter(|s| s.visible)
            .any(|s| s.contains(feature_id))
    }

    /// Get the style for a selected feature (from the first matching set)
    pub fn get_style_for(&self, feature_id: u32) -> Option<&SelectionStyle> {
        self.sets.values()
            .filter(|s| s.visible && s.contains(feature_id))
            .map(|s| &s.style)
            .next()
    }

    /// Set hover feature (for hover highlighting)
    pub fn set_hover(&mut self, feature_id: Option<u32>) {
        self.hover_feature = feature_id;
    }

    /// Get hover feature
    pub fn hover_feature(&self) -> Option<u32> {
        self.hover_feature
    }

    /// Get all visible selection sets
    pub fn visible_sets(&self) -> impl Iterator<Item = &SelectionSet> {
        self.sets.values().filter(|s| s.visible)
    }

    /// Get all set names
    pub fn set_names(&self) -> Vec<&str> {
        self.sets.keys().map(|s| s.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selection_set_operations() {
        let mut set = SelectionSet::new("test");
        
        set.add(1);
        set.add(2);
        assert!(set.contains(1));
        assert!(set.contains(2));
        assert_eq!(set.len(), 2);

        set.remove(1);
        assert!(!set.contains(1));
        assert_eq!(set.len(), 1);

        set.toggle(2);
        assert!(!set.contains(2));
        set.toggle(2);
        assert!(set.contains(2));
    }

    #[test]
    fn test_selection_manager() {
        let mut manager = SelectionManager::new();
        
        // Single select mode
        manager.handle_pick(1, false);
        assert_eq!(manager.get_selection(), vec![1]);
        
        manager.handle_pick(2, false);
        assert_eq!(manager.get_selection(), vec![2]);

        // Multi-select mode
        manager.set_multi_select(true);
        manager.handle_pick(3, true);
        let selection = manager.get_selection();
        assert!(selection.contains(&2));
        assert!(selection.contains(&3));
    }
}
