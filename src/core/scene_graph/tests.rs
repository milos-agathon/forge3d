use super::*;
use glam::Vec4Swizzles;

#[test]
fn test_scene_graph_basic_operations() {
    let mut graph = SceneGraph::new();

    let root = graph.create_node("root".to_string());
    let child1 = graph.create_node("child1".to_string());
    let child2 = graph.create_node("child2".to_string());

    assert_eq!(graph.node_count(), 3);
    assert_eq!(graph.root_count(), 3);

    graph.add_child(root, child1).unwrap();
    graph.add_child(root, child2).unwrap();

    assert_eq!(graph.root_count(), 1);

    let root_node = graph.get_node(root).unwrap();
    assert_eq!(root_node.children.len(), 2);

    let child1_node = graph.get_node(child1).unwrap();
    assert_eq!(child1_node.parent, Some(root));
}

#[test]
fn test_scene_graph_transform_inheritance() {
    let mut graph = SceneGraph::new();

    let root = graph.create_node("root".to_string());
    let child = graph.create_node("child".to_string());

    graph
        .get_node_mut(root)
        .unwrap()
        .local_transform
        .set_translation(glam::Vec3::new(1.0, 0.0, 0.0));
    graph
        .get_node_mut(child)
        .unwrap()
        .local_transform
        .set_translation(glam::Vec3::new(0.0, 1.0, 0.0));

    graph.add_child(root, child).unwrap();
    graph.update_transforms().unwrap();

    let root_node = graph.get_node(root).unwrap();
    let child_node = graph.get_node(child).unwrap();

    assert_eq!(
        root_node.world_matrix.w_axis.xyz(),
        glam::Vec3::new(1.0, 0.0, 0.0)
    );
    assert_eq!(
        child_node.world_matrix.w_axis.xyz(),
        glam::Vec3::new(1.0, 1.0, 0.0)
    );
}

#[test]
fn test_scene_graph_cycle_detection() {
    let mut graph = SceneGraph::new();

    let node1 = graph.create_node("node1".to_string());
    let node2 = graph.create_node("node2".to_string());
    let node3 = graph.create_node("node3".to_string());

    graph.add_child(node1, node2).unwrap();
    graph.add_child(node2, node3).unwrap();

    assert!(graph.add_child(node3, node1).is_err());
}

#[test]
fn test_scene_graph_removal() {
    let mut graph = SceneGraph::new();

    let root = graph.create_node("root".to_string());
    let child1 = graph.create_node("child1".to_string());
    let grandchild = graph.create_node("grandchild".to_string());

    graph.add_child(root, child1).unwrap();
    graph.add_child(child1, grandchild).unwrap();

    assert_eq!(graph.node_count(), 3);

    graph.remove_node(child1).unwrap();

    assert_eq!(graph.node_count(), 1);
    assert!(!graph.nodes.contains_key(&child1));
    assert!(!graph.nodes.contains_key(&grandchild));
}
