use glam::Vec2;

use super::*;

#[test]
fn test_vector_api_basic() {
    let mut api = VectorApi::new();

    let polygon = PolygonDef {
        exterior: vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.5, 1.0),
        ],
        holes: vec![],
        style: VectorStyle::default(),
    };

    let ids = api.add_polygons(vec![polygon], CrsType::Planar).unwrap();
    assert_eq!(ids.len(), 1);
    assert_eq!(api.get_counts(), (1, 0, 0, 0));
}

#[test]
fn test_crs_validation() {
    let mut api = VectorApi::new();

    let polygon = PolygonDef {
        exterior: vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(0.5, 1.0),
        ],
        holes: vec![],
        style: VectorStyle::default(),
    };

    let result = api.add_polygons(vec![polygon], CrsType::WebMercator);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("planar CRS"));
}

#[test]
fn test_polygon_validation() {
    let mut api = VectorApi::new();

    let invalid_polygon = PolygonDef {
        exterior: vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)],
        holes: vec![],
        style: VectorStyle::default(),
    };

    let result = api.add_polygons(vec![invalid_polygon], CrsType::Planar);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("at least 3 vertices"));
}

#[test]
fn test_graph_validation() {
    let mut api = VectorApi::new();

    let valid_graph = GraphDef {
        nodes: vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0)],
        edges: vec![(0, 1)],
        node_style: VectorStyle::default(),
        edge_style: VectorStyle::default(),
    };

    let id = api.add_graph(valid_graph, CrsType::Planar).unwrap();
    assert!(id.0 > 0);

    let invalid_graph = GraphDef {
        nodes: vec![Vec2::new(0.0, 0.0)],
        edges: vec![(0, 1)],
        node_style: VectorStyle::default(),
        edge_style: VectorStyle::default(),
    };

    let result = api.add_graph(invalid_graph, CrsType::Planar);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("exceeds node count"));
}
