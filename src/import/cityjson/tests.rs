use super::*;

fn sample_cityjson() -> &'static [u8] {
    br#"{
        "type": "CityJSON",
        "version": "1.1",
        "transform": {
            "scale": [0.001, 0.001, 0.001],
            "translate": [0.0, 0.0, 0.0]
        },
        "vertices": [
            [0, 0, 0], [10000, 0, 0], [10000, 10000, 0], [0, 10000, 0],
            [0, 0, 5000], [10000, 0, 5000], [10000, 10000, 5000], [0, 10000, 5000]
        ],
        "CityObjects": {
            "building1": {
                "type": "Building",
                "attributes": { "measuredHeight": 5.0 },
                "geometry": [{
                    "type": "Solid",
                    "lod": "1",
                    "boundaries": [[
                        [[0, 1, 2, 3]], [[4, 5, 6, 7]], [[0, 1, 5, 4]],
                        [[1, 2, 6, 5]], [[2, 3, 7, 6]], [[3, 0, 4, 7]]
                    ]]
                }]
            }
        }
    }"#
}

#[test]
fn test_parse_simple_cityjson() {
    let (buildings, meta) = parse_cityjson(sample_cityjson()).unwrap();

    assert_eq!(meta.version, "1.1");
    assert_eq!(meta.scale, [0.001, 0.001, 0.001]);
    assert_eq!(buildings.len(), 1);

    let building = &buildings[0];
    assert_eq!(building.id, "building1");
    assert_eq!(building.lod, 1);
    assert_eq!(building.height, Some(5.0));
    assert!(building.vertex_count() > 0);
    assert!(building.triangle_count() > 0);
}

#[test]
fn test_invalid_cityjson() {
    assert!(parse_cityjson(b"not json").is_err());
    assert!(parse_cityjson(br#"{"type": "NotCityJSON"}"#).is_err());
}

#[test]
fn test_cityjson_surface_with_hole_preserves_void() {
    let data = br#"{
        "type": "CityJSON",
        "version": "1.1",
        "vertices": [
            [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
            [4, 4, 0], [6, 4, 0], [6, 6, 0], [4, 6, 0]
        ],
        "CityObjects": {
            "courtyard": {
                "type": "Building",
                "geometry": [{
                    "type": "MultiSurface",
                    "lod": "2",
                    "boundaries": [
                        [[0, 1, 2, 3], [4, 5, 6, 7]]
                    ]
                }]
            }
        }
    }"#;

    let (buildings, _) = parse_cityjson(data).unwrap();
    let building = &buildings[0];

    assert!(building.vertex_count() >= 8);
    assert!(building.triangle_count() >= 4);
    for tri in building.indices.chunks(3) {
        let centroid = tri.iter().fold([0.0f32; 2], |mut acc, index| {
            let base = *index as usize * 3;
            acc[0] += building.positions[base] / 3.0;
            acc[1] += building.positions[base + 1] / 3.0;
            acc
        });
        assert!(
            centroid[0] <= 4.0 || centroid[0] >= 6.0 || centroid[1] <= 4.0 || centroid[1] >= 6.0,
            "triangle centroid {:?} fell inside the CityJSON interior ring",
            centroid
        );
    }
}
