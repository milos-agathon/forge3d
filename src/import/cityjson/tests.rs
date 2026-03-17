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
