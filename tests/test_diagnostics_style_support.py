from forge3d.style import parse_style, validate_style_support
from forge3d.terrain_params import PrimitiveType, VectorOverlayConfig


def test_style_support_accepts_local_fill_line_and_circle_layers():
    style = {
        "version": 8,
        "layers": [
            {"id": "land", "type": "fill", "paint": {"fill-color": "#447744"}},
            {"id": "roads", "type": "line", "paint": {"line-color": "#ffffff", "line-width": 2}},
            {"id": "cities", "type": "circle", "paint": {"circle-color": "#ff0000", "circle-radius": 4}},
        ],
    }

    report = validate_style_support(style)

    assert report.status == "ok"
    assert report.render_blocked() is False
    assert report.supported_features == {
        "style.layer.circle": "supported",
        "style.layer.fill": "supported",
        "style.layer.line": "supported",
        "style.local_provided_features": "supported",
    }


def test_style_support_reports_unsupported_layer_types_and_fields():
    style = {
        "version": 8,
        "layers": [
            {
                "id": "roads",
                "type": "line",
                "paint": {"line-color": "#ffffff", "line-gradient": ["get", "speed"]},
                "layout": {"line-sort-key": 10},
            },
            {"id": "heat", "type": "heatmap", "paint": {"heatmap-radius": 12}},
        ],
    }

    report = validate_style_support(style)
    payloads = [diag.to_dict() for diag in report.diagnostics]

    assert report.status == "error"
    assert [payload["code"] for payload in payloads] == [
        "unsupported_style_layer_type",
        "unsupported_style_field",
        "unsupported_style_field",
    ]
    assert payloads[0]["layer_id"] == "heat"
    assert payloads[0]["details"]["layer_type"] == "heatmap"
    assert payloads[1]["layer_id"] == "roads"
    assert payloads[1]["details"]["fields"] == ["line-gradient"]
    assert payloads[2]["details"]["fields"] == ["line-sort-key"]


def test_style_support_accepts_parsed_style_spec_without_claiming_mvt():
    parsed = parse_style(
        {
            "version": 8,
            "layers": [
                {"id": "water", "type": "fill", "paint": {"fill-opacity": 0.5}},
            ],
        }
    )

    report = validate_style_support(parsed)

    assert report.status == "ok"
    assert report.supported_features["style.local_provided_features"] == "supported"
    assert report.unsupported_features["style.streamed_mvt"] == "non-goal"


def test_parsed_style_support_reports_preserved_unsupported_fields():
    parsed = parse_style(
        {
            "version": 8,
            "layers": [
                {
                    "id": "roads",
                    "type": "line",
                    "paint": {"line-color": "#ffffff", "line-gradient": ["get", "speed"]},
                    "layout": {"visibility": "visible", "line-sort-key": 10},
                },
            ],
        }
    )

    report = validate_style_support(parsed)
    payloads = [diag.to_dict() for diag in report.diagnostics]

    assert report.status == "warning"
    assert [payload["code"] for payload in payloads] == [
        "unsupported_style_field",
        "unsupported_style_field",
    ]
    assert payloads[0]["layer_id"] == "roads"
    assert payloads[0]["details"] == {"fields": ["line-gradient"], "section": "paint"}
    assert payloads[1]["layer_id"] == "roads"
    assert payloads[1]["details"] == {"fields": ["line-sort-key"], "section": "layout"}


def test_symbol_style_support_matches_underdeveloped_matrix_truth():
    report = validate_style_support(
        {
            "version": 8,
            "layers": [
                {
                    "id": "place-labels",
                    "type": "symbol",
                    "source-layer": "place_label",
                    "layout": {"text-field": "{name}", "text-size": 14},
                    "paint": {"text-color": "#333333"},
                }
            ],
        }
    )

    assert report.status == "warning"
    assert [(diag.code, diag.support_level, diag.layer_id, diag.details["feature"]) for diag in report.diagnostics] == [
        ("experimental_feature", "experimental", "place-labels", "symbol text layer")
    ]
    assert report.layer_summaries[0].support_level == "underdeveloped"
    assert report.unsupported_features["style.layer.symbol"] == "underdeveloped"


def test_supported_style_output_feeds_vector_overlay_config():
    from forge3d.style import vector_overlay_configs_from_style

    style = {
        "version": 8,
        "layers": [
            {"id": "roads", "type": "line", "paint": {"line-color": "#ff0000", "line-width": 3}},
            {"id": "cities", "type": "circle", "paint": {"circle-color": "#00ff00", "circle-radius": 5}},
        ],
    }
    features = [
        {
            "type": "Feature",
            "properties": {"kind": "road"},
            "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1], [2, 1]]},
        },
        {
            "type": "Feature",
            "properties": {"kind": "city"},
            "geometry": {"type": "Point", "coordinates": [3, 4]},
        },
    ]

    overlays = vector_overlay_configs_from_style(style, features, name_prefix="styled")

    assert all(isinstance(overlay, VectorOverlayConfig) for overlay in overlays)
    assert [overlay.primitive for overlay in overlays] == [PrimitiveType.LINES, PrimitiveType.POINTS]
    assert overlays[0].line_width == 3.0
    assert overlays[0].vertices[0].r == 1.0
    assert overlays[1].point_size == 10.0
    assert overlays[1].vertices[0].g == 1.0
    assert overlays[0].to_ipc_dict()["cmd"] == "add_vector_overlay"


def test_symbol_style_output_feeds_future_label_layer_contract():
    from forge3d.style import label_layer_contracts_from_style

    contracts = label_layer_contracts_from_style(
        {
            "version": 8,
            "layers": [
                {
                    "id": "place-labels",
                    "type": "symbol",
                    "source-layer": "place_label",
                    "layout": {"text-field": "{name}", "text-size": 14},
                    "paint": {"text-color": "#333333", "text-halo-color": "#ffffff"},
                }
            ],
        }
    )

    assert contracts == [
        {
            "layer_id": "place-labels",
            "source_layer": "place_label",
            "text_field": "{name}",
            "support_level": "underdeveloped",
            "label_style": {
                "size": 14.0,
                "color": (0.2, 0.2, 0.2, 1.0),
                "halo_color": (1.0, 1.0, 1.0, 1.0),
                "halo_width": 1.5,
                "offset": (0.0, 0.0),
            },
        }
    ]
