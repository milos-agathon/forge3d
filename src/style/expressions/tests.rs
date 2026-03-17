use super::*;

fn empty_props() -> serde_json::Map<String, Value> {
    serde_json::Map::new()
}

fn props_with(key: &str, val: Value) -> serde_json::Map<String, Value> {
    let mut m = serde_json::Map::new();
    m.insert(key.to_string(), val);
    m
}

#[test]
fn test_get_property() {
    let props = props_with("name", Value::String("Test".to_string()));
    let ctx = EvalContext::new(&props, 10.0);

    let expr = serde_json::json!(["get", "name"]);
    let result = evaluate_expression(&expr, &ctx);
    assert_eq!(result, Some(Value::String("Test".to_string())));
}

#[test]
fn test_zoom() {
    let props = empty_props();
    let ctx = EvalContext::new(&props, 12.5);

    let expr = serde_json::json!(["zoom"]);
    let result = evaluate_expression(&expr, &ctx);
    assert_eq!(result.and_then(|v| v.as_f64()), Some(12.5));
}

#[test]
fn test_interpolate_linear() {
    let props = empty_props();
    let ctx = EvalContext::new(&props, 10.0);

    let expr = serde_json::json!(["interpolate", ["linear"], ["zoom"], 5, 1, 15, 10]);
    let result = evaluate_expression(&expr, &ctx);
    assert!((result.and_then(|v| v.as_f64()).unwrap() - 5.5).abs() < 0.01);
}

#[test]
fn test_step() {
    let props = empty_props();
    let expr = serde_json::json!(["step", ["zoom"], "small", 10, "medium", 15, "large"]);

    let ctx = EvalContext::new(&props, 5.0);
    assert_eq!(
        evaluate_expression(&expr, &ctx),
        Some(Value::String("small".to_string()))
    );

    let ctx = EvalContext::new(&props, 12.0);
    assert_eq!(
        evaluate_expression(&expr, &ctx),
        Some(Value::String("medium".to_string()))
    );

    let ctx = EvalContext::new(&props, 20.0);
    assert_eq!(
        evaluate_expression(&expr, &ctx),
        Some(Value::String("large".to_string()))
    );
}

#[test]
fn test_match() {
    let props = props_with("type", Value::String("highway".to_string()));
    let ctx = EvalContext::new(&props, 10.0);

    let expr = serde_json::json!([
        "match",
        ["get", "type"],
        "highway",
        "#ff0000",
        "street",
        "#00ff00",
        "#888888"
    ]);
    let result = evaluate_expression(&expr, &ctx);
    assert_eq!(result, Some(Value::String("#ff0000".to_string())));
}

#[test]
fn test_case() {
    let props = props_with("population", Value::Number(50000.into()));
    let ctx = EvalContext::new(&props, 10.0);

    let expr = serde_json::json!([
        "case",
        [">", ["get", "population"], 100000],
        "large",
        [">", ["get", "population"], 10000],
        "medium",
        "small"
    ]);
    let result = evaluate_expression(&expr, &ctx);
    assert_eq!(result, Some(Value::String("medium".to_string())));
}

#[test]
fn test_math_operators() {
    let props = empty_props();
    let ctx = EvalContext::new(&props, 10.0);

    let expr = serde_json::json!(["+", 1, 2, 3]);
    assert_eq!(
        evaluate_expression(&expr, &ctx).and_then(|v| v.as_f64()),
        Some(6.0)
    );

    let expr = serde_json::json!(["*", 2, 3]);
    assert_eq!(
        evaluate_expression(&expr, &ctx).and_then(|v| v.as_f64()),
        Some(6.0)
    );

    let expr = serde_json::json!(["/", 10, 2]);
    assert_eq!(
        evaluate_expression(&expr, &ctx).and_then(|v| v.as_f64()),
        Some(5.0)
    );
}

#[test]
fn test_comparison() {
    let props = empty_props();
    let ctx = EvalContext::new(&props, 10.0);

    let expr = serde_json::json!([">", 5, 3]);
    assert_eq!(evaluate_expression(&expr, &ctx), Some(Value::Bool(true)));

    let expr = serde_json::json!(["<=", 5, 5]);
    assert_eq!(evaluate_expression(&expr, &ctx), Some(Value::Bool(true)));
}

#[test]
fn test_coalesce() {
    let props = props_with("alt_name", Value::String("Alternative".to_string()));
    let ctx = EvalContext::new(&props, 10.0);

    let expr = serde_json::json!(["coalesce", ["get", "name"], ["get", "alt_name"], "Unknown"]);
    let result = evaluate_expression(&expr, &ctx);
    assert_eq!(result, Some(Value::String("Alternative".to_string())));
}
