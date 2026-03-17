use crate::viewer::viewer_enums::ViewerCmd;

pub(super) fn parse_float_or_query<F, Q>(
    line: &str,
    set_cmd: F,
    query_cmd: Q,
    usage: &str,
) -> Option<Vec<ViewerCmd>>
where
    F: FnOnce(f32) -> ViewerCmd,
    Q: FnOnce() -> ViewerCmd,
{
    let mut it = line.split_whitespace();
    let _ = it.next();
    if let Some(val_str) = it.next() {
        if let Ok(val) = val_str.parse::<f32>() {
            Some(vec![set_cmd(val)])
        } else {
            println!("Usage: :{}", usage);
            Some(vec![])
        }
    } else {
        Some(vec![query_cmd()])
    }
}

pub(super) fn parse_u32_or_query<F, Q>(
    line: &str,
    set_cmd: F,
    query_cmd: Q,
    usage: &str,
) -> Option<Vec<ViewerCmd>>
where
    F: FnOnce(u32) -> ViewerCmd,
    Q: FnOnce() -> ViewerCmd,
{
    let mut it = line.split_whitespace();
    let _ = it.next();
    if let Some(val_str) = it.next() {
        if let Ok(val) = val_str.parse::<u32>() {
            Some(vec![set_cmd(val)])
        } else {
            println!("Usage: :{}", usage);
            Some(vec![])
        }
    } else {
        Some(vec![query_cmd()])
    }
}

pub(super) fn parse_bool_or_query<F, Q>(
    line: &str,
    set_cmd: F,
    query_cmd: Q,
    usage: &str,
) -> Option<Vec<ViewerCmd>>
where
    F: FnOnce(bool) -> ViewerCmd,
    Q: FnOnce() -> ViewerCmd,
{
    if let Some(tok) = line.split_whitespace().nth(1) {
        let state = if tok.eq_ignore_ascii_case("on")
            || tok == "1"
            || tok.eq_ignore_ascii_case("true")
        {
            Some(true)
        } else if tok.eq_ignore_ascii_case("off") || tok == "0" || tok.eq_ignore_ascii_case("false")
        {
            Some(false)
        } else {
            None
        };

        if let Some(on) = state {
            Some(vec![set_cmd(on)])
        } else {
            println!("Usage: :{}", usage);
            Some(vec![])
        }
    } else {
        Some(vec![query_cmd()])
    }
}

pub(super) fn print_help() {
    println!(
        "Commands:\n  :gi <ssao|ssgi|ssr> <on|off>\n  :viz <material|normal|depth|gi|lit>\n  :viz-depth-max <float>\n  :ibl <on|off|load <path>|intensity <f>|rotate <deg>|cache <dir>|res <u32>>\n  :brdf <lambert|phong|ggx|disney>\n  :snapshot [path]\n  :obj <path> | :gltf <path>\n  :sky off|on|preetham|hosek-wilkie | :sky-turbidity <f> | :sky-ground <f> | :sky-exposure <f> | :sky-sun <f>\n  :fog <on|off> | :fog-density <f> | :fog-g <f> | :fog-steps <u32> | :fog-shadow <on|off> | :fog-temporal <0..0.9> | :fog-mode <raymarch|froxels> | :fog-preset <low|med|high>\n  :oit <auto|wboit|dual_source|off> (Order-Independent Transparency)\n  Lit:  :lit-sun <float> | :lit-ibl <float>\n  SSAO: :ssao-technique <ssao|gtao> | :ssao-radius <f> | :ssao-intensity <f> | :ssao-composite <on|off> | :ssao-mul <0..1>\n  SSGI: :ssgi-steps <u32> | :ssgi-radius <f> | :ssgi-half <on|off> | :ssgi-temporal <on|off> | :ssgi-temporal-alpha <0..1> | :ssgi-edges <on|off> | :ssgi-upsample-sigma-depth <f> | :ssgi-upsample-sigma-normal <f>\n  SSR:  :ssr-max-steps <u32> | :ssr-thickness <f>\n  P5:   :p5 <cornell|grid|sweep|ssgi-cornell|ssgi-temporal|ssr-glossy|ssr-thickness>\n  :quit"
    );
}
