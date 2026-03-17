use crate::viewer::event_loop::stdin_reader::helpers::parse_float_or_query;
use crate::viewer::viewer_enums::ViewerCmd;

pub(super) fn parse_gi_core_command(line: &str) -> Option<Vec<ViewerCmd>> {
    parse_gi_seed(line)
        .or_else(|| parse_gi_toggle(line))
        .or_else(|| parse_gi_weights(line))
}

fn parse_gi_seed(line: &str) -> Option<Vec<ViewerCmd>> {
    if line.starts_with(":gi-seed") || line.starts_with("gi-seed ") {
        let mut it = line.split_whitespace();
        let _ = it.next();
        if let Some(val_str) = it.next() {
            if let Ok(seed) = val_str.parse::<u32>() {
                Some(vec![ViewerCmd::SetGiSeed(seed)])
            } else {
                println!("Usage: :gi-seed <u32>");
                Some(vec![])
            }
        } else {
            Some(vec![ViewerCmd::QueryGiSeed])
        }
    } else {
        None
    }
}

fn parse_gi_toggle(line: &str) -> Option<Vec<ViewerCmd>> {
    if !(line.starts_with(":gi") || line.starts_with("gi ")) {
        return None;
    }

    let toks: Vec<&str> = line.trim_start_matches(':').split_whitespace().collect();
    if toks.len() == 2 && toks[1] == "status" {
        return Some(vec![ViewerCmd::GiStatus]);
    }
    if toks.len() == 2 && toks[1] == "off" {
        return Some(vec![
            ViewerCmd::GiToggle("ssao", false),
            ViewerCmd::GiToggle("ssgi", false),
            ViewerCmd::GiToggle("ssr", false),
        ]);
    }
    if toks.len() < 3 {
        println!("Usage: :gi <ssao|ssgi|ssr|off|status> [on|off]");
        return Some(vec![]);
    }

    let effect = match toks[1] {
        "ssao" => "ssao",
        "ssgi" => "ssgi",
        "ssr" => "ssr",
        "gtao" => "gtao",
        _ => {
            println!("Unknown effect '{}'", toks[1]);
            return Some(vec![]);
        }
    };
    let on = match toks[2] {
        "on" | "1" | "true" => true,
        "off" | "0" | "false" => false,
        _ => {
            println!("Unknown state '{}', expected on/off", toks[2]);
            return Some(vec![]);
        }
    };

    if effect == "gtao" {
        let mut cmds = vec![ViewerCmd::GiToggle("ssao", on)];
        if on {
            cmds.push(ViewerCmd::SetSsaoTechnique(1));
        }
        return Some(cmds);
    }

    Some(vec![ViewerCmd::GiToggle(effect, on)])
}

fn parse_gi_weights(line: &str) -> Option<Vec<ViewerCmd>> {
    if line.starts_with(":ao-weight") || line.starts_with("ao-weight ") {
        return parse_float_or_query(
            line,
            ViewerCmd::SetGiAoWeight,
            || ViewerCmd::QueryGiAoWeight,
            "ao-weight <float 0..1>",
        );
    }
    if line.starts_with(":ssgi-weight") || line.starts_with("ssgi-weight ") {
        return parse_float_or_query(
            line,
            ViewerCmd::SetGiSsgiWeight,
            || ViewerCmd::QueryGiSsgiWeight,
            "ssgi-weight <float 0..1>",
        );
    }
    if line.starts_with(":ssr-weight") || line.starts_with("ssr-weight ") {
        return parse_float_or_query(
            line,
            ViewerCmd::SetGiSsrWeight,
            || ViewerCmd::QueryGiSsrWeight,
            "ssr-weight <float 0..1>",
        );
    }
    None
}
