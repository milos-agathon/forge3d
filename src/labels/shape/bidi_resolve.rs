use super::brackets::bracket;
use super::explicit::Unit;
use crate::labels::unicode::BidiClass as C;

pub(super) fn resolve_sequences(characters: &[char], units: &mut [Unit], paragraph_level: u8) {
    let explicit_levels: Vec<_> = units.iter().map(|unit| unit.level).collect();
    let runs = level_runs(units);
    for sequence in isolating_sequences(units, &runs) {
        let indices: Vec<_> = sequence
            .iter()
            .flat_map(|&run| runs[run].iter().copied())
            .collect();
        if indices.is_empty() {
            continue;
        }
        let sos = direction(sequence_start_level(
            units,
            &explicit_levels,
            &indices,
            paragraph_level,
        ));
        let eos = direction(sequence_end_level(
            units,
            &explicit_levels,
            &indices,
            paragraph_level,
        ));
        resolve_weak(units, &indices, sos);
        resolve_brackets(characters, units, &indices, sos);
        resolve_neutral(units, &indices, sos, eos);
        resolve_implicit(units, &indices);
    }
}

fn level_runs(units: &[Unit]) -> Vec<Vec<usize>> {
    let mut runs: Vec<Vec<usize>> = Vec::new();
    for (index, unit) in units.iter().enumerate().filter(|(_, unit)| !unit.removed) {
        if runs
            .last()
            .and_then(|run| run.last())
            .is_none_or(|&previous| units[previous].level != unit.level)
        {
            runs.push(Vec::new());
        }
        runs.last_mut().unwrap().push(index);
    }
    runs
}

fn isolating_sequences(units: &[Unit], runs: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut finished = Vec::new();
    let mut stack: Vec<Vec<usize>> = vec![Vec::new()];
    for (run_index, run) in runs.iter().enumerate() {
        let first = units[run[0]].original;
        let last = units[*run.last().unwrap()].original;
        let mut sequence = if first == C::Pdi && stack.len() > 1 {
            stack.pop().unwrap()
        } else {
            Vec::new()
        };
        sequence.push(run_index);
        if matches!(last, C::Lri | C::Rli | C::Fsi) {
            stack.push(sequence);
        } else {
            finished.push(sequence);
        }
    }
    finished.extend(
        stack
            .into_iter()
            .rev()
            .filter(|sequence| !sequence.is_empty()),
    );
    finished
}

fn sequence_start_level(units: &[Unit], levels: &[u8], indices: &[usize], paragraph: u8) -> u8 {
    let index = indices[0];
    let predecessor = units[..index]
        .iter()
        .rposition(|unit| !unit.removed)
        .map_or(paragraph, |position| levels[position]);
    levels[index].max(predecessor)
}

fn sequence_end_level(units: &[Unit], levels: &[u8], indices: &[usize], paragraph: u8) -> u8 {
    let index = *indices.last().unwrap();
    let successor = if matches!(units[index].original, C::Lri | C::Rli | C::Fsi) {
        paragraph
    } else {
        units[index + 1..]
            .iter()
            .position(|unit| !unit.removed)
            .map_or(paragraph, |position| levels[index + 1 + position])
    };
    levels[index].max(successor)
}

fn resolve_weak(units: &mut [Unit], indices: &[usize], sos: C) {
    let mut previous = sos;
    for &index in indices {
        if units[index].class == C::Nsm {
            units[index].class = if matches!(previous, C::Lri | C::Rli | C::Fsi | C::Pdi) {
                C::On
            } else {
                previous
            };
        }
        previous = units[index].class;
    }
    let mut strong = sos;
    for &index in indices.iter() {
        match units[index].class {
            C::R | C::L | C::Al => strong = units[index].class,
            C::En if strong == C::Al => units[index].class = C::An,
            _ => {}
        }
    }
    for &index in indices.iter() {
        if units[index].class == C::Al {
            units[index].class = C::R;
        }
    }
    for window in indices.windows(3) {
        let (left, middle, right) = (window[0], window[1], window[2]);
        units[middle].class = match (units[left].class, units[middle].class, units[right].class) {
            (C::En, C::Es, C::En) => C::En,
            (C::En, C::Cs, C::En) => C::En,
            (C::An, C::Cs, C::An) => C::An,
            (_, class, _) => class,
        };
    }
    let mut cursor = 0;
    while cursor < indices.len() {
        if units[indices[cursor]].class != C::Et {
            cursor += 1;
            continue;
        }
        let start = cursor;
        while cursor < indices.len() && units[indices[cursor]].class == C::Et {
            cursor += 1;
        }
        let adjacent_en = start > 0 && units[indices[start - 1]].class == C::En
            || cursor < indices.len() && units[indices[cursor]].class == C::En;
        if adjacent_en {
            for &index in &indices[start..cursor] {
                units[index].class = C::En;
            }
        }
    }
    for &index in indices.iter() {
        if matches!(units[index].class, C::Es | C::Et | C::Cs) {
            units[index].class = C::On;
        }
    }
    let mut strong = sos;
    for &index in indices.iter() {
        match units[index].class {
            C::L | C::R => strong = units[index].class,
            C::En if strong == C::L => units[index].class = C::L,
            _ => {}
        }
    }
}

fn resolve_brackets(characters: &[char], units: &mut [Unit], indices: &[usize], sos: C) {
    let mut stack: Vec<(char, usize)> = Vec::new();
    let mut pairs = Vec::new();
    for (position, &index) in indices.iter().enumerate() {
        if units[index].class != C::On {
            continue;
        }
        let Some((skeleton, opening)) = bracket(characters[index]) else {
            continue;
        };
        if opening {
            if stack.len() >= 63 {
                break;
            }
            stack.push((skeleton, position));
        } else if let Some(at) = stack.iter().rposition(|&(open, _)| open == skeleton) {
            pairs.push((stack[at].1, position));
            stack.truncate(at);
        }
    }
    pairs.sort_unstable();
    for (open_position, close_position) in pairs {
        let open = indices[open_position];
        let close = indices[close_position];
        let embedding = direction(units[open].level);
        let opposite = if embedding == C::L { C::R } else { C::L };
        let mut found_embedding = false;
        let mut found_opposite = false;
        for &index in &indices[open_position + 1..close_position] {
            match strong_type(units[index].class) {
                class if class == embedding => found_embedding = true,
                class if class == opposite => found_opposite = true,
                _ => {}
            }
        }
        let resolved = if found_embedding {
            Some(embedding)
        } else if found_opposite {
            let preceding = indices[..open_position]
                .iter()
                .rev()
                .map(|&index| strong_type(units[index].class))
                .find(|class| matches!(class, C::L | C::R))
                .unwrap_or(sos);
            Some(if preceding == opposite {
                opposite
            } else {
                embedding
            })
        } else {
            None
        };
        if let Some(class) = resolved {
            units[open].class = class;
            units[close].class = class;
            for position in [open_position, close_position] {
                let mut next = position + 1;
                while next < indices.len() && units[indices[next]].original == C::Nsm {
                    units[indices[next]].class = class;
                    next += 1;
                }
            }
        }
    }
}

fn resolve_neutral(units: &mut [Unit], indices: &[usize], sos: C, eos: C) {
    let mut cursor = 0;
    while cursor < indices.len() {
        if !neutral(units[indices[cursor]].class) {
            cursor += 1;
            continue;
        }
        let start = cursor;
        while cursor < indices.len() && neutral(units[indices[cursor]].class) {
            cursor += 1;
        }
        let before = if start == 0 {
            sos
        } else {
            strong_type(units[indices[start - 1]].class)
        };
        let after = if cursor == indices.len() {
            eos
        } else {
            strong_type(units[indices[cursor]].class)
        };
        let resolved = if before == after {
            before
        } else {
            direction(units[indices[start]].level)
        };
        for &index in &indices[start..cursor] {
            units[index].class = resolved;
        }
    }
}

fn resolve_implicit(units: &mut [Unit], indices: &[usize]) {
    for &index in indices {
        let increment = if units[index].level & 1 == 0 {
            match units[index].class {
                C::R => 1,
                C::An | C::En => 2,
                _ => 0,
            }
        } else {
            match units[index].class {
                C::L | C::An | C::En => 1,
                _ => 0,
            }
        };
        units[index].level += increment;
    }
}

fn neutral(class: C) -> bool {
    matches!(
        class,
        C::B | C::S | C::Ws | C::On | C::Lri | C::Rli | C::Fsi | C::Pdi
    )
}

fn strong_type(class: C) -> C {
    match class {
        C::En | C::An => C::R,
        C::Lri | C::Rli | C::Fsi | C::Pdi => C::On,
        other => other,
    }
}

fn direction(level: u8) -> C {
    if level & 1 == 0 {
        C::L
    } else {
        C::R
    }
}
