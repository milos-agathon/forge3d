use crate::labels::unicode::BidiClass;

#[derive(Clone, Copy)]
pub(super) struct Unit {
    pub original: BidiClass,
    pub class: BidiClass,
    pub level: u8,
    pub removed: bool,
}

#[derive(Clone, Copy)]
struct Status {
    level: u8,
    override_class: Option<BidiClass>,
    isolate: bool,
}

pub(super) fn paragraph_level(classes: &[BidiClass]) -> u8 {
    let mut isolate_depth = 0usize;
    for class in classes {
        match class {
            BidiClass::Lri | BidiClass::Rli | BidiClass::Fsi => isolate_depth += 1,
            BidiClass::Pdi if isolate_depth > 0 => isolate_depth -= 1,
            BidiClass::L if isolate_depth == 0 => return 0,
            BidiClass::R | BidiClass::Al if isolate_depth == 0 => return 1,
            _ => {}
        }
    }
    0
}

pub(super) fn resolve_explicit(classes: &[BidiClass], paragraph_level: u8) -> Vec<Unit> {
    let mut units: Vec<_> = classes
        .iter()
        .copied()
        .map(|class| Unit {
            original: class,
            class,
            level: paragraph_level,
            removed: removed_by_x9(class),
        })
        .collect();
    let mut stack = vec![Status {
        level: paragraph_level,
        override_class: None,
        isolate: false,
    }];
    let mut overflow_isolates = 0usize;
    let mut overflow_embeddings = 0usize;
    let mut valid_isolates = 0usize;
    for index in 0..units.len() {
        let class = units[index].original;
        let current = *stack.last().unwrap();
        match class {
            BidiClass::Rle
            | BidiClass::Lre
            | BidiClass::Rlo
            | BidiClass::Lro
            | BidiClass::Rli
            | BidiClass::Lri
            | BidiClass::Fsi => {
                units[index].level = current.level;
                let isolate = matches!(class, BidiClass::Rli | BidiClass::Lri | BidiClass::Fsi);
                if isolate {
                    apply_override(&mut units[index], current.override_class);
                }
                let rtl = class == BidiClass::Rli
                    || matches!(class, BidiClass::Rle | BidiClass::Rlo)
                    || class == BidiClass::Fsi && first_strong_is_rtl(classes, index + 1);
                let new_level = next_level(current.level, rtl);
                if let Some(new_level) =
                    new_level.filter(|_| overflow_isolates == 0 && overflow_embeddings == 0)
                {
                    stack.push(Status {
                        level: new_level,
                        override_class: match class {
                            BidiClass::Rlo => Some(BidiClass::R),
                            BidiClass::Lro => Some(BidiClass::L),
                            _ => None,
                        },
                        isolate,
                    });
                    if isolate {
                        valid_isolates += 1;
                    } else {
                        units[index].level = new_level;
                    }
                } else if isolate {
                    overflow_isolates += 1;
                } else if overflow_isolates == 0 {
                    overflow_embeddings += 1;
                }
            }
            BidiClass::Pdi => {
                if overflow_isolates > 0 {
                    overflow_isolates -= 1;
                } else if valid_isolates > 0 {
                    overflow_embeddings = 0;
                    while stack.pop().is_some_and(|status| !status.isolate) {}
                    valid_isolates -= 1;
                }
                let current = *stack.last().unwrap();
                units[index].level = current.level;
                apply_override(&mut units[index], current.override_class);
            }
            BidiClass::Pdf => {
                if overflow_isolates == 0 {
                    if overflow_embeddings > 0 {
                        overflow_embeddings -= 1;
                    } else if stack.len() > 1 && !current.isolate {
                        stack.pop();
                    }
                }
                units[index].level = stack.last().unwrap().level;
            }
            BidiClass::B => {}
            _ => {
                units[index].level = current.level;
                if class != BidiClass::Bn {
                    apply_override(&mut units[index], current.override_class);
                }
            }
        }
    }
    units
}

fn first_strong_is_rtl(classes: &[BidiClass], start: usize) -> bool {
    let mut depth = 0usize;
    for class in &classes[start..] {
        match class {
            BidiClass::Lri | BidiClass::Rli | BidiClass::Fsi => depth += 1,
            BidiClass::Pdi if depth == 0 => break,
            BidiClass::Pdi => depth -= 1,
            BidiClass::L if depth == 0 => return false,
            BidiClass::R | BidiClass::Al if depth == 0 => return true,
            _ => {}
        }
    }
    false
}

fn next_level(level: u8, rtl: bool) -> Option<u8> {
    let next = if rtl {
        (level + 1) | 1
    } else {
        (level + 2) & !1
    };
    (next <= 125).then_some(next)
}

fn apply_override(unit: &mut Unit, override_class: Option<BidiClass>) {
    if let Some(class) = override_class {
        unit.class = class;
    }
}

pub(super) fn removed_by_x9(class: BidiClass) -> bool {
    matches!(
        class,
        BidiClass::Rle
            | BidiClass::Lre
            | BidiClass::Rlo
            | BidiClass::Lro
            | BidiClass::Pdf
            | BidiClass::Bn
    )
}
