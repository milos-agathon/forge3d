//! Full VSOP87D term evaluation, bounded by SIDERA's 2000–2050 oracle.
//!
//! The compact asset preserves every IMCCE VSOP87D term for Earth and
//! Mercury through Saturn; the declared truncation is by body coverage, not
//! by dropping terms.

use super::AstroError;
use glam::DVec3;
use std::io::{Cursor, Read};
use std::sync::OnceLock;

const DATA: &[u8] = include_bytes!("../../assets/astro/vsop87d.bin");
const J2000: f64 = 2_451_545.0;

#[derive(Debug)]
struct Theory {
    bodies: Vec<BodySeries>,
    sections: Vec<Section>,
}

#[derive(Debug)]
struct BodySeries {
    name: [u8; 3],
    sections: Vec<usize>,
}

#[derive(Debug)]
struct Section {
    coordinate: usize,
    power: i32,
    terms: Vec<Term>,
}

#[derive(Debug)]
struct Term {
    amplitude: f64,
    phase: f64,
    frequency: f64,
}

#[derive(Clone, Copy)]
pub(crate) enum VsopBody {
    Mercury,
    Venus,
    Earth,
    Mars,
    Jupiter,
    Saturn,
}

pub(crate) fn heliocentric_ecliptic(body: VsopBody, jd_tt: f64) -> Result<DVec3, AstroError> {
    let name = match body {
        VsopBody::Mercury => *b"mer",
        VsopBody::Venus => *b"ven",
        VsopBody::Earth => *b"ear",
        VsopBody::Mars => *b"mar",
        VsopBody::Jupiter => *b"jup",
        VsopBody::Saturn => *b"sat",
    };
    let theory = theory()?;
    let body = theory
        .bodies
        .iter()
        .find(|body| body.name == name)
        .ok_or(AstroError::InvalidData("vsop87d.bin"))?;
    let t = (jd_tt - J2000) / 365_250.0;
    let mut lbr = [0.0; 3];
    for index in &body.sections {
        let section = theory
            .sections
            .get(*index)
            .ok_or(AstroError::InvalidData("vsop87d.bin"))?;
        let sum = section
            .terms
            .iter()
            .map(|term| term.amplitude * (term.phase + term.frequency * t).cos())
            .sum::<f64>();
        lbr[section.coordinate] += sum * t.powi(section.power);
    }
    let (sin_l, cos_l) = lbr[0].sin_cos();
    let (sin_b, cos_b) = lbr[1].sin_cos();
    Ok(DVec3::new(
        lbr[2] * cos_b * cos_l,
        lbr[2] * cos_b * sin_l,
        lbr[2] * sin_b,
    ))
}

pub fn earth_velocity(jd_tt: f64) -> Result<DVec3, AstroError> {
    const STEP_DAYS: f64 = 0.01;
    Ok((heliocentric_ecliptic(VsopBody::Earth, jd_tt + STEP_DAYS)?
        - heliocentric_ecliptic(VsopBody::Earth, jd_tt - STEP_DAYS)?)
        / (2.0 * STEP_DAYS))
}

fn theory() -> Result<&'static Theory, AstroError> {
    static THEORY: OnceLock<Result<Theory, AstroError>> = OnceLock::new();
    THEORY.get_or_init(parse).as_ref().map_err(Clone::clone)
}

fn parse() -> Result<Theory, AstroError> {
    let mut input = Cursor::new(DATA);
    let mut magic = [0; 8];
    input
        .read_exact(&mut magic)
        .map_err(|_| AstroError::InvalidData("vsop87d.bin"))?;
    if &magic != b"F3DVSOP1" {
        return Err(AstroError::InvalidData("vsop87d.bin"));
    }
    let body_count = read_u32(&mut input)? as usize;
    let section_count = read_u32(&mut input)? as usize;
    let mut bodies = Vec::with_capacity(body_count);
    for _ in 0..body_count {
        let mut name = [0; 3];
        input
            .read_exact(&mut name)
            .map_err(|_| AstroError::InvalidData("vsop87d.bin"))?;
        let count = read_u8(&mut input)? as usize;
        let mut sections = Vec::with_capacity(count);
        for _ in 0..count {
            sections.push(read_u32(&mut input)? as usize);
        }
        bodies.push(BodySeries { name, sections });
    }
    let mut sections = Vec::with_capacity(section_count);
    for _ in 0..section_count {
        let coordinate = read_u8(&mut input)? as usize;
        let power = read_u8(&mut input)? as i32;
        let _reserved = read_u16(&mut input)?;
        let count = read_u32(&mut input)? as usize;
        if coordinate > 2 {
            return Err(AstroError::InvalidData("vsop87d.bin"));
        }
        let mut terms = Vec::with_capacity(count);
        for _ in 0..count {
            terms.push(Term {
                amplitude: read_f64(&mut input)?,
                phase: read_f64(&mut input)?,
                frequency: read_f64(&mut input)?,
            });
        }
        sections.push(Section {
            coordinate,
            power,
            terms,
        });
    }
    if input.position() != DATA.len() as u64 {
        return Err(AstroError::InvalidData("vsop87d.bin"));
    }
    Ok(Theory { bodies, sections })
}

fn read_u8(input: &mut Cursor<&[u8]>) -> Result<u8, AstroError> {
    let mut bytes = [0; 1];
    input
        .read_exact(&mut bytes)
        .map_err(|_| AstroError::InvalidData("vsop87d.bin"))?;
    Ok(bytes[0])
}

fn read_u16(input: &mut Cursor<&[u8]>) -> Result<u16, AstroError> {
    let mut bytes = [0; 2];
    input
        .read_exact(&mut bytes)
        .map_err(|_| AstroError::InvalidData("vsop87d.bin"))?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_u32(input: &mut Cursor<&[u8]>) -> Result<u32, AstroError> {
    let mut bytes = [0; 4];
    input
        .read_exact(&mut bytes)
        .map_err(|_| AstroError::InvalidData("vsop87d.bin"))?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_f64(input: &mut Cursor<&[u8]>) -> Result<f64, AstroError> {
    let mut bytes = [0; 8];
    input
        .read_exact(&mut bytes)
        .map_err(|_| AstroError::InvalidData("vsop87d.bin"))?;
    Ok(f64::from_le_bytes(bytes))
}
