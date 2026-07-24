//! Integer-only, byte-aligned, two-lane rANS.

use super::{F3dzError, F3dzResult};

pub const SCALE_BITS: u32 = 12;
pub const SCALE: u32 = 1 << SCALE_BITS;
const RANS_L: u32 = 1 << 23;
const LAYER_MAGIC: [u8; 4] = *b"RAN2";
const LAYER_VERSION: u16 = 1;
const LAYER_HEADER_LEN: usize = 32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RansEncoded {
    pub decoded_len: u32,
    pub states: [u32; 2],
    pub frequencies: [u16; 256],
    pub lanes: [Vec<u8>; 2],
}

impl RansEncoded {
    pub fn encode(data: &[u8]) -> F3dzResult<Self> {
        if data.is_empty() {
            return Err(F3dzError::InvalidArgument(
                "rANS layer cannot be empty".to_string(),
            ));
        }
        let frequencies = normalize_frequencies(data)?;
        let cumulative = cumulative(&frequencies)?;
        let mut states = [RANS_L; 2];
        let mut reversed = [Vec::new(), Vec::new()];
        for index in (0..data.len()).rev() {
            let lane = index & 1;
            let symbol = data[index] as usize;
            let frequency = u32::from(frequencies[symbol]);
            let start = cumulative[symbol];
            let threshold = ((RANS_L >> SCALE_BITS) << 8)
                .checked_mul(frequency)
                .ok_or_else(|| {
                    F3dzError::InvalidArgument("rANS renormalization overflow".to_string())
                })?;
            while states[lane] >= threshold {
                reversed[lane].push(states[lane] as u8);
                states[lane] >>= 8;
            }
            states[lane] =
                ((states[lane] / frequency) << SCALE_BITS) + (states[lane] % frequency) + start;
        }
        reversed[0].reverse();
        reversed[1].reverse();
        Ok(Self {
            decoded_len: u32::try_from(data.len()).map_err(|_| {
                F3dzError::InvalidArgument("rANS layer exceeds u32 length".to_string())
            })?,
            states,
            frequencies,
            lanes: reversed,
        })
    }

    pub fn decode(&self) -> F3dzResult<Vec<u8>> {
        validate_frequencies(&self.frequencies)?;
        if self.states.iter().any(|&state| state < RANS_L) {
            return Err(F3dzError::InvalidArgument(
                "rANS initial state is below normalization lower bound".to_string(),
            ));
        }
        let cumulative = cumulative(&self.frequencies)?;
        let mut lookup = [0u8; SCALE as usize];
        for symbol in 0..256 {
            let start = cumulative[symbol] as usize;
            let end = start + usize::from(self.frequencies[symbol]);
            for slot in &mut lookup[start..end] {
                *slot = symbol as u8;
            }
        }
        let mut states = self.states;
        let mut positions = [0usize; 2];
        let mut out = Vec::with_capacity(self.decoded_len as usize);
        for index in 0..self.decoded_len as usize {
            let lane = index & 1;
            let slot = states[lane] & (SCALE - 1);
            let symbol = lookup[slot as usize] as usize;
            let frequency = u32::from(self.frequencies[symbol]);
            if frequency == 0 {
                return Err(F3dzError::InvalidArgument(
                    "rANS state selected a zero-frequency symbol".to_string(),
                ));
            }
            states[lane] = frequency
                .checked_mul(states[lane] >> SCALE_BITS)
                .and_then(|value| value.checked_add(slot))
                .and_then(|value| value.checked_sub(cumulative[symbol]))
                .ok_or_else(|| {
                    F3dzError::InvalidArgument(format!(
                        "rANS state arithmetic overflow in lane {lane}"
                    ))
                })?;
            while states[lane] < RANS_L {
                let byte = *self.lanes[lane].get(positions[lane]).ok_or_else(|| {
                    F3dzError::InvalidArgument(format!(
                        "rANS lane {lane} ended before normalization completed"
                    ))
                })?;
                positions[lane] += 1;
                states[lane] = (states[lane] << 8) | u32::from(byte);
            }
            out.push(symbol as u8);
        }
        if positions[0] != self.lanes[0].len() || positions[1] != self.lanes[1].len() {
            return Err(F3dzError::InvalidArgument(
                "rANS layer contains trailing renormalization bytes".to_string(),
            ));
        }
        if states != [RANS_L; 2] {
            return Err(F3dzError::InvalidArgument(
                "rANS terminal states do not match the canonical lower bound".to_string(),
            ));
        }
        Ok(out)
    }

    pub fn to_bytes(&self) -> F3dzResult<Vec<u8>> {
        validate_frequencies(&self.frequencies)?;
        let nonzero = self.frequencies.iter().filter(|&&value| value != 0).count();
        let mut out = vec![0u8; LAYER_HEADER_LEN];
        out[0..4].copy_from_slice(&LAYER_MAGIC);
        put_u16(&mut out, 4, LAYER_VERSION);
        out[6] = SCALE_BITS as u8;
        put_u16(
            &mut out,
            8,
            u16::try_from(nonzero).expect("frequency alphabet has at most 256 symbols"),
        );
        put_u32(&mut out, 12, self.decoded_len);
        put_u32(&mut out, 16, self.states[0]);
        put_u32(&mut out, 20, self.states[1]);
        put_u32(
            &mut out,
            24,
            u32::try_from(self.lanes[0].len()).map_err(|_| {
                F3dzError::InvalidArgument("rANS lane 0 exceeds u32 length".to_string())
            })?,
        );
        put_u32(
            &mut out,
            28,
            u32::try_from(self.lanes[1].len()).map_err(|_| {
                F3dzError::InvalidArgument("rANS lane 1 exceeds u32 length".to_string())
            })?,
        );
        for (symbol, &frequency) in self.frequencies.iter().enumerate() {
            if frequency != 0 {
                out.push(symbol as u8);
                out.extend_from_slice(&frequency.to_le_bytes());
            }
        }
        out.extend_from_slice(&self.lanes[0]);
        out.extend_from_slice(&self.lanes[1]);
        Ok(out)
    }

    pub fn from_bytes(data: &[u8]) -> F3dzResult<(Self, usize)> {
        require_len(data, LAYER_HEADER_LEN)?;
        if data[0..4] != LAYER_MAGIC {
            return Err(F3dzError::InvalidArgument(
                "rANS layer magic must be RAN2".to_string(),
            ));
        }
        let version = get_u16(data, 4);
        if version != LAYER_VERSION {
            return Err(F3dzError::InvalidArgument(format!(
                "unsupported rANS layer version {version}"
            )));
        }
        if data[6] != SCALE_BITS as u8 || data[7] != 0 || get_u16(data, 10) != 0 {
            return Err(F3dzError::InvalidArgument(
                "invalid rANS scale/reserved fields".to_string(),
            ));
        }
        let nonzero = usize::from(get_u16(data, 8));
        if !(1..=256).contains(&nonzero) {
            return Err(F3dzError::InvalidArgument(
                "rANS sparse table must contain 1..=256 symbols".to_string(),
            ));
        }
        if get_u32(data, 12) == 0 {
            return Err(F3dzError::InvalidArgument(
                "rANS decoded length must be non-zero".to_string(),
            ));
        }
        let table_end = LAYER_HEADER_LEN
            .checked_add(nonzero * 3)
            .ok_or_else(|| F3dzError::InvalidArgument("rANS table overflow".to_string()))?;
        require_len(data, table_end)?;
        let mut frequencies = [0u16; 256];
        let mut offset = LAYER_HEADER_LEN;
        let mut previous = None;
        for _ in 0..nonzero {
            let symbol = data[offset];
            if previous.is_some_and(|previous| symbol <= previous) {
                return Err(F3dzError::InvalidArgument(
                    "rANS sparse symbols must be strictly increasing".to_string(),
                ));
            }
            let frequency = get_u16(data, offset + 1);
            if frequency == 0 {
                return Err(F3dzError::InvalidArgument(
                    "rANS sparse table contains a zero frequency".to_string(),
                ));
            }
            frequencies[symbol as usize] = frequency;
            previous = Some(symbol);
            offset += 3;
        }
        validate_frequencies(&frequencies)?;
        let lane0_len = get_u32(data, 24) as usize;
        let lane1_len = get_u32(data, 28) as usize;
        let lane0_end = offset
            .checked_add(lane0_len)
            .ok_or_else(|| F3dzError::InvalidArgument("rANS lane 0 overflow".to_string()))?;
        let end = lane0_end
            .checked_add(lane1_len)
            .ok_or_else(|| F3dzError::InvalidArgument("rANS lane 1 overflow".to_string()))?;
        require_len(data, end)?;
        Ok((
            Self {
                decoded_len: get_u32(data, 12),
                states: [get_u32(data, 16), get_u32(data, 20)],
                frequencies,
                lanes: [
                    data[offset..lane0_end].to_vec(),
                    data[lane0_end..end].to_vec(),
                ],
            },
            end,
        ))
    }
}

fn normalize_frequencies(data: &[u8]) -> F3dzResult<[u16; 256]> {
    let mut counts = [0u64; 256];
    for &symbol in data {
        counts[symbol as usize] += 1;
    }
    let total = data.len() as u64;
    let mut frequencies = [0u16; 256];
    let mut remainders = [0u64; 256];
    let mut sum = 0u32;
    for symbol in 0..256 {
        if counts[symbol] == 0 {
            continue;
        }
        let scaled = counts[symbol] * u64::from(SCALE);
        let frequency = (scaled / total).max(1);
        frequencies[symbol] = u16::try_from(frequency).map_err(|_| {
            F3dzError::InvalidArgument("normalized frequency exceeds u16".to_string())
        })?;
        remainders[symbol] = scaled % total;
        sum += frequency as u32;
    }
    while sum < SCALE {
        let symbol = (0..256)
            .filter(|&symbol| counts[symbol] != 0)
            .max_by_key(|&symbol| (remainders[symbol], counts[symbol], 255 - symbol))
            .expect("non-empty data has at least one symbol");
        frequencies[symbol] += 1;
        remainders[symbol] = 0;
        sum += 1;
    }
    while sum > SCALE {
        let symbol = (0..256)
            .filter(|&symbol| frequencies[symbol] > 1)
            .max_by_key(|&symbol| (frequencies[symbol], counts[symbol], 255 - symbol))
            .ok_or_else(|| {
                F3dzError::InvalidArgument(
                    "too many nonzero symbols for 12-bit rANS table".to_string(),
                )
            })?;
        frequencies[symbol] -= 1;
        sum -= 1;
    }
    validate_frequencies(&frequencies)?;
    Ok(frequencies)
}

fn cumulative(frequencies: &[u16; 256]) -> F3dzResult<[u32; 256]> {
    validate_frequencies(frequencies)?;
    let mut cumulative = [0u32; 256];
    let mut sum = 0u32;
    for symbol in 0..256 {
        cumulative[symbol] = sum;
        sum += u32::from(frequencies[symbol]);
    }
    Ok(cumulative)
}

fn validate_frequencies(frequencies: &[u16; 256]) -> F3dzResult<()> {
    let sum: u32 = frequencies.iter().map(|&value| u32::from(value)).sum();
    if sum != SCALE {
        return Err(F3dzError::InvalidArgument(format!(
            "rANS frequencies sum to {sum}, expected {SCALE}"
        )));
    }
    Ok(())
}

fn require_len(data: &[u8], needed: usize) -> F3dzResult<()> {
    if data.len() < needed {
        Err(F3dzError::Truncated {
            needed,
            available: data.len(),
        })
    } else {
        Ok(())
    }
}

fn get_u16(data: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([data[offset], data[offset + 1]])
}

fn get_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

fn put_u16(data: &mut [u8], offset: usize, value: u16) {
    data[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(data: &mut [u8], offset: usize, value: u32) {
    data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_bytes(length: usize) -> Vec<u8> {
        let mut state = 0x1234_5678u32;
        (0..length)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                match state % 100 {
                    0..=69 => 0,
                    70..=89 => 1,
                    90..=96 => 2,
                    _ => 3,
                }
            })
            .collect()
    }

    #[test]
    fn two_lane_rans_round_trips_one_million_symbols() {
        let source = synthetic_bytes(1_000_000);
        let encoded = RansEncoded::encode(&source).unwrap();
        assert_eq!(encoded.decode().unwrap(), source);
        let serialized = encoded.to_bytes().unwrap();
        let (parsed, consumed) = RansEncoded::from_bytes(&serialized).unwrap();
        assert_eq!(consumed, serialized.len());
        assert_eq!(parsed.decode().unwrap(), source);
    }

    #[test]
    fn payload_bitrate_is_within_one_percent_of_normalized_entropy() {
        let source = synthetic_bytes(1_000_000);
        let encoded = RansEncoded::encode(&source).unwrap();
        let entropy: f64 = encoded
            .frequencies
            .iter()
            .filter(|&&frequency| frequency != 0)
            .map(|&frequency| {
                let probability = f64::from(frequency) / f64::from(SCALE);
                -probability * probability.log2()
            })
            .sum();
        let payload_bits = ((encoded.lanes[0].len() + encoded.lanes[1].len() + 8) * 8) as f64;
        let bitrate = payload_bits / source.len() as f64;
        assert!(
            (bitrate - entropy).abs() / entropy <= 0.01,
            "bitrate={bitrate} entropy={entropy}"
        );
    }

    #[test]
    fn corrupt_lane_fails_closed() {
        let source = synthetic_bytes(4096);
        let mut encoded = RansEncoded::encode(&source).unwrap();
        encoded.lanes[0].pop();
        assert!(encoded.decode().is_err());
    }
}
