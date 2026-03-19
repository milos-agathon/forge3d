//! Ed25519 offline license signature verification.
//!
//! The public key is embedded at compile time.  The private key lives in a
//! secure signing pipeline and is never shipped in source or binary form.
//! Verification happens entirely offline — no network calls.

use ed25519_dalek::{Signature, Verifier, VerifyingKey};

/// Embedded Ed25519 public key used to verify license signatures.
///
/// Replace this with the production public key before cutting release builds.
/// The corresponding private key must be kept in a secure offline signing
/// environment.
const PUBLIC_KEY_BYTES: [u8; 32] = [
    0x9a, 0x99, 0x5d, 0x11, 0xc2, 0xda, 0x9d, 0xf6, 0xb7, 0x34, 0xe7, 0xaa, 0x98, 0xd7, 0x87, 0x7b,
    0xb3, 0x26, 0x91, 0x09, 0x98, 0x66, 0x7b, 0xef, 0x34, 0x9e, 0xb5, 0x1e, 0x16, 0x73, 0x82, 0xf7,
];

/// Verify an Ed25519 signature over `message` using the embedded public key.
///
/// Returns `true` when the signature is valid, `false` otherwise.
/// All error paths (bad key bytes, malformed signature) collapse to `false`.
pub fn verify_signature(message: &[u8], signature_bytes: &[u8]) -> bool {
    let Ok(key) = VerifyingKey::from_bytes(&PUBLIC_KEY_BYTES) else {
        return false;
    };
    if signature_bytes.len() != 64 {
        return false;
    }
    let Ok(sig) = Signature::from_slice(signature_bytes) else {
        return false;
    };
    key.verify(message, &sig).is_ok()
}

/// Return the hex-encoded public key so Python can read it for diagnostics.
pub fn public_key_hex() -> String {
    PUBLIC_KEY_BYTES
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_signature() {
        assert!(!verify_signature(b"F3D-PRO-forge3d-ci-20991231", &[]));
    }

    #[test]
    fn rejects_garbage_signature() {
        assert!(!verify_signature(
            b"F3D-PRO-forge3d-ci-20991231",
            &[0u8; 64]
        ));
    }

    #[test]
    fn rejects_wrong_length() {
        assert!(!verify_signature(
            b"F3D-PRO-forge3d-ci-20991231",
            &[0u8; 32]
        ));
    }

    #[test]
    fn public_key_hex_is_64_chars() {
        assert_eq!(public_key_hex().len(), 64);
    }
}
