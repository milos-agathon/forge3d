use super::*;

pub(super) fn register_license_py_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(verify_license_signature, m)?)?;
    m.add_function(wrap_pyfunction!(license_public_key_hex, m)?)?;
    Ok(())
}

/// Verify an Ed25519 license signature.
///
/// Parameters
/// ----------
/// message : bytes
///     The message that was signed (e.g. ``b"F3D-PRO-acme-co-20991231"``).
/// signature : bytes
///     The 64-byte Ed25519 signature.
///
/// Returns
/// -------
/// bool
///     ``True`` when the signature is valid.
#[pyfunction]
#[pyo3(signature = (message, signature))]
fn verify_license_signature(message: Vec<u8>, signature: Vec<u8>) -> bool {
    crate::license::verify_signature(&message, &signature)
}

/// Return the hex-encoded Ed25519 public key embedded in the binary.
#[pyfunction]
fn license_public_key_hex() -> String {
    crate::license::public_key_hex()
}
