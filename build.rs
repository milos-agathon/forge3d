fn main() {
    // On macOS, when building the Python extension (feature `extension-module`),
    // ensure unresolved CPython symbols are looked up at runtime from the host
    // interpreter. This mirrors the typical -undefined dynamic_lookup settings
    // used for PyO3 extension modules.
    #[cfg(target_os = "macos")]
    {
        if std::env::var_os("CARGO_FEATURE_EXTENSION_MODULE").is_some() {
            println!("cargo:rustc-link-arg=-undefined");
            println!("cargo:rustc-link-arg=dynamic_lookup");
        }
    }
}
