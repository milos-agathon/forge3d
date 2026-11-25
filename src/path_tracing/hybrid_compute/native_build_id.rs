//! Identity from the loaded image, independent of its installation path and
//! any cached Cargo build-script output.

use sha2::{Digest, Sha256};
use std::sync::OnceLock;

pub(super) fn checkpoint_build_id() -> Option<[u8; 32]> {
    static BUILD: OnceLock<Option<[u8; 32]>> = OnceLock::new();
    *BUILD.get_or_init(|| {
        let identifier = platform::loaded_build_id()?;
        let mut digest = Sha256::new();
        digest.update(b"forge3d.continua.linked-image.v1");
        digest.update(std::env::consts::OS.as_bytes());
        digest.update(identifier);
        Some(digest.finalize().into())
    })
}

#[cfg(all(windows, target_env = "msvc"))]
mod platform {
    use std::ffi::c_void;

    #[repr(C)]
    struct ModuleInfo {
        base: *const u8,
        image_size: u32,
        entry_point: *const c_void,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn GetModuleHandleExW(flags: u32, address: *const u16, module: *mut *mut c_void) -> i32;
        fn K32GetModuleInformation(
            process: *mut c_void,
            module: *mut c_void,
            info: *mut ModuleInfo,
            size: u32,
        ) -> i32;
    }

    fn word(bytes: &[u8], offset: usize) -> Option<u32> {
        Some(u32::from_le_bytes(bytes.get(offset..offset + 4)?.try_into().ok()?))
    }

    pub(super) fn loaded_build_id() -> Option<[u8; 20]> {
        const FROM_ADDRESS: u32 = 4;
        const UNCHANGED_REFCOUNT: u32 = 2;
        const DEBUG_DIRECTORY_INDEX: usize = 6;
        const DEBUG_ENTRY_SIZE: usize = 28;
        const CODEVIEW: u32 = 2;

        let mut module = std::ptr::null_mut();
        let mut info = std::mem::MaybeUninit::<ModuleInfo>::uninit();
        // The address belongs to this module, which stays loaded while it executes.
        let info = unsafe {
            if GetModuleHandleExW(
                FROM_ADDRESS | UNCHANGED_REFCOUNT,
                super::checkpoint_build_id as *const () as *const u16,
                &mut module,
            ) == 0 || K32GetModuleInformation(
                GetCurrentProcess(),
                module,
                info.as_mut_ptr(),
                std::mem::size_of::<ModuleInfo>() as u32,
            ) == 0 {
                return None;
            }
            info.assume_init()
        };
        let mapped = |offset: usize, len: usize| -> Option<&[u8]> {
            if offset.checked_add(len)? > info.image_size as usize {
                return None;
            }
            // Only used for the loaded PE headers and the linker's read-only
            // debug data, which are mapped at these relative virtual addresses.
            Some(unsafe { std::slice::from_raw_parts(info.base.add(offset), len) })
        };
        let dos = mapped(0, 64)?;
        if !dos.starts_with(b"MZ") {
            return None;
        }
        let pe_offset = word(dos, 60)? as usize;
        let pe = mapped(pe_offset, 24)?;
        if !pe.starts_with(b"PE\0\0") {
            return None;
        }
        let optional_size = usize::from(u16::from_le_bytes(pe[20..22].try_into().ok()?));
        let optional = mapped(pe_offset.checked_add(pe.len())?, optional_size)?;
        let directories = match optional.get(..2)? {
            [0x0b, 0x01] => 96,  // PE32
            [0x0b, 0x02] => 112, // PE32+
            _ => return None,
        };
        if word(optional, directories - 4)? <= DEBUG_DIRECTORY_INDEX as u32 {
            return None;
        }
        let debug = directories + DEBUG_DIRECTORY_INDEX * 8;
        let entries = mapped(word(optional, debug)? as usize, word(optional, debug + 4)? as usize)?;
        for entry in entries.chunks_exact(DEBUG_ENTRY_SIZE) {
            if word(entry, 12)? != CODEVIEW || word(entry, 16)? < 24 {
                continue;
            }
            let rsds = mapped(word(entry, 20)? as usize, 24)?;
            if rsds.starts_with(b"RSDS") {
                // The CodeView identifier is the 16-byte GUID and 4-byte age.
                return rsds[4..].try_into().ok();
            }
        }
        None
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use std::ffi::{c_char, c_int, c_void};

    #[repr(C)]
    struct ProgramHeader {
        kind: u32,
        #[cfg(target_pointer_width = "64")]
        flags: u32,
        offset: usize,
        address: usize,
        physical_address: usize,
        file_size: usize,
        memory_size: usize,
        #[cfg(target_pointer_width = "32")]
        flags: u32,
        alignment: usize,
    }

    #[repr(C)]
    struct ImageInfo {
        base: usize,
        name: *const c_char,
        headers: *const ProgramHeader,
        header_count: u16,
    }

    extern "C" {
        fn dl_iterate_phdr(
            callback: unsafe extern "C" fn(*mut ImageInfo, usize, *mut c_void) -> c_int,
            data: *mut c_void,
        ) -> c_int;
    }

    const PT_LOAD: u32 = 1;
    const PT_NOTE: u32 = 4;
    const PF_READ: u32 = 4;
    const NT_GNU_BUILD_ID: u32 = 3;

    fn read_note(mut bytes: &[u8]) -> Option<[u8; 20]> {
        let word = |bytes: &[u8]| -> Option<u32> {
            Some(u32::from_ne_bytes(bytes.get(..4)?.try_into().ok()?))
        };
        while let Some(header) = bytes.get(..12) {
            let name_size = word(header)? as usize;
            let desc_size = word(&header[4..])? as usize;
            let kind = word(&header[8..])?;
            bytes = &bytes[12..];
            let name = bytes.get(..name_size)?;
            let desc_start = name_size.checked_next_multiple_of(4)?;
            let desc_end = desc_start.checked_add(desc_size)?;
            let desc = bytes.get(desc_start..desc_end)?;
            if kind == NT_GNU_BUILD_ID && name == b"GNU\0" {
                return desc.try_into().ok();
            }
            bytes = bytes.get(desc_end.checked_next_multiple_of(4)?..)?;
        }
        None
    }

    unsafe extern "C" fn visit(info: *mut ImageInfo, size: usize, data: *mut c_void) -> c_int {
        if size < std::mem::size_of::<ImageInfo>() {
            return 0;
        }
        // The loader owns the program headers during this callback.
        let info = unsafe { &*info };
        if info.header_count == 0 {
            return 0;
        }
        let headers =
            unsafe { std::slice::from_raw_parts(info.headers, usize::from(info.header_count)) };
        let contains = |header: &ProgramHeader, address: usize, len: usize| {
            header.kind == PT_LOAD
                && header.flags & PF_READ != 0
                && address >= header.address
                && (address - header.address)
                    .checked_add(len)
                    .is_some_and(|end| end <= header.memory_size)
        };
        let address = super::checkpoint_build_id as *const () as usize;
        let Some(relative_address) = address.checked_sub(info.base) else {
            return 0;
        };
        if !headers.iter().any(|h| contains(h, relative_address, 1)) {
            return 0;
        }
        for note in headers.iter().filter(|h| h.kind == PT_NOTE) {
            if !headers.iter().any(|h| contains(h, note.address, note.file_size)) {
                continue;
            }
            let Some(address) = info.base.checked_add(note.address) else {
                continue;
            };
            // This note is contained in a readable segment of the loaded image.
            let bytes = unsafe { std::slice::from_raw_parts(address as *const u8, note.file_size) };
            if let Some(build_id) = read_note(bytes) {
                unsafe { *data.cast::<Option<[u8; 20]>>() = Some(build_id) };
                break;
            }
        }
        1
    }

    pub(super) fn loaded_build_id() -> Option<[u8; 20]> {
        let mut result = None;
        unsafe { dl_iterate_phdr(visit, std::ptr::from_mut(&mut result).cast()) };
        result
    }
}

#[cfg(not(any(all(windows, target_env = "msvc"), target_os = "linux")))]
mod platform {
    pub(super) fn loaded_build_id() -> Option<[u8; 20]> {
        None
    }
}
