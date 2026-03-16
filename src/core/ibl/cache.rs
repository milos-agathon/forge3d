use super::*;

impl IBLRenderer {
    pub fn configure_cache<P: AsRef<Path>>(
        &mut self,
        dir: P,
        hdr_path: &Path,
    ) -> Result<(), String> {
        let canonical = hdr_path
            .canonicalize()
            .unwrap_or_else(|_| hdr_path.to_path_buf());
        let dir = dir.as_ref();
        if !dir.exists() {
            fs::create_dir_all(dir).map_err(|e| format!("Failed to create cache dir: {e}"))?;
        }
        self.cache = Some(IblCacheConfig {
            dir: dir.to_path_buf(),
            hdr_path: canonical.to_string_lossy().into_owned(),
            hdr_width: 0,
            hdr_height: 0,
            cache_key: None,
        });
        Ok(())
    }

    fn cache_key(&self) -> Option<String> {
        self.cache.as_ref().and_then(|cfg| cfg.cache_key.clone())
    }

    pub(super) fn invalidate_cache_key(&mut self) {
        if let Some(ref mut cfg) = self.cache {
            cfg.cache_key = None;
        }
    }

    fn ensure_cache_key(&mut self) {
        if let Some(ref mut cfg) = self.cache {
            if cfg.cache_key.is_some() || cfg.hdr_width == 0 || cfg.hdr_height == 0 {
                return;
            }
            let mut hasher = Sha256::new();
            hasher.update(&cfg.hdr_path);
            hasher.update(cfg.hdr_width.to_le_bytes());
            hasher.update(cfg.hdr_height.to_le_bytes());
            hasher.update(self.base_resolution.to_le_bytes());
            hasher.update(self.quality.irradiance_size().to_le_bytes());
            hasher.update(self.quality.specular_size().to_le_bytes());
            hasher.update(self.quality.specular_mip_levels().to_le_bytes());
            hasher.update(self.quality.brdf_size().to_le_bytes());
            cfg.cache_key = Some(format!("{:x}", hasher.finalize()));
        }
    }

    fn cache_path(&mut self) -> Option<PathBuf> {
        self.ensure_cache_key();
        self.cache.as_ref().and_then(|cfg| {
            cfg.cache_key
                .as_ref()
                .map(|key| cfg.dir.join(format!("{key}.iblcache")))
        })
    }

    pub(super) fn try_load_cache(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<bool, String> {
        let path = match self.cache_path() {
            Some(path) => path,
            None => return Ok(false),
        };
        if !path.exists() {
            return Ok(false);
        }

        let mut reader = BufReader::new(
            File::open(&path)
                .map_err(|e| format!("Failed to open IBL cache '{}': {e}", path.display()))?,
        );

        let mut magic = [0u8; 8];
        reader
            .read_exact(&mut magic)
            .map_err(|e| format!("Failed to read IBL cache magic: {e}"))?;
        if &magic != CACHE_MAGIC {
            return Err(format!("Invalid IBL cache magic in '{}'", path.display()));
        }

        let mut version_bytes = [0u8; 4];
        reader
            .read_exact(&mut version_bytes)
            .map_err(|e| format!("Failed to read IBL cache version: {e}"))?;
        let version = u32::from_le_bytes(version_bytes);
        if version != CACHE_VERSION {
            return Ok(false);
        }

        let mut meta_len_bytes = [0u8; 4];
        reader
            .read_exact(&mut meta_len_bytes)
            .map_err(|e| format!("Failed to read IBL cache metadata length: {e}"))?;
        let meta_len = u32::from_le_bytes(meta_len_bytes) as usize;
        let mut meta_buf = vec![0u8; meta_len];
        reader
            .read_exact(&mut meta_buf)
            .map_err(|e| format!("Failed to read IBL cache metadata: {e}"))?;
        let metadata: IblCacheMetadata = serde_json::from_slice(&meta_buf)
            .map_err(|e| format!("Failed to parse IBL cache metadata: {e}"))?;

        if metadata.hdr_width == 0
            || metadata.hdr_height == 0
            || metadata.irradiance_size != self.quality.irradiance_size()
            || metadata.specular_size != self.quality.specular_size()
            || metadata.specular_mips != self.quality.specular_mip_levels()
            || metadata.brdf_size != self.quality.brdf_size()
        {
            return Ok(false);
        }

        // Validate sha256 hash (spec requirement: graceful invalidate on mismatch)
        self.ensure_cache_key();
        if let Some(expected_key) = self.cache_key() {
            if metadata.sha256 != expected_key {
                warn!(
                    "IBL cache sha256 mismatch: expected {}, got {}",
                    expected_key, metadata.sha256
                );
                return Ok(false);
            }
        }

        let irradiance_bytes = read_blob(&mut reader)?;
        let specular_bytes = read_blob(&mut reader)?;
        let brdf_bytes = read_blob(&mut reader)?;

        let (irr_tex, irr_view) = Self::upload_cubemap(
            device,
            queue,
            metadata.irradiance_size,
            1,
            &irradiance_bytes,
        )?;
        self.irradiance_map = Some(irr_tex);
        self.irradiance_view = Some(irr_view);
        let (spec_tex, spec_view) = Self::upload_cubemap(
            device,
            queue,
            metadata.specular_size,
            metadata.specular_mips,
            &specular_bytes,
        )?;
        self.specular_map = Some(spec_tex);
        self.specular_view = Some(spec_view);
        let (brdf_tex, brdf_view) = Self::upload_2d(
            device,
            queue,
            metadata.brdf_size,
            metadata.brdf_size,
            &brdf_bytes,
        )?;
        self.brdf_lut = Some(brdf_tex);
        self.brdf_view = Some(brdf_view);

        let cache_size_mib = (irradiance_bytes.len() + specular_bytes.len() + brdf_bytes.len())
            as f32
            / (1024.0 * 1024.0);
        info!(
            "IBL cache hit: '{}' ({:.2} MiB) - irradiance_{}.cube, prefilter_mips.cube, brdf_{}.png",
            path.display(),
            cache_size_mib,
            metadata.irradiance_size,
            metadata.brdf_size
        );
        Ok(true)
    }

    pub(super) fn write_cache(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), String> {
        let path = match self.cache_path() {
            Some(path) => path,
            None => return Ok(()),
        };
        let irradiance_tex = self
            .irradiance_map
            .as_ref()
            .ok_or("Irradiance texture unavailable")?;
        let specular_tex = self
            .specular_map
            .as_ref()
            .ok_or("Specular texture unavailable")?;
        let brdf_tex = self
            .brdf_lut
            .as_ref()
            .ok_or("BRDF LUT texture unavailable")?;

        let metadata = IblCacheMetadata {
            version: CACHE_VERSION,
            hdr_path: self
                .cache
                .as_ref()
                .map(|cfg| cfg.hdr_path.clone())
                .unwrap_or_default(),
            hdr_width: self.cache.as_ref().map(|c| c.hdr_width).unwrap_or(0),
            hdr_height: self.cache.as_ref().map(|c| c.hdr_height).unwrap_or(0),
            quality: format!("{:?}", self.quality),
            base_resolution: self.base_resolution,
            irradiance_size: self.quality.irradiance_size(),
            specular_size: self.quality.specular_size(),
            specular_mips: self.quality.specular_mip_levels(),
            brdf_size: self.quality.brdf_size(),
            created_unix_secs: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            sha256: self
                .cache
                .as_ref()
                .and_then(|cfg| cfg.cache_key.clone())
                .unwrap_or_default(),
        };

        let irradiance_bytes = self.download_cubemap(
            device,
            queue,
            irradiance_tex,
            self.quality.irradiance_size(),
            1,
        )?;
        let specular_bytes = self.download_cubemap(
            device,
            queue,
            specular_tex,
            self.quality.specular_size(),
            self.quality.specular_mip_levels(),
        )?;
        let brdf_bytes = self.download_2d(
            device,
            queue,
            brdf_tex,
            self.quality.brdf_size(),
            self.quality.brdf_size(),
        )?;

        let mut writer = BufWriter::new(
            File::create(&path)
                .map_err(|e| format!("Failed to create IBL cache '{}': {e}", path.display()))?,
        );

        writer
            .write_all(CACHE_MAGIC)
            .map_err(|e| format!("Failed to write IBL cache magic: {e}"))?;
        writer
            .write_all(&CACHE_VERSION.to_le_bytes())
            .map_err(|e| format!("Failed to write IBL cache version: {e}"))?;

        let meta_json = serde_json::to_vec(&metadata)
            .map_err(|e| format!("Failed to serialise metadata: {e}"))?;
        writer
            .write_all(&(meta_json.len() as u32).to_le_bytes())
            .map_err(|e| format!("Failed to write metadata length: {e}"))?;
        writer
            .write_all(&meta_json)
            .map_err(|e| format!("Failed to write metadata: {e}"))?;

        write_blob(&mut writer, &irradiance_bytes)?;
        write_blob(&mut writer, &specular_bytes)?;
        write_blob(&mut writer, &brdf_bytes)?;
        writer
            .flush()
            .map_err(|e| format!("Failed to flush cache: {e}"))?;

        info!(
            "Wrote IBL cache '{}' ({:.2} MiB)",
            path.display(),
            (irradiance_bytes.len() + specular_bytes.len() + brdf_bytes.len()) as f32
                / (1024.0 * 1024.0)
        );
        Ok(())
    }
}
