# SIDERA source-asset manifest

Runtime validity: UTC 2000-01-01 through 2050-12-31. Positions outside this interval are rejected.
Catalog V magnitudes map to relative flux as `10^(-0.4 V)` with the V=0
reference (about 3630 Jy in the [UKIRT zero-magnitude table](https://about.ifa.hawaii.edu/ukirt/calibration-and-standards/astronomical-utilities/zero-mag-fluxes-and-conversions/)).
The display renderer has no absolute radiometric calibration. Catalogue records
without B−V use a declared display-colour default of 0.65 in `tools/sidera_assets.py`.
The coefficient sets are complete for the included bodies, with zero term truncation. `tools/sidera_assets.py` packs the published source terms without using the Horizons acceptance gates as a cutoff. The 280-row, 40-site/epoch Horizons oracle with the corrected historical UT1 model measured maxima of Sun 1.868", Moon 1.773", Mercury 1.853", Venus 1.775", Mars 1.645", Jupiter 1.377", Saturn 1.386", phase 0.0000225 and lunar semidiameter 0.090". These are empirical maxima for the frozen vectors, not global error guarantees. Future UTC uses the declared constant-DUT1 projection below.

Earth-rotation time uses [IERS EOP 20u24 C04 daily UT1−UTC](https://datacenter.iers.org/data/254/eopc04_20u24.1962-now.txt) from 2000-01-01 through 2026-08-25. `tools/sidera_ut1.py` stores monthly TT−UT1 knots and the final daily point; all 9,734 source days reconstruct UT1−UTC within 0.005776 s. Beyond the final measured day, the runtime holds its final UT1−UTC at +0.0070542 s as an explicitly unbounded compatibility projection through 2050. Future leap seconds and Earth-rotation drift are unknown, so the 1 s residual is established for the historical interval only. The pinned source SHA-256 was `7e39bb43bd1e1920517ed9316d3c5b14f898fe68afe9e0891bf18c4180b9d272` when the asset was generated.

| Runtime asset | Bytes | SHA-256 | Description |
| --- | ---: | --- | --- |
| `vsop87d.bin` | 692801 | `366f2d95fa86c403d02ce4698a3d30a67194c672ceb48f08a156e86cddc1962b` | IMCCE VSOP87D; all 25,659 heliocentric ecliptic terms for Earth, Mercury, Venus, Mars, Jupiter and Saturn |
| `elp82b.bin` | 1045519 | `9cc28d677a028284e59dd65095eb3dff52ceeea3869882a860c9958228f88b1a` | IMCCE ELP2000-82B; all 2,645 main and 35,227 secondary lunar terms |
| `ybsc5.bin` | 145544 | `146b5dafdeff15c248f14420f5098ea816024ddf67b9b57af0335c2648cbd889` | CDS V/50 Yale Bright Star Catalogue, 5th edition; 9,096 J2000 RA/Dec, V, B−V records, no proper motion |
| `nut00b.bin` | 4089 | `ea3b092df130cd248a2d4760b483bc24d7666a6538d993bff087374b702ec547` | ERFA IAU 2000B; 77 lunisolar terms, BSD license copied as ERFA-LICENSE |
| `ut1_utc.bin` | 1300 | `62a749513dd53139e8e006a40a7db86b85b6d4ec7818bc7ab1675c8414f7141f` | IERS C04 historical UT1−UTC represented as monthly TT−UT1 knots, plus declared constant-DUT1 projection |
| `moon_albedo.bin` | 65536 | `5b5dc8adb6a2e6636f392a40b21ee5d8c31053a5b87c014dcb281e51a5527f25` | 256×256 single-channel relative display albedo from NASA SVS Moon Mosaic 5001; IAU lunar pole orientation, no libration model |

The Moon texture comes from [NASA SVS Moon Mosaic 5001](https://svs.gsfc.nasa.gov/5001/), a 1,231-image LRO NAC nearside mosaic. NASA SVS requests credit to NASA's Scientific Visualization Studio. `tools/sidera_moon_texture.py` verifies source JPEG SHA-256 `36fbe604043f1403acbc4db6fbd36d04db1db1c8a3749aa81d1e6ffea072a5fa` before generating the packed display texture. The generator maps source radius 0.96 to the disc edge, excluding the dark photographic rim while keeping the outer surface. It uses a shoulder tone curve to preserve highlights; the viewer generates mip levels for minification. Its lunar pole follows the IAU WGCCRE model, but its fixed nearside markings do not model libration or measured reflectance.

| Source file | SHA-256 | Source |
| --- | --- | --- |
| `moon_mosaic_print.jpg` | `36fbe604043f1403acbc4db6fbd36d04db1db1c8a3749aa81d1e6ffea072a5fa` | https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005001/moon_mosaic_print.jpg |
| `catalog.gz` | `3dc44b1e90be8fbe5bcc7656032560f51275f985c7e3f783c9028e1838ec7bed` | https://cdsarc.cds.unistra.fr/ftp/V/50/catalog.gz |
| `catalog_ReadMe` | `44fd9c73e2eecad0beb47bdfa3f01c60fd43f93d6964198e31fcd48732de5b33` | https://cdsarc.cds.unistra.fr/ftp/V/50/ReadMe |
| `ee00b.c` | `93f5977283fc78e9b9253d48b038900a4e7bf6b3f7a70fd16c0ae2d6725d0f65` | https://raw.githubusercontent.com/liberfa/erfa/master/src/ee00b.c |
| `ELP1` | `ae30cbffb83a7bd4582a83a32a322d08a48ba057a4df7bf9dd5df9f06b1688fa` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP1 |
| `ELP10` | `dbd82ddc6064e4cc7b4f08fa27b2fcb48f82456ad36a850a0d3ddae098c3e2e6` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP10 |
| `ELP11` | `0ad7a914c9f98008a648881783c9dd4a14692ec14e2e1bfce4709c68d17bd659` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP11 |
| `ELP12` | `8ed7be0ab70f4ffae6b1f711cc4e915257ae5e269fbbfc5a7060f7e952728ba8` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP12 |
| `ELP13` | `643295b3894023b4b1bd6ee2b0ecf5d3ff23d703baccc6302caa05fa8b84f76c` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP13 |
| `ELP14` | `b59d8b9bbef282f2bead538d6906781257a7fb5b8699bb6adcacb070a76f1e89` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP14 |
| `ELP15` | `17ab0d521c178187a5de4847b6696fbcb7a55d77d776568a0c16f33fd3be342a` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP15 |
| `ELP16` | `2bef867d8aad4075bc2711559cf1bc42757501bb10c307ff121152bddd344a66` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP16 |
| `ELP17` | `6cf0746d034ac75ed60d4d16ed0de790fe5b7aad9df7b462091c38020e1b1bfc` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP17 |
| `ELP18` | `b1d93931f6016023c83354cd54a2614978de3b6bc5b7234537d461200f7f4753` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP18 |
| `ELP19` | `dd0b0bd5f5c354683f035ee8a09d82e9c138baaf27758ca311d07c76983bdd2e` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP19 |
| `ELP2` | `c91e5585b0a9e7bd091304b164ce89a6461acd0e439d47957c890aec1e031e08` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP2 |
| `ELP20` | `0f1d571879dc9b1a6f7698b403bef26ab151c3ec2ed42cbdec5b297cb0464a8e` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP20 |
| `ELP21` | `1546d0e8af01f759f9dfb7a3bf4334a3e579f59e5c22edab29d660dede2ed4b4` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP21 |
| `ELP22` | `44263bb254c2b6c0df963bddfd1ecfd01950668fd913f9e6ba6da6ff729f3e41` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP22 |
| `ELP23` | `38917cf2cbe0f9afcd271444a47066b098f9a8792836ec85dac359f4c11b9464` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP23 |
| `ELP24` | `ca67e5db5933887130709767c1eb3ac009e4bce596978bad7995c416ee71c59f` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP24 |
| `ELP25` | `fd0cb03d496cbf23bf7bebef40aa09019c6a50072fc7fd2573645f26a56ab635` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP25 |
| `ELP26` | `d5b2a33974b099448a35536987b2f08aff5b11d5801ffa55c87ae8a1d26e9f4f` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP26 |
| `ELP27` | `648379d85e1753cc37bc3852b63899d414a05d3aea1c8621dc44e9fdfff234f1` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP27 |
| `ELP28` | `0785b8e002887799bc303e8be1abd71c37ae9de671bffbdeb8f50998f892f18e` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP28 |
| `ELP29` | `816fd1a94b1cb4e2e5e6cec72971f27ad0f2bf987735d184586fdadd033c3d02` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP29 |
| `ELP3` | `862a8e4c8e70ce8b28383be4c9f2e2c025a8f633d7b7a811eb3afdab4ed9f354` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP3 |
| `ELP30` | `cff5ab4c84a6a36855e5b2e2f47e1e0e1d605e789ff2954755dc64d188067a13` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP30 |
| `ELP31` | `c2fc53c2442c1b61404991f31c859cc5eb300eb66bb764e1f16c70fe8d199dc3` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP31 |
| `ELP32` | `7a07397b63d1ade0909c12be9024632de6ff27fd9e8e410e5f97f821d2390a60` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP32 |
| `ELP33` | `459ea9eff9a9d7b5c224245f5edb113060174b4991d3adf7d27079719bcf2339` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP33 |
| `ELP34` | `b83178e98bd33e8f26ef6662e03455761ffac7dae399ad1ccb4bf028c5f0e774` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP34 |
| `ELP35` | `692d1752a7ea28c7157dbf694750af6ea2cbf74c744c4b792a7e8bf1bb5ad7d7` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP35 |
| `ELP36` | `1f8eec292def5ceb4fff9a09ca678bd81e9cdc23ff7bdc802f2281c837357bda` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP36 |
| `ELP4` | `f27ea439bf8f4fd35bed31c0a42de5414db07f9fcc3587237f6908891d43d773` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP4 |
| `ELP5` | `6803422481e4decae4a59f89d4f94c7b33af21d293d9bc807d565f51c29b9915` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP5 |
| `ELP6` | `2a6be4d33dfce4cf2351d295b4aade8f34cc476a97747b3eccb12763658e1fd1` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP6 |
| `ELP7` | `35491a0c73ff6bcb136d8f54db89d8df2fb741af43aed9707618e3a925df474d` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP7 |
| `ELP8` | `f3e7f4c851e7f9ac1a0556fe7e685e44612f3dca317ab605eb35f565251e020e` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP8 |
| `elp82b_1` | `edd0f96d7baa4344497584eeec6e8abf615d7d9f9e8d397a341b4eb5e16872c9` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/elp82b_1 |
| `elp82b_2` | `751308209d00ce7f1ddc6b8ee1d02dcec4aac706607026f5eb63934df1bdb886` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/elp82b_2 |
| `ELP9` | `574347346363df52c7602f56747b790e9cbe60152127d24451c6a1633bc79f0e` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/ELP9 |
| `elp_README` | `5f33e4d27afddda2d9c6c260e0467f68ff789381521ceab7464fe446e28d83f6` | https://ftp.imcce.fr/pub/ephem/moon/elp82b/README |
| `fw2m.c` | `480d1ae7780566980b8f80f4c6daa2be330a80e99861d892dc5a9f7b11da47c0` | https://raw.githubusercontent.com/liberfa/erfa/master/src/fw2m.c |
| `gmst06.c` | `5adeede2ed23b37209870778e52083bd57a7870c5b4779c425cf254d26d2e420` | https://raw.githubusercontent.com/liberfa/erfa/master/src/gmst06.c |
| `nut00b.c` | `a5246d525e987eb587bbe786b2846160e01be85b5d12af573870b89235695f49` | https://raw.githubusercontent.com/liberfa/erfa/master/src/nut00b.c |
| `obl06.c` | `550260874d5e5fa3be77296b85b92c54c726fee6fbb1bb44748d04c17ffb37c2` | https://raw.githubusercontent.com/liberfa/erfa/master/src/obl06.c |
| `pfw06.c` | `085f49861026fab98633d8479795090b3278a0d246a938ebdf10fde326ef1987` | https://raw.githubusercontent.com/liberfa/erfa/master/src/pfw06.c |
| `pr00.c` | `4b5a5ad183f69d7b62b2adbfefe742dea52e2a244bb22a4314af35c3c5a0d511` | https://raw.githubusercontent.com/liberfa/erfa/master/src/pr00.c |
| `VSOP87D.ear` | `8b160c859136d467f2be7fc29efa8a9652e95516dfbde00e4c739d7ddc90ca91` | https://ftp.imcce.fr/pub/ephem/planets/vsop87/VSOP87D.ear |
| `VSOP87D.jup` | `3f3dfbc7d117ecad2b2dadf2fc626b260a3cd5efa98e7d4c6b26cd682fc48090` | https://ftp.imcce.fr/pub/ephem/planets/vsop87/VSOP87D.jup |
| `VSOP87D.mar` | `b1184df9553d85ffcf904c16bd437ab668804fa98859f27fe2e7bf6cfa6bc07e` | https://ftp.imcce.fr/pub/ephem/planets/vsop87/VSOP87D.mar |
| `VSOP87D.mer` | `f468481b5a05080a943ad4746ff7ea7e0ff6652b71a46d83c9c636cb69485e34` | https://ftp.imcce.fr/pub/ephem/planets/vsop87/VSOP87D.mer |
| `VSOP87D.sat` | `2e49e19396f24c17298f0b667e7763ee5c28b60d549c89d72a17dfd5f8d46b05` | https://ftp.imcce.fr/pub/ephem/planets/vsop87/VSOP87D.sat |
| `VSOP87D.ven` | `cb2f3a738289ed45f69fec1845e480baf4b32d481eccc21b8629a2d0d10e8261` | https://ftp.imcce.fr/pub/ephem/planets/vsop87/VSOP87D.ven |

The disposable source cache is regenerated by `tools/sidera_fetch_sources.py`; `tools/sidera_assets.py` packs the complete theory and catalogue tables. `tools/sidera_horizons.py` generates the independent Horizons oracle. The old `tools/sidera_delta_t.py` output is no longer shipped or used for Earth rotation.
JPL oracle settings and the aggregate raw response digest are stored in `tests/data/horizons_vectors.dat`. The runtime assets do not call the network.

References: [IMCCE VSOP87](https://ftp.imcce.fr/pub/ephem/planets/vsop87/), [IMCCE ELP82B](https://ftp.imcce.fr/pub/ephem/moon/elp82b/), [CDS Yale V/50](https://cdsarc.cds.unistra.fr/ftp/V/50/ReadMe), [ERFA](https://github.com/liberfa/erfa), [JPL Horizons](https://ssd.jpl.nasa.gov/horizons/).
