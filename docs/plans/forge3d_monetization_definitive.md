# forge3d Monetization Plan
## The definitive version

*Merged from two independent analyses. Stripped of TAM rhetoric. Ranked by fastest cash, specific buyers, and lowest build effort.*

---

## 1. The organizing question

This plan answers one question: **who will pay in the next 30–90 days because forge3d removes pain from their workflow?**

Not "what is the total addressable market for 3D rendering." Not "what large companies could eventually be built on top of forge3d." Those questions are interesting. They are not urgent. The urgent question is about the first euros, because the first euros validate everything that follows.

### Five principles that govern this plan

1. **Monetize packaging, not the engine.** Keep the core renderer useful and visible. Charge for the layer that creates commercial value: exports, templates, batch workflows, branded deliverables.
2. **Sell to professionals before hobbyists.** Hobbyists bring attention. Professionals bring money. Every pricing decision should be legible to someone billing a client.
3. **Start with narrow paid wedges.** Do not launch a giant Pro suite with 20 features. Launch 3–5 painful-to-replicate premium capabilities and see what people actually buy.
4. **Use services to discover product demand.** Consulting is not a distraction. It is the fastest validation loop. Every engagement tells you which features deserve productization.
5. **Prioritize channels you already control.** 70k+ Instagram, 30k+ X, a YouTube channel, a consulting pipeline, an upcoming book, and your own apps. Every near-term euro flows through that ecosystem, not through cold PyPI discovery.

---

## 2. Three blockers that gate every euro

These findings come from a code-level audit of the forge3d repository. **Nothing ships until all three are resolved.** Budget 10–16 person-days total across Weeks 1–2.

### Blocker 1: Pro license enforcement is a placeholder

The `_verify_signature()` function accepts any non-empty string. The key parser expects `F3D-TIER-YYYYMMDD-signature` and recognizes `PRO` and `ENTERPRISE` tiers, but signature verification is explicitly stubbed. Any developer who inspects the source — and your target users are developers who routinely inspect Python source — can generate their own valid-looking key and unlock all Pro features without paying.

**Fix:** Implement Ed25519 offline signature verification in the Rust binary (not Python). Ship a signing pipeline that generates keys tied to a customer identifier and expiration date. The verification code must be in the compiled binary, where it is harder to patch than a Python module. Estimated effort: 5–8 person-days.

### Blocker 2: Distribution funnel is severed

`pip install forge3d` returns 404. The configured homepage returns 502. The repository contains a tag-triggered PyPI publish workflow via OIDC, but the package is not currently live. A potential buyer who hears about forge3d cannot install it, cannot read documentation, and cannot navigate to a purchase page. The entire discover → install → evaluate → buy funnel is broken.

**Fix:** Trigger a PyPI publish and verify the package installs on all target platforms (Linux x86_64, Linux aarch64, macOS universal2, Windows). Restore the homepage. Verify all links from README, PyPI metadata, and the Pro key prompt resolve correctly. Add uptime monitoring. Estimated effort: 3–5 person-days.

### Blocker 3: License metadata inconsistency

The LICENSE file says MIT. The package metadata declares Apache-2.0 OR MIT. The repo URL in metadata points to `github.com/forge3d/forge3d` (404) instead of the actual repository at `github.com/milos-agathon/forge3d`. Documentation labels version 0.39.0 while the package declares 1.12.2. Enterprise procurement will flag every one of these discrepancies.

**Fix:** Align the license (choose one or ship both full texts with clear scope). Correct the repository URL in all metadata. Sync documentation and package version numbers. Estimated effort: 2–3 person-days.

> **Non-negotiable:** These blockers are not "commercial cleanup." They are showstoppers. A Pro tier with a bypassable license, an uninstallable package, and inconsistent metadata will not convert a single paying user. Weeks 1–2 are about fixing these, full stop.

---

## 3. Who pays, and why

Abstract market sizes are deliberately excluded. Instead, each segment is described by its specific pain and what it will pay to relieve.

| Segment | Their pain | What they'll buy | Price sensitivity |
|---------|-----------|-----------------|-------------------|
| **GIS consultants** | Standard GIS outputs look flat. Production workflows are manual, inconsistent, and hard to reproduce across projects. | Batch rendering, premium templates, report exports, consulting + license bundles | Low. They bill clients €800–2,000/day. A €99/mo tool that saves a day pays for itself instantly. |
| **Environmental / planning firms** | Need presentation-grade terrain, flood, visibility, and infrastructure visuals for reports. Current tools produce analysis layers, not communication materials. | Consultancy templates, scene automation, reproducible report graphics | Low–medium. Report budgets are substantial. Software is a rounding error next to staff costs. |
| **LiDAR / DEM vendors** | Their data is strong but their visual presentation is weak. Marketplace previews and sample packs look generic. | Automated dataset preview packs, branded render pipelines, white-label deliverable generation | Medium. Marketing budgets exist but are smaller than service budgets. |
| **Technical creators / educators** | Want striking 3D map visuals without building a rendering pipeline from scratch. | Pro templates, creator presets, export packs, notebook/Quarto integration | Higher. Individual budgets. But strong distribution value and brand fit. |
| **Research / data journalism teams** | Hard to produce unique, publication-grade 3D figures reproducibly. Current workflow: open a GUI, manually export, paste into paper. Non-reproducible. | Publication presets, figure export workflow, reproducible scene configs | High for individuals, low for institutional licenses. Strong showcase value. |

---

## 4. Honest competitive positioning

The previous report's weakest claim was "nobody else serves this." Several adjacent tools and workflows overlap with forge3d on at least one dimension. The defensible claim is narrower but real: no other tool combines headless Rust/wgpu rendering, pip-installability, true server-side operation without xvfb or a browser, and PDF/SVG vector export in a single Python-native package.

That's a genuine gap. It's not a vacuum. Here is the precise map:

| Tool | Overlaps on | forge3d's edge | Their edge | Threat level |
|------|------------|----------------|------------|-------------|
| **PyVista** | 3D visualization from Python | Headless without xvfb; PDF/SVG export; Rust performance | Mature ecosystem; large community; VTK backend | Medium. Different rendering approach, but Python GIS users may choose it by default. |
| **pymgl** | Server-side map rendering | 3D terrain, elevation data, shadow mapping | MapLibre style support; proven production use | Low. Explicitly no 3D terrain support. Complementary, not competitive. |
| **Cesium** | 3D geospatial visualization | Headless batch; pip-installable; no browser or DOM required | Globe-scale; 3D Tiles; massive enterprise adoption | Low–medium. Different scale and deployment model. |
| **QGIS 3D** | 3D terrain rendering | Scriptable headless; embeddable in pipelines; vector export | Free; full GIS stack; enormous community | Medium. Free and familiar. But desktop-only, not server-safe. |
| **BlenderProc** | Headless 3D rendering | 10–50 MB container vs 300 MB+; faster cold start; PyO3 zero-copy | Full PBR; physics; massive format support | Low for geo workflows. High if forge3d enters synthetic data. |
| **Open3D** | Point cloud / 3D processing | Terrain-focused; map-specific workflows; vector export | Broader 3D processing; reconstruction; algorithms | Low. Different focus area. |
| **rayshader (R)** | 3D terrain maps | Python-native; server-safe; Rust speed; no R dependency | Mature; beautiful defaults; loyal R community | High for mindshare. Low for direct competition (different language). |

**Positioning line:** *"forge3d is the fastest way for Python teams to produce premium 3D geospatial deliverables — headless, reproducible, and export-ready — without xvfb, without a browser, and without leaving Python."*

---

## 5. Eight opportunities ranked by fastest cash

Each opportunity is evaluated on: how quickly it generates revenue, who the specific buyer is, how much new code it requires, and how it connects to the next opportunity in the sequence. Revenue estimates are conservative and denominated in euros.

| # | Opportunity | First buyer | Price point | New code | Cash in | Validates |
|---|-----------|------------|------------|---------|--------|-----------|
| 1 | Consulting accelerator | GIS teams, agencies, municipalities | €1.5–10k/project | None | Week 3 | Which features to productize |
| 2 | 3D map print storefront | Your 100k+ audience | €19–199/print | Minimal | Weeks 4–6 | Pro export pipeline end-to-end |
| 3 | Forge3D Pro: narrow export tier | Consultants, publishers, creators | €29–99/mo | Low | Month 2–3 | Developer-tool product-market fit |
| 4 | Template & style packs | Book readers, GIS devs, creators | €29–79/pack | Low | Month 2–3 | Willingness to pay for taste |
| 5 | Batch CLI for agencies | GIS consultancies, data vendors | €99–299/mo | Medium | Month 3–4 | Production workflow lock-in |
| 6 | Quarto / R Markdown bridge | R users, research teams, analysts | Pro-only feature | Medium | Month 3–5 | Stickiness in report workflows |
| 7 | Sunlight/shadow for buurt-check | buurt-check users, urban planners | Premium feature | Medium–High | Month 4–6 | Internal demand for external API |
| 8 | Python terrain library (rayshader equiv.) | Python GIS community globally | €49–199/mo Pro | High | Month 6–9 | Long-term growth engine |

### 1. Consulting accelerator

**What it is.** Sell done-for-you or done-with-you client work: 3D terrain maps, LiDAR visual products, automated render pipelines, planning/report graphics, branded geospatial scenes. You deliver the outcome. forge3d is the engine under the hood.

**Why it's first.** Zero new code required. You have the technical credibility (700+ maps), the audience, and the consulting practice already. Every engagement doubles as product research: you learn which features clients value, which workflows break, and what they'd pay for as self-service.

**Specific buyers.** Environmental consulting firms needing automated terrain renders for impact reports. Real estate analytics teams needing 3D neighborhood visualizations. Government mapping agencies with reproducibility requirements. LiDAR vendors needing QA previews of point cloud deliverables.

**Revenue.** €1.5–10k per engagement, 2–4 engagements per month once pipeline is warm. €3–40k/month with zero product overhead. This funds everything else.

**Risk.** Doesn't scale beyond your time. That's fine — it's a bridge, not a destination. Every service engagement that you find yourself repeating is a product feature waiting to be extracted.

### 2. 3D map print storefront

**What it is.** On-demand store: customer picks a location, chooses a style preset, receives a 3D-rendered terrain map as digital download or physical print. Fourthwall already handles checkout and fulfillment. This adds a "Design Your Map" step powered by forge3d.

**Why it's second.** Your social audience is the distribution channel. Every map you post is a product sample with zero customer acquisition cost. Digital downloads carry ~95% margin. Print-on-demand means zero inventory. The only new work is a lightweight frontend (React + Leaflet map selector) and a rendering pipeline script.

**Specific buyers.** Your existing followers: outdoor enthusiasts, cartography fans, data-viz people. Secondary: hikers, travelers, real estate agents who want neighborhood gifts, corporate gifting buyers. These people already engage with your content.

**Revenue.** €2–10k/month within 8 weeks of launch. More importantly, this stress-tests the Pro export pipeline (SVG/PDF) end-to-end with real customers before any developer-tool launch.

**Pricing tiers.** Digital PNG €19–29, SVG+PDF bundle €39–49, unframed poster €59–79, framed/canvas €99–199, limited edition signed runs €149–299.

**Risk.** Rendering speed for custom locations. Mitigate by pre-rendering popular locations (major cities, national parks, iconic peaks) for instant delivery and queuing custom requests.

### 3. Forge3D Pro: narrow export tier

**What it is.** The existing Pro boundary — MapPlate composition, SVG/PDF vector export, scene bundles, style spec workflows — sold as a monthly subscription. Free tier: learning, experimentation, basic scenes, PNG snapshots. Pro: production exports, premium presets, batch workflows, branded deliverables.

**Why it matters.** This is the actual product-market fit test. If consultants and publishers will pay €29–99/month for export capabilities, that validates the library model. If they won't, you lean harder into consulting and prints. Either answer is valuable.

**Critical:** Launch with 3–5 painful-to-replicate premium capabilities, not 20 features. Keep the boundary clean and easy to explain: "Free to explore, Pro to deliver."

**Revenue.** €3–8k/month by month 4–5 if the rendering quality is compelling. Scales with installs and community growth.

**Dependency.** Requires all three blockers resolved. The license must actually work before you sell it.

### 4. Template and style packs

**What it is.** Curated bundles of forge3d style presets, color palettes, camera angles, and rendering recipes packaged as downloadable config files. Examples: terrain storytelling pack, environmental impact pack, LiDAR preview pack, report/publication pack, poster/print aesthetic pack, alpine/relief cartography pack.

**Why it works.** Your 700+ maps represent years of aesthetic refinement that others cannot easily replicate. This turns your taste into a product. Low engineering burden, strong leverage of design sensibility, and a natural cross-sell with the rayshader book audience.

**Revenue.** €1–5k/month. Small but nearly pure margin, and it reinforces the brand.

### 5. Batch CLI for agencies

**What it is.** A paid command-line tool that takes a YAML/JSON config file and batch-renders dozens or hundreds of terrain maps unattended. The "render recipe → deliverable pack" pattern from forge3d's scene bundle architecture, productized.

**Why agencies pay.** A consultancy producing 50 terrain renders for a regional planning report wants a config file and a command, not 50 interactive sessions. Reproducibility matters for compliance and audit trails. Once embedded in production workflows, churn drops.

**Revenue.** €99–299/month per seat. 20–50 seats in year one is realistic. €2–15k/month.

**Best packaging.** Not a standalone product. Part of Pro Commercial or Team tier.

### 6. Quarto / R Markdown bridge

**What it is.** A bridge that makes forge3d easy to embed into Quarto reports, R Markdown documents, Jupyter notebooks, and reproducible analytical documents. A figure factory mode where forge3d renders 3D figures during document builds.

**Why it works.** This connects forge3d to a workflow people already budget for. It aligns with your strongest audience overlap (R/Python GIS community) and the rayshader book's readership. It makes forge3d harder to replace once embedded.

**Revenue.** Pro-only feature. Drives subscription retention rather than direct revenue.

**Timing.** Choose this OR the batch CLI as the first "sticky workflow" feature. Not both initially. Ship whichever consulting engagements indicate higher demand.

### 7. Sunlight/shadow analysis for buurt-check

**What it is.** Use forge3d as the rendering engine for automated shadow studies on buurt-check property listings. 3D BAG building geometry + coordinates + date/time range → sunlight exposure heatmap and shadow animation.

**Why it's here and not higher.** The rendering work (sun position calculation, shadow mapping, heatmap generation) is real engineering. But it has captive internal demand and external demand from urban planners who currently pay €500–5,000 per manual shadow study.

**Revenue path.** Initially a premium buurt-check feature. Later, potentially an API product. Don't build the API until buurt-check usage validates demand.

### 8. Python terrain library (the rayshader equivalent)

**What it is.** The full-featured pip-installable Python library that replicates and extends rayshader: elevation matrix in, publication-quality 3D terrain out, headless, server-safe, with shadow mapping, ambient occlusion, water detection, and composable hillshading.

**Why it's last despite being the biggest.** Because it's 6–9 months of focused development before it competes with rayshader's output quality. Everything above it generates cash while you build this. When it ships, it becomes the growth engine that pulls all other products forward. But it ships with revenue already flowing, not as a bet-the-company first move.

**Competitive honesty.** PyVista handles some 3D terrain workflows (with xvfb pain). QGIS 3D exists for interactive desktop use. Cesium covers globe-scale. None combine headless + pip-installable + Rust-fast + vector export. That's the real gap. "No one else does this" is too strong. "No one else bundles these specific capabilities in a Python-native package" is accurate.

---

## 6. The 90-day execution timeline

Every row answers: what do you do, what does it cost, and when does money arrive?

| When | Action | Revenue | Effort | Validates |
|------|--------|---------|--------|-----------|
| **Weeks 1–2** | Fix all three blockers. Ed25519 signing, PyPI publish, metadata alignment, homepage restoration, end-to-end funnel test. | €0 | 10–16 person-days | Can people install and buy? |
| **Week 3** | Launch consulting: reach out to 3–5 GIS/environmental/real-estate contacts who need terrain deliverables. Ship first engagement. | €1.5–10k first project | 0 new code | What do clients actually value? |
| **Weeks 3–6** | Build map print pipeline + storefront integration. Create 5–10 style presets. Soft launch to Instagram/X with early-bird pricing. | €2–10k/mo | ~2 weeks dev | Pro export pipeline under real load |
| **Weeks 6–8** | Launch narrow Pro tier with working license. Announce on social channels. Early-bird annual pricing. | €1–5k/mo | ~1 week packaging | Will devs pay for exports? |
| **Weeks 6–10** | Ship 3–5 template/style packs. Cross-sell to book pre-order list and consulting pipeline. | €1–5k/mo | ~1 week curation | Willingness to pay for taste |
| **Weeks 8–12** | Build batch CLI. Beta with 2–3 agency contacts from consulting. Decide: Quarto bridge or batch CLI first based on demand signals. | €2–8k/mo | ~3 weeks dev | Production workflow stickiness |
| **Month 4+** | Begin sunlight/shadow work for buurt-check. Start terrain library funded by revenue above. | Reinvest | Ongoing | Long-term engine viability |

> **Conservative 90-day total:** €10–40k from consulting + €5–20k from prints + €3–10k from Pro/templates = €18–70k in the first three months with minimal new code. Not venture scale. Validation capital.

---

## 7. The product ladder

This is what forge3d should sell, structured so a buyer can answer "what do I get if I pay?" in under 10 seconds.

### Free tier

- Basic rendering, learning, experimentation
- Simple scenes, camera/sun control, PNG snapshots
- Enough quality to attract people and build habit

### Forge3D Pro (€29–99/month)

- Premium export resolution (SVG, PDF, high-res PNG)
- Premium preset library
- Batch export mode
- Commercial-use template bundles
- Publication/report presets
- Advanced scene controls (MapPlate compositor, style spec)

### Forge3D Pro Commercial (€149–299/month)

- Everything in Pro
- Batch CLI for config-driven renders
- Team/seat licensing
- White-label deliverable generation
- Priority support

### Services layer

- "We build your render pipeline"
- "We produce your premium 3D map/report visuals"
- "We create your branded scene system"
- Per-project or retainer pricing

### Add-ons (one-time purchases)

- Template packs (terrain storytelling, environmental impact, LiDAR preview, poster aesthetics)
- Vertical style packs (Dutch planning, alpine cartography)
- Quarto/R Markdown bridge (when ready)
- Dataset preview generator

---

## 8. What not to build yet

These are explicitly parked, not discarded. They are valid directions once forge3d has revenue, users, and a proven rendering core. They are wrong first moves.

### Server-side rendering API platform

Billing, queues, retries, auth, GPU orchestration, observability, and customer support is a company, not a feature. Revisit at €20k+ MRR.

### Scientific 3D figure platform

Real market, but academic sales cycles are 6–12 months and budgets are glacial. Build after the terrain library proves out.

### BI dashboard 3D layer

Requires deep integration (Power BI SDK, Tableau Extensions API) far from your current architecture and audience. Year 2+.

### E-commerce product rendering

Different sales motion, different buyer profile, different quality bar (PBR materials, studio lighting). Poor fit with current brand.

### Synthetic data rendering engine

Interesting technical overlap with forge3d's wgpu core. But 9–15 months to market, deep ML tooling expectations, and direct competition with NVIDIA's ecosystem. Year 3.

### Climate / environmental platform

Valuable but requires domain-specific data pipelines (NetCDF, GRIB), regulatory knowledge, and government procurement cycles. Not a quick win.

### "Stripe for 3D rendering" positioning

Aspirational framing that will confuse buyers today. Earn this narrative after you have paying customers, not before.

---

## 9. The real moat

The previous report framed forge3d's moat as a technical capability stack: Rust-native, headless wgpu, pip-installable, multi-backend GPU support. That's necessary but not sufficient. In the first 12 months, the actual moat is the ecosystem around the code:

- **A 100k+ audience** that already cares about 3D cartography and trusts your visual taste
- **A consulting practice** that generates cash while validating product decisions
- **A book launch (July 2026)** that creates a natural cross-sell moment for Pro and template packs
- **A property intelligence app (buurt-check)** that provides captive internal demand for 3D rendering features
- **700+ maps worth of aesthetic DNA** that competitors cannot easily replicate even if they match the technical stack
- **An open-source community entry point** where the free tier builds trust and distribution that Pro monetizes

forge3d's Rust/wgpu architecture makes all of this possible. But the revenue comes from the ecosystem around it. Technical excellence is the floor, not the ceiling.

---

## 10. Bottom line

*The fastest money is not "sell a graphics engine." The fastest money is sell outcomes, productize the most repeatable parts, and wrap them into Pro features, templates, and workflow integrations.*

The monetization ladder:

**Consulting → Prints → Pro → Templates → Batch CLI → Workflow bridge → Terrain library**

Each step generates cash or validates the next step. No step requires venture funding or a giant market thesis. Fix the foundation, sell outcomes, and build from there.
