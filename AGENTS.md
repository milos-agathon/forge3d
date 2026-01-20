# forge3d – Agent Orientation Guide

Welcome to the **forge3d** repository! This guide equips AI coding agents with a concise, code-grounded overview of the project so they can navigate, modify and debug the codebase effectively without breaking tests or violating architectural assumptions. It consolidates the essential content of the original guide, removes duplication and reorganises the material for clarity, while preserving all key rules and principles.

## 1. Project orientation and constraints

`forge3d` implements a cross-platform rendering engine in Rust with a Python facade. To succeed as a contributor you must internalise the following constraints:

* **Layering:** The Rust crate under `src/` contains the core rendering engine and performance-critical code. The Python package under `python/forge3d/` is a thin, high-level wrapper. Maintain this layering: core logic and GPU interaction belong in Rust; Python coordinates and tests it. Tests are written in Python, never in Rust.
* **Platforms and build system:** The project supports `win_amd64`, `linux_x86_64` and `macos_universal2`. Builds use CMake ≥ 3.24, Cargo/Rustc, PyO3 and VMA (Vulkan Memory Allocator). Respect the GPU budget (≤ 512 MiB host-visible heap) and design for Vulkan ≥ 1.2 and WebGPU/WGSL. Use RAII to manage resources; avoid leaks and fragmentation.
* **Tests and documentation:** Behavior is defined by the Python tests in `tests/` and the Sphinx documentation in `docs/`. When changing semantics or signatures, update tests and docs accordingly. Never break existing tests; new features must be covered by tests.
* **Feature flags and QA:** Honour memory budgets, GPU feature toggles and acceptance tests. Keep code portable across platforms. Monitor GPU memory usage and avoid host-visible memory fragmentation.

## 2. Clean code fundamentals

Clean code is readable, predictable and easy to change. Bad code compounds costs over time. This section distils the core practices you must follow.

### 2.1 Discipline and professionalism

* **Treat cleanliness as mandatory:** Maintaining tidy code is a professional obligation, not a luxury. Messy code slows progress and hurts maintainability.
* **Keep files small:** New source files must remain under about 300 lines. Split large files into logical modules.
* **Leave things better:** Whenever you touch a file, improve at least one small thing. Small refactorings accumulate and pay off.
* **Balance shipping and quality:** Even under deadline pressure, carve out time for minimal refactoring. Cleanliness now saves time later.

### 2.2 Small units and naming

* **Single responsibility:** Keep functions, classes and modules tiny and focused on one thing. Each class must have one dominant reason to change.
* **Meaningful names:** Names must reveal intent. Use nouns for data structures (e.g. `mesh`, `camera_config`) and verbs for functions (e.g. `load_texture`, `submit_frame`). Avoid misleading, ambiguous or inconsistent terminology, unnecessary abbreviations and overloaded words. Names must be pronounceable and searchable.
* **Consistent vocabulary:** Use one term per concept throughout the codebase. Distinguish GPU concepts (e.g. `command_buffer`, `descriptor_set`) from domain concepts (e.g. `elevation_tile`, `landcover_class`).

### 2.3 Comments and documentation

* **Comment why, not what:** Prefer expressing intent through clear code. Comments must explain *why* decisions were made, describe non-obvious consequences or document legal/regulatory notes. They must not restate the code.
* **Use API documentation:** Public interfaces deserve carefully written docstrings or Sphinx documentation. Keep these up-to-date when signatures or behavior change.
* **Avoid clutter:** Do not leave stale comments, commented-out code, journal notes (“changed this on date X”) or noise comments like `// increment i`. Source control tracks history, and the code must speak for itself.

### 2.4 Formatting and whitespace

* **Vertical whitespace:** Group related lines into blocks separated by blank lines. Keep related code vertically close and separate unrelated concepts.
* **Horizontal spacing and indentation:** Use spaces to break up complex expressions. Maintain consistent indentation to reflect scope. Let automated formatters enforce the team style and avoid manual alignment that breaks easily when code changes.

### 2.5 Functions and argument lists

* **Tiny functions:** Functions must be very small and operate at one level of abstraction. Code must read like well-structured prose. Respect the step-down rule: high-level policy functions call lower-level detail functions in top-down order.
* **Manage parameters:** Limit argument lists. Group related parameters into structs or tuples; avoid flag arguments and output parameters. Functions must either perform an action (command) or answer a question (query), not both.
* **Minimise side effects:** Side effects must be explicit or minimised. Use Rust’s `Result`/`Option` types (or exceptions in other languages) rather than scattering error codes. Isolate error handling so it does not pollute main logic.

### 2.6 Refactoring methods

* **Correctness first:** When improving a method, ensure behavior is correct and covered by tests before refactoring.
* **Extract and clarify:** Break big functions into smaller ones with meaningful names. Clarify names, remove duplication and separate concerns. Use extraction to discover latent classes and domain concepts.
* **File structure:** Organise files like a newspaper: high-level concepts at the top, then details. Apply the step-down rule at the file level; avoid mixing high-level orchestration with low-level operations.

## 3. Design, testing, and AI assistance

### 3.0 AI Evidence & Stop-Conditions

1. **No “DONE/PASS” without pasted command output.** Every checklist item must include the raw terminal output block.
2. **Always include a “Remaining checklist” section** at the end of every report; any unchecked item blocks “done”.
3. **Never add new scripts/tools/logging** unless the task explicitly asks—or you first justify it as unavoidable and minimal.
4. **Treat presets as schema-bound**: before editing, print/quote the exact preset keys read by the loader and where they’re parsed.
5. **For any new CLI flag**: update *all three* (argparse definition, preset param_map, CLI override precedence map) or explain why not.
6. **Boolean flags require explicit precedence semantics**: define how `--flag`, `--no-flag`, defaults, and preset values interact.
7. **A/B tests must specify**: identical scene, identical camera, identical preset baseline, *only one variable changed*.
8. **IBL/GI validation requires a “forced impact” scene**: if delta is near-zero, create a test case where GI must change output (e.g., kill sun, boost IBL, or use controlled material).
9. **Logs are not proof of effect.** Require numeric deltas (ROI metrics, histogram, luminance stats) or pixel diffs.
10. **Whenever you modify shader branching**, add a runtime “feature bit” readback or visible debug mode (only if requested) to prove branch execution.
11. **Do not cite line numbers as proof** unless you paste the relevant snippet from the current tree.
12. **Warnings policy**: if cargo produces warnings, state whether any are new; if new, fix or justify explicitly.
13. **Regression policy**: define the canonical verification suite (exact commands) and run all of it before concluding.
14. **Add a test when you fix a bug** (esp. precedence bugs): unit test or integration test that fails on the old behavior.
15. **Preserve baselines by construction**: new features default off; presets pin old behavior; comparisons must include the pinned preset.
16. **Don’t “handwave minimal impact.”** If a feature is claimed to work, provide at least one case where it measurably changes output.
17. **Separate “config honored” vs “render changed”** as two different acceptance checks.
18. **Document assumptions**: if anything is assumed (adapter support, sRGB format selection, etc.), list it + how to verify.
19. **Edits must be minimal and attributable**: prefer one layer at a time; if multiple layers change, explain the dependency forcing it.
20. **Always end with “What remains / risks”** even when everything is green (e.g., “no automated test yet for X”, “GI delta small in this scene”).
21. **No “root cause found” without a falsifiable hypothesis + disproof of alternatives.** Provide (a) hypothesis, (b) minimal experiment, (c) result, (d) what it rules out. If any of these is missing, stop.
22. **Do not generalize from a single scene.** If a rendering feature seems to have “no effect,” you must add a *forced-edge/forced-impact* scene that amplifies the expected difference (e.g., hard shadow edge, occluder silhouette). Otherwise, stop.
23. **Never “fix” by changing product meaning.** Do not remove/rename user-facing modes (e.g., `vsm/evsm/msm/csm`) unless the task explicitly instructs it. If you believe removal is necessary, stop and produce a decision memo with tradeoffs and a migration plan.
24. **Shadow terminology must be explicit.** Distinguish **pipeline** vs **filter** vs **technique** (e.g., “CSM pipeline + PCF filter”). Any ambiguity in code/CLI/help must be resolved with clear naming and docs.
25. **Uniform-layout changes require hard proof.** If you touch a Rust↔WGSL uniform, you must provide: (a) Rust `size_of` + `align_of`, (b) WGSL struct snippet, (c) `offset_of!` table for all fields used by shader, and (d) at least one runtime wire-test proving the shader reads the intended value.
26. **Shader-compile provenance is mandatory.** When shader edits are involved, include a “wire test” (sentinel-driven visible output) plus the exact rebuild/install commands and their raw output. If you cannot prove the new shader is in the built extension, stop.
27. **Do not introduce debug modes as permanent API surface.** Debug toggles must be (a) behind an existing debug flag mechanism, (b) default-off, (c) documented as temporary, and (d) removed or guarded before final merge unless explicitly requested.
28. **No “hash-only” conclusions for perceptual claims.** If claiming “looks flatter / more dramatic,” you must back it with at least one numeric perceptual proxy: edge-width histogram on shadow boundaries, ROI gradient magnitude stats, or pixel-diff heatmap summary. Hashes alone are insufficient.
29. **Camera semantics must be printed, not assumed.** For any camera issue (phi/theta/fov), print computed `eye`, `target`, `up`, and derived basis vectors (or matrix) for at least 2 settings, and show that only the intended variable changed.
30. **Perspective vs orthographic claims require a geometry probe.** Provide a lighting-independent probe render (e.g., view-space depth visualization) demonstrating FOV/theta changes in *geometry*, not just shading. Without this probe, stop.
31. **Reference matching must be parameterized.** When comparing to Blender/reference images, provide a mapping table: coordinate system, FOV, camera elevation, azimuth, exposure/tonemap, sun direction, units (meters/world scale). If you cannot map parameters, stop and state what is unknown.
32. **Do not “explain away” user observations with heuristics.** E.g., “camera-sun alignment causes flatness” is not acceptable unless you show an A/B where only alignment changes and the specific missing effect appears/disappears.
33. **Avoid scope creep via “helpful” new features.** Do not add new rendering paths (mesh mode, ray reconstruction, etc.) unless asked. If you believe it’s required, provide a strict minimal-change plan and stop before implementing until the plan is acknowledged in the task.
34. **Respect the frozen default behavior.** Any new path must be opt-in with defaults unchanged, and must include a pinned preset demonstrating the old behavior. If you changed defaults inadvertently, stop and revert.
35. **No silent fallback behavior.** If a requested technique is not implemented end-to-end, error early with a clear message that includes: what is missing (format/passes/blur), where it would be implemented, and what to use instead.
36. **Tests must prove the *user-facing* promise.** If the promise is “technique changes shadows,” add an integration test that renders a forced-edge scene and asserts a non-trivial difference between techniques. Unit tests of parsing/validation do not satisfy this.
37. **Stop on contradictory evidence.** If an experiment contradicts your hypothesis (e.g., hardcoded shader return doesn’t change output), you must stop, summarize the contradiction, and propose the next smallest diagnostic step—no large refactors.
38. **No “it’s cached” without evidence.** If you claim caching/build reuse, you must show build logs or timestamps proving the artifact didn’t rebuild, then show the fix that forces rebuild. Otherwise, do not mention caching as cause.
39. **Do not change CLI value sets casually.** If you change accepted values (e.g., `--shadows csm`), you must: update help text, presets, docs, and tests in one commit, and provide a migration note. If not, stop.
40. **Explicit deliverables for every decision.** When a decision is made (remove feature vs implement), list concrete deliverables: files, functions, tests, acceptance renders, and stop-conditions. If you can't enumerate deliverables, stop.
41. **Buffer size mismatches require systematic analysis, not iteration.** If a GPU buffer size error appears (e.g., "Buffer is bound with size X where shader expects Y"), do NOT blindly increment padding. Instead: (a) calculate the expected size from std430 rules, (b) verify with `std::mem::size_of`, (c) fix BOTH Rust and WGSL in one edit. Iterative guessing wastes builds.
42. **Verify struct sizes before full builds.** When modifying GPU uniform/storage buffer structs, write a minimal standalone Rust program to verify `size_of` matches the target BEFORE running `cargo build` or `maturin develop`. This saves minutes per iteration.
43. **Never assume viewer uses main renderer code.** The interactive viewer (`src/viewer/`) has its own terrain scene, shaders, and bind groups separate from the main `TerrainRenderer`. When debugging viewer issues, trace through `src/viewer/terrain/` specifically—do not assume fixes to `src/terrain/` apply.
44. **Silent crashes require stderr capture.** If a GPU application "fails silently" or "shows no window," capture stderr separately: `./binary 2> stderr.log`. WGPU validation errors appear on stderr and explain the exact cause. Do not speculate without this output.
45. **IPC viewer debugging pattern.** For interactive viewer crashes: (a) start binary with `--ipc-port 0`, (b) parse port from stdout, (c) send IPC commands via socket, (d) check if process is still alive after each command, (e) read stderr on crash. This isolates which command triggers the failure.

### 3.1 Test-driven practices

* **TDD and TCR:** Embrace test-driven development (write a failing test → make it pass → refactor) and Test & Commit or Revert (small changes must keep the main branch green). Work in small, reversible increments.
* **Tests are code:** Tests deserve the same cleanliness as production code. Keep them fast, isolated, repeatable, self-validating and timely (F.I.R.S.T.). Use helpers to build a domain-specific language for tests. Each test must usually perform one action and assert one essential condition.
* **Acceptance tests:** Define clear user-level acceptance criteria and automate them. Integrate acceptance tests into continuous builds. They catch cross-cutting issues that unit tests may miss.
* **High coverage goal:** Aim for asymptotically high coverage (approaching 100 % lines and branches). High coverage is the enabling factor for aggressive refactoring: you cannot maximise expression or minimise duplication and size without tests.

### 3.2 AI-assisted development

* **Sceptical use of AI:** Treat AI-generated code as a starting point, not a final answer. Expect it to be incomplete or wrong. Review and refactor AI output rigorously.
* **Prompt-driven programming:** Use prompts as a tool to generate ideas, but always verify the design and architecture yourself. AI does not replace engineering judgement.

### 3.3 Simplicity and YAGNI

* **Untangled design:** Simplicity means untangled design. Achieving simplicity requires effortful refactoring, not just fewer lines.
* **YAGNI:** Question speculative hooks. Only add extension points when you have strong, concrete reasons. Don’t add code for imagined future requirements.
* **Four simple design rules:** In order of priority: (1) code is covered by tests; (2) code maximises expression (intent is clear); (3) duplication is minimised; (4) size is minimised without compromising the first three rules. Follow this sequence when improving design.

### 3.4 Interactive Viewer Integration

Every new rendering feature must be incorporated into the interactive viewer flow. Features are not complete until they are demonstrable in the interactive viewer.

**Required User Journeys:**
1. **Interactive Session:** User renders the scene in the interactive viewer, moves/rotates the camera or issues commands in the terminal, then calls `snap` in the terminal to save the snapshot locally as an image.
2. **One-shot Snapshot:** User renders the scene and takes the snapshot immediately running a single line of code (e.g., `--snapshot out.png`).

**Implementation Requirements:**
* **Model Example:** Examples for new features must be modeled after `examples/terrain_viewer_interactive.py`.
* **Terminal Control:** Ensure parameters can be adjusted via terminal commands during the session.
* **Enrich Knowledge Base:** After every debugging, implementation or any other action in the codebase, you must always add a reflection to `AGENTS.md` in order to improve your efficiency and avoid repeating the same mistakes in the future. After you make changes you must make sure to deduplicate the file. Never remove the existing content, which is not duplicated.

### 3.5 Documentation and Release Standards

* **Documentation:** With every new feature, documentation must be created and added to the appropriate `docs/` subfolder.
* **Versioning:** Bump the project version for new features by incrementing the third number (patch version, e.g., `1.2.0` → `1.2.1`). Update `Cargo.toml` and `pyproject.toml` (if applicable).
* **Changelog:** Report all recent changes in `CHANGELOG.md` under the new version header.
* **README Updates:** Update the "Current release:" line in `README.md` with a single sentence summarizing the latest release. Maintain the tone and formatting of existing examples and **preserve** them.
* **Zero Warnings:** The code must compile cleanly after changes. Resolve **ALL** warnings and errors when building with `maturin develop --release` before finishing.

## 4. SOLID and component principles

Mid-level principles govern how to organise functions and data into cohesive, reusable components. They apply regardless of whether your language uses classes.

### 4.1 SOLID principles

* **Single Responsibility (SRP):** A module must have one, and only one, reason to change. Base responsibilities on actors (groups of people with distinct concerns). Do not bundle accounting, HR and DBA logic into a single class.
* **Open/Closed (OCP):** Software artefacts must be open for extension but closed for modification. Extend behaviour by adding new modules or types rather than editing existing ones.
* **Liskov Substitution (LSP):** Subtypes must be substitutable for their base types without changing client behaviour. Ensure that overrides respect the expectations of base contracts.
* **Interface Segregation (ISP):** Clients must not depend on methods they do not use. Split large interfaces into smaller ones so each consumer depends only on what it needs.
* **Dependency Inversion (DIP):** Depend on abstractions, not concretions. High-level modules must not directly import low-level details. Insert interfaces so low-level implementations depend on high-level policies.

### 4.2 Component cohesion and coupling

* **Release/Reuse Equivalence (REP):** Reuse and release go hand-in-hand. Package reusable classes/functions as a single unit with a clear version and release process.
* **Common Closure (CCP):** Group classes that change for the same reasons into the same component. Align components with the forces that cause change.
* **Common Reuse (CRP):** Users must not depend on code they do not reuse. Avoid bundling unrelated functionalities into one component; split components to reduce unnecessary dependencies.
* **Acyclic Dependencies (ADP):** The component dependency graph must be acyclic. Cycles complicate understanding and builds.
* **Stable Dependencies (SDP):** Depend in the direction of stability. Highly stable components (with many dependents and infrequent change) must not depend on volatile ones.
* **Stable Abstractions (SAP):** A component must be as abstract as it is stable. Very stable components must contain mostly interfaces or policies, not concrete details. Volatile components can be more concrete.

## 5. Architecture, boundaries and continuous design

### 5.1 Continuous design

* **Code is the design:** Over decades of software engineering, the consensus is that source code and configuration *are* the definitive design. Diagrams and documents are secondary and may become stale.
* **Four C’s:** Use **Clarity**, **Conciseness**, **Confirmability** and **Cohesion** as continuous design criteria. Clarify intent after getting things working; remove redundancy; ensure behavior is testable; keep modules tightly cohesive. These criteria interact—optimising one at the expense of others is a mistake.
* **Design is ongoing:** Design is not a phase that ends before coding. It happens every time you write code, test, and refactor. Plan to revisit the design continuously as requirements evolve.

### 5.2 Policy vs. details and keeping options open

* **Two values of software:** Delivering features is one value; the architecture’s shape is another. The latter enables change and longevity.
* **Separate policy from details:** Decompose systems into high-level policy (business rules) and low-level details (databases, UI, frameworks). Treat policy as the core value to protect.
* **Delay detail decisions:** Keep options open by deferring decisions about which database, web server, protocol or framework to use. Use interfaces to decouple policy from details; details plug into policy via inversion of control.
* **Plug-in architecture:** Consider UI, databases and external integrations as plug-ins behind interfaces. High-level policy depends on abstract interfaces; low-level details implement those interfaces and depend on high-level policies.
* **Dependency rule:** Across architectural boundaries, dependencies must point from details to policies. Business rules must not depend on UI, DB or frameworks. Use dependency inversion to enforce this direction.

### 5.3 Clean architecture layers

The Clean Architecture organises code into concentric rings. The exact number of circles is flexible, but the dependency direction is not:

1. **Entities:** Enterprise-wide critical business rules. Encapsulate the most stable rules and data structures, independent of operational details. Entities must be unaffected by UI, databases or frameworks.
2. **Use Cases / Interactors:** Application-specific business rules that orchestrate entities. Use cases coordinate the flow of data to achieve application goals but do not know about external frameworks.
3. **Interface Adapters:** Controllers, presenters and gateways that convert data between external forms (web requests, UIs, database records) and the internal models used by use cases and entities. They translate and format data but carry little logic.
4. **Frameworks & Drivers:** The outermost ring containing UI code, database access, web servers, file systems and other infrastructure. It must be mostly glue code. Details in this ring depend on interfaces defined in the inner rings.

Data that crosses boundaries must be simple, framework-agnostic structures (primitives or plain structs), not ORM types or framework-specific objects. Use interfaces and inversion to pass information both ways without reversing dependencies.

### 5.4 Operational concerns

* **Support use cases, operation, development and deployment:** Architecture is not solely about supporting features. It must also support operational arrangements (monolith vs processes vs distribution), independent team development and easy deployment. Partition the system so teams can act independently and deployments can happen soon after a build.
* **Leave options open while balancing concerns:** Balancing use cases, operational scalability, development and deployment is hard. Always leave options open rather than binding to one pattern too early.

### 5.5 Boundaries with third-party code and concurrency

* **Isolate vendors:** Wrap vendor or third-party libraries behind thin adapters. Localise vendor-specific calls in a few places. Business logic, UI and concurrency concerns must not know about vendor APIs.
* **Encapsulate concurrency:** Confine concurrency mechanisms to low-complexity modules that change rarely. Business logic must be thread- and process-agnostic.
* **Separate creation and binding:** Object creation and binding must be separate from usage. At initialisation time, create concrete objects and bind them together on a need-to-know basis.
* **Learn before integration:** Use learning tests to explore third-party APIs before integrating them. Write tests that call the library as you intend to use it and verify behaviour. Manage unknown or evolving APIs by defining your own interfaces and wrapping the real API behind adapters.
* **Clear seams:** Boundaries are where change happens. Code at boundaries must be clearly separated and have tests that define and enforce behaviour. Minimise the number of places that touch third-party code.

## 6. Safe refactoring practices

### 6.1 General refactoring guidance

* **Refactor in tiny steps:** Make a small structural change, run tests, commit or revert. Code must never stay broken for more than a couple of minutes.
* **Always keep a working baseline:** Have tests or a harness that can validate each incremental change. Even small examples deserve repeatable checks.
* **Extract small functions:** Continually extract well-named functions to clarify intent and remove duplication. Introduce intermediate data structures deliberately to clarify pipelines, phases and iterator chains.
* **Separate computation from presentation:** Build a data-rich representation first, then have dedicated renderers for text, HTML, etc. Keep formatting and I/O separate from calculations.
* **Move variation into polymorphism:** When type-conditionals proliferate, introduce strategy objects or polymorphic hierarchies instead of repeated `if`/`match` statements.
* **Let refactoring reveal the domain model:** Use naming and extraction to surface domain concepts (e.g., “volume credits”, “amount”). Replace primitive obsession with domain types.
* **Parallel change (expand/contract):** For public APIs or DB schemas, add new API/column, migrate callers/data, then remove the old one. Avoid all-at-once breaks.
* **Shared ownership:** Within a domain, any team member must be able to refactor any module. Ownership is enforced via review, not file locks. Avoid long-lived feature branches in a refactoring-heavy codebase.
* **Legacy code:** For untested legacy systems, create seams, add tests around those seams, and improve code gradually as you touch it.

### 6.2 Refactoring must-nots

* **Don’t break behaviour and call it refactoring:** If behaviour changes, it’s a feature or bug fix. Mixing behaviour changes with refactoring complicates reasoning and reviewing.
* **Don’t refactor code you never touch:** Ugly but stable APIs you rarely change can stay ugly. Focus on the areas under active change.
* **Don’t plan giant refactoring projects:** Big-bang redesigns are risky. Prefer incremental, opportunistic improvements embedded in normal work.
* **Don’t add speculative flexibility:** Avoid extension points “just in case.” Extra indirection adds cognitive load. Add it only with a concrete need.
* **Don’t rely on long-lived branches:** Long-lived branches make merges painful and discourage structural improvements.

### 6.3 Code smells and heuristics

Use code smells as heuristics for improvement, not rigid rules:

* **Improve names relentlessly:** Rename functions, variables, fields and classes to make them clearer. Naming is a cheap, high-ROI refactoring.
* **Remove duplication:** Use extraction (functions, methods, classes) to centralise logic. Even “almost the same” code can often be aligned and unified.
* **Prefer small, focused functions:** Long, multi-concern functions are smells. Extract responsibilities into their own functions or classes.
* **Shrink parameter lists:** Use `Preserve Whole Object` when multiple parameters come from the same object. Introduce parameter objects or combine functions into classes when several functions share the same parameter group.
* **Encapsulate and limit globals:** Hide global state behind accessors or modules; minimise scope. Favour immutability when feasible.
* **Separate queries from commands:** Queries return data and have no side effects; commands mutate state. Centralise writes via setters or commands.
* **Address divergent change and shotgun surgery:** When a module changes for many reasons, split responsibilities (Extract Class, Split Phase, Move Function). When a small change requires edits across many modules, consolidate related behaviour.
* **Upgrade data clumps and primitive obsession:** Replace groups of repeated fields with proper structs or classes. Replace bare primitives with domain objects (e.g., `Money`, `Coordinate`).
* **Replace repeated switches with polymorphism:** Conditionally repeated logic belongs in polymorphic types or strategies.
* **Prefer pipelines and iterators:** Use higher-order functions (`map`, `filter`, `reduce`) when they clarify intent better than manual loops.
* **Remove lazy and speculative abstractions:** Inline tiny functions that add no value. Remove unused parameters, methods and extension points that never materialised.

## 7. Final notes for forge3d agents

As you contribute to `forge3d`, always align with the guidance above. A few final reminders specific to this project:

* **Respect the Rust/Python boundary:** Core rendering and GPU interaction belongs in Rust; Python orchestrates and tests. Use PyO3 carefully to expose Rust functionality to Python without leaking implementation details.
* **Manage GPU resources carefully:** Use RAII for resource acquisition and release. Allocate memory via VMA and stay within the 512 MiB host-visible budget. Avoid fragmentation and track memory usage.
* **Maintain build compatibility:** Keep CMake (≥ 3.24), Cargo and Python build scripts consistent across supported platforms. Ensure CI runs successfully on `win_amd64`, `linux_x86_64` and `macos_universal2`.
* **Update documentation and tests:** Any change in behaviour or API must be reflected in Sphinx docs and Python tests. Acceptance tests must remain green across all platforms.

By adhering to these principles you will help keep the `forge3d` project robust, maintainable and adaptable to change. Continuous attention to cleanliness, design and architecture will enable you—and future agents—to carry out tasks confidently and effectively.

## 8. Claude model efficiency and compliance guidelines

The following guidelines are derived from best practices for making Claude models more accurate, efficient, and compliant when working on coding tasks.

### 8.1 Response efficiency

* **Skip flattery and preamble:** Never start responses with positive adjectives like "Good question!", "Great idea!", or "That's a fascinating problem." Jump straight into addressing the request.
* **Be concise:** Minimize output tokens while maintaining helpfulness, quality, and accuracy. Avoid verbose explanations in huge blocks of text or long nested lists. Prefer concise bullet points and short paragraphs.
* **Scale effort to complexity:** Simple questions deserve concise responses; complex and open-ended questions warrant thorough responses. Do not over-engineer simple tasks.
* **Bold key facts:** When providing explanations or reports, bold critical information for scannability.

### 8.2 Tool and search efficiency

* **Avoid unnecessary tool calls:** If you can answer confidently from existing knowledge, do so without invoking tools. Most queries about stable, well-established concepts do not require searches or external lookups.
* **Scale tool calls to query complexity:** Use minimal tool calls for simple factual queries; use multiple coordinated calls for complex research tasks requiring synthesis from multiple sources.
* **Batch independent calls:** When multiple independent tool calls are needed, invoke them in parallel rather than sequentially.
* **Plan before complex research:** For multi-step investigations, develop a brief research plan identifying which tools are needed and in what order, then execute systematically.

### 8.3 Code generation accuracy

* **Generate complete, runnable code:** Every code artifact must be immediately executable. Include all necessary imports, dependencies, and boilerplate. Never leave placeholders like `// TODO` or `...` unless explicitly discussing incomplete designs.
* **Prefer updates over rewrites:** When modifying existing code, use targeted edits (fewer than 20 lines, fewer than 5 distinct locations) rather than rewriting entire files. This reduces risk and makes changes easier to review.
* **Use concise variable names where appropriate:** In tight loops or lambda expressions, short names like `i`, `j`, `e`, `el` are acceptable. In domain logic, prefer descriptive names.
* **Handle errors explicitly:** Always include proper error handling. Use `Result`/`Option` in Rust, exceptions or explicit checks in Python. Do not assume happy paths.

### 8.4 Correction and verification

* **Think before acknowledging corrections:** If the user corrects you or claims you made a mistake, first verify the claim carefully. Users sometimes make errors themselves. Do not blindly accept corrections.
* **Verify before asserting:** When making claims about code behavior, file contents, or system state, verify with tools rather than assuming from memory or context.
* **Cite sources precisely:** When referencing code, use exact file paths and line numbers. Do not cite line numbers as proof unless you paste the relevant snippet.

### 8.5 Communication style

* **Match format to context:** Use markdown and structured formatting for technical documentation and code explanations. Avoid markdown, lists, and excessive formatting in casual conversation or brief clarifications.
* **Offer alternatives when declining:** If you cannot or will not help with something, briefly explain and offer helpful alternatives if possible. Keep refusals to 1–2 sentences without lecturing.
* **One question at a time:** In conversational exchanges, avoid overwhelming with multiple questions. Ask one clarifying question per response when needed.
* **State limitations upfront:** If you cannot complete part of a request, say so at the start of your response, not buried in the middle.

### 8.6 Safety and compliance

* **Assume legitimate intent:** When a request is ambiguous, interpret it charitably as having a legal and legitimate purpose, unless clear red flags are present.
* **Never reproduce copyrighted material:** Do not reproduce large chunks (20+ words) of copyrighted content. Use very short quotes (under 15 words) in quotation marks with citations when referencing external sources.
* **Refuse harmful requests succinctly:** If a request has clear harmful intent (malware, weapons, exploitation), decline without extensive explanation or alternative suggestions that could enable harm.
* **Protect privacy:** Do not make assumptions about individuals' identities, emails, or personal information. When uncertain, ask for clarification rather than guessing.

### 8.7 Incremental artifact management

* **One artifact per response:** When creating substantial code or documentation, produce one complete artifact rather than fragmenting across multiple outputs.
* **Update incrementally:** For subsequent modifications, use precise find-and-replace edits rather than regenerating entire artifacts. This preserves context and reduces errors.
* **Maintain working state:** After any edit, the code must compile and pass existing tests. Never leave the codebase in a broken state between edits.

### 8.8 Research and investigation methodology

* **Start broad, then narrow:** When investigating unfamiliar code or debugging, begin with broad exploration (file structure, module organization) before diving into specific implementations.
* **Document findings as you go:** For complex investigations, maintain a mental or explicit checklist of what you've verified and what remains. Update your understanding as new information emerges.
* **Stop on contradictions:** If evidence contradicts your hypothesis, stop, summarize the contradiction, and propose the next smallest diagnostic step. Do not proceed with large refactors based on uncertain assumptions.
* **Distinguish "config honored" from "behavior changed":** When verifying features, separately confirm that (a) configuration is being read correctly and (b) the configuration actually affects output as expected.

## 9. GPU buffer layout and WGSL alignment rules

This section documents critical lessons learned about Rust↔WGSL buffer layout mismatches that cause silent crashes.

### 9.1 std430 storage buffer alignment (CRITICAL)

WGSL storage buffers use **std430 layout rules**, which differ significantly from Rust's `#[repr(C)]`. Mismatches cause `wgpu::ValidationError` crashes with messages like:

```
Buffer is bound with size X where the shader expects Y in group[N] compact index M
```

**Key std430 alignment rules:**
- Scalars (`f32`, `u32`, `i32`): 4-byte alignment
- `vec2<f32>`: 8-byte alignment
- `vec3<f32>`: **16-byte alignment** (NOT 12 bytes!)
- `vec4<f32>`: 16-byte alignment
- `mat4x4<f32>`: 16-byte alignment (column-major, each column is vec4)
- Arrays: Element stride rounded up to **16-byte alignment** for vec3/vec4/mat elements
- Structs: Aligned to their largest member; size padded to multiple of alignment

**Common pitfalls:**
1. `vec3` in WGSL takes 16 bytes, not 12. Always use `vec4` or add explicit padding in Rust.
2. Arrays of structs containing `mat4x4` require 16-byte struct alignment.
3. The **total struct size** must be a multiple of 16 bytes for storage buffers.

### 9.2 Debugging buffer size mismatches

When you see a buffer size mismatch error:

1. **Read the error carefully:** It tells you exact expected vs actual sizes.
2. **Calculate struct size in Rust:**
   ```rust
   println!("Size: {} bytes", std::mem::size_of::<MyStruct>());
   ```
3. **Match WGSL padding exactly:** Add `_padding` arrays to reach the expected size.
4. **Verify with a test program:** Create a minimal Rust program to verify `size_of` before building the full project.
5. **Update BOTH Rust and WGSL:** Padding must match in both the Rust struct and WGSL struct definition.

**Example fix pattern:**
```rust
// Rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MyUniforms {
    // ... fields ...
    pub last_field: f32,
    pub _padding: [f32; N],  // Adjust N until size_of matches shader expectation
}
```
```wgsl
// WGSL
struct MyUniforms {
    // ... fields ...
    last_field: f32,
    _padding: array<f32, N>,  // Same N as Rust
}
```

### 9.3 Interactive viewer vs main renderer architecture

The codebase has **two separate terrain rendering paths**:

1. **`TerrainScene` / `TerrainRenderer`** (`src/terrain/`): Full-featured renderer with complete shadow pipeline (CSM depth passes, moment generation, all techniques).

2. **`ViewerTerrainScene`** (`src/viewer/terrain/`): Simplified interactive viewer with its own shaders and bind groups. **Shadow infrastructure may be incomplete.**

**When debugging viewer-specific issues:**
- Check `src/viewer/terrain/scene.rs` for initialization
- Check `src/viewer/terrain/render.rs` for rendering and bind group creation
- Check `src/viewer/terrain/shader_pbr.rs` for WGSL shader code
- Do NOT assume viewer uses the same code paths as the main renderer

### 9.4 Silent GPU crash debugging

WGPU validation errors can cause the application to crash without obvious error output. To diagnose:

1. **Capture stderr separately:**
   ```bash
   ./binary 2> stderr.log &
   sleep 3
   cat stderr.log
   ```

2. **Use IPC testing pattern:**
   ```python
   # Start viewer, wait for READY, send commands, check if still running
   proc = subprocess.Popen([binary], stdout=PIPE, stderr=PIPE)
   # ... send IPC commands ...
   if proc.poll() is not None:
       print("CRASHED:", proc.stderr.read())
   ```

3. **Add diagnostic logging at key points:**
   - Window creation
   - GPU adapter/device initialization
   - Bind group creation
   - First render frame

4. **Check for these common crash causes:**
   - Buffer size mismatches (see 9.1)
   - Missing texture views in bind groups
   - Shader compilation errors (check for WGSL syntax)
   - Incompatible texture formats

### 9.5 Bytemuck and GPU data transfer

When using `bytemuck` for Rust→GPU data transfer:

1. **All fields must be Pod:** Use `#[derive(bytemuck::Pod, bytemuck::Zeroable)]` and ensure all nested types are also Pod.
2. **Use `#[repr(C)]`:** Required for predictable memory layout.
3. **Verify Default impl matches struct:** If you change padding size, update `Default::default()` to match.
4. **Write to buffer correctly:**
   ```rust
   queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&[my_uniforms]));
   ```

### 9.6 Checklist for uniform/storage buffer changes

Before modifying any Rust struct that maps to a WGSL buffer:

- [ ] Print `std::mem::size_of::<Struct>()` and record expected size
- [ ] Verify WGSL struct has identical field order and padding
- [ ] Update `Default` impl if padding changed
- [ ] Rebuild with `cargo build --release`
- [ ] Test with actual GPU rendering (not just compilation)
- [ ] Capture stderr to see validation errors
- [ ] If size mismatch persists, calculate field-by-field offsets

## 10. IPC protocol and large payload handling

This section documents critical lessons learned about IPC communication between Python and the Rust viewer, especially for large data transfers like vector overlays.

### 10.1 JSON payload size limits (CRITICAL)

The IPC protocol uses newline-delimited JSON (NDJSON) over TCP sockets. Large payloads cause failures:

**Symptoms of payload-too-large issues:**
- Python receives empty response: `"JSON parse error: expected value at line 1 column 1"`
- Viewer appears to hang or not respond
- No error message from viewer (silent failure)

**Size thresholds observed:**
- **< 1 MB:** Generally reliable
- **1-5 MB:** May work but requires extended timeouts
- **> 5 MB:** Likely to fail; requires decimation or chunking

**Example: Vector overlay with 84,964 vertices**
```
84964 vertices × 7 floats × ~10 chars/float ≈ 6 MB JSON
Result: IPC failure, empty response
```

**Solutions:**
1. **Decimate data before sending:** Reduce vertex count to manageable levels (e.g., `--max-vertices 20000`)
2. **Increase BufReader capacity in Rust:** Use `BufReader::with_capacity(1024 * 1024, stream)` instead of default 8KB
3. **Extend socket timeouts:** Set 60-120 second timeouts for large payloads
4. **Log payload size for debugging:** Print message size before sending to identify large payloads

### 10.2 IPC debugging checklist

When IPC commands fail silently:

1. **Log message size on Python side:**
   ```python
   msg_size = len(json.dumps(cmd).encode())
   if msg_size > 100000:
       print(f"Warning: Large IPC message ({msg_size / 1024 / 1024:.1f} MB)")
   ```

2. **Add debug logging to Rust IPC server:**
   ```rust
   if n > 100000 {
       eprintln!("[IPC] Received large message: {} bytes", n);
   }
   ```

3. **Check for parse errors:**
   ```rust
   Err(e) => {
       eprintln!("[IPC] Parse error (msg len={}): {}", trimmed.len(), e);
       IpcResponse::error(e)
   }
   ```

4. **Verify viewer is still running:** After IPC send, check `proc.poll()` to see if viewer crashed

### 10.3 Socket timeout configuration

**Python side:**
```python
# Extend timeout for large payloads
old_timeout = sock.gettimeout()
sock.settimeout(120.0)  # 2 minutes
resp = send_ipc(sock, large_cmd)
sock.settimeout(old_timeout)
```

**Rust side (server.rs):**
```rust
let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(300)));
let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(30)));
```

## 11. Python API design for IPC-bound features

When creating Python API classes that serialize to IPC commands, consider these lessons.

### 11.1 Validation ranges must match all use cases

**Mistake made:** `VectorOverlayConfig.line_width` was validated to `[1.0, 10.0]` assuming GPU line pixel widths, but the example used world-unit widths (e.g., 25.0 meters) for triangle-quad "thick lines."

**Correct approach:**
- Consider ALL contexts where a parameter is used
- Document the unit and valid range clearly in docstrings
- Use permissive validation for parameters with multiple interpretations:
  ```python
  # Bad: Too restrictive
  if not 1.0 <= self.line_width <= 10.0:
      raise ValueError("line_width must be in [1.0, 10.0]")
  
  # Good: Permissive with clear documentation
  if self.line_width < 0.1:
      raise ValueError("line_width must be >= 0.1")
  ```

### 11.2 Provide fallback paths for large data

**Pattern:** Skip expensive Python object creation for large datasets

```python
# For large overlays (>10000 vertices), use raw dict directly to avoid
# memory overhead of VectorVertex object conversion
use_raw_dict = len(vertices) > 10000 or not HAS_VECTOR_API

if not use_raw_dict:
    # Use typed API classes for validation and documentation
    vertex_objects = [VectorVertex(x=v[0], ...) for v in vertices]
    config = VectorOverlayConfig(vertices=vertex_objects, ...)
    ipc_cmd = config.to_ipc_dict()
else:
    # Send raw dict directly (more efficient for large data)
    ipc_cmd = {"cmd": "add_vector_overlay", "vertices": vertices, ...}
```

### 11.3 to_ipc_dict() method pattern

Every Python config class that maps to an IPC command should implement `to_ipc_dict()`:

```python
@dataclass
class VectorOverlayConfig:
    name: str
    vertices: List[VectorVertex]
    # ... other fields ...
    
    def to_ipc_dict(self) -> dict:
        """Convert to IPC request dictionary format."""
        return {
            "cmd": "add_vector_overlay",
            "name": self.name,
            "vertices": [v.to_array() for v in self.vertices],
            # ... other fields ...
        }
```

## 12. Vector overlay system architecture

### 12.1 Rendering pipeline requirements

For vector overlays to render, ALL of these conditions must be true:

1. **`vector_overlay_stack` is `Some`:** Stack must be initialized
2. **`stack.is_enabled()` returns `true`:** Global enable flag
3. **`stack.visible_layer_count() > 0`:** At least one visible layer
4. **`stack.pipelines_ready()` returns `true`:** GPU pipelines initialized
5. **`stack.bind_group.is_some()`:** Bind group created with textures

**Debugging invisible overlays:**
```rust
// Check each condition
let has_stack = self.vector_overlay_stack.is_some();
let is_enabled = stack.is_enabled();
let visible_count = stack.visible_layer_count();
let pipelines_ok = stack.pipelines_ready();
let bind_group_ok = stack.bind_group.is_some();
eprintln!("[debug] stack={} enabled={} visible={} pipelines={} bind_group={}",
    has_stack, is_enabled, visible_count, pipelines_ok, bind_group_ok);
```

### 12.2 Feature flag pattern

New rendering features must follow the opt-in pattern:

```rust
// In ViewerTerrainPbrConfig
pub vector_overlays_enabled: bool,  // Field

// In Default impl
vector_overlays_enabled: false,  // DEFAULT OFF per plan

// Check before rendering
let has_vector_overlays = if let Some(ref stack) = self.vector_overlay_stack {
    stack.is_enabled() && stack.visible_layer_count() > 0
} else {
    false
};
```

### 12.3 Coordinate system alignment

When loading external geometry (GeoPackage, GeoJSON) to overlay on terrain:

1. **Identify source CRS:** Read from file metadata (e.g., `gdf.crs`)
2. **Identify terrain CRS:** From DEM file or rendering config
3. **Reproject if needed:** Use pyproj or geopandas
4. **Normalize to terrain bounds:** Map world coordinates to terrain-local coordinates

**Common mistake:** Assuming GeoPackage uses same CRS as DEM. Always verify and reproject.

```python
# Check and reproject
if gdf.crs != target_crs:
    print(f"Reprojecting from {gdf.crs} to {target_crs}")
    gdf = gdf.to_crs(target_crs)
```

## 13. Example script patterns

### 13.1 Graceful dependency handling

```python
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    gpd = None

# Later in code
if HAS_GEOPANDAS and gpd is not None:
    gdf = gpd.read_file(path)
else:
    # Fallback: native reader or demo data
    vertices, indices = load_native_or_demo(path)
```

### 13.2 IPC send helper with error handling

```python
def send_ipc(sock: socket.socket, cmd: dict) -> dict:
    """Send an IPC command and receive response."""
    msg = json.dumps(cmd) + "\n"
    msg_bytes = msg.encode()
    
    # Log large messages for debugging
    if len(msg_bytes) > 100000:
        print(f"  Sending large IPC message: {len(msg_bytes) / 1024 / 1024:.1f} MB")
    
    try:
        sock.sendall(msg_bytes)
    except Exception as e:
        return {"ok": False, "error": f"Send failed: {e}"}
    
    data = b""
    while True:
        try:
            chunk = sock.recv(8192)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break
        except socket.timeout:
            if not data:
                return {"ok": False, "error": "Timeout waiting for response"}
            break
    
    line = data.decode().strip()
    if not line:
        return {"ok": False, "error": "Empty response from viewer"}
    
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"Invalid JSON response: {e}"}
```

### 13.3 Data decimation for large datasets

```python
# Decimate if too many vertices (IPC can't handle very large JSON payloads)
if vertices and len(vertices) > max_vertices:
    ratio = len(vertices) / max_vertices
    step = max(1, int(ratio))
    print(f"Decimating: {len(vertices)} -> ~{len(vertices)//step} vertices")
    
    # Keep every Nth primitive (e.g., every Nth quad for thick lines)
    new_vertices = []
    new_indices = []
    primitives_per_unit = 4  # e.g., 4 vertices per quad
    for i in range(0, len(vertices) // primitives_per_unit, step):
        base_old = i * primitives_per_unit
        base_new = len(new_vertices)
        new_vertices.extend(vertices[base_old:base_old + primitives_per_unit])
        # Remap indices...
    
    vertices = new_vertices
    indices = new_indices
```

### 13.4 Argument parser structure

Follow this pattern for example scripts to maintain consistency:

```python
def parse_args():
    parser = argparse.ArgumentParser(
        description="Short description",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required/positional arguments first
    parser.add_argument("--dem", type=str, required=True, help="DEM path or shortcut")
    
    # Configuration arguments grouped logically
    parser.add_argument("--rings", type=int, default=4, help="Number of rings")
    
    # Output arguments
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    return parser.parse_args()
```

### 13.5 DEM shortcut resolution

Standardize DEM path resolution to support common test assets:

```python
DEM_SHORTCUTS = {
    "fuji": "assets/tif/Mount_Fuji_30m.tif",
    "rainier": "assets/tif/Mount_Rainier_30m.tif",
}

def resolve_dem_path(dem_arg: str) -> Path:
    """Resolve DEM argument to actual path."""
    if dem_arg.lower() in DEM_SHORTCUTS:
        return Path(DEM_SHORTCUTS[dem_arg.lower()])
    return Path(dem_arg)
```

### 13.6 Visualization image generation

```python
def create_visualization_image(width: int, height: int) -> np.ndarray:
    """Create RGB image as numpy array."""
    return np.zeros((height, width, 3), dtype=np.uint8)

def save_image(img: np.ndarray, path: Path):
    """Save numpy RGB array as PNG."""
    from PIL import Image
    Image.fromarray(img).save(path)
```

### 13.7 Combined visualization layout

For multi-panel visualizations (e.g., in `clipmap_demo.py`):

```python
def create_combined_visualization(panel1, panel2, panel3, output_path):
    """Create 3-panel side-by-side visualization."""
    panel_height, panel_width = panel1.shape[:2]
    
    # Add padding and labels
    padding = 10
    label_height = 30
    
    total_width = panel_width * 3 + padding * 4
    total_height = panel_height + label_height + padding * 2
    
    combined = np.zeros((total_height, total_width, 3), dtype=np.uint8)
    
    # Place panels
    x_offset = padding
    for i, (panel, label) in enumerate([(panel1, "Panel 1"), ...]):
        y_start = label_height + padding
        combined[y_start:y_start + panel_height, x_offset:x_offset + panel_width] = panel
        # Add label text...
        x_offset += panel_width + padding
    
    save_image(combined, output_path)
```

### 13.8 Statistics output formatting

```python
def print_section_header(title: str, width: int = 50):
    """Print centered section header."""
    print(f"\n{'=' * width}")
    print(f"{title:^{width}}")
    print(f"{'=' * width}\n")

def print_stat(label: str, value, unit: str = "", width: int = 20):
    """Print aligned statistic."""
    print(f"  {label:<{width}}: {value}{unit}")
```

## 14. Picking and selection system (Plan 3)

This section documents the picking/selection system architecture, IPC protocol, and common pitfalls.

### 14.1 Vector overlay vertex format (CRITICAL)

The `AddVectorOverlay` IPC command expects vertices with **8 components**, not 7:

```
[x, y, z, r, g, b, a, feature_id]
 0  1  2  3  4  5  6      7
```

**Protocol definition in `src/viewer/ipc/protocol.rs`:**
```rust
AddVectorOverlay {
    name: String,
    vertices: Vec<[f32; 8]>,  // <-- 8 components required
    indices: Vec<u32>,
    // ...
}
```

**Common mistake:** Sending 7-component vertices `[x, y, z, r, g, b, a]` causes:
```
[VIEWER] [IPC] Parse error (msg len=9838): JSON parse error: invalid length 7, expected an array of length 8
```

**Python side must include feature_id:**
```python
# CORRECT: 8 components
vertex = [x, y, z, r, g, b, a, feature_id]

# WRONG: 7 components (missing feature_id)
vertex = [x, y, z, r, g, b, a]  # Will cause parse error
```

### 14.2 BVH feature ID extraction

When building a BVH for picking in `src/viewer/cmd/handler.rs`, feature IDs must be extracted from vertex data, not the layer ID.

**Incorrect (assigns all triangles to layer ID):**
```rust
// BAD: All triangles get the same ID (layer ID)
layer_data.cpu_feature_ids = bvh.tri_indices.iter().map(|_| id).collect();
```

**Correct (extracts feature_id from vertex data):**
```rust
// GOOD: Extract feature_id from vertex[7] for each triangle
layer_data.cpu_feature_ids = bvh.tri_indices.iter().map(|&tri_idx| {
    let idx = tri_idx as usize;
    if idx < tris_indices.len() {
        let ti = tris_indices[idx];
        let v0_idx = ti[0] as usize;
        if v0_idx < vertices.len() {
            vertices[v0_idx][7] as u32  // feature_id is at index 7
        } else {
            id // fallback to layer id
        }
    } else {
        id // fallback to layer id
    }
}).collect();
```

### 14.3 Picking system components

The picking system has three main components:

1. **`UnifiedPickingSystem`** (`src/picking/unified.rs`):
   - Manages BVH data per layer
   - Performs ray-BVH intersection
   - Tracks selection sets
   - Emits `PickEvent` results

2. **`LassoState`** (`src/picking/lasso.rs`):
   - Manages lasso/box selection polygon
   - Point-in-polygon containment tests
   - Douglas-Peucker simplification

3. **`LayerBvhData`** (`src/picking/mod.rs`):
   - Per-layer BVH storage
   - CPU-side triangle and node data
   - Feature ID mapping

**Key data flow:**
```
Python IPC → AddVectorOverlay → Build BVH → Register with UnifiedPickingSystem
                                    ↓
User Click → Ray cast → BVH intersection → PickEvent → PollPickEvents IPC
```

### 14.4 Picking IPC commands

| Command | Purpose | Response |
|---------|---------|----------|
| `poll_pick_events` | Get pending pick events | `{"ok": true, "pick_events": [...]}` |
| `set_lasso_mode` | Enable/disable lasso selection | `{"ok": true}` |
| `get_lasso_state` | Query current lasso state | `{"ok": true, "state": "active\|inactive"}` |
| `clear_selection` | Clear current selection | `{"ok": true}` |

**Example pick event response:**
```json
{
  "ok": true,
  "pick_events": [
    {
      "feature_id": 42,
      "layer_name": "Places",
      "world_pos": [138.72, 4000.0, 35.36],
      "event_type": "click"
    }
  ]
}
```

### 14.5 Testing picking functionality

**Test file locations:**
- `tests/test_picking_premium.py` - Unit tests for Python picking types
- `tests/test_picking_ipc.py` - Integration tests for IPC-based picking
- `examples/picking_demo.py` - Interactive demo script
- `examples/picking_test_interactive.py` - Automated verification script

**Running picking tests:**
```bash
# Unit tests
python -m pytest tests/test_picking_premium.py -v

# IPC integration tests (launches viewer)
python -m pytest tests/test_picking_ipc.py -v

# Interactive verification
python examples/picking_test_interactive.py --verbose
```

**Test checklist for picking changes:**
- [ ] 8-component vertex format accepted without parse error
- [ ] 7-component vertex format rejected with clear error
- [ ] Lasso mode toggle works (enable → get state → disable)
- [ ] Pick event polling returns list (empty is OK if no clicks)
- [ ] Clear selection command succeeds
- [ ] Feature IDs in BVH match vertex data (not layer ID)

### 14.6 Debugging picking issues

**Symptom: JSON parse error with vertex data**
- Check vertex array length is exactly 8
- Print first vertex to verify: `print(f"First vertex: {vertices[0]}, len={len(vertices[0])}")`

**Symptom: All picked features return same ID**
- Check `cpu_feature_ids` extraction in `handler.rs`
- Verify feature_id at vertex index 7 is set correctly in Python
- Add debug logging: `println!("Feature IDs: {:?}", &layer_data.cpu_feature_ids[..5.min(len)]);`

**Symptom: No pick events returned**
- Verify BVH was built: look for `[picking] Built BVH for layer N` in viewer output
- Check `primitive="triangles"` in AddVectorOverlay (BVH only built for triangles)
- Verify layer is visible and not empty

**Symptom: Lasso mode doesn't activate**
- Check IPC response for `{"ok": true}`
- Verify `UnifiedPickingSystem::set_lasso_enabled` is called
- Check shared lasso state mutex is not poisoned

## 15. Terrain coordinate systems and draping

### 15.1 Coordinate conventions

The terrain viewer uses a specific coordinate mapping:

| Axis | GIS Convention | Terrain Viewer |
|------|----------------|----------------|
| X | Longitude (East) | Longitude |
| Y | Latitude (North) | **Elevation** |
| Z | Elevation (Up) | Latitude |

**When creating overlay vertices from geo-coordinates:**
```python
# GIS: (lon, lat, elev)
# Viewer vertex: [lon, elev, lat, r, g, b, a, feature_id]
#                  X    Y     Z
vertex = [longitude, elevation, latitude, r, g, b, a, fid]
```

### 15.2 Draping overlays onto terrain

When `drape=True` in AddVectorOverlay, the viewer samples terrain elevation at each vertex XZ position:

```python
add_vector_overlay(
    sock, "Layer",
    vertices, indices,
    drape=True,         # Enable terrain draping
    drape_offset=20.0,  # Meters above terrain surface
)
```

**Drape offset purpose:** Prevents z-fighting by lifting geometry above terrain surface. Typical values: 5-50 meters depending on scale.

### 15.3 Common coordinate mistakes

1. **Swapped Y/Z:** Vertices appear at wrong elevation or underground
2. **Missing drape:** Flat geometry floats at arbitrary height
3. **Wrong CRS:** Overlay appears in wrong location (reproject to terrain CRS first)
4. **Scale mismatch:** Overlay too small/large (check units match terrain)

## 16. Codebase Navigation and Discovery Patterns

This section documents critical lessons learned about efficiently navigating the forge3d codebase and avoiding wasted effort from assuming features need implementation.

### 16.1 Always verify existing implementations FIRST (CRITICAL)

**Mistake made repeatedly:** Assuming a roadmap phase (e.g., "P4: Map Plate Compositor" or "P5: 3D Tiles") needs implementation when it was already complete. This wastes significant time planning work that doesn't need to be done.

**Before starting ANY implementation task:**

1. **Search for existing modules:**
   ```bash
   # Search for key class/struct names from the task
   grep -r "MapPlate\|Legend\|ScaleBar" python/forge3d/ src/
   grep -r "Tileset\|B3dm\|Pnts" src/
   ```

2. **Search for existing tests:**
   ```bash
   # Tests define behavior - if tests exist, feature likely exists
   find tests/ -name "test_map_plate*" -o -name "test_3dtiles*"
   ```

3. **Run existing tests to verify:**
   ```bash
   python -m pytest tests/test_map_plate_layout.py tests/test_3dtiles_parse.py -v
   ```

4. **Check docs/plan.md status notes:** Look for "STATUS" or "implemented" annotations.

**Example of wasted effort:** Starting to implement P4 and P5 when 59 tests already passed, indicating full implementation.

### 16.2 Module organization patterns

The codebase follows consistent organization patterns:

| Feature Area | Rust Location | Python Location | Test Location |
|--------------|---------------|-----------------|---------------|
| Terrain rendering | `src/terrain/` | `python/forge3d/terrain*.py` | `tests/test_terrain*.py` |
| Clipmaps/LOD | `src/terrain/clipmap/` | N/A (Rust-only) | `tests/test_clipmap*.py`, `tests/test_gpu_lod*.py` |
| COG streaming | `src/terrain/cog/` | `python/forge3d/cog.py` | `tests/test_cog*.py` |
| 3D Tiles | `src/tiles3d/` | `python/forge3d/tiles3d.py` | `tests/test_3dtiles*.py` |
| Map plates | N/A (Python-only) | `python/forge3d/map_plate.py` | `tests/test_map_plate*.py` |
| Legends | N/A (Python-only) | `python/forge3d/legend.py` | `tests/test_map_plate*.py` |
| Scale bars | N/A (Python-only) | `python/forge3d/scale_bar.py` | `tests/test_map_plate*.py` |
| Shadows | `src/shadows/` | N/A | `tests/test_shadow*.py` |
| TAA | `src/core/taa.rs`, `src/core/jitter.rs` | N/A | `tests/test_taa*.py`, `tests/test_jitter*.py` |
| OIT | `src/core/dual_source_oit.rs` | N/A | `tests/test_oit*.py` |
| Viewer/IPC | `src/viewer/` | `python/forge3d/viewer_ipc.py` | `tests/test_viewer*.py` |
| Vector overlays | `src/vector/` | `python/forge3d/vector_overlay.py` | `tests/test_vector*.py` |

### 16.3 Feature discovery search patterns

**Quick feature existence check:**
```bash
# Check if Python API exists
grep -l "class.*Config\|def.*load_\|def.*render_" python/forge3d/*.py | head -20

# Check if Rust module exists
ls -la src/*/mod.rs | head -20

# Check test coverage
ls tests/test_*.py | wc -l
```

**Deep feature search:**
```bash
# Find where a feature is exposed to Python
grep -r "def_class\|#\[pyclass\]\|#\[pyfunction\]" src/ --include="*.rs" | grep -i "feature_name"

# Find IPC command handlers
grep -r "\"cmd\":\|ViewerCmd::" src/viewer/ --include="*.rs"
```

### 16.4 Test file naming conventions

Tests follow predictable naming that maps to features:

| Pattern | Examples | Covers |
|---------|----------|--------|
| `test_{feature}.py` | `test_taa_convergence.py` | Single feature deep tests |
| `test_{feature}_{aspect}.py` | `test_shadow_techniques.py` | Feature aspect tests |
| `test_{phase}_{feature}.py` | `test_p4_water_reflections.py` | Phase-specific integration |
| `test_{format}_*.py` | `test_cog_streaming.py` | Data format handling |

**To find tests for a roadmap phase:**
```bash
# P1 (TAA) tests
ls tests/test_motion*.py tests/test_jitter*.py tests/test_taa*.py

# P2 (Clipmaps) tests
ls tests/test_clipmap*.py tests/test_geomorph*.py tests/test_gpu_lod*.py

# P3 (COG) tests
ls tests/test_cog*.py

# P4 (Map Plate) tests
ls tests/test_map_plate*.py tests/test_legend*.py tests/test_plate*.py

# P5 (3D Tiles) tests
ls tests/test_3dtiles*.py
```

### 16.5 Roadmap phase to code mapping

| Phase | Deliverables | Key Files |
|-------|--------------|-----------|
| **P0.1** OIT | Transparency modes | `src/core/dual_source_oit.rs` |
| **P0.2** Shadows | VSM/EVSM/MSM | `src/shadows/`, `src/shaders/shadow*.wgsl` |
| **P0.3** Sun | Ephemeris | `python/forge3d/sun_ephemeris.py` |
| **P1.1** Motion Vectors | Velocity buffer | `src/core/gbuffer.rs`, `src/shaders/velocity.wgsl` |
| **P1.2** Jitter | Halton sequence | `src/core/jitter.rs` |
| **P1.3** TAA Resolve | History + clamp | `src/core/taa.rs`, `src/shaders/taa.wgsl` |
| **P2.1** Clipmaps | Nested rings | `src/terrain/clipmap/mod.rs` |
| **P2.2** Geomorphing | Seam blending | `src/terrain/clipmap/geomorph.rs` |
| **P2.3** GPU LOD | Compute culling | `src/terrain/clipmap/gpu_lod.rs` |
| **P3.1** COG Range | HTTP ranges | `src/terrain/cog/range_reader.rs` |
| **P3.2** COG Overview | IFD parsing | `src/terrain/cog/ifd_parser.rs` |
| **P3.3** COG Cache | LRU eviction | `src/terrain/cog/cache.rs` |
| **P4.1** Map Plate | Layout regions | `python/forge3d/map_plate.py` |
| **P4.2** Legend | Colormap ticks | `python/forge3d/legend.py` |
| **P4.3** Scale Bar | Geodetic calc | `python/forge3d/scale_bar.py` |
| **P5.1** 3D Tiles Parse | tileset.json | `src/tiles3d/tileset.rs`, `src/tiles3d/b3dm.rs` |
| **P5.2** 3D Tiles SSE | LOD traversal | `src/tiles3d/sse.rs`, `src/tiles3d/traversal.rs` |
| **P5.3** 3D Tiles Render | Batched draw | `src/tiles3d/renderer.rs` |

### 16.6 Verification before claiming "needs implementation"

**Checklist before starting new feature work:**

- [ ] Searched `python/forge3d/` for Python API classes
- [ ] Searched `src/` for Rust modules with feature name
- [ ] Checked `tests/` for existing test files
- [ ] Ran existing tests to see if they pass
- [ ] Checked `docs/plan.md` for status annotations
- [ ] Checked `CHANGELOG.md` for feature mentions
- [ ] Grepped for key class/function names in codebase

**If tests exist and pass, the feature is likely complete.** Read the tests to understand what's implemented before planning new work.

## 17. Common Mistakes and Anti-Patterns

This section documents specific mistakes made during development to help future agents avoid them.

### 17.1 Assuming work from roadmap phases

**Mistake:** Reading `docs/plan.md` phase descriptions and assuming implementation is needed.

**Reality:** Roadmap phases often describe *completed* work. The plan document may not be updated to reflect completion status.

**Fix:** Always verify with tests:
```bash
python -m pytest tests/test_{phase_feature}.py -v --tb=short
```

### 17.2 Missing Python-only features

**Mistake:** Searching only in `src/` (Rust) for features like Map Plate, Legend, Scale Bar.

**Reality:** Some features are implemented purely in Python under `python/forge3d/`.

**Fix:** Always search both:
```bash
grep -r "ClassName" src/ python/forge3d/
```

### 17.3 Confusing viewer and main renderer

**Mistake:** Assuming fixes to `src/terrain/` apply to the interactive viewer.

**Reality:** The viewer (`src/viewer/terrain/`) has **separate** shaders and bind groups.

**Fix:** When debugging viewer issues, trace through `src/viewer/` specifically.

### 17.4 Not running tests before claiming completion

**Mistake:** Claiming a phase is "implemented" based on code review alone.

**Reality:** Tests define behavior. Code without passing tests is incomplete.

**Fix:** Always run and paste test output:
```bash
python -m pytest tests/test_feature.py -v 2>&1 | tail -30
```

### 17.5 Over-engineering simple tasks

**Mistake:** Creating elaborate plans for tasks that are already done.

**Reality:** A quick grep + test run can confirm completion in seconds.

**Fix:** Start with discovery, not planning:
1. Search for existing code (30 seconds)
2. Run existing tests (30 seconds)
3. Only then plan new work if needed

### 17.6 Ignoring test file as documentation

**Mistake:** Reading only implementation code to understand features.

**Reality:** Test files are the **best documentation** of expected behavior and API usage.

**Example:** `tests/test_map_plate_layout.py` shows exactly how to use `MapPlate`, `Legend`, `ScaleBar` classes with realistic examples.

### 17.7 Duplicate memory/TODO entries

**Mistake:** Creating multiple memory entries for the same task or context.

**Reality:** This clutters context and wastes tokens on redundant information.

**Fix:** Before creating a new memory, check if a semantically related one exists and update it instead.

## 18. Python API Quick Reference

### 18.1 Map Plate Compositor (P4)

```python
from forge3d.map_plate import MapPlate, MapPlateConfig, BBox
from forge3d.legend import Legend, LegendConfig
from forge3d.scale_bar import ScaleBar, ScaleBarConfig

# Create plate
plate = MapPlate(MapPlateConfig(width=1600, height=1000))

# Set main map region
plate.set_map_region(rendered_image, BBox(west=-122, south=46, east=-121, north=47))

# Add elements
plate.add_title("My Terrain Map", font_size=24)
plate.add_legend(Legend(colormap, domain=(0, 4000)).render())
plate.add_scale_bar(ScaleBar(meters_per_pixel=100).render())

# Export
plate.export_png("output.png")
plate.export_jpeg("output.jpg", quality=90)
```

### 18.2 3D Tiles (P5)

```python
from forge3d.tiles3d import load_tileset, Tiles3dRenderer, SseParams

# Load tileset
tileset = load_tileset("path/to/tileset.json")
print(f"Tiles: {tileset.tile_count}, Max depth: {tileset.max_depth}")

# Create renderer with SSE threshold
renderer = Tiles3dRenderer(sse_threshold=16.0)

# Get visible tiles for camera position
visible = renderer.get_visible_tiles(tileset, camera=(0, 0, 500))
for vt in visible:
    print(f"  {vt.tile.content_uri()} at depth {vt.depth}")
```

### 18.3 COG Streaming (P3)

```python
from forge3d.cog import open_cog, is_cog_available

if is_cog_available():
    ds = open_cog("path/to/dem.tif")
    print(f"Bounds: {ds.bounds()}")
    print(f"Overviews: {ds.overview_count()}")
    
    # Read tile
    tile = ds.read_tile(x=0, y=0, overview=0)
    
    # Get stats
    stats = ds.stats()
    print(f"Cache hits: {stats.cache_hits}")
```

### 18.4 Vector Overlays

```python
from forge3d.vector_overlay import VectorOverlayConfig, VectorVertex

# Create overlay config
config = VectorOverlayConfig(
    name="my_overlay",
    vertices=[VectorVertex(x=0, y=0, z=0, r=1, g=0, b=0, a=1, feature_id=0)],
    indices=[0, 1, 2],
    line_width=2.0,
    primitive="triangles",
)

# Convert to IPC command
ipc_cmd = config.to_ipc_dict()
```

### 18.5 Terrain Demo CLI Flags

Common CLI flags for `terrain_demo.py` and similar scripts:

```bash
# Basic rendering
--dem path/to/dem.tif
--width 1920 --height 1080
--output render.png

# Shadows
--shadow-technique pcf|pcss|vsm|evsm|msm
--shadow-resolution 2048
--cascade-count 4

# TAA
--taa
--taa-history-weight 0.9

# OIT
--oit wboit|dual_source|auto

# Sun position
--sun-lat 46.8 --sun-lon -121.7
--sun-datetime "2024-06-21T12:00:00"

# Presets
--preset high_quality|fast|default
```

## 19. Clipmap Terrain System (P2)

This section documents the clipmap terrain LOD system architecture, geo-morphing concepts, and visualization patterns.

### 19.1 Clipmap architecture overview

The clipmap system uses **nested concentric rings** of terrain mesh, each ring doubling in cell size (halving resolution) from center outward:

```
┌─────────────────────────────────────┐
│           Ring 3 (coarsest)         │
│   ┌─────────────────────────────┐   │
│   │       Ring 2                │   │
│   │   ┌─────────────────────┐   │   │
│   │   │     Ring 1          │   │   │
│   │   │   ┌─────────────┐   │   │   │
│   │   │   │Center Block │   │   │   │
│   │   │   │  (finest)   │   │   │   │
│   │   │   └─────────────┘   │   │   │
│   │   └─────────────────────┘   │   │
│   └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

**Key files:**
- `src/terrain/clipmap/mod.rs` - Module exports
- `src/terrain/clipmap/config.rs` - `ClipmapConfig` struct
- `src/terrain/clipmap/level.rs` - `ClipmapLevel`, `ClipmapMesh` generation
- `src/terrain/clipmap/ring.rs` - `make_center_block()`, `make_ring()` functions
- `src/terrain/clipmap/geomorph.rs` - Geo-morphing and seam correction (P2.2)
- `src/terrain/clipmap/gpu_lod.rs` - GPU LOD selection compute shader (P2.3)
- `src/shaders/clipmap_terrain.wgsl` - Vertex shader with morph blending
- `src/shaders/clipmap_lod_select.wgsl` - Compute shader for frustum culling

### 19.2 Clipmap vertex format

Each clipmap vertex contains position, UV, and morph data:

```rust
#[repr(C)]
pub struct ClipmapVertex {
    pub position: [f32; 2],    // X, Z world position
    pub uv: [f32; 2],          // Texture coordinates [0,1]
    pub morph_data: [f32; 2],  // [morph_weight, ring_index]
}
```

**Morph weight interpretation:**
- `0.0` = Use fine (current ring) height
- `1.0` = Use coarse (outer ring) height  
- `-1.0` = Skirt vertex (no morphing, dropped below terrain)

### 19.3 Geo-morphing (P2.2) concepts

Geo-morphing prevents "popping" artifacts when LOD changes by smoothly blending vertex heights at ring boundaries.

**Key parameters:**
- `morph_range` (0.0-1.0): Fraction of ring width where morphing occurs
- Default: `0.3` (morphing in outer 30% of each ring)

**Shader morph blending:**
```wgsl
// In clipmap_terrain.wgsl
let fine_height = sample_height(uv);
let coarse_height = sample_height(snap_to_coarser_grid(uv, ring_index));
let final_height = mix(fine_height, coarse_height, morph_weight);
```

**Seam correctness:** Vertices at ring boundaries snap UVs to coarser grid to prevent T-junctions (gaps between LOD levels).

### 19.4 GPU LOD selection (P2.3) concepts

The GPU LOD selection compute shader performs:

1. **Frustum culling:** Reject tiles outside view frustum
2. **Screen-space error calculation:** `sse = (cell_size * viewport_height) / (distance * 2 * tan(fov/2))`
3. **LOD selection:** Choose coarsest LOD where SSE < threshold

**Key parameters:**
- `viewport_height`: Screen height in pixels
- `fov_y`: Vertical field of view in radians
- `sse_threshold`: Maximum allowed pixel error (typically 1-4 pixels)

### 19.5 DEM loading patterns

When loading DEM files (GeoTIFF), handle geographic vs projected CRS correctly:

```python
def get_pixel_size_meters(dem_path: Path) -> float:
    """Get pixel size in meters, handling geographic CRS."""
    with rasterio.open(dem_path) as ds:
        crs = ds.crs
        pixel_x = abs(ds.transform.a)
        
        if crs and crs.is_geographic:
            # Convert degrees to meters at center latitude
            center_lat = (ds.bounds.top + ds.bounds.bottom) / 2
            meters_per_degree = 111320 * math.cos(math.radians(center_lat))
            return pixel_x * meters_per_degree
        else:
            return pixel_x  # Already in meters
```

**Common DEM shortcuts in examples:**
```python
DEM_SHORTCUTS = {
    "fuji": "assets/tif/Mount_Fuji_30m.tif",
    "rainier": "assets/tif/Mount_Rainier_30m.tif",
    # ...
}
```

### 19.6 Visualization and statistics patterns

When implementing visualization for terrain systems:

**1. Ring structure visualization:**
```python
# Color each LOD ring distinctly
RING_COLORS = [
    (0, 100, 255),   # Blue - center
    (0, 200, 100),   # Green - ring 1
    (255, 200, 0),   # Yellow - ring 2
    (255, 100, 0),   # Orange - ring 3
    (255, 0, 0),     # Red - ring 4
]
```

**2. Morph weight heatmap:**
```python
# Blue (0.0) → Green (0.5) → Red (1.0)
def morph_weight_to_color(weight: float) -> tuple:
    if weight < 0:  # Skirt vertex
        return (128, 128, 128)
    if weight < 0.5:
        t = weight * 2
        return (0, int(255 * t), int(255 * (1 - t)))
    else:
        t = (weight - 0.5) * 2
        return (int(255 * t), int(255 * (1 - t)), 0)
```

**3. Screen-space error calculation for stats:**
```python
def calculate_sse(distance: float, cell_size: float, 
                  viewport_height: int, fov_y: float) -> float:
    """Calculate screen-space error in pixels."""
    if distance <= 0:
        return float('inf')
    pixels_per_meter = viewport_height / (2 * distance * math.tan(fov_y / 2))
    return cell_size * pixels_per_meter
```

**4. LOD statistics output format:**
```
==========P2.3 LOD Selection Statistics===========

  Triangle Distribution by LOD Ring:
    Center    :  4745 verts ( 75.3%), ~7,706 triangles
    Ring 1    :   520 verts (  8.2%), ~844 triangles
    ...

  Screen-Space Error Analysis:
    Projected Pixel Size by Distance:
      100m: 13.04 px/m, cell=91.0px → LOD 4
      500m: 2.61 px/m, cell=18.2px → LOD 4
      ...
```

### 19.7 Clipmap demo CLI flags

The `clipmap_demo.py` example supports:

```bash
# Basic usage
python examples/clipmap_demo.py --dem fuji

# Custom configuration
--rings N              # Number of LOD rings (default: 4)
--resolution N         # Vertices per ring side (default: 64)
--morph-range 0.3      # Geo-morph blend range [0.0-1.0]
--skirt-depth N        # Skirt depth in meters (auto if not set)

# Visualization (P2.2/P2.3)
--visualize            # Generate combined visualization image
--lod-stats            # Print detailed LOD statistics
--output FILE          # Output path for visualization image

# LOD analysis parameters
--camera-height N      # Simulated camera height (default: 1000m)
--viewport-height N    # Viewport height for SSE (default: 1080)
--fov N                # Camera FOV in degrees (default: 45)
```

### 19.8 Common clipmap mistakes

1. **Forgetting morph_data in vertex format:** Clipmap vertices need `[morph_weight, ring_index]` for shader blending.

2. **Wrong UV snapping:** Seam vertices must snap to coarser grid: `uv = floor(uv * coarse_res) / coarse_res`

3. **Skirt vertices with positive morph weight:** Skirts use `morph_weight = -1.0` to indicate "no morphing, drop below surface."

4. **Not accounting for ring doubling:** Cell size doubles each ring: `cell_size(ring) = base_cell_size * 2^ring`

5. **Geographic CRS pixel size:** Always convert degrees to meters when DEM uses EPSG:4326.

## Reflections

- Keep viewer TAA plumbing aligned with core structs (use setters/getters) and avoid borrow conflicts in render passes by splitting field borrows or moving temporary views out before mutating other fields.

