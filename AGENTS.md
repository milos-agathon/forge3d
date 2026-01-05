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
40. **Explicit deliverables for every decision.** When a decision is made (remove feature vs implement), list concrete deliverables: files, functions, tests, acceptance renders, and stop-conditions. If you can’t enumerate deliverables, stop.

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