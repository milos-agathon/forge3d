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
