# forge3d – Agent Orientation Guide

This document is for **AI coding agents** working inside the `forge3d` repository. It gives you a fast, *code-grounded* overview so you can navigate, modify, and debug the project effectively without breaking tests or architecture assumptions.

If you only remember a few things, remember:

- **Rust crate first, Python package second** – Rust `src/` is the rendering engine, Python `python/forge3d/` is a high-level, well-validated facade.
- **Tests and docs define behavior** – Always consult `tests/` and `docs/` before changing semantics or signatures.
- **Memory, GPU features, and QA are strict** – Don’t ignore the memory budget, feature flags, or acceptance tests.

## Coding rules

Code is not going away; bad code destroys productivity; clean code is about readability and ease of change. Professionals adopt the attitude that keeping code clean is their job, especially under pressure.

### Must do

* Treat **cleanliness as a professional obligation**, not a luxury.
* Recognize that **bad code’s cost compounds**: every messy decision today slows you more tomorrow.
* Remember we **read far more than we write**:

  * Optimize for reading: small functions, clear names, consistent structure.
* Adopt the “leave it cleaner than you found it” habit:

  * Any time you touch a file, improve at least one small thing.
* Push back on the “primal conundrum” (pressure to ship vs. clean):

  * Acknowledge time pressure but still carve out minimal refactoring steps.

### Must don’t

* Don’t assume tooling, higher-level languages, or AI will make code “go away.”
* Don’t justify messy code with deadlines (“we’ll clean it later”).
* Don’t treat clean code as an aesthetic preference; it’s directly tied to **productivity and livability**.
* Don’t accept a culture where “heroic” hacks are praised but maintainers suffer later.

---

You don’t write clean code in one shot. You first get it working, then you **systematically clean it**. There is explicit emphasis on a disciplined cleaning process, and an AI-assisted refactoring vignette in the postscript.

### Must do

* Follow the two-phase loop:

  * **Phase 1:** Get it working (even if a bit messy).
  * **Phase 2:** Refactor systematically into clean code while tests are green.
* Make the **cleaning process explicit**:

  * Plan time to simplify functions, clarify names, remove duplication.
* Work in **small refactoring steps** with tests between steps:

  * Rename → run tests.
  * Extract function → run tests.
  * Simplify logic → run tests.
* Use AI (as in the Grok3 postscript) as a **helper**:

  * Let it suggest refactors, but you decide structure, naming, and boundaries.

### Must don’t

* Don’t skip the cleaning phase once “it works.”
* Don’t refactor without safety nets:

  * No large restructuring without tests.
* Don’t treat AI’s refactored output as authoritative:

  * Never merge AI code that you haven’t checked for clarity, correctness, and design.
* Don’t bundle huge logical changes and refactors into a single opaque diff.

---

### Must do

* Strive for **small, well-named, well-organized units**:

  * Modules, classes, and functions should be tiny and focused.
* Treat **function design as foundational**:

  * Functions are where “small and well-named” matters the most.
* Think in terms of **independent deployability**:

  * Group behavior so that components can be built, tested, and deployed in isolation (Rust crates, Python packages, shader modules).
* Apply the same principles across **all layers**:

  * Python services, Rust core, WGSL shaders all benefit from small, ordered code.

### Must don’t

* Don’t accept large, tangled modules because “that’s how it grew.”
* Don’t write functions that require reading half the codebase to understand.
* Don’t architect components so tightly coupled that **deploying one** requires rebuilding the entire system.
* Don’t treat these principles as only applicable to “big systems” – they apply even to small utilities and shaders.

---

### Must do

* Make names **reveal intent**:

  * A reader should know *what* something represents and *why it exists* from the name.
* Avoid **misleading or ambiguous names**:

  * Don’t use domain terms incorrectly; avoid words that suggest the wrong behavior.
* Ensure names are **pronounceable and searchable**:

  * This matters for discussion and for searching through the codebase.
* Use **appropriate length**:

  * Important abstractions can have longer names; minor locals can be short but still clear.
* Drop **obsolete encodings**:

  * Don’t prefix types with Hungarian notation-style tags; rely on types and context.
* Use **parts of speech correctly**:

  * Nouns for objects/data (`mesh`, `camera_config`),
  * Verbs for functions (`load_texture`, `submit_frame`).
* Use **consistent vocabulary**:

  * One term per concept across the codebase (e.g., always “tile”, not “tile/chunk/block” randomly).
* Mix **solution-domain** and **problem-domain** names appropriately:

  * GPU concepts: `command_buffer`, `descriptor_set`.
  * Domain concepts: `elevation_tile`, `landcover_class`.

### Must don’t

* Don’t use misleading names that imply wrong units, spaces, or semantics (e.g. `lat`/`lon` when they’re local x,y).
* Don’t create meaningless distinctions (`data1`, `data2`; `result`, `result2`).
* Don’t rely on cryptic abbreviations to “save typing.”
* Don’t reuse the same word for multiple unrelated things (`config` meaning 10 different structures).
* Don’t overuse clever or “cute” names that obscure purpose.

---

### Must do

* Recognize comments as **compensation for shortcomings** in code clarity:

  * Prefer improving the code first.
* Keep comments focused on:

  * **Intent and reasons** for decisions.
  * **Non-obvious consequences** and constraints.
  * When necessary, **legal or regulatory** information.
* Use **API-level documentation** where appropriate:

  * Public interfaces warrant carefully written docs.
* Use TODO-style comments **sparingly and specifically**:

  * They should refer to concrete follow-ups, ideally with issue IDs.

### Must don’t

* Don’t leave **hidden or obscure comments** that only make sense to the author.
* Don’t write **lying comments** that no longer match the code.
* Don’t write comments that are **too intimate** with implementation details that change frequently.
* Don’t assume comments can **make up for bad code**:

  * If code is unclear, refactor; don’t just explain the mess.
* Don’t fill code with **journal comments** (“changed this on date X”) – version control already does this.
* Don’t create **noise comments** that add no information (e.g. `// increment i`).
* Don’t leave **commented-out code** lying around; delete it and rely on VCS.
* Don’t push large or irrelevant blocks of text (HTML comments, big essays) into source files.
* Don’t write overly generic **function headers** that just restate the name.

---

### Must do

* Use **vertical whitespace** to signal conceptual separation:

  * Group related lines into blocks and separate unrelated ones.
* Control **vertical density**:

  * Don’t cram everything; don’t scatter a single concept across many pages.
* Keep **related code vertically close**:

  * Declarations near use; dependent functions near each other.
* Use **horizontal spacing** intentionally:

  * Break up complex expressions into readable pieces.
* Maintain **consistent indentation**:

  * Indentation shows scope; make it trustworthy.
* Align certain constructs where it helps readability:

  * Struct fields, uniform declarations, configuration tables.
* Follow **team formatting rules** consistently:

  * Let tools enforce them (formatters) and avoid human bike-shedding.

### Must don’t

* Don’t let formatting be arbitrary or purely personal style.
* Don’t place highly related code far apart vertically without reason.
* Don’t rely on clever horizontal alignment that becomes brittle when code changes.
* Don’t break indentation conventions to “squeeze code in.”
* Don’t wage style wars once the team rules are established and automated.

---

### Must do

* Keep functions **very small**:

  * This chapter strongly reinforces smallness as a primary property.
* Write functions like **well-structured prose**:

  * Each function expresses a conceptual sentence; the file reads like a story.
* Keep each function at **one level of abstraction**:

  * Don’t mix high-level policy with low-level details in the same function.
* Respect the **stepdown rule**:

  * Readers should be able to read from top to bottom, descending logically in abstraction.
* Treat switch-like constructs carefully:

  * Isolate them and avoid scattering them across the codebase.
* Aim for functions that are:

  * **Contextual**: parameters and location make sense.
  * **Nameable & descriptive**: you can describe their purpose in a short phrase.
  * **Convenient**: easy and natural to call.
  * **Insulated**: internal changes don’t force wide changes.
  * **Homogeneous**: one coherent responsibility.
  * **Pure or nearly pure**: side effects are explicit or minimized.

### Must don’t

* Don’t write functions that span many concepts or abstractions.
* Don’t bury low-level operations inside high-level orchestration code.
* Don’t force readers to constantly jump up and down abstraction levels (“abstraction roller coaster” at function level).
* Don’t litter switch statements all over; they should be centralized or encapsulated behind polymorphism/strategies.
* Don’t accept functions whose behavior you can’t easily summarize in a single, coherent sentence.

---

### Must do

* Keep **argument lists short**:

  * Avoid more than a small number of parameters; group related ones into objects/structs.
* Use **keyword arguments** (or named parameters) where supported to clarify calls.
* Avoid **flag arguments**:

  * Use separate functions or configuration types instead.
* Avoid **output arguments**:

  * Return data instead; in Rust this may mean returning structs or tuples with clear meaning.
* Prefer **exceptions (or equivalent mechanisms)** over error codes:

  * Centralize handling; avoid scattering error checks everywhere.
* Isolate and refactor **explicit error-handling blocks** (try/catch, match arms) so they don’t pollute main logic.
* Apply **Command–Query Separation**:

  * Functions either do something (command) or answer something (query), not both.
* Apply **DRY** carefully:

  * Consolidate true duplication, but distinguish between trivial repetition vs. conceptual duplication.
* Be mindful of **side effects**:

  * Minimize them; consider more functional style for complex logic.
* Keep structured programming basics in mind:

  * Favor clear sequence, selection, and iteration structures.

### Must don’t

* Don’t pass long lists of unrelated parameters.
* Don’t hide multiple different behaviors behind a boolean flag or magic numeric argument.
* Don’t scatter error-code checks everywhere; they become a dependency magnet.
* Don’t force callers into confusing patterns with output arguments.
* Don’t blindly DRY everything; merging superficially similar code can introduce wrong abstractions.
* Don’t let side effects be implicit or hidden behind benign-looking functions.

---

### Must do

* Start by **making behavior correct and test-covered** before big refactors.
* Then **make the method right**:

  * Extract smaller functions,
  * Clarify names,
  * Remove duplication,
  * Separate concerns.
* Consider **design and architecture** implications:

  * As you refactor a method, you may discover better class/module boundaries.
* Use a **stepwise approach**:

  * Each refactoring step should be small and verifiable.

### Must don’t

* Don’t “clean” methods without verifying behavior first.
* Don’t refactor large methods in one giant leap with no intermediate checkpoints.
* Don’t ignore signs from a method that suggest deeper architectural issues (wrong class, wrong module).
* Don’t stop halfway when the structure is still clearly muddled.

---

### Must do

* Use **extract-method refactoring** aggressively:

  * Break big functions into smaller ones that each do one thing.
* Treat **single responsibility** for functions as non-negotiable.
* Recognize that **large functions hide intent**:

  * Extracting them clarifies the story and often clarifies class responsibilities.
* Use function extraction to discover **latent classes**:

  * Related extracted functions and data may belong together as a type.

### Must don’t

* Don’t accept “drowning in small functions” as a real objection:

  * Properly named, they make the code more navigable, not less.
* Don’t hide behind performance concerns unless you have hard evidence; modern compilers and runtimes inline small functions well.
* Don’t let “bouncing around” (following calls) become a problem of poor naming and structure; fix the design rather than giving up on extractions.
* Don’t ignore function size as a smell – large functions are almost always doing too much.

---

### Must do

* Use the **newspaper metaphor**:

  * Top: big picture.
  * Then: sections with increasing detail.
* Structure files so that **high-level concepts come first**, then the details.
* Apply the **stepdown rule** at file level:

  * Each step down in the file should reveal a lower abstraction level.
* Tame the **abstraction roller coaster**:

  * Keep jumps between high and low level to a minimum; group similar levels together.

### Must don’t

* Don’t write in the order you discover things and leave it like that.
* Don’t force readers to constantly bounce between unrelated details and high-level orchestration.
* Don’t bury core ideas deep in the file under heaps of low-level helpers.
* Don’t be “impolite” to readers by making them work hard to reconstruct structure that could have been laid out clearly.

### Must do

* Understand the **antisymmetry between objects and data structures**:

  * Objects hide data and expose behavior.
  * Data structures expose data with minimal behavior.
* Use **data abstraction**:

  * Hide representation; expose clear operations that express intent.
* Respect the **Law of Demeter**:

  * Avoid “trainwreck” call chains (`a.b().c().d()` style).
  * Limit how far an object reaches into others.
* Use **DTOs** where appropriate:

  * Simple structs or records for transferring data, e.g. across processes or API boundaries.
* Be deliberate about **OO vs procedural** trade-offs:

  * Sometimes procedural, data-centric code is better; sometimes object-centric is better.
* Consider **performance** without letting it justify bad abstractions:

  * Optimize where needed, but keep interfaces clean.

### Must don’t

* Don’t create **hybrids** that are neither proper objects nor clean data structures.
* Don’t violate the Law of Demeter with deep navigation through object graphs.
* Don’t leak internal representation details all over the codebase.
* Don’t overuse switch statements when polymorphism (or strategy objects, or enums with methods in Rust) would make behavior easier to extend.
* Don’t assume OO is always better than procedural, or vice versa – the chapter explicitly frames this as a trade-off.

---
### Must do

* Distinguish **classes/modules** from mere files:

  * Organize around responsibilities, not file boundaries.
* Design classes that are:

  * **Cohesive**: all methods relate to a single purpose.
  * **Closed** to unrelated changes.
  * **Focused on a single responsibility**.
* Look for **reasons to change**:

  * Each class should have just one dominant reason.
* Keep classes **small**:

  * When they grow too large, split responsibilities.
* Keep policies (the rules) visible in code:

  * Clarify where decisions live, and group related policy decisions together.
* Leverage clean classes for **simpler testing**:

  * Smaller, cohesive classes are easier to test in isolation.
* Treat AI assistance skeptically:

  * Expect AI-generated class designs to be wrong or incomplete; adjust based on principles in this chapter.

### Must don’t

* Don’t hide multiple unrelated reasons to change inside one “do everything” class.
* Don’t let classes grow unchecked until they become untestable god objects.
* Don’t confuse “file organization” with “class design” – putting things in the same file doesn’t automatically make good classes.
* Don’t accept AI-generated class designs without verifying responsibilities, dependencies, and change reasons.

---

### Must do

* Understand and practice **TDD**:

  * Short cycles of write a failing test → make it pass → refactor.
* Understand **TCR (Test && Commit || Revert)**:

  * Emphasize very small steps that always keep the mainline green.
* Work in **small change bundles**:

  * Keep each change tiny to maintain control and easy rollback.
* Use tests as a **design tool**:

  * They drive interfaces and modularity.
* Treat testing as a **discipline**, even if it feels tedious:

  * The payoff is in debugging reduction, reliability, and documentation.
* Keep tests **clean**:

  * They must obey similar cleanliness standards as production code.
* Regard tests as enabling the **“-ilities”**:

  * Maintainability, flexibility, etc., come from a strong test safety net.

### Must don’t

* Don’t dismiss testing disciplines as “too slow”; the chapter explicitly argues the opposite.
* Don’t treat tests as optional or merely an afterthought.
* Don’t let tests become messy or unmaintainable – dirty tests undermine confidence and design.
* Don’t carve out “exceptions” to discipline lightly; every loophole weakens the entire practice.

---

### Must do

* Evolve a **domain-specific language for tests**:

  * Helpers and abstractions that make tests read like the problem domain.
* Use **composed assertions and results**:

  * Build expressive utilities rather than repeating low-level checks.
* Apply a **“dual standard”**:

  * Production and test code both need to be clean, but tests are allowed different trade-offs (e.g., more duplication if it improves clarity).
* Consider the **Single Assert / Single Act** rules:

  * Tests should usually do one thing and check one essential condition.
* Apply **F.I.R.S.T.**:

  * Tests should be Fast, Isolated, Repeatable, Self-Validating, Timely.
* Design tests thoughtfully:

  * Structure them clearly; treat test design as seriously as production design.

### Must don’t

* Don’t let tests become overly clever or opaque; they’re supposed to clarify behavior.
* Don’t stuff many unrelated assertions and acts into one test.
* Don’t tolerate slow, flaky tests; they’ll be skipped or ignored.
* Don’t defer tests until “after coding”; timeliness is part of their value.

---

### Must do

* Treat **acceptance testing** as a discipline in its own right:

  * Define clear, user-level acceptance criteria.
* Develop and maintain **automated acceptance tests**:

  * They verify that the system meets agreed behavior end-to-end.
* Integrate acceptance tests into a **continuous build**:

  * Changes are frequently validated against these tests.
* Collaborate across roles:

  * Developers, testers, and stakeholders align on what acceptance means.

### Must don’t

* Don’t rely exclusively on unit tests for system behavior; acceptance tests catch cross-cutting issues.
* Don’t allow acceptance testing to become entirely manual and ad-hoc.
* Don’t treat acceptance tests as one-time artifacts; they must evolve with the system.
* Don’t ignore failures of acceptance tests as “just integration flakiness”; those are important signals.

---

### Must do

* Understand **prompt-driven programming** as a new tool, not a replacement for understanding:

  * You still need to reason about correctness, design, and architecture.
* Treat LLM-generated code as **immature**:

  * It’s in its early stages; expect limitations and errors.
* Make a **disciplined, skeptical use** of AI:

  * Review and refactor its output rigorously.
* Continuously revisit how AI fits into your workflow as it evolves:

  * Expect changes; be ready to adjust your practices.

### Must don’t

* Don’t assume AI can safely replace core engineering judgment.
* Don’t treat AI suggestions as ground truth for design or architecture.
* Don’t relax clean-code standards because “the AI wrote it.”
* Don’t bet your system’s correctness on speculative future AI capabilities.

### Must do

* **Treat simplicity as “untangled,” not “quick.”**
  The chapter explicitly says simple means untangled, and untangling things is hard work. Expect that making design simple is effortful refactoring, not just fewer lines.

* **Apply YAGNI as a serious design question.**
* Simple design is governed by **four explicit rules**, in order of priority:

  1. **Covered by tests.**
  2. **Maximize expression.**
  3. **Minimize duplication.**
  4. **Minimize size.**

### Must do

* **Treat simplicity as “untangled,” not “quick.”**
  The chapter explicitly says simple means untangled, and untangling things is hard work. Expect that making design simple is effortful refactoring, not just fewer lines.

* **Apply YAGNI as a serious design question.**
  Every time you’re tempted to add a “hook” for a possible future requirement, *ask yourself explicitly*:

  > “What if I’m not gonna need it?”
  > Only add that hook if you have strong, concrete reasons—not vague “we might need this someday.”

* **Aim for asymptotically high test coverage, anchored on 100%.**
  The chapter argues that:

  * The only sensible coverage target is **100%** (line + branch),
  * But you treat it as an **asymptotic goal** (you may never reach it, but you always move closer).
    That high coverage is the **first rule of simple design**.

* **Use tests to force decoupled, testable code.**
  It explicitly says: *testable code is decoupled code*.
  To cover a unit well, you must design it so it can be tested in isolation. That pushes you to design better boundaries and APIs.

* **Recognize that the other three rules only work if you have tests.**
  The chapter is explicit:

  * Coverage is first because the other rules (expression, duplication, size) are **refactoring rules**,
  * And meaningful refactoring is “virtually impossible” without a good suite of tests.

* **Maximize expression: make code read like well-written prose.**
  The chapter contrasts older, cryptic code full of comments with modern expressive languages. It explicitly states that, with discipline, code can read like “well-written prose.”
  You should:

  * Choose names and structure so the **intent** is clearly visible in the code.
  * Make the algorithm and concepts obvious without needing heavy comments.

* **Ask what the underlying abstraction is.**
  In the “Underlying Abstraction” section, the chapter shows how expressive code arises from:

  * Finding the **right abstraction**, not just nicer names.
  * Refining that abstraction over time under the safety of tests.

* **Minimize duplication (“third in the priority list”).**
  The chapter walks through the history of code reuse and shows that:

  * Duplication leads to fragility, because changes are needed in multiple places.
  * A simple design reduces duplication *after* you have good tests and expressive code.
    It explicitly says eliminating duplication is **third in priority**: tests → expression → duplication.

* **Minimize size after tests, expression, duplication.**
  Under “Minimize Size” the chapter says:

  * A simple design uses **as few simple elements as possible** without compromising tests or expression.
  * Only after tests, expression, and duplication are addressed should you work to decrease **modules, classes, functions, and lines**.

### Must don’t

* **Don’t equate “simple” with “easy” or “minimal effort.”**
  The chapter explicitly warns: simple is untangled, and untangling is hard. If you choose the “easy” route (don’t untangle), you’re not doing simple design.

* **Don’t add speculative hooks just because you can imagine a future need.**
  YAGNI is not “never think of the future”; it’s **“question speculative features.”**
  Don’t add extension points or configuration just because “we might need this someday” when the chance you’ll actually need it is low.

* **Don’t settle for arbitrary coverage targets like 80–90% as “good enough.”**
  The chapter explicitly pushes back on treating coverage percentages as quota goals. It says the only reasonable coverage goal is **100%**, treated asymptotically.

* **Don’t try to apply expression/duplication/size rules without tests.**
  It states directly that the other three rules become impractical without good tests. Don’t refactor aggressively to remove duplication or shrink code if you don’t have tests to protect behavior.

* **Don’t use comments as a substitute for expression.**
  The example of 1960s code full of comments is there to show that:

  * In modern languages, heavy commenting around opaque code is a smell.
    You should change the code to be expressive instead.

* **Don’t remove duplication blindly.**
  The chapter is clear that duplication is problematic, but it also talks about the difficulty of identifying what’s truly “the same thing.”
  Don’t merge code paths that merely look similar but represent different concepts.

* **Don’t chase small size at the expense of tests, expression, and duplication.**
  “Minimize size” is **last** in priority. Don’t compress or over-consolidate code in ways that make it less expressive or harder to test just to save lines.

---

* SOLID is a set of principles for **mid-level software structures**: how to organize functions and data into modules, classes, or similar groupings.
* These principles apply whether or not your language literally has “classes.”
* The goals (explicitly listed) are structures that:

  * **Tolerate change**,
  * **Are easy to understand**,
  * **Form the basis for components**.

### Must do

* **Use SOLID to shape mid-level structures.**
  The chapter says SOLID explains:

  * How to package functions and data into coherent groupings,
  * And how those groupings (classes/modules/etc.) should depend on each other.

* **SRP – base responsibilities on actors (reasons to change).**
  The chapter explicitly refines SRP to:

  > A module should be responsible to one, and only one, *actor*.
  > Where an *actor* is a group of people (e.g. accounting, HR, DBAs) that require changes.

  * Identify which actor each module serves.
  * Split modules when they serve multiple distinct actors.

* **OCP – make artifacts open for extension, closed for modification.**
  The chapter quotes Meyer’s formulation:

  > A software artifact should be open for extension but closed for modification.
  > That means:

  * Behavior should be extendible via new code (e.g. new types, new modules),
  * Without editing the existing artifact itself.

* **LSP – ensure substitutability.**
  The chapter quotes Liskov’s definition: substituting subtype instances must **not change correct behavior** of clients.

  * When you have a base and “subtypes”, ensure the client code doesn’t depend on which subtype it gets, as long as it talks to the base contract.
  * Violations mean the subtype is not a true subtype.

* **ISP – segregate interfaces to avoid depending on unused things.**
  The chapter’s example is a class with operations `op1`, `op2`, `op3`, and users that need only a subset.

  * Split operations into interfaces so each user depends only on what it actually calls.
  * This isolates users from changes in operations they don’t care about.

* **DIP – invert dependencies so details depend on policies.**
  The chapter explicitly contrasts:

  * Old style: source code dependencies follow the flow of control (high-level modules depend on low-level modules they call).
  * With DIP style: insert an interface so the low-level module depends on the interface, and the high-level module calls through that interface.
    This gives architects **control over every source code dependency**.

### Must don’t

* **Don’t confuse SRP with “modules should do just one *thing* at implementation level.”**
  The chapter says explicitly:

  * That “do one thing” principle (for functions) is real, but **it is not SRP**.
  * SRP is about one **reason to change / one actor**, not “one operation.”

* **Don’t lump responsibilities for multiple actors into one class/module.**
  The `Employee` example shows:

  * `calculatePay()` (accounting),
  * `reportHours()` (HR),
  * `save()` (DBA)
    all in one class—this couples unrelated org units and creates change hazards.

* **Don’t implement extensibility by constantly editing the same core module.**
  That breaks OCP.

  * If adding a new case always means editing a central switch or modifying an existing class, you’re not truly open/closed.

* **Don’t violate substitutability for convenience.**
  If a subtype:

  * Breaks client assumptions,
  * Throws different kinds of errors in cases where the base wouldn’t,
  * Or requires extra knowledge from the client,
    then it violates LSP.

* **Don’t force clients to depend on operations they don’t call.**
  This is exactly what ISP warns against: one big interface/class used by multiple clients with different needs creates unnecessary coupling.

* **Don’t let your high-level policy modules depend directly on low-level implementation details.**
  DIP explicitly says low-level details should depend on high-level policies, not the other way around.

  * Don’t have business rules import concrete database/IO/etc. types.

---

### Must do

#### Components and history

* **Think of components as the granule of deployment.**
  A component is:

  * The smallest entity you deploy as part of the system,
  * The unit at which you manage versions and releases.

* **Understand why components emerged.**
  The history section explains the evolution:

  * Early libraries were kept in source and compiled together,
  * Compilation time and device limitations drove separation into independently compiled units,
  * That separation evolved into the components we manage and deploy now.

#### Cohesion principles

* **REP – treat “reuse” and “release” as inseparable.**
  The chapter states:

  > The granule of reuse is the granule of release.

  * If you want to reuse a set of classes/functions, you package and version them as a unit.
  * Reused artifacts should have a clear version and release process.

* **CCP – group classes that close together under the same kind of change.**
  The chapter presents CCP as a principle for grouping classes:

  * Classes that tend to change for the same reasons should be in the same component.
  * That way, when requirements change, you can often modify just one component.

* **CRP – avoid forcing users to depend on what they don’t reuse.**
  CRP says:

  * Classes/modules that are reused together should live together.
  * Don’t bundle things that are rarely used together in the same component, or consumers will pull in unnecessary stuff.

#### Coupling principles

* **ADP – maintain an acyclic component dependency graph.**
  The Acyclic Dependencies Principle is stated explicitly:

  > Allow no cycles in the component dependency graph.
  > This:

  * Simplifies understanding and builds,
  * Prevents tangled interdependencies between components.

* **SDP – depend in the direction of stability.**
  The chapter states:

  > Depend in the direction of stability.

  * Stable components (many depend on them, they change rarely) should not depend on unstable ones.
  * Design for some components to be volatile, others to be stable, and orient dependencies accordingly.

* **SAP – match abstraction to stability.**
  The chapter states:

  > A component should be as abstract as it is stable.

  * Very stable components should be highly abstract (interfaces, policies), not full of concrete details.
  * Volatile components should be more concrete and easier to change.

* **Recognize the theory vs practice distinction.**
  The conclusion explicitly says:

  * You’re not expected to gather metrics and do full math in daily work.
  * These principles set target shapes; you apply them pragmatically.

### Must don’t

* **Don’t treat “component” as a purely conceptual grouping.**
  The chapter anchors it in real deployment units (jars, DLLs, etc.). Don’t ignore the deployment aspect.

* **Don’t mix classes that change for different reasons in the same component.**
  That violates CCP: now that component will constantly change for multiple distinct reasons, harming stability.

* **Don’t force users to pull in heavy transitive dependencies just to get a small piece.**
  Violates CRP: if consumers need only a subset but must depend on a big component, cohesion is wrong.

* **Don’t allow cycles in component dependencies.**
  ADP explicitly forbids cycles. They complicate builds, understanding, and independent deployment.

* **Don’t make stable, widely used components depend on volatile, frequently changing ones.**
  This inverts SDP and makes change expensive: every volatility in leaf components ripples into foundational ones.

* **Don’t make ultra-stable components concrete.**
  SAP warns against stable-but-concrete components. If something is very stable, you want it abstract so it can support multiple implementations and evolution.

* **Don’t obsessively “do the math” instead of using these as guiding heuristics.**
  The conclusion explicitly says you don’t normally gather stats and generate plots; the principles are guides, not bureaucratic rules.

---

### Must do

* **Treat the code as “the design.”**
  The chapter opens by emphasizing:

  * Over decades, experience showed that code and configuration *are* the definitive design.
  * Documentation and diagrams are, at best, secondary.

* **Accept that change is continuous.**
  Sections like “Continuous Change” and “Continuous Design” stress:

  * Product owners will keep reshaping what they want.
  * Therefore, your design must continuously adapt; it’s not done up front.

* **Use the Four Cs as continuous design criteria.**

  1. **Clarity**

     * The code should make the programmer’s intentions clearly visible.
     * After getting things working, you’re expected to revisit the code to make intent clearer, not just move on.

  2. **Conciseness**

     * Intent should be implemented with a **minimal amount of code**.
     * Remove redundancy and noise once clarity is in place, so that every line earns its keep.

  3. **Confirmability**

     * Every unit behavior should be **easily verifiable by tests**.
     * Tests act as “living documentation” of decisions and make it possible to safely change the design.

  4. **Cohesion**

     * Each module should have a **high level of cohesion**: its elements strongly relate to one another.
     * This echoes earlier chapters: align responsibilities so modules don’t mix unrelated concerns.

* **Balance the Four Cs; they interact.**
  The chapter explicitly points out:

  * These considerations don’t live in isolation.
  * You have to think about how they support or oppose one another as you evolve the design.

* **Recognize that design is done “all the time,” not just in a big initial push.**
  In “When Else Do We Design?” the chapter makes it clear:

  * We design when we write code, when we refactor, when we test.
  * Design is continuous, in small increments.

### Must don’t

* **Don’t think of design as something you finish before coding.**
  The chapter explicitly rejects the idea that design is mostly up-front. Code is the design, and both evolve together.

* **Don’t ship “it works” and never revisit clarity.**
  The Clarity section directly pokes at the instinct to stop once the code runs. Continuous design demands you loop back and clarify.

* **Don’t let verbosity and duplication accumulate.**
  Conciseness means you don’t leave bloated, repetitive code in place once the behavior is stable. You’re expected to take passes to compress and simplify.

* **Don’t treat tests as optional for design.**
  Confirmability ties design to testability. If unit behaviors aren’t easily testable, the design is lacking in confirmability.

* **Don’t throw unrelated responsibilities into the same module.**
  Cohesion as a design consideration means you don’t mix unrelated concerns just because it’s convenient; it erodes continuous design.

* **Don’t treat the Four Cs as independent checklists.**
  The chapter explicitly warns that they interact. Optimizing one (e.g. extreme conciseness) at the expense of others (e.g. clarity) is a misuse.

### Must do

* **Recognize the two separate values of software.**
  It’s not just about “features that work”; the *shape* of the system (architecture) is a value on its own.

* **Decompose systems into policy and details.**
  The text says every system can be split into:

  * **Policy** – high-level rules and behavior where the true value lives.
  * **Details** – the things needed to enable humans and machines (devices, DBs, frameworks, protocols).

* **Treat policy as the core value to protect.**
  The goal of architecture is explicitly:

  * Shape the system so the policy is central and stable,
  * And details can be delayed and swapped.

* **Leave as many options open as possible, for as long as possible.**
  The “Keeping Options Open” section and the final boxed quote hammer this:

  * Delay choices about details (DB, web server, REST, DI framework) while still building and testing policy.
  * More time → more information → better decisions.

* **Actively defer decisions about details.**
  The chapter gives concrete examples of decisions that **do not need to be made early**:

  * Which **database** (relational, distributed, hierarchical, flat files, etc.).
  * Which **web server** (or even whether it’s on the Web).
  * Whether to adopt **REST**.
  * Whether to use a particular **dependency injection framework**.
    High-level policy should not care about these.

* **Exploit the ability to experiment with details.**
  By not hardwiring details:

  * You can try multiple DBs, web frameworks, or even “the Web itself” to test applicability and performance before committing.

* **Ignore pre-made decisions when they pollute architecture.**
  There’s a section about “What if decisions have already been made?” (e.g., corporate standard database or framework).
  The advice is: behave architecturally **as if they hadn’t been made**, because you never know when those constraints will change.

### Must don’t

* **Don’t treat architecture as only “making it work now.”**
  If you optimize solely for behavior and ignore structure, you lose the “soft” part of software.

* **Don’t mix policy and details.**
  If high-level rules directly depend on database types, frameworks, or protocols, you’re binding policy to details and closing options.

* **Don’t commit early to details that don’t matter yet.**
  The chapter explicitly lists these as decisions you *can* defer. Committing early without need is exactly what it warns against.

* **Don’t stop experimenting just because an early choice “kind of works.”**
  If you lock in too soon, you lose the ability to try alternatives later when you have more knowledge.

* **Don’t accept external “standards” as immutable architectural boundaries.**
  Even if someone else chose “we’re a X database shop,” the chapter says you should still design so that choice can be reversed.

---

### Must do

* **Support use cases explicitly in the architecture.**
  The “Use Cases” section says:

  * Architecture must be shaped so the **intent of the system is visible** at the architectural level.
  * In the shopping cart example, you should see components named by use cases (e.g., cart, checkout) with clear responsibilities.

* **Structure for operation, not just code layout.**
  The “Operation” section says architecture plays a larger role here:
* The development
* The deployment

### Must do

* **Support use cases explicitly in the architecture.**
  The “Use Cases” section says:

  * Architecture must be shaped so the **intent of the system is visible** at the architectural level.
  * In the shopping cart example, you should see components named by use cases (e.g., cart, checkout) with clear responsibilities.

* **Structure for operation, not just code layout.**
  The “Operation” section says architecture plays a larger role here:

  * Decide how processing elements are arranged: monolith vs processes vs threads vs distribution.
  * The architecture should allow these operational arrangements to evolve as needs change.

* **Partition to enable independent development.**
  In “Development”, it says:

  * Systems built by many teams “with many concerns” require architecture that allows **independent actions** by those teams.
  * This is achieved by **proper partitioning** so teams don’t interfere with each other’s work.

* **Partition to ease deployment.**
  The “Deployment” section states:

  * Architecture should make it easy to deploy the system, ideally “immediately after each build.”
  * Again, this is achieved via proper partitioning and infrastructure so each component can be started, integrated, and supervised predictably.

* **Balance these concerns while still leaving options open.**
  The final section “Leaving Options Open” says:

  * Good architecture balances use cases, operation, development, deployment with a **continuous emphasis** on leaving options open.
  * And it explicitly concludes: good architecture makes the system easy to change in all the ways it must change by leaving options open.

### Must don’t

* **Don’t think of architecture as only supporting use cases.**
  The chapter is explicit: architecture must also support **operation, development, and deployment**, not just functionality.

* **Don’t design structures that hinder operational changes.**
  If changing from monolith to distributed, or altering scaling patterns, is painful because of tight coupling, you’re violating the “Operation” aim.

* **Don’t structure the system so teams constantly step on each other.**
  If multiple teams can’t work independently because architecture intertwines their concerns, you’ve failed the “Development” aim.

* **Don’t make deployment a special pain-phase separate from development.**
  If you can’t easily deploy after a build, the architecture isn’t serving deployment.

* **Don’t ignore the “options open” constraint while balancing these four concerns.**
  The chapter explicitly says this balancing is **hard** and that concerns are “indistinct and inconstant,” but still: leaving options open is mandatory, not optional.

---

### Must do

* **Draw boundaries to separate what matters (business rules) from what doesn’t (details).**
  In “What Lines Do You Draw, and When?”:

  * It states there should be a boundary between **business rules** and **databases**, **UI**, and other details.
  * You separate those “because they involve different things that matter.”

### Must do

* **Draw boundaries to separate what matters (business rules) from what doesn’t (details).**
  In “What Lines Do You Draw, and When?”:

  * It states there should be a boundary between **business rules** and **databases**, **UI**, and other details.
  * You separate those “because they involve different things that matter.”

* **Treat the database as a tool, not as business rules.**
  The chapter explicitly attacks the idea that “the database is the embodiment of the business rules” and calls it misguided:

  * The DB is a **tool** used by the business rules to store and retrieve data.
  * So the DB can (and should) be put **behind an interface**.

* **Make boundaries explicit through interfaces and dependency direction.**
  In the database diagrams:

  * There’s a `BusinessRules` component and a `Database` component.
  * The boundary runs through an interface (e.g., `DatabaseInterface`).
  * The **dependency arrow** points from `Database` (detail) **to** `BusinessRules` (policy).

* **Use plugin architecture: details plug in to core.**
  In “Plug-in Architecture”:

  * GUI and DB components are shown as **plug-ins** to the `BusinessRules` component.
  * UI and DB are considered **optional** or **multi-implementable** details.

* **Delay detail decisions via these boundaries.**
  The FitNesse case study shows:

  * They avoided committing to a DB for a long time by putting all data access behind an interface, stubbing it initially, then trying flat files, and eventually an optional MySQL implementation.
  * They also wrote their own minimal web server rather than adopting a big one early, guided by “Download and Go”.

* **Use boundaries to make experiments and replacements practical.**
  The chapter notes that:

  * Replacing DB or UI may **not be trivial**, but boundaries make it *practical*.
  * The FitNesse story ends with: drawing these lines helped them delay and defer many decisions and saved time and headaches.

* **Enforce the architectural Dependency Rule across boundaries.**
  The conclusion explicitly states:

  * Dependencies that cross architectural boundaries must point from **lower-level details** to **higher-level abstractions**.
  * This is called the **Dependency Rule of Architecture**.

### Must don’t

* **Don’t let detail decisions pollute core business logic.**
  The opening warns against coupling business rules to choices like DB, web frameworks, etc., especially when those decisions are premature.

* **Don’t treat the database as the “center” or the model of your system.**
  The chapter directly rejects that view; it breaks the policy-vs-detail discipline.

* **Don’t draw boundaries that point the wrong way.**
  If your `BusinessRules` depend on `Database` classes directly, you’ve reversed the arrow the chapter insists on: details must depend on policies.

* **Don’t treat UI and DB as core rather than plug-ins.**
  If UI and DB are not replaceable or optional, you’ve broken the plug-in idea described.

* **Don’t think “deferring decisions” means you never make them.**
  The FitNesse story shows they did make decisions (e.g. flat files, then a MySQL plugin)—just **after** getting more information and value.

---

### Must do

* **Isolate vendor/third-party infrastructure behind thin adapters.**
  In the IoT example:

  * The team builds on an off-the-shelf IoT platform.
  * They ensure vendor-specific complexity is localized in small, low-complexity functions/classes (e.g., `HydraNetwork`).
  * Hydra “knows nothing about” the vendor; the adapter hides details.

* **Keep business logic free of knowledge about infrastructure.**
  In the “important attributes” bullet list:

  * Business logic is **free from radio system knowledge** (IoT isolation).
  * Business logic is separate from UI (UI isolation).
  * Business logic is free from concurrency concerns (concurrency bullets).

* **Centralize concurrency in low-complexity code.**
  The concurrency bullets emphasize:

  * Concurrency mechanisms are confined to low-complexity functions that are unlikely to change often.
  * Business logic stays thread-/process-agnostic.

* **Separate object creation and binding from usage.**
  The object creation bullets explicitly say:

  * Creation is separate from use.
  * During initialization, concrete objects are created and bound together.
  * Binding is done on a “need-to-know” basis.

* **Use learning tests for third-party APIs.**
  “Exploring and Learning Boundaries”:

  * Learning the library and integrating it at the same time is “doubly hard.”
  * Instead, write **learning tests** that:

    * Call the third-party API as you *expect* to use it.
    * Capture the behavior you care about.
    * Help you know exactly how it works and if it changes between versions.

* **Use seams and adapters for code that doesn’t yet exist.**
  In “Using Code That Does Not Yet Exist”:

  * They define their **own interface** to a not-yet-designed transmitter/ADC hardware.
  * They treat this interface as “the one we wish we had.”
  * They wrap the real API later via an adapter/HAL behind that interface.
  * This gives them a seam and a single place to update if the underlying API changes.

* **Ensure code at boundaries has clear separation and tests.**
  In the final “Clean Boundaries” section:

  * Boundaries are where change happens.
  * Code at boundaries must be **clearly separated** and have tests that define and enforce behavior.

* **Minimize the number of places that touch third-party code.**
  The last paragraph explicitly says:

  * Manage third-party boundaries by having very **few places** in your code that interact with third-party packages.
  * This reduces maintenance points when the third-party changes.

### Must don’t

* **Don’t spread vendor/framework-specific calls all over your code.**
  That would make updates and replacements painful and violate the “few places” rule.

* **Don’t mix business logic with infrastructure, UI, or concurrency code.**
  The case study repeatedly isolates these: any mixing undermines the clean boundaries you’re supposed to maintain.

* **Don’t try to simultaneously learn and integrate third-party APIs in production code.**
  The chapter calls doing both at once “doubly hard” and recommends learning tests instead.

* **Don’t “just call the vendor API directly everywhere” because it’s easy.**
  That gives the vendor control over your architecture; the chapter warns explicitly against letting something you don’t control “end up controlling you.”

* **Don’t skip seams/interfaces for unknown or evolving code.**
  If you couple directly to a not-yet-solid (or not-yet-existent) API, you lose the ability to cheaply change when that API arrives or evolves.

### Must do

* **Aim for the shared properties of “clean” architectures.**
  The chapter explicitly lists what these architectures yield:

  * **Independent of frameworks** – architecture doesn’t depend on framework feature constraints.
  * **Testable** – business rules can be tested without UI, DB, web server, or any external element.
  * **Independent of UI** – UI can change without touching business rules.
  * **Independent of database** – DB can be swapped without affecting business rules.
  * **Independent of any external agency** – business rules know nothing about external interfaces.

* **Apply the Dependency Rule.**
  The rule, stated explicitly:

  > Source code dependencies must only point inward, toward higher-level policies.

  * Inner layers must not know about outer ones (no references to their classes/functions/variables).
  * Data formats of outer layers must not impose on inner layers.

* **Separate system into concentric policy vs mechanisms.**
  The circles are:

  * Inner circles = **policies**.
  * Outer circles = **mechanisms**.
    You design so mechanisms depend on policies, not the other way around.

* **Define and keep the Entity layer stable and enterprise-wide.**
  Entities:

  * Encapsulate **enterprise-wide critical business rules**.
  * Can be reused by many applications.
  * Must be unaffected by operational concerns (navigation, security, etc.).

* **Keep application-specific rules in the Use Case layer.**
  Use Cases:

  * Contain **application-specific business rules** that orchestrate entities to achieve goals.
  * Should not depend on external frameworks.
  * Can change when application operations change, without affecting entities.

* **Use Interface Adapters to convert data formats across boundaries.**
  Interface Adapters:

  * Convert data between external forms (e.g., web, UI, DB) and the formats most convenient for use cases/entities.
  * Contain controllers, presenters, gateways, etc.

* **Confine frameworks and drivers to the outermost layer.**
  Frameworks & Drivers:

  * Web, UI, DB, external devices, etc.
  * Contain mostly glue code.
  * Are considered **details**.

* **Use interfaces and dependency inversion to cross boundaries.**
  In “Crossing Boundaries”:

  * The chapter shows how to use interfaces (e.g., an output boundary implemented by a presenter) so:

    * Flow of control can go both ways,
    * But source code dependencies still point **inward**.

* **Pass simple data structures across boundaries.**
  In “What Data Crosses the Boundaries”:

  * Data to/from inner layers should be **simple data structures** (fields, primitives, simple lists) and not framework-specific types.
  * Data is shaped in the form most convenient for the inner circle.

* **Follow the typical scenario pattern for web systems.**
  The “Typical Scenario” section describes:

  * Controller receives input → calls Interactor (use case) → Interactor calls an output boundary → Presenter builds a ViewModel → View renders it.
  * All dependencies still point inward.

### Must don’t

* **Don’t let frameworks dictate your architecture.**
  The benefits section explicitly says systems should be “independent of frameworks.” If you shape the core around a particular framework’s constraints, you’ve missed the goal.

* **Don’t let business rules depend on UI, DB, or external systems.**
  If entities or use cases import those things or know their types, you’ve violated independence and the Dependency Rule.

* **Don’t allow dependencies to point outward across boundaries.**
  Inner circles must know nothing of outer circles. Any direction from policy to mechanism across architectural boundaries is a violation.

* **Don’t let database or ORM data structures leak into inner layers.**
  The chapter specifically warns about “convenient” DB-returned structures being passed around:

  * Doing so forces inner layers to know about outer frameworks.

* **Don’t put a lot of logic into the frameworks & drivers layer.**
  It should be mostly glue. Piling logic there makes upgrades/replacements painful and undermines the whole purpose of the outer-most circle.

* **Don’t overspecify the number of layers.**
  The “Only Four Circles?” section says four is schematic.
  Don’t rigidly cling to four; you may need more or fewer, as long as the dependency direction and policy/mechanism separation are preserved.

* **Don’t think conforming to these rules is optional.**
  The conclusion states:

  * Following these simple rules is not hard and will save you distress when web frameworks, DBs, or other technologies change; ignoring them ensures trouble later.

## Refactoring rules

### Must-do

* **Refactor in tiny, safe steps**

  * Make a small structural change → compile/run tests → repeat.
  * Code should *never* stay broken for more than a couple of minutes.

* **Always keep a working baseline with tests or a harness**

  * Even in small examples, each step is validated by some repeatable check, not hand-waving.

* **Continuously extract small, well-named functions**

  * Extract things like `amountFor`, `volumeCreditsFor`, etc.
  * Use extraction to clarify intent and remove duplication.
  * Treat extraction as your default move when a block of code needs a name or a test of its own.

* **Separate concerns: calculation vs. presentation**

  * First, build a data-rich representation (e.g., `createStatementData`) that computes all totals.
  * Then have separate renderers (text, HTML, etc.) that *only* format that already-computed data.

* **Introduce intermediate data structures deliberately**

  * Build intermediate objects that group related values (e.g., enriched performance entries).
  * This makes your pipeline clearer and later refactors (pipelines, iterator chains, phases) easier.

* **Move variation into polymorphism when type-conditionals proliferate**

  * When you see `if play.type == "tragedy" … else if "comedy" …` everywhere, introduce a polymorphic calculator hierarchy or equivalent strategy.

* **Let refactoring reveal the domain model**

  * Use naming and extraction to surface domain concepts (“volume credits”, “performance”, “amount”), not technical noise (“val1”, “tmp”).

### Must-not-do

* **Don’t call long-running, broken-build changes “refactoring”**

  * If the code is broken for days, that’s a big-bang redesign, not refactoring.

* **Don’t mix computation with formatting and I/O**

  * Avoid functions that both compute totals and generate UI strings/HTML or touch the UI.

* **Don’t tolerate duplicate logic across the code path**

  * If the same way of computing an amount or credits appears in more than one place, factor it out.

* **Don’t keep giant monolithic functions “for performance”**

  * That’s premature optimization; optimize *after* measuring bottlenecks.

* **Don’t introduce polymorphism before there’s a real need**

  * If there’s only one type or branch, keep it simple; introduce polymorphism when duplication and variation appear.

* **Don’t refactor without any safety checks**

  * Even tiny refactors should be backed by tests or at least a deterministic harness.

### Must-do

* **Use the precise definition of refactoring**

  * Refactoring = *a small change to internal structure that preserves observable behavior*.

* **Wear the right “hat” at the right time**

  * *Feature hat*: add new behavior; tests change because behavior changes.
  * *Refactoring hat*: change structure only; behavior (and tests) stay the same unless interfaces change.

* **Refactor opportunistically in the flow of work**

  * While adding a feature or fixing a bug, clean the area you’re touching.
  * “Make the change easy, then make the easy change.”

* **Use refactoring to maintain design stamina**

  * Continuous refactoring keeps architecture from rotting and maintains high change velocity over time.

* **Base refactoring on self-testing code**

  * Have a fast, automated test suite you trust to catch mistakes.
  * Treat “self-testing code” as a core property of the system.

* **Integrate frequently (CI) to support refactoring**

  * Refactorings often span multiple files and are prone to semantic merge conflicts.
  * Short-lived branches + CI make structural work safe and sustainable.

* **Practice YAGNI (You Aren’t Gonna Need It)**

  * Design for current, concrete needs.
  * Use refactoring to evolve design later as requirements shift, instead of baking in speculative flexibility.

* **Optimize performance *after* clarifying design**

  * Start from clean, readable, well-factored code.
  * Then profile, find hotspots, and optimize in small, measured steps.

* **Use parallel change (expand/contract) for public surfaces**

  * For public APIs, DB schemas, cross-team contracts:

    * Add new API/column → migrate callers/data → remove old one.
  * This avoids massive all-at-once breaks.

* **Prefer team/shared ownership within a domain**

  * Any team member should be able to refactor any module in their domain.
  * Ownership is enforced via review, not file locks.

* **Treat legacy refactoring as incremental and test-driven**

  * For legacy systems, create seams, add tests around those seams, and improve code gradually as you touch it.

### Must-not-do

* **Don’t break behavior and call it refactoring**

  * If behavior changes, it’s a feature/bugfix change. Mixing it with refactoring makes reasoning and reviewing much harder.

* **Don’t refactor code you never touch**

  * Ugly but stable APIs you rarely change can stay ugly; focus refactoring efforts where change pressure is real.

* **Don’t plan giant refactoring “projects” by default**

  * Big-bang refactors are risky and often unnecessary.
  * Most refactoring should be opportunistic micro-steps embedded in normal work.

* **Don’t add flexibility “just in case”**

  * Extra indirection and extension points add complexity and cognitive load.
  * Add them only when you have a concrete new use case or when refactoring would become significantly harder later.

* **Don’t rely on long-lived feature branches in a refactoring-heavy codebase**

  * Long-lived branches make merges painful and discourage structural improvements.

* **Don’t assume refactoring never affects performance**

  * Refactoring can modestly slow code initially; its payoff is easier, more targeted optimization later.

* **Don’t enforce rigid per-file code ownership**

  * Strict ownership blocks cross-cutting refactors and leads to bizarre workarounds.

* **Don’t refactor large untested systems recklessly**

  * Without tests, stick to safe, tool-supported refactorings and micro-steps—or write tests first.

* **Don’t sell refactoring as “extra work”**

  * It’s not optional; it’s part of how you deliver features quickly *and* safely.
  * Avoid “refactor week” theater; just refactor continuously as you go.

### Must-do

* **Treat smells as heuristics, not laws**

  * They’re prompts for investigation and refactoring ideas, not strict rules.

* **Relentlessly improve names**

  * Use Rename refactorings heavily: functions, variables, fields, classes.
  * Good names are the cheapest, highest-ROI refactorings.

* **Eliminate duplicated code**

  * Use Extract Function, Pull Up Method, and other moves to centralize logic.
  * Even “almost the same” code can often be aligned and unified.

* **Prefer many small, focused functions**

  * Short functions with clear responsibilities improve readability, reuse, and testability.
  * Long, multi-concern functions are a smell and a refactoring target.

* **Shrink long parameter lists**

  * Use:

    * **Preserve Whole Object** when multiple params come from the same object.
    * **Introduce Parameter Object** or **Combine Functions into Class** when several functions share the same parameter group.

* **Encapsulate and limit global data**

  * Wrap global state behind accessors or modules; constrain scope.
  * Favor immutability when possible.

* **Tame mutable data**

  * Centralize writes via setters/accessors.
  * Separate queries (no side effects) from commands (mutations).
  * Use immutability where feasible.

* **Address divergent change & shotgun surgery**

  * *Divergent change*: one module changed for many reasons → split responsibilities (Extract Class, Split Phase, Move Function).
  * *Shotgun surgery*: small change requires edits to many modules → consolidate related behavior.

* **Upgrade data clumps & primitive obsession to real domain types**

  * Replace repeated groups of fields with proper classes/structs.
  * Replace “bare” primitives (`string`, `int`) with domain objects (`Money`, `Coordinate`, `Extent`).

* **Replace repeated switches with polymorphism**

  * If the same conditional logic appears in multiple places, move that logic into polymorphic types.

* **Favor pipelines/iterators over manual loops when they clarify intent**

  * Use map/filter/reduce constructs where available; they often tell the “story” better than index-based loops.

* **Remove lazy elements and speculative generality**

  * Inline tiny functions that add no value.
  * Remove unused parameters, methods, and extension points that never became real.

* **Reserve comments for “why,” not “what”**

  * Use Extract Function / better names instead of comments explaining what the code does.
  * Use comments to explain rationale, constraints, and uncertainties.

### Must-not-do

* **Don’t enforce rigid, mechanical rules (e.g., “all functions < 20 lines”)**

  * Smells are contextual; use them as guides, not blunt metrics.

* **Don’t live with “mysterious names” because renaming feels like work**

  * Modern tools make renaming safe and cheap; avoiding it is far more expensive over time.

* **Don’t accept copy-paste as normal practice**

  * Every duplication multiplies bug-fix effort and cognitive load.

* **Don’t let functions/classes accrete responsibilities indefinitely**

  * Large classes, god objects, and multi-concern methods are prime candidates for Extract Class/Function.

* **Don’t rely on global mutable data as a shortcut**

  * The downstream debugging pain massively outweighs the immediate convenience.

* **Don’t keep “data classes” anemic when they clearly own behavior**

  * Behavior that depends on a data structure should usually live with that structure, not in unrelated helpers.

* **Don’t hide poor design behind heavy commenting**

  * Comments can’t fix structural problems; refactor first, then keep only the comments that still add value.

### Must-do

* **Make tests fully automatic and self-checking**

  * No manual log inspection; use assertions for pass/fail.

* **Run tests frequently**

  * Run affected tests every few minutes and the full suite at least daily.
  * Running tests should be as routine as compiling.

* **Aim for “self-testing code” as the norm**

  * Your codebase should have enough automated tests that most regressions show up quickly as failing tests.

* **Target tests at behavior and risk, not raw API density**

  * Focus on logic that’s likely to break or that would hurt if it broke.
  * Don’t waste time testing trivial getters/setters or boilerplate.

* **Use fresh fixtures to keep tests isolated**

  * Build a new fixture per test; avoid shared mutable state between tests.
  * This prevents order-dependent flakiness.

* **Structure tests with clear phases**

  * Setup → Exercise → Verify (→ Teardown).
  * Make these phases visually obvious in your test code.

* **Probe boundary conditions aggressively**

  * Empty collections, zero, negative values, invalid input, extremes.
  * These edges often reveal design and robustness issues.

* **Reproduce bugs with failing tests first**

  * When a bug appears, write a test that exposes it *before* fixing.
  * This prevents regressions and documents the failure.

* **Follow the green-bar / red-bar discipline**

  * Don’t refactor when tests are red.
  * Get back to green before restructuring.

* **Use TDD when it fits your workflow**

  * Short cycles: write a failing test → implement minimal code → refactor.
  * This tends to produce lean, testable designs by construction.

* **Favor “good enough and running” over imaginary test perfection**

  * An imperfect but useful test suite you run constantly beats the perfect one that never gets written.

* **Keep tests fast and deterministic**

  * Avoid slow I/O in unit tests.
  * Use mocks/doubles around external systems so the core suite runs quickly and reliably.

### Must-not-do

* **Don’t rely on human eyeballs for validation**

  * “Print and inspect” doesn’t scale, and humans get sloppy as repetition increases.

* **Don’t share mutable fixtures between tests**

  * Shared mutable state is the #1 cause of flaky, order-dependent tests.

* **Don’t test everything indiscriminately**

  * Over-testing trivial code creates noise and makes refactoring painful when tests are over-coupled to implementation.

* **Don’t refactor while the test suite is red**

  * If tests are failing, fix behavior first; structure comes second.

* **Don’t treat coverage percentage as a direct quality metric**

  * Coverage helps find untested areas, but doesn’t measure how valuable the tests are.

* **Don’t cram many unrelated assertions into one test**

  * One main behavior per test is a good default; it isolates failures and simplifies diagnosis.

* **Don’t postpone tests until “after the feature is done”**

  * Tests written after the fact are harder, less accurate, and don’t guide design.

* **Don’t attempt large-scale refactoring without a safety net**

  * For serious structural changes, a trustworthy test suite is non-negotiable.

### Must-do

* **Build and use a shared vocabulary of refactorings**

  * Use names like *Extract Function*, *Split Phase*, *Move Field* in discussions and reviews.
  * “This wants Extract Function” is faster and clearer than “maybe move some code out.”

* **Think about refactorings as structured entries**

  * For each common transformation, know:

    * Its **motivation** (what problem it solves).
    * Its **mechanics** (safe steps).
    * A **mental example** that makes it recognizable.

* **Treat mechanics as step-by-step checklists**

  * Fowler’s mechanics are written as terse, safe, micro-steps.
  * When a refactor feels risky, revert to these small steps and test after each.

* **Use the catalog as a reference, not something to memorize**

  * Skim enough to know what’s in it; look up details when needed.

* **Prioritize high-leverage refactorings**

  * The catalog is explicitly curated to the most useful refactorings—lean on them first.

* **Recognize that many refactorings have inverses**

  * E.g., Encapsulate Variable vs. removing accessors.
  * Even if inverses aren’t fully documented, understand that “undo” moves exist and when they’re appropriate.

### Must-not-do

* **Don’t treat the catalog as exhaustive**

  * It’s a high-value subset, not the complete space.
  * Be willing to invent new refactorings for recurring patterns in your domain.

* **Don’t follow mechanics as rigid scripts**

  * They are “one good way,” not the *only* way.
  * As you gain experience, you can compress steps, but fall back to small steps when things go wrong.

* **Don’t assume the example code is the perfect end state**

  * Each example usually only applies a single refactoring; further improvement would use other entries.

* **Don’t let the vocabulary stay unused in your team culture**

  * If no one uses these names in reviews/design meetings, you lose a lot of their value.

### Must-do

* **Extract functions aggressively for clarity and reuse**

  * Use *Extract Function* whenever a code block:

    * Represents a distinct concept.
    * Is repeated.
    * Makes a high-level flow noisy.

* **Extract variables to clarify complex expressions**

  * Give names to subexpressions so logic is easier to read, debug, and test.
  * This also gives natural hooks for assertions.

* **Inline functions/variables that add no real value**

  * When a helper does nothing but forward to another function or is no clearer than the original code, inline it to reduce indirection.

* **Change function declarations in small, safe stages**

  * Evolve signatures via *Change Function Declaration*:

    * Add new parameters with defaults or overloads.
    * Migrate callers incrementally.
    * Remove old parameters/names once all callers are updated.

* **Encapsulate widely-used or mutable variables**

  * Wrap fields behind accessors so you can later:

    * Change representation.
    * Add validation/logging.
    * Enforce invariants.

* **Improve variable names to reflect meaning, not history**

  * Apply *Rename Variable* freely, especially in domain-heavy areas (limits, thresholds, camera parameters, etc.).

* **Introduce parameter objects for recurring parameter clumps**

  * When a group of parameters always travels together across multiple functions, group them into a small type (struct, dataclass, etc.).

* **Combine functions into a class or transform when they share data**

  * If multiple related functions operate on the same data shape, move the data and functions together into a class/struct, or into a transform object mapping input → output.

* **Split multi-phase logic using *Split Phase***

  * Isolate distinct phases:

    * e.g., “parse/collect data” vs “compute/render”.
  * Makes each phase clearer and easier to evolve independently.

### Must-not-do

* **Don’t keep monolithic functions “for performance”**

  * Extract now for clarity; inline later only if profiling proves the call overhead is meaningful.

* **Don’t change function signatures via a single massive edit**

  * Blind search/replace across the codebase is brittle.
  * Use incremental *Change Function Declaration* instead.

* **Don’t introduce parameter objects prematurely**

  * Only bundle parameters that naturally belong together and recur, or you’ll end up with meaningless “bags of stuff.”

* **Don’t over-encapsulate trivial local variables**

  * Local, short-lived variables inside a small function rarely need getters/setters.

* **Don’t leave grouping refactors half-done**

  * Once you start *Combine Functions into Class* or *Split Phase*, finish the re-grouping; a mixed old/new style is worse than either alone.


### Must-do

* **Encapsulate records (structs) behind methods / properties**

  * Use *Encapsulate Record* so changes to field structure/types don’t force global edits.
  * Put domain behavior close to the data it uses.

* **Encapsulate collections**

  * Hide raw `List`/`Vec`/`HashMap` behind methods (`add_x`, `remove_x`, `items()`):

    * Control mutation.
    * Enforce invariants (no duplicates, ordering, etc.).

* **Promote important primitives to domain objects**

  * *Replace Primitive with Object* for concepts like `Money`, `Resolution`, `Extent`, `WorldSpacePosition`, `Color` etc.
  * Use the new type to centralize rules and prevent mixing mismatched units.

* **Replace temporary variables with queries when values are derivable**

  * If a variable can always be computed from other data, prefer a query method over stored mutable state.
  * Includes *Replace Derived Variable with Query*.

* **Extract classes when a class is doing too much**

  * If a subset of fields/methods form a coherent concept (camera, material, viewport), extract them into their own type.

* **Inline classes that no longer justify their overhead**

  * After refactoring, if a class is just a tiny wrapper, inline it back into its consumers to simplify the design.

* **Use Hide Delegate / Remove Middle Man appropriately**

  * *Hide Delegate*: expose simple forwarding methods to avoid callers knowing about deep internal structure (no long message chains).
  * *Remove Middle Man*: if an object only forwards calls without adding value, delete it and let callers talk directly to what they actually need.

* **Substitute simpler algorithms when possible**

  * *Substitute Algorithm*: once an algorithm is isolated, replace custom complex logic with a clearer or library-provided alternative.

### Must-not-do

* **Don’t expose raw mutable data structures as part of your public API**

  * Public fields or unguarded mutable collections make invariants unenforceable and refactors risky.

* **Don’t duplicate derived data unless you have a rigorous invalidation strategy**

  * Storing both base and derived values invites inconsistency; prefer computed queries or immutability.

* **Don’t keep domain behavior scattered when it clearly belongs on a data type**

  * Bringing logic into the type that owns the data reduces Feature Envy and message chains.

* **Don’t keep meaningless delegation layers**

  * Layers that add no behavior or abstraction but just forward calls should be removed.

* **Don’t cling to complex homegrown algorithms when simpler ones exist**

  * If a refactor allows you to replace 50 lines with a single library call and a clear loop, do it.


### Must-do

* **Regularly ask: “Is this function living in the right place?”**

  * *Move Function* when a function uses other module’s data more than its own, or when its callers live elsewhere.

* **Move fields to the types that conceptually own them**

  * Use *Move Field* when properties belong more naturally to another object (e.g., `interest_rate` on `AccountType`, not `Account` instance).

* **Use Move/Slide Statements to clean up ordering and grouping**

  * Move statements into/out of functions or up/down within them to:

    * Group related work.
    * Fix ordering issues.
    * Remove duplication and confusion.

* **Replace repeated inline code with function calls**

  * If inline code duplicates an existing helper’s behavior, call the helper instead.

* **Split loops by responsibility**

  * *Split Loop* so each loop does one specific job (compute sum, compute max, build list) instead of tackling multiple tasks in one pass.

* **Replace loops with pipelines when it clarifies intent**

  * Transformers like `map/filter/reduce` (or Rust iterators) often tell the story better than manual index-based loops.

* **Aggressively remove dead code**

  * Delete unused functions, branches, constants, and fields once you’re sure they’re truly unused.
  * Rely on version control for history, not dead code in the codebase.

### Must-not-do

* **Don’t treat original class/module boundaries as sacred**

  * When understanding and requirements evolve, move functions/fields; don’t fossilize old structure.

* **Don’t keep loops that do several unrelated things**

  * Multi-purpose loops are harder to read, test, and modify; they hide dependencies and bugs.

* **Don’t avoid pipelines just because they look “too functional”**

  * If they express the transformation clearly, they’re a win.

* **Don’t hoard dead code “just in case”**

  * It adds noise and confusion. If you ever need it, you can recover it from version control.

### Must-do

* **Ensure each variable has a single conceptual responsibility**

  * Use *Split Variable* when one variable is reused for different meanings over its lifetime (e.g., first a count, then a flag).

* **Rename fields carefully and globally via encapsulation**

  * First *Encapsulate Record*, then *Rename Field* through its constructor, accessors, and storage, updating callers in small steps.

* **Replace derived variables with queries wherever you can**

  * If a field is always derivable from other fields, remove the stored value and compute it on demand, unless you have a *deliberate* caching strategy.

* **Choose value vs. reference semantics consciously**

  * *Change Reference to Value* when copies are cheap and independent (small immutable configs, vectors, colors).
  * *Change Value to Reference* when identity matters and shared mutation must be visible across the system (customers, materials, textures, shared configs).

* **Use repositories/managers for shared references**

  * When you decide on reference semantics, centralize entity management in a repository (e.g., maps keyed by ID) to prevent accidental duplicates and confusion.

* **Exploit immutability to make duplication safe**

  * Immutable data structures can be safely copied; derived variants can exist as values without synchronization issues.

### Must-not-do

* **Don’t reuse a variable for multiple conceptual roles**

  * It harms readability and makes refactoring much riskier and more confusing.

* **Don’t rename fields with ad-hoc search/replace on large codebases**

  * That’s fragile. Use encapsulation and structured refactorings to rename safely.

* **Don’t store derived numbers that can drift out of sync by default**

  * Cache derived values only as a conscious optimization, with a clear invalidation/refresh strategy.

* **Don’t model inherently identity-based concepts as pure values**

  * If multiple objects must refer to the *same* underlying thing, treat it as a shared reference and manage it accordingly.

* **Don’t overuse references where values would do**

  * References everywhere increase coupling. Keep simple, small, immutable data as plain values.

### Must-do

* **Decompose complex conditionals into named helpers**

  * Extract both:

    * The condition (`is_special_rate()`).
    * Each branch (`calculate_special_rate()`, `calculate_regular_rate()`).

### Must-do

* **Decompose complex conditionals into named helpers**

  * Extract both:

    * The condition (`is_special_rate()`).
    * Each branch (`calculate_special_rate()`, `calculate_regular_rate()`).

* **Consolidate multiple conditions that have the same outcome**

  * If several `if` checks lead to the same result, consolidate with logical operators and a single predicate function.

* **Use guard clauses for exceptional/early-exit cases**

  * For rare or invalid states, use early returns instead of deep nesting.
  * This makes the “happy path” visually dominant and flatter.

* **Reserve symmetric `if/else` for genuinely equal alternatives**

  * Use `if/else` when both branches represent normal, equally important outcomes.
  * Use guard clauses where one branch is clearly a bail-out.

* **Replace repeated “type-code” conditionals with polymorphism where appropriate**

  * When a type code drives behavior in many places, move that behavior into subtype objects or strategies and use polymorphic dispatch.

* **Introduce special-case objects instead of scattered sentinel checks**

  * Replace repeated checks like `if (customer == unknown)` with an “unknown” customer object that encapsulates default behavior.

* **Use assertions to enforce invariants, not to validate external input**

  * Introduce assertions for conditions that must be true if the code is correct.
  * External data validation should use normal error handling.

* **Lean on Extract Function around conditional logic**

  * Extract complex conditions and branch contents early to make future polymorphism or special-case refactors easy.

### Must-not-do

* **Don’t tolerate long, deeply nested `if/else` jungles**

  * If you see multiple levels of nesting in core logic, treat it as a refactoring signal.

* **Don’t duplicate the same result or condition in multiple separate `if`s**

  * Consolidate them into a single predicate and a single return/branch.

* **Don’t assume every conditional should become polymorphism**

  * Only move to polymorphism when it actually clarifies behavior and reduces duplicated decision logic.

* **Don’t scatter special-case handling across the codebase**

  * Centralize special-case behavior in one object or one enrichment step.

* **Don’t use assertions for normal runtime errors or input validation**

  * Assertions are for programmer-level invariants, not for user/IO errors.

* **Don’t keep conditional expressions dense and opaque “for performance”**

  * Clarity first; you can inline or micro-optimize later if measurement shows a hotspot.

### Must-do

* **Apply command–query separation**

  * Functions that *return* information should generally not have observable side effects.
  * If a function both returns and mutates, split it into a pure query and a separate modifier.

### Must-do

* **Apply command–query separation**

  * Functions that *return* information should generally not have observable side effects.
  * If a function both returns and mutates, split it into a pure query and a separate modifier.

* **Make queries obviously safe and side-effect free**

  * Once separated, query methods should be callable from anywhere without fear of hidden mutations.

* **Unify near-duplicate functions by parameterizing**

  * When logic is identical except for constants/configuration, turn those differences into parameters and collapse the duplication.

* **Eliminate flag arguments that switch behavior**

  * If callers pass literal booleans/enums to change behavior, replace them with distinct functions or APIs representing each behavior.

* **Preserve whole objects instead of shredding them into many parameters**

  * When several parameters always come from the same object, pass the object.
  * This reduces signature clutter and makes extension safer.

* **Balance “Replace Parameter with Query” vs “Replace Query with Parameter”**

  * If the callee can easily look up what it needs, drop the parameter and query it.
  * If the caller should control the input or you want to decouple from a global dependency, pass it as a parameter instead.

* **Make immutability a design goal for data and APIs**

  * Use *Remove Setting Method* to eliminate unused or creation-only setters.
  * Move essential values into constructors and keep them immutable after construction.

* **Wrap constructors in named factory functions when you need flexibility**

  * *Replace Constructor with Factory Function* to:

    * Return different subtypes/proxies based on conditions.
    * Give clearer names than generic constructors.
    * Fit into contexts that expect regular functions, not language-level `new`.

* **Use command objects for genuinely complex, stateful operations**

  * When a function is large, needs plenty of parameters, or must support advanced scenarios (undo, custom configuration, multiple steps), refactor it into a command object.
  * Then use Extract Function on its `execute` or equivalent to tame internal complexity.

* **Collapse command objects back to functions when the complexity disappears**

  * If the command’s logic becomes small and straightforward, inline it back into a plain function.

### Must-not-do

* **Don’t let “getter-looking” methods secretly mutate state**

  * This breaks command–query separation and makes API behavior unpredictable.

* **Don’t hide distinct behaviors behind flag arguments**

  * `do_x(true)` vs `do_x(false)` is unclear at call sites and hard to reason about.

* **Don’t expose “bags of setters” as your public design**

  * Objects that are built via many arbitrary setter calls are hard to understand.
  * Prefer well-defined construction followed by a smaller set of meaningful operations.

* **Don’t over-parameterize everything “for flexibility”**

  * Long lists of scalars instead of domain objects are a data-clump smell and lead to error-prone calls.

* **Don’t keep command objects once they no longer justify their weight**

  * Unnecessary indirection makes code harder to follow; switch back to a function when appropriate.

* **Don’t assume API designs are fixed forever**

  * The whole purpose of these refactorings is to *change* APIs safely as you learn more about your domain and use cases.

### Must-do

* **Use Pull Up Method/Field to remove duplication across subclasses**

  * When subclasses share identical or very similar methods, move those methods to the base class.
  * Do the same for common fields and constructor logic.

* **Use Push Down Method/Field to localize behavior**

  * If a base class member is only meaningful on some subclasses, push it down to those subclasses.

* **Replace type codes with real subclasses when behavior diverges by type**

  * If a field encodes type and you see behavior branching on it, introduce subclasses and move that behavior into them.

* **Remove subclasses that no longer provide value**

  * If a subclass adds no new state or behavior, or only functions as a tag, fold it back into its parent class.

* **Extract a superclass when classes share real conceptual commonality**

  * When separate classes share fields and behavior with the same intent, factor that into a new superclass.

* **Collapse hierarchies that have become redundant**

  * If a class and its parent have nearly identical implementations after refactors, merge them.

* **Replace misuse of inheritance with delegation**

  * If not all superclass methods make sense on a subclass, or if you have type/instance confusion (e.g., concrete instance as subtype of a “model”), delegate to the other class instead of inheriting.

* **Use Liskov Substitution Principle as your inheritance gatekeeper**

  * A subclass must truly be a kind-of its superclass: all superclass operations must make sense, and contracts must be honored.
  * If that isn’t true, prefer delegation/composition to inheritance.

### Must-not-do

* **Don’t use inheritance solely for code reuse when the subtype isn’t truly a kind-of the supertype**

  * You’ll inherit methods that don’t make sense and distort your domain model.

* **Don’t leave duplicated logic scattered across subclasses**

  * Align differing implementations, parameterize where needed, and pull shared logic up into the parent.

* **Don’t keep empty or nearly-empty subclasses around as “maybe future” hooks**

  * They add noise and confusion; collapse or re-model them as data/flags if needed.

* **Don’t cling to outdated hierarchies when the domain changes**

  * Treat hierarchies as malleable: extract new superclasses, collapse old ones, or switch to delegation when the domain model evolves.

* **Don’t overreact by banning inheritance altogether**

  * When semantics fit (true subtype relationship), inheritance can be clean and effective. You can always refactor to delegation later if it becomes problematic.

* **Don’t let base classes become dumping grounds**

  * If only some subclasses use a field or method, push it down.
  * Keep base classes focused on what is *truly* common to all descendants.

---

## 1. High-level Architecture

### 1.1 Core idea

forge3d is a **Rust + wgpu/WebGPU renderer** with **PyO3 bindings** and a rich **Python API**. It targets:

- **Headless deterministic rendering** (PNG ↔ NumPy)
- **Terrain rendering** (DEM/heightmaps, PBR+POM, colormaps)
- **Path tracing** (GPU-oriented design with CPU fallbacks)
- **Screen-space GI & postfx** (AO, SSGI, SSR, bloom, tonemap)
- **Vector graphics & overlays** (OIT points/lines/polygons, text)

Execution paths:

- **Rust core** (`src/`) implements GPU pipelines, memory systems, and low-level rendering.
- **PyO3 extension** (`forge3d._forge3d`) exposes selected Rust types to Python.
- **Python package** (`python/forge3d/`) layers high-level APIs, validation, fallbacks, and integrations.
- **Tests** (`tests/`) are exhaustive and define many invariants (quality, performance envelopes, API contracts).

### 1.2 Major domains

- **Terrain & raster** – `src/terrain*`, `src/terrain_renderer.rs`, `python/forge3d/terrain_params.py`, examples.
- **Path tracing** – `src/path_tracing/`, `python/forge3d/path_tracing.py`, `python/forge3d/render.py` (raytrace mesh API).
- **Lighting / PBR / shadows / IBL** – `src/lighting`, `src/core/*` (clouds, shadows, ibl, dof, etc.), `python/forge3d/pbr.py`, `python/forge3d/shadows.py`, `python/forge3d/lighting.py`.
- **Screen-space effects (P5)** – `src/core/screen_space_effects.rs`, `src/p5/*`, `src/passes/*`, `src/shaders/ssao.wgsl` and GI/SSR-related shaders, Python helpers in `python/forge3d/screen_space_gi.py`.
- **Vector & overlays** – `src/vector/*`, `src/core/overlays.rs`, `src/core/text_overlay.rs`, `python/forge3d/vector.py`.
- **Memory & streaming** – `src/core/memory_tracker.rs`, `src/core/virtual_texture.rs`, `python/forge3d/mem.py`, `python/forge3d/memory.py`, `python/forge3d/streaming.py`, docs `docs/memory_budget.rst`.

---

## 2. Directory Map (Agent-centric)

### 2.1 Root

- `Cargo.toml` – Rust crate definition, **features** crucial for behavior (`enable-pbr`, `enable-ibl`, `enable-renderer-config`, `enable-staging-rings`, `weighted-oit`, etc.).
- `pyproject.toml` – Python packaging via `maturin`, ABI3 config, optional-deps groups.
- `README.md` – Short intro and quickstart; good sanity check for triangle/terrain.
- `prompt.md` – A high-level **task prompt** used for some workstreams (e.g., P5.4 GI composition). Treat it as a spec, not as code.
- `AGENTS.md` – This file.

### 2.2 Rust core – `src/`

- `src/lib.rs`
  - Crate root, PyO3 bindings, and module exports.
  - Re-exports many **core types** for Python (e.g. `RendererConfig`, `RendererGiMode`, `IBLRenderer`, `CloudRenderer`, etc.).
  - Defines PyO3 classes like `Frame`, `PyScreenSpaceGI` (Python GI manager), and utility functions for vector OIT.
- `src/core/mod.rs`
  - High-level **engine subsystems**: framegraph, GPU timing, postfx, bloom, memory/virtual textures, tonemap, matrix stack, scene graph, async compute, envmap/ibl, PBR material, shadows, reflections, clouds, ground plane, water surface, soft light radius, text, render bundles, screen-space effects (`gbuffer`, `screen_space_effects`).
- `src/terrain_renderer.rs`
  - `#[pyclass] TerrainRenderer` – PBR+POM terrain pipeline.
  - Bind group layouts for heightmap, materials, colormap, overlay, IBL; MSAA selection; light buffer integration.
- Other key modules:
  - `src/terrain/*` – Heightmap terrain pipeline implementation.
  - `src/lighting/*` – Lighting types and BRDF integration used by terrain+PBR.
  - `src/path_tracing/mod.rs` and submodules – GPU path tracing infrastructure (Megakernel/Wavefront).
  - `src/scene/mod.rs` – `Scene` PyO3 class (terrain scene, SSAO resources, toggles for reflections/DOF/clouds/etc.).
  - `src/render/mod.rs` + `src/render/params.rs` – Renderer configuration types used on both Rust + Python side.
  - `src/viewer/mod.rs` – Interactive viewer loop and integration with `ScreenSpaceEffectsManager`.
  - `src/passes/*`, `src/p5/*`, `src/shaders/*` – GI passes, SSR/SSGI/AO/tonemap, etc.

### 2.3 Python package – `python/forge3d/`

- `__init__.py`
  - **Public Python entrypoint**, layered:
    - Top section: imports `_native`, `_gpu`, memory facade `mem`, colormaps, terrain params, presets.
    - Re-exports native types when the extension module exists (`Scene`, `TerrainRenderer`, `IBL`, lighting/atmosphere, etc.).
    - Public helpers: memory metrics (`memory_metrics`, `budget_remaining`, `utilization_ratio`, `override_memory_limit`), GPU helpers (`enumerate_adapters`, `device_probe`, `has_gpu`, `get_device`), vector OIT demo wrappers, `composite_rgba_over`, matrix-stack QA helper (`c9_push_pop_roundtrip`).
  - Second section: **high-level rendering facade**:
    - Imports `RendererConfig` and config helpers.
    - Imports `PathTracer`, `make_camera`, rendering helpers from `.render`.
    - Imports `PbrMaterial`, `textures`, `geometry`, `io`, SDF wrappers, offscreen helpers, IPython display, frame dumper.
    - Defines **fallback `Renderer` class** used in many tests: triangle rendering, config/preset plumbing, lighting/shading config caching.
    - Exposes `render_triangle_rgba`, `render_triangle_png`, `numpy_to_png`, `png_to_numpy`, DEM stats/normalize, `open_viewer`, sampler utilities.
- `config.py`
  - **RendererConfig** tree for lighting/shading/shadows/GI/atmosphere and normalization of BRDF/techniques/GI modes.
  - Mirrors and validates Rust `src/render/params.rs`. Changes here must stay in sync with Rust side.
- `render.py`
  - High-level **rayshader-like APIs**:
    - `render_raytrace_mesh` – ingest mesh (OBJ or numpy), build BVH, attempt GPU path tracing via native `_pt_render_gpu_mesh`, fallback to CPU `PathTracer`.
    - DEM / vector ingestion helpers (rasterio, geopandas, shapely), palette resolution, camera autoframing, AOV export.
- `path_tracing.py`
  - Deterministic **CPU fallback path tracer** and AOV generator.
  - Used heavily in tests for conformance, firefly clamp behavior, AOV shapes/types.
- `mem.py` / `_memory.py`
  - Python facade over native memory tracker, exposing `MEMORY_LIMIT_BYTES`, `memory_metrics`, and budget helpers.
- `_gpu.py` / `_native.py`
  - GPU adapter detection, device probe, and fallback `MockDevice` when native not available.
- Other important submodules: `pbr.py`, `shadows.py`, `lighting.py`, `screen_space_gi.py`, `vector.py`, `postfx.py`, `streaming.py`, `terrain_params.py`, `memory.py`, `tiles/`, etc.

### 2.4 Tests – `tests/`

- Thousands of unit/integration tests across Rust and Python.
- Key patterns:
  - `test_api.py`, `smoke_test.py` – minimal API contracts for `Renderer` and triangle rendering.
  - Workstream-specific suites: `test_b*` (lighting/postfx), `test_p*` (PBR, GI, P5), `test_t*` (terrain), `test_f*` (geometry/mesh ops), `test_m*` (media/sky), `test_workstream_*`.
  - Rust tests under `tests/*.rs` and golden image harnesses (`golden_images.rs`, `scripts/generate_golden_images.py`).

### 2.5 Docs – `docs/`

- `index.rst` – documentation index, highlights **Core**, **Advanced Features**, **Integrations**, **Examples**, **Troubleshooting**.
- `quickstart.rst` – minimal Python usage path (triangle, terrain, vector graphics, GPU detection).
- `api_reference.rst` – Sphinx API docs; good map of *intended* top-level Python API.
- `memory_budget.rst` – authoritative source on **512 MiB host-visible memory budget** and patterns.
- `docs/api/*.md` / `docs/user/*.md` – deeper feature-specific docs.

### 2.6 Examples – `examples/`

- Python examples – galleries, terrain demos, SSGI demo, raytrace demos.
- Rust examples – interactive viewer, P5 SSR/SSGI tools, GI ablation harnesses.
- Useful for:
  - Sanity-checking rendering after changes.
  - Understanding how Py and Rust pieces are expected to compose.

---

## 3. Main Workflows & Data Flow

### 3.1 Basic Python triangle

1. User imports `forge3d`:
   - `python/forge3d/__init__.py` initializes shims and re-exports native objects if available.
2. `Renderer(width, height)` creates Python fallback renderer (`RendererConfig` + state cached in Python).
3. `render_triangle_rgba()` synthesizes an RGBA gradient triangle image entirely in Python; used for A1.4 acceptance tests.
4. `numpy_to_png` writes the PNG (Pillow or raw bytes fallback).

**Impact for agents:**
- Don’t break the fallback `Renderer` semantics: tests assert shape, dtype, non-empty output.

### 3.2 Terrain rendering

Typical path:

- Python side:
  - User uses higher-level API (`examples/terrain_demo.py` or `Scene` + `TerrainRenderer` in Python).
  - Terrain params built in Python (`terrain_params.py`), serialized to native configs.
- Rust side:
  - `TerrainRenderer` (PyO3 class) orchestrates heightmap upload, texture layout, PBR+POM shader, IBL. See `src/terrain_renderer.rs` + terrain shaders.

**Agent note:** when touching terrain:

- Keep Rust `TerrainRenderer` bind group layouts in sync with WGSL shader resource bindings.
- Ensure `terrain_render_params` and Python `terrain_params.py` stay aligned.

### 3.3 Path tracing and raytrace mesh

- Python `render_raytrace_mesh`:
  - Loads mesh (`forge3d.io.load_obj` or numpy/mesh dict inputs).
  - Validates via `forge3d.mesh` helpers and builds BVH.
  - Attempts GPU rendering via native `_pt_render_gpu_mesh`; falls back to CPU `PathTracer.render_rgba`.
  - Optionally writes AOVs, returns final RGBA + metadata.

**Agent note:**

- Maintain deterministic output where tests expect it (path_tracing CPU fallback is synthetic but stable under seed).
- GPU path is opportunistic; tests are usually written to tolerate CPU-only environments.

### 3.4 Screen-space effects (P5): AO / SSGI / SSR

- Manager: `core::screen_space_effects::ScreenSpaceEffectsManager` (Rust).
- Python binding: `PyScreenSpaceGI` in `src/lib.rs` with methods `enable_ssao`, `enable_ssgi`, `enable_ssr`, `disable`, `resize`, `execute`.
- AO path example: `Scene` holds `SsaoResources`, creates compute pipelines from `shaders/ssao.wgsl`, dispatches SSAO + composite into color buffer.

For more advanced GI work (e.g. P5.4 described in `prompt.md`):

- GI composition logic should live in GI-specific WGSL (e.g. `shaders/gi/composite.wgsl`).
- Orchestration is in `src/passes/gi.rs` and integrated into viewer/examples.
- Tests enforce energy and component-isolation constraints via P5-specific suites in `tests/`.

**Agent note:** when editing GI:

- Don’t smear AO/SSGI/SSR semantics across files; keep composition in designated shader and wiring in `passes/gi.rs`.
- Always re-read P5 prompt and the relevant tests (e.g. `tests/test_p5_screen_space_effects.py`, `tests/test_p53_ssr_status.py`) before structural changes.

### 3.5 Interactive viewer

- Python `open_viewer` delegates to native `open_viewer` in `_forge3d`.
- Rust `viewer` module sets up winit loop, GPU device, Scene/GBuffer, and integrates screen-space effects and overlays.

**Agent note:**
- Viewer code is sensitive to event loop, device lifetime, and pipeline ordering; avoid large refactors unless guided by a clear spec and backed by tests/examples.

---

## 4. Build, Test, and CI Expectations

### 4.1 Local builds

- **Rust only**:
  - `cargo check --workspace --all-features`
  - `cargo test --workspace --all-features -- --test-threads=1`
- **Python extension via maturin** (from repo root):
  - `pip install -U maturin`
  - `maturin develop --release`

### 4.2 Python tests

- Install built wheel or `maturin develop` first.
- Run Python tests:
  - `pytest tests/ -v --tb=short`

### 4.3 CI snapshot

From `.github/workflows/ci.yml`:

- Rust: `cargo check`, `cargo test --workspace --all-features -- --test-threads=1`, `cargo clippy` (warnings as errors, but allowed to fail in CI).
- Wheels: `maturin build` on Windows/Linux/macOS.
- Python tests on matrix of OS/Python versions.
- Golden images, shader param tests, example sanity runs.

**Agent note:**

- Keep Rust code generally **Clippy-clean** (warnings matter, even if CI allows some slack).
- Never introduce Python dependencies not reflected in `pyproject.toml` optional groups.

---

## 5. Design & API Conventions

### 5.1 Python API policy

See `python/forge3d/api_policy.md`:

- **Core module** (`import forge3d as f3d`):
  - Only exports stable, often-used symbols: `Renderer`, `Scene`, utility functions (`numpy_to_png`, `png_to_numpy`, DEM helpers, GPU helpers, vector helpers).
- **Specialized functionality** lives in submodules (`forge3d.pbr`, `forge3d.shadows`, `forge3d.path_tracing`, etc.).
- **Stability levels**:
  - Stable: exported in `__all__`, tests + docs, semver guarantees.
  - Experimental: submodules that may change but are documented.
  - Internal: not exported; no stability guarantees.

**Agent rule:**
- Don’t add new public top-level names casually. If you need a new user-facing feature, consider submodule placement and update API docs/tests accordingly.

### 5.2 Memory budget

From `docs/memory_budget.rst` and `python/forge3d/mem.py`:

- Default **host-visible budget**: **512 MiB**.
- Memory tracker distinguishes host-visible vs GPU-only allocations; only host-visible counts against budget.
- Exposed metrics: `memory_metrics()`, `budget_remaining()`, `utilization_ratio()`, `override_memory_limit()`.

**Agent rule:**

- Any new buffers/textures that are host-visible must be accounted for in memory tracking.
- Prefer reusing buffers / ring buffers where possible.

### 5.3 Feature detection and fallbacks

- Many GPU features are **optional** (e.g. weighted OIT, shadows, some PBR/IBL modes).
- Python and Rust sides have **feature-detection utilities**:
  - `forge3d.has_gpu()`, `forge3d.enumerate_adapters()`, `forge3d.device_probe()`.
  - Checks in shaders and Rust for optional extensions.

**Agent rule:**

- Never assume GPU or advanced features are always present; preserve or extend the existing detection/guard patterns.

---

## 6. Debugging Strategy for Agents

When debugging or modifying behavior, prefer this order:

1. **Find the contract**:
   - Look for tests: `grep` test name or feature in `tests/`.
   - Read relevant docs in `docs/`.
   - Inspect Python facade (often easier than diving straight into Rust).
2. **Locate the Rust core**:
   - Use `src/lib.rs` exports to locate types.
   - For PBR/lighting: `src/core/` and `src/lighting/`.
   - For GI/SSR/SSGI: `src/core/screen_space_effects.rs`, `src/passes/*`, `src/shaders/*`.
3. **Trace data flow**:
   - For terrain: `Scene` → `TerrainRenderer` → WGSL.
   - For path tracing: Python `render_raytrace_mesh` → native GPU path / CPU `PathTracer`.
   - For GI: GBuffer → AO/SSGI/SSR intermediates → final composite.
4. **Check memory & GPU environment**:
   - Use `forge3d.memory_metrics()` and `forge3d.has_gpu()` in tests or debugging snippets.
5. **Prefer small, localized changes**:
   - Modify a single module/shader at a time.
   - If new data is needed by shaders, propagate via minimal new fields and keep struct layout compatible.

Common failure modes:

- **Shape/dtype mismatches** between Python and Rust (e.g. NumPy arrays not C-contiguous, wrong dtype). Many tests exist explicitly to catch this.
- **Feature gate mismatches** – forgetting to enable a Cargo feature needed by a code path, or assuming it is always enabled.
- **Breaking energy/quality invariants** – especially around PBR, GI, and tonemapping.
- **Exceeding memory budget** – large textures or readback buffers without budget checks.

---

## 7. How to Safely Extend forge3d (Agent Checklist)

When you implement a new feature or change behavior:

1. **Identify scope**
   - Is this a Python-only helper? A Rust pipeline change? A shader tweak? A new example?
2. **Align with existing patterns**
   - Follow existing naming, module placement, and config patterns (e.g. `RendererConfig` for renderer options, `TerrainRenderParams` for terrain).
3. **Wire both sides if needed**
   - For new GPU features:
     - Rust struct + device/pipeline code.
     - WGSL shader changes.
     - PyO3 bindings in `src/lib.rs` or relevant module.
     - Python wrappers / validation.
4. **Update tests and docs (if public API)**
   - Add or extend tests under `tests/`.
   - If user-facing, update `docs/` and/or `api_policy.md`.
5. **Respect CI constraints**
   - Keep `cargo test --workspace --all-features` and `pytest tests/` passing.
   - Don’t add heavy new dependencies unless absolutely necessary.

---

## 8. Quick Pointers by Task Type

- **You’re asked to change GI / SSR / SSGI**
  - Start from `prompt.md` (if P5-related), `src/core/screen_space_effects.rs`, `src/passes/*`, and shaders under `src/shaders/` (especially `ssao.wgsl`, `gi/*`, `ssr/*`).
  - Mirror any new GI controls into Python config only if explicitly required.
- **You’re asked to modify terrain appearance**
  - Look first at `src/terrain_renderer.rs`, terrain shaders, and `python/forge3d/terrain_params.py`.
- **You’re asked to adjust Python API behavior**
  - Check `python/forge3d/__init__.py`, `api_policy.md`, and tests like `test_api.py`, `test_renderer_config.py`, `test_presets.py`.
- **You’re asked to optimize memory or streaming**
  - Inspect `src/core/memory_tracker.rs`, `src/core/virtual_texture.rs`, and Python `mem.py`, `memory.py`, `streaming.py`.

---

This document is intentionally high-level but code-grounded. When in doubt, prefer:

- Reading tests and docs over making assumptions.
- Adding small, focused changes over broad refactors.
- Preserving GPU feature and memory constraints.

If a future task prompt (like `prompt.md`) conflicts with this file, treat the prompt + tests as the authoritative spec and use this guide only as orientation.
