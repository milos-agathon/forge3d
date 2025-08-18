#!/usr/bin/env python3
"""
Local no-API multi-agent orchestrator for vulkan-forge

Flow
----
1) Build a unified diff vs default branch
2) Open Gemini (Reviewer #1) with a clipboard-primed prompt
3) Open ChatGPT (Reviewer #2) with a clipboard-primed prompt that consumes Gemini JSON
4) Ask Claude Code CLI (implementer) for a unified diff patch
5) Sanitize to ensure "diff-only"; optionally append docs/TODO-UNCERTAIN.md when reviewers list missing sources
6) Apply patch
7) (Gate) Run pytest; if tests pass -> commit; else revert changes and abort

No paid APIs are used. Gemini/ChatGPT are opened for you to paste prompts/replies.
Claude Code is invoked via the included CLI in Claude Pro/Max.

Prereqs
-------
- Claude Code CLI installed & logged in: `npm i -g @anthropic-ai/claude-code` (or Anthropic installer)
- macOS: pbcopy/pbpaste; Linux: xclip or xsel; Windows: clip + PowerShell Get-Clipboard
- Python 3.10+
- pytest installed in your venv if you enable the gate (AI_RUN_TESTS=1)

Env knobs
---------
DEFAULT_BRANCH=main            # base branch to diff against
CLAUDE_MODEL=sonnet            # Claude Code CLI model alias
GEMINI_URL=https://gemini.google.com/app
CHATGPT_URL=https://chat.openai.com/
AI_RUN_TESTS=1                 # 1=run pytest gate, 0=skip
AI_PYTEST_ARGS="-q -m not gpu" # default args for pytest gate
AI_PYTEST_TIMEOUT=900          # seconds
"""

import json
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
import webbrowser
from typing import List, Set

DEFAULT_BASE = os.getenv("DEFAULT_BRANCH", "main")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "sonnet")
OPEN_GEMINI_URL = os.getenv("GEMINI_URL", "https://gemini.google.com/app")
OPEN_CHATGPT_URL = os.getenv("CHATGPT_URL", "https://chat.openai.com/")
AI_RUN_TESTS = os.getenv("AI_RUN_TESTS", "1") == "1"
AI_PYTEST_ARGS = os.getenv("AI_PYTEST_ARGS", "-q -m not gpu")
AI_PYTEST_TIMEOUT = int(os.getenv("AI_PYTEST_TIMEOUT", "900"))

# ---------- small utils ----------

def run(cmd, check=True, **kw):
    return subprocess.run(cmd, text=True, capture_output=True, check=check, **kw)

def git(*args, check=True):
    return run(["git", *args], check=check).stdout

def copy_to_clipboard(txt: str):
    sysname = platform.system()
    if sysname == "Darwin":
        p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
        p.stdin.write(txt.encode()); p.stdin.close(); p.wait()
    elif sysname == "Windows":
        p = subprocess.Popen(["clip"], stdin=subprocess.PIPE, shell=True)
        p.stdin.write(txt.encode()); p.stdin.close(); p.wait()
    else:
        try:
            subprocess.run(["xclip","-selection","clipboard"], input=txt.encode(), check=True)
        except Exception:
            subprocess.run(["xsel","--clipboard","--input"], input=txt.encode(), check=True)

def read_clipboard() -> str:
    sysname = platform.system()
    if sysname == "Darwin":
        return run(["pbpaste"]).stdout
    elif sysname == "Windows":
        return run(["powershell","-Command","Get-Clipboard"]).stdout
    else:
        r = run(["bash","-lc","xclip -selection clipboard -o || xsel --clipboard --output"], check=False)
        return r.stdout

def parse_json_loose(s: str):
    """Parse JSON that may be wrapped in code fences / include stray text."""
    if not s:
        raise ValueError("Empty string")
    s2 = re.sub(r"^```(json)?\s*|\s*```$", "", s.strip(), flags=re.MULTILINE)
    start = s2.find("{"); end = s2.rfind("}")
    if start != -1 and end != -1 and end > start:
        window = s2[start:end+1]
        try:
            return json.loads(window)
        except Exception:
            pass
    return json.loads(s2)

def sanitize_to_unified_diff(output: str) -> str:
    """Strip prose/fences; return text starting at first diff header."""
    if not output:
        return ""
    txt = re.sub(r"^```(diff|patch)?\s*|\s*```$", "", output.strip(), flags=re.MULTILINE)
    m = re.search(r"(?m)^(diff --git .+|---\s*/dev/null|---\s+a/)", txt)
    if m: return txt[m.start():].strip()
    m2 = re.search(r"(?m)^(Index: .+|\*\*\* .+)", txt)
    if m2: return txt[m2.start():].strip()
    return ""

def build_uncertain_todo_diff(items: List[str]) -> str:
    if not items:
        return ""
    body = "# UNCERTAIN items from Reviewer #2\n\nThese require explicit sources before finalizing implementation:\n\n"
    for it in items[:200]:
        body += f"- {it}\n"
    diff = textwrap.dedent(f"""\
        diff --git a/docs/TODO-UNCERTAIN.md b/docs/TODO-UNCERTAIN.md
        new file mode 100644
        index 0000000..1111111
        --- /dev/null
        +++ b/docs/TODO-UNCERTAIN.md
        @@ -0,0 +1,@@
        +{body.replace('\n', '\\n')}
    """)
    return diff.replace("\\n", "\n")

def parse_new_files_from_patch(diff_text: str) -> Set[str]:
    """Return set of newly added file paths from a unified diff."""
    new_files: Set[str] = set()
    lines = diff_text.splitlines()
    add_mode = False
    for i, line in enumerate(lines):
        if line.startswith("diff --git "):
            add_mode = False
        if line.startswith("--- /dev/null"):
            add_mode = True
        if line.startswith("new file mode"):
            add_mode = True
        if line.startswith("+++ b/"):
            path = line[6:].strip()
            if add_mode:
                new_files.add(path)
            add_mode = False
    return new_files

def pytest_available() -> bool:
    try:
        r = run([sys.executable, "-c", "import pytest; print(pytest.__version__)"], check=True)
        return r.returncode == 0
    except Exception:
        return False

def run_pytests(args: str, timeout: int) -> int:
    """Run pytest with provided args; return exit code."""
    argv = [sys.executable, "-m", "pytest"]
    if args:
        argv += shlex.split(args)
    try:
        r = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
        print(r.stdout)
        if r.stderr.strip():
            print(r.stderr, file=sys.stderr)
        return r.returncode
    except subprocess.TimeoutExpired:
        print(f"[ai] pytest timed out after {timeout}s", file=sys.stderr)
        return 124

# ---------- Agent prompts (roles/objectives/boundaries baked in) ----------

GEMINI_PROMPT_TEMPLATE = """\
# ROLE
You are a senior engineer reviewing a Rust-first, cross-platform wgpu/WebGPU renderer exposed to Python for fast, headless 3D rendering (project: vulkan-forge). Built in Rust, shipped as Python wheels. You know Claude Code, Vulkan 1.2, WebGPU, RAII, Rust, Python ≥3.8, CMake ≥3.24, VMA, and Sphinx. Your job is a *code review only* (no code edits).

# TECH (project constraints)
• Vulkan 1.2 only — no mesh/ray-trace extensions
• Platforms: win_amd64 · linux_x86_64 · macos_universal2
• GPU budget: ≤ 512 MiB host-visible heap
• Build: CMake 3.24 → pybind11 → VMA; Rust; WebGPU

# STYLE
• Python: PEP 8 + Black  • C++: Google + clang-format
• Docs: Doxygen (C++), NumPy-style (Py)
• Every Vulkan call ⇒ CHECK_VK_RESULT(); return VkResult; throw std::runtime_error on fatal.

# VALIDATION
• Validation layers ON in Debug; any VUID = test fail
• Zero leaks (VMA stats & ASan)
• `python -c "import vulkan_forge"` must do nothing except import.

# UNCERTAIN RULE
If any claim can’t be traced to spec or repo content here, respond **UNCERTAIN** and request the specific source.

# TASK
You are Reviewer #1 (Gemini 2.5 Pro). Read the unified diff and produce an objective, actionable review **in STRICT JSON** with this schema:

{
  "general_feedback": "one concise paragraph",
  "line_comments": [
    { "path": "relative/file/path", "comment": "actionable, testable observation or fix hint" }
  ],
  "risk_checklist": {
    "gil_release": true,
    "numpy_boundary": true,
    "abi3_packaging": true,
    "padded_alignment_256": true,
    "single_gpu_context": true,
    "error_taxonomy": true,
    "docs_sync": true
  },
  "missing_sources": []
}

# REVIEW FOCUS (acceptance criteria)
python/ — Memory for Python surface & packaging
• Release the GIL around encode/submit/map/upload/download via py.allow_threads.
• Enforce dtype/shape/contiguity at the boundary; precise “expected vs got” errors; C-contiguous only.
• Version coherence across pyproject.toml, vulkan_forge/__init__.py, Cargo.toml, README, CHANGELOG.
• No hidden threads/globals; zero-/single-copy NumPy↔Rust paths. Small metrics helpers.

src/ — Memory for Rust core (wgpu + PyO3)
• One global GPU context (OnceCell). No per-object device creation in production paths.
• Correct 256B alignment for both readbacks and Queue::write_texture uploads (multi-row).
• Sampler rule: R32F heights use non-filtering sampler.
• Central error enum → consistent PyErr.
• PyO3 returns use Bound<'py, PyArray…>; use .to_pyarray_bound/.into_pyarray_bound.
• Validation layers in Debug; label resources; RAII.

# INPUT
<<PR_UNIFIED_DIFF>>

# OUTPUT
Return STRICT JSON only. No prose outside JSON.
"""

CHATGPT_PROMPT_TEMPLATE = """\
# ROLE
You are a senior engineer performing a second-pass review for a Rust-first, cross-platform wgpu/WebGPU renderer exposed to Python (vulkan-forge). Same expertise and constraints as Reviewer #1. Your job is to **tighten and complete** the first review—merge duplicates, add missing edge cases and tests, and turn it into an implementable plan. No code edits; JSON only.

# TECH / STYLE / VALIDATION / UNCERTAIN RULE
(Identical to Reviewer #1; see above constraints.)
If anything is unverifiable from the diff, mark **UNCERTAIN** and specify the exact file/line needed.

# TASK
You receive:
1) The PR unified diff.
2) Reviewer #1 JSON.

Produce an improved review **in STRICT JSON** with this schema:

{
  "general_feedback": "updated concise paragraph",
  "line_comments": [
    { "path": "relative/file/path", "comment": "deduped + sharpened guidance, concrete fix hints" }
  ],
  "tests_to_add": [
    "pytest test name + purpose (e.g., test_upload_r32f_enforces_256B_alignment)"
  ],
  "acceptance_criteria": [
    "objective checks required for merge (e.g., GIL released in render_rgba; NumPy arrays Bound<'py>; abi3 build; version bump in 4 files)"
  ],
  "risk_checklist": {
    "gil_release": true,
    "numpy_boundary": true,
    "abi3_packaging": true,
    "padded_alignment_256": true,
    "single_gpu_context": true,
    "error_taxonomy": true,
    "docs_sync": true
  },
  "missing_sources": []
}

# REVIEW FOCUS
python/ layer:
• GIL release around GPU work; precise boundary errors; contiguity via PyUntypedArrayMethods; accepted dtypes; PNG IO shapes; PyRuntimeError::new_err only (no typos).

src/ layer:
• 256B bytes_per_row for uploads *and* readbacks; shared helpers; single OnceCell GPU context.
• Non-filtering sampler for R32F; uniform layout doc; RenderError → PyErr; resource labels; validation layers ON; no panics across FFI; py.allow_threads at Python boundary.

# INPUTS
<<PR_UNIFIED_DIFF>>
<<REVIEWER1_JSON>>

# OUTPUT
Return STRICT JSON only. No prose outside JSON.
"""

CLAUDE_PROMPT_TEMPLATE = """\
# ROLE
You are a senior engineer implementing changes in a Rust-first, cross-platform wgpu/WebGPU renderer exposed to Python (vulkan-forge). Built in Rust, shipped as Python wheels. You know Vulkan 1.2, WebGPU, RAII, Rust, Python ≥3.8, CMake ≥3.24, VMA, Sphinx—and you are excellent at *surgical code edits*.

# OBJECTIVE
Implement the second-pass review precisely and safely. Produce a single **unified diff** that applies cleanly to the repo. No extra commentary.

# BOUNDARIES & CONSTRAINTS
• Vulkan 1.2 only; no mesh/ray-trace extensions.
• Platforms: win_amd64 · linux_x86_64 · macos_universal2.
• GPU budget: ≤ 512 MiB host-visible heap.
• Build: CMake 3.24 → pybind11 → VMA; Rust; WebGPU.
• Python style: PEP 8 + Black. C++: Google + clang-format. Docs: Doxygen (C++), NumPy-style (Py).
• Every Vulkan call guarded: CHECK_VK_RESULT(); return VkResult; throw std::runtime_error on fatal.
• Validation layers ON in Debug; any VUID implies a test must fail.
• No panics across FFI; consistent error taxonomy → PyErr.
• If you cannot verify a claim from the diff/repo content, insert a **TODO(UNCERTAIN: …)** comment adjacent to the affected code and proceed minimally.

# MUST-MEET ACCEPTANCE CRITERIA
python/ (PyO3 surface & packaging)
1) Release the GIL around GPU work at Python boundaries using `py.allow_threads(|| { … })`.
2) Boundary checks: float32/float64 accepted for DEM; enforce C-contiguous shapes; raise PyRuntimeError with “expected vs got”.
3) All NumPy returns use `Bound<'py, numpy::PyArrayN<T>>` and `.to_pyarray_bound(py)` (or `.into_pyarray_bound(py)` with trait imports).
4) PNG helpers: (H,W,4) uint8 for reads; (H,W)/(H,W,3)/(H,W,4) uint8 for writes; reject non-contiguous arrays clearly.
5) Version coherence across pyproject.toml, vulkan_forge/__init__.py, Cargo.toml, CHANGELOG.md (+ README if behavior changes).
6) Packaging: abi3 (py>=3.10); manylinux2014; Cargo `lto="thin"` and `strip=true`.
7) Tests/markers: gpu/camera/terrain; adapter-aware skips; `python -c "import vulkan_forge"` must only import.

src/ (wgpu + core Rust)
8) Single global GPU context (OnceCell) reused; no per-object device creation in production.
9) 256B alignment for **both** readbacks and multi-row uploads. Provide shared helpers: `padded_bpr(width,bpp)`, `upload_r32f_padded(...)`, `read_rgba_unpadded(...)`.
10) R32F heights use **non-filtering** sampler; document once where created and reuse.
11) Error taxonomy: `RenderError` (Device/Upload/Render/Readback/IO) + `to_py_err()` mapping.
12) Resource labels; validation layers in Debug; no panics across FFI.
13) Clip-space invariants: centralize GL↔WGPU remap; document uniform layout; WGSL normals robust with epsilon-guard divisions.

# INPUTS
A) PR unified diff:
<<PR_UNIFIED_DIFF>>

B) Second-pass review JSON:
<<REVIEWER2_JSON>>

# OUTPUT FORMAT (STRICT)
Output **only** a single *unified diff* patch that can be applied with:
`git apply --reject --whitespace=fix <patch>`

Rules:
• Begin with `diff --git` or `--- /dev/null` or `--- a/…`.
• Include new files with `--- /dev/null` → `+++ b/path`.
• No prose before/after the diff. No code fences.
• Prefer surgical edits. For unverifiable spots, add inline `// TODO(UNCERTAIN: …)` or `# TODO(UNCERTAIN: …)` adjacent to the change.
"""

# ---------- main flow ----------

def main():
    # fetch base & compute diff
    try:
        run(["git","fetch","origin",DEFAULT_BASE], check=False)
    except Exception:
        pass
    diff = git("diff","--binary",f"origin/{DEFAULT_BASE}...HEAD")
    if not diff.strip():
        print("[ai] No changes vs base; nothing to review.")
        return 0

    # Reviewer #1 (Gemini)
    gemini_prompt = GEMINI_PROMPT_TEMPLATE.replace("<<PR_UNIFIED_DIFF>>", diff[:170000])
    copy_to_clipboard(gemini_prompt)
    print("\n[ai] Reviewer #1 prompt copied to clipboard.")
    print("[ai] Opening Gemini… Paste (Cmd/Ctrl+V), press Enter, then copy the FULL JSON reply and return here.")
    webbrowser.open(OPEN_GEMINI_URL)
    input("[ai] After you COPY Gemini's JSON to clipboard, press Enter here… ")
    gemini_raw = read_clipboard()
    try:
        rev1 = parse_json_loose(gemini_raw)
    except Exception as e:
        print("[ai] Could not parse Reviewer #1 JSON:", e)
        print("Raw (first 600 chars):", gemini_raw[:600])
        return 1

    # Reviewer #2 (ChatGPT)
    chatgpt_prompt = CHATGPT_PROMPT_TEMPLATE.replace("<<PR_UNIFIED_DIFF>>", diff[:150000]).replace(
        "<<REVIEWER1_JSON>>", json.dumps(rev1)[:60000]
    )
    copy_to_clipboard(chatgpt_prompt)
    print("\n[ai] Reviewer #2 prompt copied to clipboard.")
    print("[ai] Opening ChatGPT… Paste, press Enter, then copy the FULL JSON reply and return here.")
    webbrowser.open(OPEN_CHATGPT_URL)
    input("[ai] After you COPY ChatGPT's improved JSON to clipboard, press Enter here… ")
    chatgpt_raw = read_clipboard()
    try:
        rev2 = parse_json_loose(chatgpt_raw)
    except Exception as e:
        print("[ai] Could not parse Reviewer #2 JSON:", e)
        print("Raw (first 600 chars):", chatgpt_raw[:600])
        return 1

    # Implementer (Claude Code)
    claude_prompt = CLAUDE_PROMPT_TEMPLATE.replace("<<PR_UNIFIED_DIFF>>", diff[:160000]).replace(
        "<<REVIEWER2_JSON>>", json.dumps(rev2)[:60000]
    )
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        f.write(claude_prompt)
        prompt_path = f.name

    print("\n[ai] Asking Claude Code for a unified diff patch…")
    cc = run(["claude","-p","--output-format","text","--max-turns","4","--model",CLAUDE_MODEL,f"file:{prompt_path}"], check=False)
    if cc.returncode != 0 or not cc.stdout.strip():
        print(cc.stderr); print("[ai] Claude Code did not produce output.")
        return 1

    patch = sanitize_to_unified_diff(cc.stdout)
    if not patch:
        print("[ai] Claude output lacked a recognizable unified diff.")
        print("First 400 chars of raw output:\n", cc.stdout[:400])
        return 1

    # Append a TODO file if reviewers listed missing sources and patch has no TODO markers
    missing = rev2.get("missing_sources") or []
    miss_clean = [str(x).strip() for x in missing if str(x).strip()] if isinstance(missing, list) else []
    if miss_clean and "TODO(UNCERTAIN" not in patch:
        todo_diff = build_uncertain_todo_diff(miss_clean)
        if todo_diff:
            patch = patch.rstrip() + "\n\n" + todo_diff

    # Track newly added files to remove if tests fail
    new_files = parse_new_files_from_patch(patch)

    # Apply patch
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".patch") as pf:
        pf.write(patch); patch_path = pf.name
    print("[ai] Applying patch…")
    apply = run(["git","apply","--reject","--whitespace=fix",patch_path], check=False)
    if apply.returncode != 0:
        print(apply.stderr)
        print("[ai] Patch had rejects (.rej). Resolve manually, then commit.")
        return 1

    # Optional: pytest gate
    if AI_RUN_TESTS:
        if not pytest_available():
            print("[ai] pytest not available; skipping gate (set AI_RUN_TESTS=0 to avoid this message).")
        else:
            print(f"[ai] Running pytest gate with args: {AI_PYTEST_ARGS!r}")
            rc = run_pytests(AI_PYTEST_ARGS, timeout=AI_PYTEST_TIMEOUT)
            if rc != 0:
                print(f"[ai] Tests FAILED (exit={rc}). Reverting patch…", file=sys.stderr)
                # Revert modified tracked files
                try:
                    run(["git","apply","-R",patch_path], check=False)
                except Exception:
                    pass
                # Restore tracked files
                try:
                    run(["git","checkout","--","."], check=False)
                except Exception:
                    pass
                # Remove any new files created by the patch
                for p in new_files:
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass
                print("[ai] Patch reverted. Fix review items or tests, then rerun.", file=sys.stderr)
                return 1

    # Commit if tests passed or gate disabled
    run(["git","add","-A"], check=False)
    run(["git","commit","-m","AI: apply reviewer-suggested fixes (+ UNCERTAIN TODOs if any)"], check=False)
    print("[ai] Fixes committed locally. Review `git diff origin/%s...HEAD` then push." % DEFAULT_BASE)
    return 0

if __name__ == "__main__":
    sys.exit(main())
