#!/usr/bin/env python3
"""
Local no-API multi-agent orchestrator for vulkan-forge
Adds full provenance (Run-ID, prompt/output hashes), optional repo ledger,
pytest gate, and UNCERTAIN handling.

Env knobs:
  DEFAULT_BRANCH=main
  CLAUDE_MODEL=sonnet
  GEMINI_URL=https://gemini.google.com/app
  CHATGPT_URL=https://chat.openai.com/
  AI_RUN_TESTS=1                  # run pytest gate
  AI_PYTEST_ARGS="-q -m not gpu"
  AI_PYTEST_TIMEOUT=900
  AI_PROVENANCE_LEDGER=1          # commit .ai/ledger.jsonl entry
  AI_PROVENANCE_MODE=hash         # hash | full (ledger payload)
  AI_COMMIT_RUN_DIR=0             # also git-add .ai/runs/<run-id> (privacy!)
"""

import json, os, platform, re, shlex, subprocess, sys, tempfile, textwrap, webbrowser, hashlib, time
from typing import List, Set

DEFAULT_BASE = os.getenv("DEFAULT_BRANCH", "main")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "sonnet")
OPEN_GEMINI_URL = os.getenv("GEMINI_URL", "https://gemini.google.com/app")
OPEN_CHATGPT_URL = os.getenv("CHATGPT_URL", "https://chat.openai.com/")
AI_RUN_TESTS = os.getenv("AI_RUN_TESTS", "1") == "1"
AI_PYTEST_ARGS = os.getenv("AI_PYTEST_ARGS", "-q -m not gpu")
AI_PYTEST_TIMEOUT = int(os.getenv("AI_PYTEST_TIMEOUT", "900"))
AI_PROVENANCE_LEDGER = os.getenv("AI_PROVENANCE_LEDGER", "1") == "1"
AI_PROVENANCE_MODE = os.getenv("AI_PROVENANCE_MODE", "hash")  # or "full"
AI_COMMIT_RUN_DIR = os.getenv("AI_COMMIT_RUN_DIR", "0") == "1"

# ---------- small utils ----------
def run(cmd, check=True, **kw):
    # Windows-safe: force UTF-8 and replace undecodable bytes
    kw.setdefault("text", True)
    kw.setdefault("capture_output", True)
    kw.setdefault("encoding", "utf-8")
    kw.setdefault("errors", "replace")
    return subprocess.run(cmd, check=check, **kw)

def git(*args, check=True):
    """Run a git command and return stdout text."""
    return run(["git", *args], check=check).stdout

def copy_to_clipboard(txt: str):
    sysname = platform.system()
    if sysname == "Darwin":
        p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE); p.stdin.write(txt.encode()); p.stdin.close(); p.wait()
    elif sysname == "Windows":
        p = subprocess.Popen(["clip"], stdin=subprocess.PIPE, shell=True); p.stdin.write(txt.encode()); p.stdin.close(); p.wait()
    else:
        try: subprocess.run(["xclip","-selection","clipboard"], input=txt.encode(), check=True)
        except Exception: subprocess.run(["xsel","--clipboard","--input"], input=txt.encode(), check=True)

def read_clipboard() -> str:
    sysname = platform.system()
    if sysname == "Darwin": return run(["pbpaste"]).stdout
    elif sysname == "Windows": return run(["powershell","-Command","Get-Clipboard"]).stdout
    else: return run(["bash","-lc","xclip -selection clipboard -o || xsel --clipboard --output"], check=False).stdout

def parse_json_loose(s: str):
    if not s: raise ValueError("Empty string")
    s2 = re.sub(r"^```(json)?\s*|\s*```$", "", s.strip(), flags=re.MULTILINE)
    start, end = s2.find("{"), s2.rfind("}")
    if start != -1 and end != -1 and end > start:
        window = s2[start:end+1]
        try: return json.loads(window)
        except Exception: pass
    return json.loads(s2)

def sanitize_to_unified_diff(output: str) -> str:
    if not output: return ""
    txt = re.sub(r"^```(diff|patch)?\s*|\s*```$", "", output.strip(), flags=re.MULTILINE)
    m = re.search(r"(?m)^(diff --git .+|---\s*/dev/null|---\s+a/)", txt)
    if m: return txt[m.start():].strip()
    m2 = re.search(r"(?m)^(Index: .+|\*\*\* .+)", txt)
    if m2: return txt[m2.start():].strip()
    return ""

def build_uncertain_todo_diff(items: List[str]) -> str:
    if not items: return ""
    body = "# UNCERTAIN items from Reviewer #2\n\nThese require explicit sources before finalizing implementation:\n\n"
    for it in items[:200]: body += f"- {it}\n"
    diff = f"""diff --git a/docs/TODO-UNCERTAIN.md b/docs/TODO-UNCERTAIN.md
new file mode 100644
index 0000000..1111111
--- /dev/null
+++ b/docs/TODO-UNCERTAIN.md
@@ -0,0 +1,@@
+{body.replace('\n','\\n')}
"""
    return diff.replace("\\n","\n")

def parse_new_files_from_patch(diff_text: str) -> Set[str]:
    new_files, lines, add_mode = set(), diff_text.splitlines(), False
    for line in lines:
        if line.startswith("diff --git "): add_mode = False
        if line.startswith("--- /dev/null"): add_mode = True
        if line.startswith("new file mode"): add_mode = True
        if line.startswith("+++ b/"):
            path = line[6:].strip()
            if add_mode: new_files.add(path)
            add_mode = False
    return new_files

def pytest_available() -> bool:
    try:
        r = run([sys.executable, "-c", "import pytest; print(pytest.__version__)"], check=True)
        return r.returncode == 0
    except Exception: return False

def run_pytests(args: str, timeout: int) -> int:
    argv = [sys.executable, "-m", "pytest"]
    if args: argv += shlex.split(args)
    try:
        r = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
        print(r.stdout); 
        if r.stderr.strip(): print(r.stderr, file=sys.stderr)
        return r.returncode
    except subprocess.TimeoutExpired:
        print(f"[ai] pytest timed out after {timeout}s", file=sys.stderr); return 124

def sha12(s: str) -> str: return hashlib.sha256(s.encode()).hexdigest()[:12]
def now_iso() -> str: return time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())

# ---------- Agent prompts ----------
GEMINI_PROMPT_TEMPLATE = """# ROLE
You are a senior engineer reviewing a Rust-first, cross-platform wgpu/WebGPU renderer exposed to Python for fast, headless 3D rendering (project: vulkan-forge). Built in Rust, shipped as Python wheels. You know Claude Code, Vulkan 1.2, WebGPU, RAII, Rust, Python ≥3.8, CMake ≥3.24, VMA, and Sphinx. Your job is a *code review only* (no code edits).
# TECH
• Vulkan 1.2 only — no mesh/ray-trace
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
# REVIEW FOCUS
python/: GIL release via py.allow_threads; dtype/shape/contiguity errors; version coherence; zero/single-copy NP↔Rust; small metrics.
src/: single OnceCell GPU context; 256B alignment (uploads & readbacks); non-filtering R32F sampler; central error enum→PyErr; Bound<'py,..> returns; labels; validation layers.
# INPUT
<<PR_UNIFIED_DIFF>>
# OUTPUT
STRICT JSON only.
"""

CHATGPT_PROMPT_TEMPLATE = """# ROLE
You are a senior engineer performing a second-pass review for vulkan-forge. Tighten, dedupe, and complete Reviewer #1; JSON only.
# TECH/STYLE/VALIDATION/UNCERTAIN
Same as Reviewer #1. Mark **UNCERTAIN** with exact file/line needed.
# TASK
Inputs: PR diff + Reviewer #1 JSON. Output improved review with this schema:
{
  "general_feedback": "updated concise paragraph",
  "line_comments": [
    { "path": "relative/file/path", "comment": "deduped + sharpened guidance, concrete fix hints" }
  ],
  "tests_to_add": [
    "pytest test name + purpose (e.g., test_upload_r32f_enforces_256B_alignment)"
  ],
  "acceptance_criteria": [
    "objective checks required for merge (e.g., GIL released in render_rgba; Bound<'py> arrays; abi3; version bump in 4 files)"
  ],
  "risk_checklist": {
    "gil_release": true, "numpy_boundary": true, "abi3_packaging": true,
    "padded_alignment_256": true, "single_gpu_context": true,
    "error_taxonomy": true, "docs_sync": true
  },
  "missing_sources": []
}
# REVIEW FOCUS
python/: GIL release; precise boundary errors; PyUntypedArrayMethods for contiguity; accepted dtypes; PNG IO shapes; PyRuntimeError::new_err.
src/: 256B bytes_per_row (uploads+readbacks); helpers; single OnceCell context; non-filtering R32F; uniform doc; RenderError→PyErr; labels; validation layers; no FFI panics.
# INPUTS
<<PR_UNIFIED_DIFF>>
<<REVIEWER1_JSON>>
# OUTPUT
STRICT JSON only.
"""

CLAUDE_PROMPT_TEMPLATE = """# ROLE
You implement the second-pass review for vulkan-forge. Produce a single unified diff; no prose.
# CONSTRAINTS
Vulkan 1.2; win_amd64/linux_x86_64/macos_universal2; ≤512MiB host-visible; CMake→pybind11→VMA; Rust; WebGPU; PEP8/Black; clang-format; Doxygen/NumPy-style; CHECK_VK_RESULT; validation layers ON; no FFI panics; consistent error taxonomy→PyErr.
If unverifiable, add inline `// TODO(UNCERTAIN: …)` minimal changes.
# ACCEPTANCE (must meet)
python/: py.allow_threads; strict boundary checks with clear “expected vs got”; Bound<'py,..> returns; PNG helpers; version coherence; abi3+manylinux2014; tests & markers; `import vulkan_forge` is side-effect-free.
src/: single OnceCell context; 256B alignment uploads+readbacks (helpers provided); non-filtering R32F; RenderError + to_py_err(); labels; GL↔WGPU remap; robust WGSL normals.
# INPUTS
A) PR diff:
<<PR_UNIFIED_DIFF>>
B) Second-pass review JSON:
<<REVIEWER2_JSON>>
# OUTPUT
Only a unified diff suitable for `git apply --reject --whitespace=fix`.
"""

# ---------- main flow ----------
def main():
    # fetch base & diff
    run(["git","fetch","origin",DEFAULT_BASE], check=False)
    diff = git("diff","--binary",f"origin/{DEFAULT_BASE}...HEAD")
    if not diff.strip():
        print("[ai] No changes vs base; nothing to review."); return 0

    # ===== Reviewer #1 (Gemini) =====
    gemini_prompt = GEMINI_PROMPT_TEMPLATE.replace("<<PR_UNIFIED_DIFF>>", diff[:170000])
    copy_to_clipboard(gemini_prompt)
    print("\n[ai] Reviewer #1 prompt copied. Opening Gemini… Paste/run, copy JSON, then press Enter here.")
    webbrowser.open(OPEN_GEMINI_URL)
    input("[ai] After you COPY Gemini's JSON to clipboard, press Enter… ")
    gemini_raw = read_clipboard()
    try: rev1 = parse_json_loose(gemini_raw)
    except Exception as e:
        print("[ai] Could not parse Reviewer #1 JSON:", e); print("Raw head:", gemini_raw[:600]); return 1

    # ===== Reviewer #2 (ChatGPT) =====
    chatgpt_prompt = CHATGPT_PROMPT_TEMPLATE.replace("<<PR_UNIFIED_DIFF>>", diff[:150000]).replace(
        "<<REVIEWER1_JSON>>", json.dumps(rev1)[:60000]
    )
    copy_to_clipboard(chatgpt_prompt)
    print("\n[ai] Reviewer #2 prompt copied. Opening ChatGPT… Paste/run, copy JSON, then press Enter here.")
    webbrowser.open(OPEN_CHATGPT_URL)
    input("[ai] After you COPY ChatGPT's JSON to clipboard, press Enter… ")
    chatgpt_raw = read_clipboard()
    try: rev2 = parse_json_loose(chatgpt_raw)
    except Exception as e:
        print("[ai] Could not parse Reviewer #2 JSON:", e); print("Raw head:", chatgpt_raw[:600]); return 1

    # ===== Implementer (Claude Code) =====
    claude_prompt = CLAUDE_PROMPT_TEMPLATE.replace("<<PR_UNIFIED_DIFF>>", diff[:160000]).replace(
        "<<REVIEWER2_JSON>>", json.dumps(rev2)[:60000]
    )
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
        f.write(claude_prompt); prompt_path = f.name

    print("\n[ai] Asking Claude Code for a unified diff patch…")
    cc = run(["claude","-p","--output-format","text","--max-turns","4","--model",CLAUDE_MODEL,f"file:{prompt_path}"], check=False)
    if cc.returncode != 0 or not cc.stdout.strip():
        print(cc.stderr); print("[ai] Claude Code did not produce output."); return 1

    # sanitize / augment patch
    patch = sanitize_to_unified_diff(cc.stdout)
    if not patch:
        print("[ai] Claude output lacked a recognizable unified diff."); print("Head:\n", cc.stdout[:400]); return 1
    missing = rev2.get("missing_sources") or []
    miss_clean = [str(x).strip() for x in missing if str(x).strip()] if isinstance(missing, list) else []
    if miss_clean and "TODO(UNCERTAIN" not in patch:
        todo_diff = build_uncertain_todo_diff(miss_clean)
        if todo_diff: patch = patch.rstrip() + "\n\n" + todo_diff

    new_files = parse_new_files_from_patch(patch)

    # ===== provenance (Run-ID, hashes, run dir, ledger) =====
    diff_sha, r1_sha, r2_sha, patch_sha = sha12(diff), sha12(json.dumps(rev1)), sha12(json.dumps(rev2)), sha12(patch)
    run_id = f"{now_iso()}_{diff_sha[:6]}-{r2_sha[:6]}"

    run_dir = os.path.join(".ai","runs",run_id)
    os.makedirs(run_dir, exist_ok=True)
    # write artifacts
    def w(path, content, is_json=False):
        with open(path, "w", encoding="utf-8") as fh:
            if is_json and isinstance(content, (dict,list)): json.dump(content, fh, indent=2)
            else: fh.write(content)

    w(os.path.join(run_dir,"01_gemini_prompt.txt"), gemini_prompt)
    w(os.path.join(run_dir,"01_gemini_response.json"), rev1, is_json=True)
    w(os.path.join(run_dir,"02_chatgpt_prompt.txt"), chatgpt_prompt)
    w(os.path.join(run_dir,"02_chatgpt_response.json"), rev2, is_json=True)
    w(os.path.join(run_dir,"03_claude_prompt.txt"), claude_prompt)
    w(os.path.join(run_dir,"03_claude_patch.diff"), patch)
    meta = {
        "run_id": run_id,
        "timestamp": run_id.split("_")[0].replace("_"," "),
        "models": {
            "reviewer1": "Gemini-2.5-Pro (app)",
            "reviewer2": "ChatGPT-5-Thinking (app)",
            "implementer": f"Claude-Code-{CLAUDE_MODEL} (CLI)"
        },
        "hashes": {"diff": diff_sha, "review1": r1_sha, "review2": r2_sha, "patch": patch_sha},
        "base": DEFAULT_BASE,
        "pytest": {"enabled": AI_RUN_TESTS, "args": AI_PYTEST_ARGS, "timeout": AI_PYTEST_TIMEOUT}
    }
    w(os.path.join(run_dir,"meta.json"), meta, is_json=True)

    # gitignore .ai/runs if not committing runs
    os.makedirs(".ai", exist_ok=True)
    gi_path = os.path.join(".ai",".gitignore")
    if not os.path.exists(gi_path):
        w(gi_path, "runs/\n", is_json=False)

    # optional ledger entry (committed)
    if AI_PROVENANCE_LEDGER:
        ledger_path = os.path.join(".ai","ledger.jsonl")
        payload = {
            "run_id": run_id,
            "commit": None,   # filled after commit
            "models": meta["models"],
            "hashes": meta["hashes"],
            "provenance_dir": run_dir,
            "mode": AI_PROVENANCE_MODE
        }
        if AI_PROVENANCE_MODE == "full":
            payload.update({
                "prompts": {
                    "gemini": gemini_prompt, "chatgpt": chatgpt_prompt, "claude": claude_prompt
                },
                "reviews": {"r1": rev1, "r2": rev2}
            })
        else:
            payload.update({
                "prompts_hash": {
                    "gemini": sha12(gemini_prompt),
                    "chatgpt": sha12(chatgpt_prompt),
                    "claude": sha12(claude_prompt)
                }
            })
        with open(ledger_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # ===== apply patch
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".patch") as pf:
        pf.write(patch); patch_path = pf.name
    print("[ai] Applying patch…")
    apply = run(["git","apply","--reject","--whitespace=fix",patch_path], check=False)
    if apply.returncode != 0:
        print(apply.stderr); print("[ai] Patch had rejects (.rej). Resolve manually, then commit."); return 1

    # ===== pytest gate
    if AI_RUN_TESTS:
        if not pytest_available():
            print("[ai] pytest not available; skipping gate.")
        else:
            print(f"[ai] Running pytest gate: {AI_PYTEST_ARGS!r}")
            rc = run_pytests(AI_PYTEST_ARGS, timeout=AI_PYTEST_TIMEOUT)
            if rc != 0:
                print(f"[ai] Tests FAILED (exit={rc}). Reverting patch…", file=sys.stderr)
                try: run(["git","apply","-R",patch_path], check=False)
                except Exception: pass
                try: run(["git","checkout","--","."], check=False)
                except Exception: pass
                for p in new_files:
                    try:
                        if os.path.exists(p): os.remove(p)
                    except Exception: pass
                print("[ai] Patch reverted. Fix issues and rerun.", file=sys.stderr)
                return 1

    # ===== commit (with trailers)
    run(["git","add","-A"], check=False)
    if not AI_COMMIT_RUN_DIR:
        # ensure run_dir is not added inadvertently
        run(["git","restore","--staged", run_dir], check=False)

    # fill commit trailers
    trailers = [
        f"AI-Run: {run_id}",
        f"AI-Agents: reviewer1=Gemini-2.5-Pro reviewer2=ChatGPT-5-Thinking implementer=Claude-Code-{CLAUDE_MODEL}",
        f"AI-Hashes: diff={diff_sha} r1={r1_sha} r2={r2_sha} patch={patch_sha}",
        f"AI-Provenance: {run_dir}"
    ]
    commit_msg = "AI: apply reviewer-suggested fixes (+ provenance)\n\n" + "\n".join(trailers) + "\n"
    run(["git","commit","-m", commit_msg], check=False)

    # patch ledger with commit sha
    if AI_PROVENANCE_LEDGER:
        head = git("rev-parse","HEAD").strip()
        # rewrite last line with commit populated
        ledger_path = os.path.join(".ai","ledger.jsonl")
        try:
            with open(ledger_path, "r+", encoding="utf-8") as fh:
                lines = fh.readlines()
                if lines:
                    last = json.loads(lines[-1])
                    if last.get("run_id")==run_id and last.get("commit") is None:
                        last["commit"] = head
                        lines[-1] = json.dumps(last, ensure_ascii=False) + "\n"
                        fh.seek(0); fh.truncate(0); fh.writelines(lines)
            run(["git","add", ledger_path], check=False)
            run(["git","commit","--amend","--no-edit"], check=False)
        except Exception:
            pass

    print("[ai] Commit complete.")
    print(f"[ai] Traceability:\n  - Run-ID: {run_id}\n  - Ledger: .ai/ledger.jsonl (committed={AI_PROVENANCE_LEDGER})\n  - Local artifacts: {run_dir} (committed={AI_COMMIT_RUN_DIR})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
