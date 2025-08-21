#!/usr/bin/env python3
"""
XML-mode orchestrator (no paid APIs):

- Computes diff (commit/staged/worktree)
- (Optional) gets Reviewer #1 from Gemini in the browser and captures its JSON
- Injects <diff/> and <reviewer1_json/> into task.xml (CDATA-safe)
- Feeds task.xml to Claude Code CLI (implementer)
- Sanitizes diff-only output, applies patch, pytest-gates, and commits
- Logs basic provenance via commit trailers

Windows-safe:
- Forces UTF-8 decoding for subprocess, and UTF-8 when writing temp files
- Resolves claude(.cmd) on PATH; falls back to `npx @anthropic-ai/claude-code`
"""

import os
import re
import sys
import json
import subprocess
import tempfile
import webbrowser
import platform
import shlex
import shutil

DEFAULT_BASE      = os.getenv("DEFAULT_BRANCH", "main")
TASK_XML_PATH     = os.getenv("AI_TASK_XML", "task.xml")
CLAUDE_MODEL      = os.getenv("CLAUDE_MODEL", "sonnet")
OPEN_GEMINI_URL   = os.getenv("GEMINI_URL", "https://gemini.google.com/app")
AI_GET_REVIEW1    = os.getenv("AI_GET_REVIEW1", "1") == "1"    # set 0 to skip Gemini hop
AI_RUN_TESTS      = os.getenv("AI_RUN_TESTS", "1") == "1"
AI_PYTEST_ARGS    = os.getenv("AI_PYTEST_ARGS", "-q -m not gpu")
AI_PYTEST_TIMEOUT = int(os.getenv("AI_PYTEST_TIMEOUT", "900"))
AI_DIFF_MODE      = os.getenv("AI_DIFF_MODE", "commit")        # commit | staged | worktree
AI_CLAUDE_TURNS   = int(os.getenv("AI_CLAUDE_TURNS", "30"))    # default: 30 turns (bump if needed)

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

def copy_clip(s: str):
    sysname = platform.system()
    if sysname == "Darwin":
        p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE); p.stdin.write(s.encode()); p.stdin.close(); p.wait()
    elif sysname == "Windows":
        p = subprocess.Popen(["clip"], stdin=subprocess.PIPE, shell=True); p.stdin.write(s.encode()); p.stdin.close(); p.wait()
    else:
        try:
            subprocess.run(["xclip","-selection","clipboard"], input=s.encode(), check=True)
        except Exception:
            subprocess.run(["xsel","--clipboard","--input"], input=s.encode(), check=True)

def read_clip() -> str:
    sysname = platform.system()
    if sysname == "Darwin":
        return run(["pbpaste"]).stdout
    elif sysname == "Windows":
        return run(["powershell","-Command","Get-Clipboard"]).stdout
    else:
        return run(["bash","-lc","xclip -selection clipboard -o || xsel --clipboard --output"], check=False).stdout

def parse_json_loose(s: str):
    s2 = re.sub(r"^```(json)?\s*|\s*```$", "", s.strip(), flags=re.MULTILINE)
    start, end = s2.find("{"), s2.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s2[start:end+1])
        except Exception:
            pass
    return json.loads(s2)

def sanitize_diff(out: str) -> str:
    if not out:
        return ""
    txt = re.sub(r"^```(diff|patch)?\s*|\s*```$", "", out.strip(), flags=re.MULTILINE)
    m = re.search(r"(?m)^(diff --git .+|---\s*/dev/null|---\s+a/)", txt)
    if m: return txt[m.start():].strip()
    m2 = re.search(r"(?m)^(Index: .+|\*\*\* .+)", txt)
    if m2: return txt[m2.start():].strip()
    return ""

def pytest_available() -> bool:
    try:
        r = run([sys.executable, "-c", "import pytest; print(pytest.__version__)"], check=True)
        return r.returncode == 0
    except Exception:
        return False

def run_pytests(args: str, timeout: int) -> int:
    argv = [sys.executable, "-m", "pytest"] + (shlex.split(args) if args else [])
    try:
        r = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
        print(r.stdout)
        if r.stderr.strip():
            print(r.stderr, file=sys.stderr)
        return r.returncode
    except subprocess.TimeoutExpired:
        print(f"[ai] pytest timed out after {timeout}s", file=sys.stderr)
        return 124

def sha12(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()[:12]

# ---- locate Claude Code CLI (Windows safe) ---------------------------------
def _resolve_claude_prefix():
    # 1) explicit override
    override = os.getenv("CLAUDE_BIN")
    if override:
        return [override]
    # 2) look for 'claude' or 'claude.cmd'
    for name in ("claude", "claude.cmd"):
        path = shutil.which(name)
        if path:
            return [path]
    # 3) fall back to npx (runs the package without global install)
    npx = shutil.which("npx")
    if npx:
        return [npx, "-y", "@anthropic-ai/claude-code@latest"]
    return None

CLAUDE_PREFIX = _resolve_claude_prefix()
if CLAUDE_PREFIX is None:
    print("[ai] Claude Code CLI not found. Install with `npm i -g @anthropic-ai/claude-code`, "
          "ensure %APPDATA%\\npm is on PATH, or set CLAUDE_BIN to the full path of claude(.cmd).", file=sys.stderr)
    sys.exit(1)

# ---------- main flow ----------
def main():
    # 1) build diff (commit vs base by default, with staged/worktree fallbacks)
    run(["git","fetch","origin",DEFAULT_BASE], check=False)

    def pick_diff():
        if AI_DIFF_MODE == "worktree":
            return "worktree", git("diff")
        if AI_DIFF_MODE == "staged":
            return "staged", git("diff","--staged")
        # default: commits vs base
        d = git("diff", f"origin/{DEFAULT_BASE}...HEAD")
        if d.strip():
            return "commit", d
        ds = git("diff","--staged")
        if ds.strip():
            return "staged", ds
        dw = git("diff")
        return ("worktree" if dw.strip() else "none", dw)

    mode, diff = pick_diff()
    if not diff.strip():
        print("[ai] No changes detected (commits, staged, or worktree). Nothing to do.")
        return 0
    print(f"[ai] Using {mode} diff.")

    # 2) Reviewer #1 via Gemini (optional)
    reviewer1_json = "{}"
    if AI_GET_REVIEW1:
        prompt = ("You are Reviewer #1 for forge3d. Return STRICT JSON with "
                  '{"general_feedback":"…","line_comments":[{"path":"…","comment":"…"}],'
                  '"risk_checklist":{"gil_release":true,"numpy_boundary":true,"abi3_packaging":true,'
                  '"padded_alignment_256":true,"single_gpu_context":true,"error_taxonomy":true,"docs_sync":true},'
                  '"missing_sources":[]} Diff:\n' + diff[:150000])
        copy_clip(prompt)
        print("[ai] Gemini prompt copied. Opening Gemini… Paste, run, copy JSON, then press Enter here.")
        webbrowser.open(OPEN_GEMINI_URL)
        input("[ai] After you COPY Gemini's JSON to clipboard, press Enter… ")
        try:
            reviewer1 = parse_json_loose(read_clip())
            reviewer1_json = json.dumps(reviewer1, ensure_ascii=False)
        except Exception as e:
            print("[ai] Could not parse Reviewer #1 JSON; continuing with empty {}. Error:", e)

    # 3) Ensure task.xml exists
    if not os.path.exists(TASK_XML_PATH):
        print(f"[ai] {TASK_XML_PATH} not found. Create it by asking ChatGPT to generate the XML.")
        return 1

    # 4) Load and inject live inputs into task.xml placeholders (CDATA-safe)
    with open(TASK_XML_PATH, "r", encoding="utf-8") as fh:
        xml = fh.read()

    # Encourage one-shot behavior: first reply must be the unified diff
    xml = re.sub(
        r"(?is)</instructions>",
        " FIRST REPLY MUST BE A SINGLE UNIFIED DIFF. "
        "Do not ask clarifying questions. No prose, no code fences.</instructions>",
        xml,
        count=1
    )

    def _as_cdata(text: str) -> str:
        # Split accidental CDATA terminators to keep XML valid
        return "<![CDATA[" + text.replace("]]>", "]]]]><![CDATA[>") + "]]>"

    def inject(tag: str, content: str, xml_text: str) -> str:
        """
        Replace (or insert) <tag>...</tag> with CDATA-wrapped payload.
        Use a function replacement so backslashes in content aren't parsed by re.sub.
        """
        cdata = _as_cdata(content)
        pattern = re.compile(rf"<{tag}>\s*(?:<!\[CDATA\[.*?\]\]>)?\s*</{tag}>", re.DOTALL | re.IGNORECASE)
        if pattern.search(xml_text):
            return pattern.sub(lambda m: f"<{tag}>{cdata}</{tag}>", xml_text, count=1)
        closing_inputs = re.compile(r"</inputs>", re.IGNORECASE)
        if closing_inputs.search(xml_text):
            return closing_inputs.sub(lambda m: f"  <{tag}>{cdata}</{tag}>\n</inputs>", xml_text, count=1)
        return xml_text.rstrip() + f"\n  <{tag}>{cdata}</{tag}>\n"

    # Fill placeholders if present and ensure tags are populated
    xml = xml.replace("{{DIFF}}", diff)
    xml = xml.replace("{{REVIEWER1_JSON}}", reviewer1_json)
    xml = inject("diff", diff, xml)
    xml = inject("reviewer1_json", reviewer1_json, xml)

    # 5) Call Claude Code CLI with the XML file directly (no inline fallback)
    os.makedirs(os.path.join(".ai", "tmp"), exist_ok=True)
    task_for_run = os.path.abspath(os.path.join(".ai", "tmp", "task_run.xml"))
    with open(task_for_run, "w", encoding="utf-8", newline="") as tf:
        tf.write(xml)
    print(f"[ai] Running Claude Code with task.xml at {task_for_run} …")

    cc = run(
        CLAUDE_PREFIX + [
            "-p", "--output-format", "text",
            "--max-turns", str(AI_CLAUDE_TURNS),
            "--model", CLAUDE_MODEL,
            f"file:{task_for_run}"
        ],
        check=False
    )

    patch = sanitize_diff(cc.stdout)
    if not patch:
        head = (cc.stdout or "")[:400] or (cc.stderr or "")[:400]
        print("[ai] Claude output lacked a recognizable unified diff. Head:\n", head)
        print("[ai] Tip: raise AI_CLAUDE_TURNS (e.g. 40) or shorten the diff (stage fewer files).")
        return 1

    # 6) Apply patch
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".patch", encoding="utf-8", newline="") as pf:
        pf.write(patch)
        patch_path = pf.name

    print("[ai] Applying patch…")
    apply = run(["git","apply","--reject","--whitespace=fix",patch_path], check=False)
    if apply.returncode != 0:
        print(apply.stderr)
        print("[ai] Patch had rejects (.rej). Resolve manually, then commit.")
        return 1

    # 7) pytest gate (optional)
    if AI_RUN_TESTS and pytest_available():
        print(f"[ai] Running pytest: {AI_PYTEST_ARGS!r}")
        rc = run_pytests(AI_PYTEST_ARGS, timeout=AI_PYTEST_TIMEOUT)
        if rc != 0:
            print(f"[ai] Tests FAILED (exit={rc}). Reverting patch…", file=sys.stderr)
            try: run(["git","apply","-R",patch_path], check=False)
            except Exception: pass
            try: run(["git","checkout","--","."], check=False)
            except Exception: pass
            return 1

    # 8) Commit with provenance trailers (task.xml hash)
    diff_sha  = sha12(diff)
    task_sha  = sha12(xml)
    patch_sha = sha12(patch)
    trailers = [
        f"AI-Mode: xml",
        f"AI-Hashes: diff={diff_sha} taskxml={task_sha} patch={patch_sha}",
        f"AI-Task: {TASK_XML_PATH}"
    ]
    run(["git","add","-A"], check=False)
    run(["git","commit","-m", "AI: implement via task.xml\n\n" + "\n".join(trailers)], check=False)
    print("[ai] Commit complete. Your Claude task.xml flow is fully captured and reproducible.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
