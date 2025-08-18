#!/usr/bin/env python3
"""
XML-mode orchestrator (no paid APIs):

- Computes diff vs base
- (Optional) gets Reviewer #1 from Gemini in the browser and captures its JSON
- Updates task.xml placeholders with the actual diff and review JSON
- Feeds task.xml to Claude Code CLI (implementer)
- Sanitizes diff-only output, applies patch, pytest-gates, and commits
- Logs provenance (hashes + run dir) so you can trace the prompt → patch

Prereqs:
- Claude Code CLI installed (`npm i -g @anthropic-ai/claude-code`)
- pytest installed if AI_RUN_TESTS=1
- Clipboard tools (pbcopy/pbpaste on macOS; xclip/xsel on Linux; clip/Get-Clipboard on Windows)
"""

import os, re, sys, json, time, hashlib, subprocess, tempfile, webbrowser, platform, shlex

DEFAULT_BASE     = os.getenv("DEFAULT_BRANCH", "main")
TASK_XML_PATH    = os.getenv("AI_TASK_XML", "task.xml")
CLAUDE_MODEL     = os.getenv("CLAUDE_MODEL", "sonnet")
OPEN_GEMINI_URL  = os.getenv("GEMINI_URL", "https://gemini.google.com/app")
AI_GET_REVIEW1   = os.getenv("AI_GET_REVIEW1", "1") == "1"   # set 0 to skip Gemini hop
AI_RUN_TESTS     = os.getenv("AI_RUN_TESTS", "1") == "1"
AI_PYTEST_ARGS   = os.getenv("AI_PYTEST_ARGS", "-q -m not gpu")
AI_PYTEST_TIMEOUT= int(os.getenv("AI_PYTEST_TIMEOUT", "900"))

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
        try: subprocess.run(["xclip","-selection","clipboard"], input=s.encode(), check=True)
        except Exception: subprocess.run(["xsel","--clipboard","--input"], input=s.encode(), check=True)

def read_clip() -> str:
    sysname = platform.system()
    if sysname == "Darwin": return run(["pbpaste"]).stdout
    elif sysname == "Windows": return run(["powershell","-Command","Get-Clipboard"]).stdout
    else: return run(["bash","-lc","xclip -selection clipboard -o || xsel --clipboard --output"], check=False).stdout

def parse_json_loose(s: str):
    s2 = re.sub(r"^```(json)?\s*|\s*```$", "", s.strip(), flags=re.MULTILINE)
    start, end = s2.find("{"), s2.rfind("}")
    if start != -1 and end != -1 and end > start:
        try: return json.loads(s2[start:end+1])
        except Exception: pass
    return json.loads(s2)

def sanitize_diff(out: str) -> str:
    if not out: return ""
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
    except Exception: return False

def run_pytests(args: str, timeout: int) -> int:
    argv = [sys.executable, "-m", "pytest"] + (shlex.split(args) if args else [])
    try:
        r = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
        print(r.stdout); 
        if r.stderr.strip(): print(r.stderr, file=sys.stderr)
        return r.returncode
    except subprocess.TimeoutExpired:
        print(f"[ai] pytest timed out after {timeout}s", file=sys.stderr); return 124

def sha12(s: str) -> str: import hashlib; return hashlib.sha256(s.encode()).hexdigest()[:12]

def main():
    # 1) diff vs base
    run(["git","fetch","origin",DEFAULT_BASE], check=False)
    diff = git("diff","--binary",f"origin/{DEFAULT_BASE}...HEAD")
    if not diff.strip():
        print("[ai] No changes vs base; nothing to do."); return 0

    # 2) Reviewer #1 via Gemini (optional)
    reviewer1_json = "{}"
    if AI_GET_REVIEW1:
        # lightweight prompt
        prompt = ("You are Reviewer #1 for vulkan-forge. Return STRICT JSON with "
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

    # 3) Ensure task.xml exists; if it already exists we will inject the live inputs
    if not os.path.exists(TASK_XML_PATH):
        print(f"[ai] {TASK_XML_PATH} not found. Create it by asking ChatGPT to generate the XML (see the prompt I provided).")
        return 1

    # 4) Load and inject live inputs into task.xml placeholders
    with open(TASK_XML_PATH, "r", encoding="utf-8") as fh:
        xml = fh.read()

    # Replace placeholders or existing blocks
    def inject(tag, content):
        cdata = f"<![CDATA[{content}]]>"
        if re.search(fr"<{tag}>.*?</{tag}>", xml, flags=re.S):
            return re.sub(fr"(?s)<{tag}>.*?</{tag}>", f"<{tag}>{cdata}</{tag}>", xml)
        elif re.search(fr"<{tag}><!\[CDATA\[.*?\]\]></{tag}>", xml, flags=re.S):
            return re.sub(fr"(?s)<{tag}><!\[CDATA\[.*?\]\]></{tag}>", f"<{tag}>{cdata}</{tag}>", xml)
        else:
            # If tag missing, append under <inputs>
            return re.sub(r"(?s)</inputs>", f"  <{tag}>{cdata}</{tag}>\n  </inputs>", xml)

    xml = xml.replace("{{DIFF}}", diff)
    xml = xml.replace("{{REVIEWER1_JSON}}", reviewer1_json)
    # If ChatGPT left placeholders but tags already present, ensure content is injected
    xml = inject("diff", diff)
    xml = inject("reviewer1_json", reviewer1_json)

    # Write a temp copy for this run (do not overwrite your canonical task.xml)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".xml") as tf:
        tf.write(xml)
        task_for_run = tf.name

    # 5) Call Claude Code CLI with the XML file directly
    print("[ai] Running Claude Code with task.xml…")
    cc = run(["claude","-p","--output-format","text","--max-turns","4","--model",CLAUDE_MODEL,f"file:{task_for_run}"], check=False)
    if cc.returncode != 0 or not cc.stdout.strip():
        print(cc.stderr); print("[ai] Claude Code did not produce output."); return 1

    patch = sanitize_diff(cc.stdout)
    if not patch:
        print("[ai] Claude output lacked a recognizable unified diff. Head:\n", cc.stdout[:400])
        return 1

    # 6) Apply patch
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".patch") as pf:
        pf.write(patch); patch_path = pf.name
    print("[ai] Applying patch…")
    apply = run(["git","apply","--reject","--whitespace=fix",patch_path], check=False)
    if apply.returncode != 0:
        print(apply.stderr); print("[ai] Patch had rejects (.rej). Resolve manually, then commit."); return 1

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
