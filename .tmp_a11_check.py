import csv, codecs, sys, json
row=None
try:
  with codecs.open("roadmap2.csv","r","utf-8-sig") as f:
    rdr=csv.DictReader(f)
    for r in rdr:
      wid=(r.get("Workstream ID","") or "").strip().lower()
      tid=(r.get("Task ID","") or "").strip().lower()
      ttl=(r.get("Task Title","") or "").strip().lower()
      if wid=="a" and (tid=="a11" or "a11" in ttl):
        row=r; break
except FileNotFoundError:
  print("UNCERTAIN: roadmap2.csv not found", file=sys.stderr); sys.exit(2)
if not row:
  print("UNCERTAIN: A11 not found in roadmap2.csv (Workstream A). Provide the exact row with headers.", file=sys.stderr); sys.exit(3)
meta={
  "Title": row.get("Task Title","").strip(),
  "Rationale": row.get("Rationale","").strip(),
  "Deliverables": row.get("Deliverables","").strip(),
  "Acceptance": row.get("Acceptance Criteria","").strip(),
  "Priority": row.get("Priority","").strip(),
  "Phase": row.get("Phase","").strip(),
  "Missing": row.get("Missing Features","").strip(),
  "Dependencies": row.get("Dependencies","").strip(),
  "Risks": row.get("Risks/Mitigations","").strip()
}
print("WORKSTREAM_TASK_FOUND:A11")
print("A11_META:"+json.dumps(meta, ensure_ascii=False))