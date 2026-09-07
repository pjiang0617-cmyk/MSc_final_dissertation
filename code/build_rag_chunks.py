# Unifies CAP/BCAP rules, legislation sections, and ASA rulings into one flat
# chunks.jsonl for retrieval. Common fields: chunk_id, source_type, text,
# citation, source_url, metadata (group/decision/code/rule_number/etc, varies
# by source_type). Run with no args -> writes ../data/rag/chunks.jsonl.

import json
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "rag"


def chunks_from_code(path, code_name):
    sections = json.loads(path.read_text(encoding="utf-8"))
    chunks = []
    for sec in sections:
        for rule in sec["rules"]:
            citation = f"{code_name} Code (Edition 12) rule {rule['rule_number']}"
            chunks.append({
                "chunk_id": f"{code_name.lower()}_{rule['rule_number']}",
                "source_type": f"{code_name.lower()}_rule",
                "text": rule["text"],
                "citation": citation,
                "source_url": sec["source_url"],
                "metadata": {
                    "code": code_name,
                    "section_number": sec["section_number"],
                    "section_title": sec["section_title"],
                    "rule_number": rule["rule_number"],
                    "subheading": rule["subheading"],
                },
            })
    return chunks


def chunks_from_legislation(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if not rec["text"]:
                continue
            # Acts (primary) are cited "20XX c.N"; Regulations/SIs (secondary) are cited "SI 20XX/N".
            is_secondary = rec.get("document_category") == "secondary"
            number_part = f"SI {rec['year']}/{rec['chapter']}" if is_secondary else f"{rec['year']} c.{rec['chapter']}"
            section_label = "reg." if is_secondary else "s."
            citation = f"{rec['act_title']} ({number_part}) {section_label}{rec['section_number']}"
            chunks.append({
                "chunk_id": f"leg_{rec['source_file']}_{rec['section_id']}",
                "source_type": "legislation_section",
                "text": rec["text"],
                "citation": citation,
                "source_url": rec["source_url"],
                "metadata": {
                    "act_title": rec["act_title"],
                    "document_category": rec.get("document_category"),
                    "year": rec["year"],
                    "chapter": rec["chapter"],
                    "section_number": rec["section_number"],
                    "heading_breadcrumb": rec["heading_breadcrumb"],
                    "document_status": rec["document_status"],
                    "commentary": [c["text"] for c in rec["commentary"]],
                },
            })
    return chunks


def chunks_from_asa_rulings(path, group_name):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            case_label = f"{rec.get('advertiser') or 'Unknown advertiser'} ({rec.get('complaint_ref') or 'no ref'})"

            summary_parts = [p for p in (rec.get("ad_description"), rec.get("issue")) if p]
            if summary_parts:
                chunks.append({
                    "chunk_id": f"asa_{group_name}_{rec['complaint_ref']}_summary",
                    "source_type": "asa_case_summary",
                    "text": "\n\n".join(summary_parts),
                    "citation": f"ASA ruling: {case_label}",
                    "source_url": rec["source_url"],
                    "metadata": {
                        "group": group_name,
                        "decision": rec.get("decision"),
                        "date": rec.get("date"),
                        "topics": rec.get("topics"),
                        "cited_rules": rec.get("cited_rules"),
                    },
                })

            if rec.get("assessment"):
                chunks.append({
                    "chunk_id": f"asa_{group_name}_{rec['complaint_ref']}_assessment",
                    "source_type": "asa_case_assessment",
                    "text": rec["assessment"],
                    "citation": f"ASA ruling: {case_label}",
                    "source_url": rec["source_url"],
                    "metadata": {
                        "group": group_name,
                        "decision": rec.get("decision"),
                        "date": rec.get("date"),
                        "topics": rec.get("topics"),
                        "cited_rules": rec.get("cited_rules"),
                        "code_edition": rec.get("code_edition"),
                    },
                })
    return chunks


def main():
    all_chunks = []
    all_chunks += chunks_from_code(RAW_DIR / "cap_code_sections.json", "CAP")
    all_chunks += chunks_from_code(RAW_DIR / "bcap_code_sections.json", "BCAP")
    for leg_path in sorted(PROCESSED_DIR.glob("*_sections.jsonl")):
        all_chunks += chunks_from_legislation(leg_path)
    all_chunks += chunks_from_asa_rulings(RAW_DIR / "asa_rulings_group_b.jsonl", "group_b")
    all_chunks += chunks_from_asa_rulings(RAW_DIR / "asa_rulings_group_d.jsonl", "group_d")

    from collections import Counter
    print("Chunk counts by source_type:", dict(Counter(c["source_type"] for c in all_chunks)))
    print("Total chunks:", len(all_chunks))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_chunks)} chunks -> {out_path}")


if __name__ == "__main__":
    main()
