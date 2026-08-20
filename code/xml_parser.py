"""
Parse UK legislation.gov.uk CLML XML (data.xml) into per-section JSONL records
suitable for RAG indexing / fine-tuning instruction-pair generation.

Usage:
    python xml_parser.py downloaded_legal_xmls/ukpga_2026_1.xml ...
    python xml_parser.py downloaded_legal_xmls/*.xml -o parsed_sections.jsonl
"""

import argparse
import glob
import json
import sys
from lxml import etree

LEG_NS = "http://www.legislation.gov.uk/namespaces/legislation"
UKM_NS = "http://www.legislation.gov.uk/namespaces/metadata"
NS = {"leg": LEG_NS, "ukm": UKM_NS}

# container tags we recurse through looking for headings / nested sections
CONTAINER_TAGS = {"Part", "Chapter", "Pblock", "P1group", "Schedules", "Schedule", "ScheduleBody"}


def local(tag):
    return tag.split("}")[-1] if "}" in tag else tag


def full_text(elem):
    return " ".join("".join(elem.itertext()).split())


def build_commentary_index(root):
    """Map commentary key -> plain text, so section text can carry its in-force/amendment notes as metadata."""
    index = {}
    for c in root.findall(".//leg:Commentaries/leg:Commentary", namespaces=NS):
        key = c.get("id")
        ctype = c.get("Type")
        text = full_text(c)
        if key:
            index[key] = {"type": ctype, "text": text}
    return index

def extract_metadata(root):
    # dc:title lives in the shared <Metadata> block and covers both primary
    # legislation (Acts, e.g. ukpga) and secondary legislation (Regulations/
    # Statutory Instruments, e.g. uksi) -- their body structure (Part/P1group/P1)
    # is identical, but Acts nest their title under leg:Primary/leg:PrimaryPrelims
    # while Regulations nest it under leg:Secondary/leg:SecondaryPrelims, and the
    # metadata block is ukm:PrimaryMetadata vs ukm:SecondaryMetadata respectively.
    # dc:title sidesteps needing to know which one we're looking at.
    def first_match(*xpaths):
        for xp in xpaths:
            el = root.find(xp, namespaces=NS)
            if el is not None:
                return el
        return None

    title = root.findtext(".//dc:title", namespaces={**NS, "dc": "http://purl.org/dc/elements/1.1/"})
    long_title = root.findtext(".//leg:Primary/leg:PrimaryPrelims/leg:LongTitle", namespaces=NS)
    year_el = first_match(".//ukm:PrimaryMetadata/ukm:Year", ".//ukm:SecondaryMetadata/ukm:Year")
    number_el = first_match(".//ukm:PrimaryMetadata/ukm:Number", ".//ukm:SecondaryMetadata/ukm:Number")
    status_el = root.find(".//ukm:DocumentStatus", namespaces=NS)
    enact_el = first_match(".//ukm:EnactmentDate", ".//ukm:Made")
    category_el = root.find(".//ukm:DocumentClassification/ukm:DocumentCategory", namespaces=NS)
    modified = root.findtext(".//ukm:Metadata/dc:modified", namespaces={**NS, "dc": "http://purl.org/dc/elements/1.1/"})
    return {
        "act_title": title,
        "long_title": long_title,
        # "primary" (Acts, cited as "20XX c.N") vs "secondary" (Regulations/SIs, cited as "SI 20XX/N")
        "document_category": category_el.get("Value") if category_el is not None else None,
        "year": year_el.get("Value") if year_el is not None else None,
        "chapter": number_el.get("Value") if number_el is not None else None,
        "document_status": status_el.get("Value") if status_el is not None else None,
        "enactment_date": enact_el.get("Date") if enact_el is not None else None,
        "last_modified": modified,
        "document_uri": root.get("DocumentURI"),
    }


def section_own_text(p1_elem):
    """
    Collect this section's own body text, excluding text that lives inside a
    BlockAmendment (that text is being inserted into a DIFFERENT act and is not
    this section's own content -- naive .//Text collection double-counts it).
    """
    parts = []
    for node in p1_elem.iter():
        if local(node.tag) != "Text":
            continue
        anc = node.getparent()
        inside_amendment = False
        while anc is not None and anc is not p1_elem:
            if local(anc.tag) == "BlockAmendment":
                inside_amendment = True
                break
            anc = anc.getparent()
        if not inside_amendment:
            parts.append(full_text(node))
    return " ".join(p for p in parts if p)


def collect_commentary_refs(p1_elem, commentary_index):
    notes = []
    for ref in p1_elem.findall(".//leg:CommentaryRef", namespaces=NS):
        key = ref.get("Ref")
        note = commentary_index.get(key)
        if note:
            notes.append(note)
    return notes


def walk_body(container, commentary_index, heading_stack, out, source_file):
    for child in container:
        tag = local(child.tag)
        if tag == "P1":
            pnum_el = child.find("leg:Pnumber", namespaces=NS)
            pnum = full_text(pnum_el) if pnum_el is not None else None
            para = child.find("leg:P1para", namespaces=NS)
            text = section_own_text(para) if para is not None else ""
            out.append({
                "source_file": source_file,
                "section_id": child.get("id"),
                "section_number": pnum,
                "heading_breadcrumb": [h for h in heading_stack if h],
                # NOTE: RestrictStartDate/RestrictExtent never appear on P1 itself (verified empirically,
                # 0/3712 sections across the corpus) -- they only live on ancestor containers and mean
                # "this consolidated text is valid as of this date", NOT "this section came into force on
                # this date". Real per-section commencement info is free text inside `commentary`
                # (Type="I" entries, e.g. "S. 1 in force at 22.3.2026 by S.I. 2026/XXX") and needs its
                # own regex/date parser if you want a structured in-force filter.
                "source_url": child.get("DocumentURI"),
                "commentary": collect_commentary_refs(child, commentary_index),
                "text": text,
            })
            # a P1 can itself contain nested P1group (rare, e.g. schedule paragraphs) -- recurse into P1para
            if para is not None:
                walk_body(para, commentary_index, heading_stack, out, source_file)
        elif tag in CONTAINER_TAGS:
            title = child.findtext("leg:Title", namespaces=NS)
            walk_body(child, commentary_index, heading_stack + [title], out, source_file)
        elif tag == "BlockAmendment":
            continue  # quoted text belongs to a different act; never treated as this act's own section
        # ignore leaf metadata tags (Number, Title, Pnumber, P1para handled above, etc.)


def parse_file(path):
    tree = etree.parse(path)
    root = tree.getroot()
    meta = extract_metadata(root)
    commentary_index = build_commentary_index(root)
    body = root.find(".//leg:Body", namespaces=NS)
    sections = []
    if body is not None:
        walk_body(body, commentary_index, [], sections, path.split("/")[-1])
    return meta, sections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="XML file paths (globs expanded by shell or passed literally)")
    ap.add_argument("-o", "--output", default=None, help="write JSONL here instead of stdout")
    args = ap.parse_args()

    paths = []
    for pattern in args.files:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])

    out_fh = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout

    total_sections = 0
    for path in paths:
        meta, sections = parse_file(path)
        print(f"[{path.split('/')[-1]}] {meta['act_title']} ({meta['year']} c.{meta['chapter']}) "
              f"status={meta['document_status']} -> {len(sections)} sections", file=sys.stderr)
        for sec in sections:
            record = {**meta, **sec}
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        total_sections += len(sections)

    if args.output:
        out_fh.close()
    print(f"\nTotal: {len(paths)} files, {total_sections} sections written.", file=sys.stderr)


if __name__ == "__main__":
    main()
