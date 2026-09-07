"""
Scrapes the CAP Code (non-broadcast) and BCAP Code (broadcast) full rule text
from asa.org.uk. These are the rules cited in every ASA ruling; they are not
on legislation.gov.uk, so this needs its own scraper.

Usage:
    python scrape_cap_bcap_code.py
Writes:
    ../data/raw/cap_code_sections.json
    ../data/raw/bcap_code_sections.json
"""

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

CODES = {
    "CAP": {
        "toc_url": "https://www.asa.org.uk/codes-and-rulings/advertising-codes/non-broadcast-code.html",
        "section_url_tpl": "https://www.asa.org.uk/type/non_broadcast/code_section/{:02d}.html",
        "max_sections": 22,
        "out_file": "cap_code_sections.json",
    },
    "BCAP": {
        "toc_url": "https://www.asa.org.uk/codes-and-rulings/advertising-codes/broadcast-code.html",
        "section_url_tpl": "https://www.asa.org.uk/type/broadcast/code_section/{:02d}.html",
        "max_sections": 33,
        "out_file": "bcap_code_sections.json",
    },
}


def get_section_titles(toc_url, url_prefix):
    """Pull '01 Compliance' style links off the table-of-contents page."""
    resp = requests.get(toc_url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    titles = {}
    for a in soup.find_all("a", href=True):
        m = re.search(url_prefix + r"(\d+)\.html$", a["href"])
        if m and a.get_text(strip=True):
            titles[int(m.group(1))] = a.get_text(strip=True)
    return titles


def parse_section_page(html):
    """Extract the Background text and each numbered rule from one section page."""
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find(class_="main-content") or soup

    background_parts = []
    rules = []
    current_subheading = None

    # Walk main-content children in document order so rules stay attached to
    # the nearest preceding Background heading or subgroup heading.
    mode = None
    for el in main.find_all(["h2", "h3", "div"], recursive=True):
        if el.name == "h2":
            text = el.get_text(strip=True)
            mode = "background" if text.lower() == "background" else None
        elif el.name == "h3" and "font-color-grey" in (el.get("class") or []):
            current_subheading = el.get_text(strip=True)
            mode = "rules"
        elif el.name == "div" and "well" in (el.get("class") or []):
            h4 = el.find("h4")
            if h4 is None:
                continue
            rule_number = h4.get_text(strip=True)
            paras = [p.get_text(" ", strip=True) for p in el.find_all("p")]
            rules.append({
                "rule_number": rule_number,
                "subheading": current_subheading,
                "text": " ".join(p for p in paras if p),
            })
        if mode == "background" and el.name == "div" and "content-block" in (el.get("class") or []):
            background_parts.append(el.get_text(" ", strip=True))

    return " ".join(background_parts).strip(), rules


def scrape_code(name, cfg):
    url_prefix = cfg["section_url_tpl"].split("{")[0]
    titles = get_section_titles(cfg["toc_url"], re.escape(url_prefix))
    print(f"[{name}] found {len(titles)} sections in table of contents")

    sections = []
    for num in sorted(titles):
        url = cfg["section_url_tpl"].format(num)
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        background, rules = parse_section_page(resp.text)
        sections.append({
            "code": name,
            "section_number": f"{num:02d}",
            "section_title": titles[num],
            "source_url": url,
            "background": background,
            "rules": rules,
        })
        print(f"  [{name} {num:02d}] {titles[num]!r} -> {len(rules)} rules")
        time.sleep(0.5)  # be polite to asa.org.uk

    out_path = DATA_DIR / cfg["out_file"]
    out_path.write_text(json.dumps(sections, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{name}] wrote {len(sections)} sections -> {out_path}")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in CODES.items():
        scrape_code(name, cfg)
