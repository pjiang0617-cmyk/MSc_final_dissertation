"""
Scrapes real ASA ruling case pages, filtered by topic, to get complaint/decision
text for a chosen vertical group.

Site mechanics (no public API):
  - Each ASA "Topic" has a GUID used to filter the rulings search: rulings.html?topic=<GUID>
  - The search needs an explicit date range (custom_date=1&from_date=DD/MM/YYYY&to_date=DD/MM/YYYY);
    without it the default window returns almost nothing.
  - Results paginate with a plain &page=N query param.
  - Individual ruling pages are static server-rendered HTML with consistent
    <h2>Background/Ad description/Issue/Response/Assessment/Action</h2> blocks,
    a "CAP Code (Edition N)" section listing the exact rule numbers breached
    (as links), and a title-section with decision/media/date.

Usage:
    python scrape_asa_rulings.py group_b   # medical / cosmetic
    python scrape_asa_rulings.py group_d   # gambling / alcohol / vaping / children
    python scrape_asa_rulings.py group_b group_d
"""

import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
LISTING_URL = "https://www.asa.org.uk/codes-and-rulings/rulings.html"


class PermanentHTTPError(Exception):
    """A dead link (404 / redirected to ruling-not-found.html) -- not retried."""
    pass


def get_with_retry(url, params=None, retries=4, timeout=30):
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if resp.status_code == 404 or resp.url.endswith("/ruling-not-found.html"):
                raise PermanentHTTPError(f"{url} -> not found ({resp.status_code})")
            resp.raise_for_status()
            return resp
        except PermanentHTTPError:
            raise
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            print(f"  [retry {attempt + 1}/{retries}] {url} -> {e}; sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Giving up on {url} after {retries} retries")


# GUIDs extracted from each https://www.asa.org.uk/topic/<Name>.html page
# (look for `topic=<GUID>` in the "View all articles" link on that page).
TOPIC_GUIDS = {
    "group_b": {  # medical / cosmetic
        "Cosmetic_surgery_and_procedures": "2DEAE19F-62FC-448D-8AB064C6763755FC",
        "Medicines_remedies_and_therapies": "B9497F61-870A-4053-A85BC98F42051F66",
        "Medical_procedures_and_services": "D88C3218-6E18-40A5-9D15F13327FE2A28",
        "Medical_devices": "58FA7A26-99CB-4514-AC2023E99756ECD8",
        "Health_conditions": "07D7C5DE-432F-455D-848AFF57A6094BF7",
        "Weight_and_slimming": "B9A1B16C-4DFA-41E5-BF1124CAA56B737A",
    },
    "group_d": {  # gambling / alcohol / vaping / children
        "Gambling": "7509B48E-504B-4E08-9C053A2BF9DA5021",
        "Alcohol": "21C768FB-1111-4C00-A2D191B7508910D7",
        "Vaping_smoking_and_drugs": "83B823B2-DEBA-498A-AA709E19387E58A4",
        "Children_and_the_vulnerable": "F9DA2FCE-F62F-49BE-AEC6BCF4FA07070A",
    },
}

FROM_DATE = "01/01/2015"
TO_DATE = "29/07/2026"


def list_ruling_urls(topic_guid, max_pages=50):
    urls = set()
    for page in range(1, max_pages + 1):
        params = {
            "q": "",
            "sort_order": "recent",
            "custom_date": "1",
            "from_date": FROM_DATE,
            "to_date": TO_DATE,
            "topic": topic_guid,
            "page": page,
        }
        resp = get_with_retry(LISTING_URL, params=params)
        page_links = set(re.findall(r'href="(https://www\.asa\.org\.uk/rulings/[^"]+)"', resp.text))
        before = len(urls)
        urls |= page_links
        if len(urls) == before:  # no new links on this page -> reached the end
            break
        time.sleep(0.4)
    return urls


def text_of(el):
    return el.get_text(" ", strip=True) if el else None


def parse_ruling_page(html, url):
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.find("h1", class_="heading")
    advertiser = text_of(h1)
    if advertiser:
        advertiser = re.sub(r"\s+", " ", advertiser).replace("ASA Ruling on ", "").strip()

    meta_items = [li.get_text(" ", strip=True) for li in soup.select("ul.meta-listing li.meta-listing-item")]
    decision = next((m for m in meta_items if m in ("Upheld", "Not upheld", "Partially upheld")), None)
    date = next((m for m in meta_items if re.match(r"\d{1,2} \w+ \d{4}", m)), None)
    media = next((m for m in meta_items if m not in (decision, date)), None)

    sections = {}
    for h2 in soup.find_all("h2", class_="font-color-grey"):
        heading = h2.get_text(strip=True)
        if heading.startswith("CAP Code") or heading.startswith("BCAP Code") or heading == "More on":
            continue
        parts = []
        for sib in h2.find_next_siblings():
            if sib.name == "h2":
                break
            if sib.name == "p":
                parts.append(sib.get_text(" ", strip=True))
        sections[heading] = " ".join(p for p in parts if p)

    # dedicated "CAP Code (Edition N)" / "BCAP Code (Edition N)" block: clean <a> links per rule number
    code_edition = None
    cited_rules = []
    for h2 in soup.find_all("h2", class_="font-color-grey"):
        heading = h2.get_text(strip=True)
        if heading.startswith("CAP Code") or heading.startswith("BCAP Code"):
            code_edition = heading
            nxt = h2.find_next_sibling("p")
            if nxt:
                cited_rules = [a.get_text(strip=True) for a in nxt.find_all("a")]
            break

    topics = [a.get_text(strip=True) for a in soup.select("ul.tag-listing a.tag")]

    complaint_ref = None
    ref_label = soup.find(string=re.compile("Complaint Ref"))
    if ref_label:
        strong = ref_label.find_parent("p").find("strong") if ref_label.find_parent("p") else None
        complaint_ref = strong.get_text(strip=True) if strong else None

    return {
        "source_url": url,
        "advertiser": advertiser,
        "decision": decision,
        "media": media,
        "date": date,
        "complaint_ref": complaint_ref,
        "code_edition": code_edition,
        "cited_rules": cited_rules,
        "topics": topics,
        "background": sections.get("Background"),
        "ad_description": sections.get("Ad description") or sections.get("Advertisement description"),
        "issue": sections.get("Issue"),
        "response": sections.get("Response"),
        "assessment": sections.get("Assessment"),
        "action": sections.get("Action"),
    }


def scrape_group(group_name):
    topics = TOPIC_GUIDS[group_name]
    all_urls = {}
    for topic_name, guid in topics.items():
        urls = list_ruling_urls(guid)
        print(f"[{group_name}] {topic_name}: {len(urls)} ruling URLs")
        for u in urls:
            all_urls.setdefault(u, set()).add(topic_name)

    print(f"[{group_name}] {len(all_urls)} unique rulings across {len(topics)} topics")

    out_path = DATA_DIR / f"asa_rulings_{group_name}.jsonl"
    written, skipped = 0, 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (url, matched_topics) in enumerate(sorted(all_urls.items())):
            try:
                resp = get_with_retry(url)
            except PermanentHTTPError as e:
                print(f"  [skip] {e}", file=sys.stderr)
                skipped += 1
                continue
            record = parse_ruling_page(resp.text, url)
            record["matched_search_topics"] = sorted(matched_topics)
            record["group"] = group_name
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()  # write incrementally -- one dead link should not lose already-scraped results
            written += 1
            if (i + 1) % 10 == 0:
                print(f"  parsed {i + 1}/{len(all_urls)}")
            time.sleep(0.4)
    print(f"[{group_name}] wrote {written} rulings ({skipped} broken links skipped) -> {out_path}")


if __name__ == "__main__":
    groups = sys.argv[1:] or list(TOPIC_GUIDS.keys())
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for g in groups:
        scrape_group(g)
