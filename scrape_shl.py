"""
Scrapes the SHL product catalog (Individual Test Solutions only) into a
structured JSON file that the FastAPI app consumes for retrieval.

RUN THIS LOCALLY (not in a sandbox) since it needs real internet access to
www.shl.com.

Usage:
    pip install requests beautifulsoup4 tqdm
    python scrape_shl.py --out ../data/catalog.json

The SHL catalog page (https://www.shl.com/solutions/products/product-catalog/)
renders a paginated table. It supports query params:
    type=1   -> Individual Test Solutions   (what we want)
    type=2   -> Pre-packaged Job Solutions   (out of scope per assignment)
    start=N  -> pagination offset (12 rows per page historically)

Each row typically contains:
    - Assessment name (also a link to its detail page)
    - Remote Testing support (yes/no icon)
    - Adaptive/IRT support (yes/no icon)
    - Test type badges (single letters: A, B, C, D, E, K, P, S, ...)

Because SHL's markup changes over time and can differ slightly between
environments, this script is defensive: it tries several selector strategies
and prints diagnostics so you can quickly patch it if something breaks.
It also visits each assessment's detail page to pull its description text,
which is what our retrieval index is built over (name/type alone is too
sparse to match against natural-language job descriptions).
"""

import argparse
import json
import re
import sys
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE = "https://www.shl.com"
CATALOG_URL = f"{BASE}/solutions/products/product-catalog/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def get_soup(url, params=None, retries=3, backoff=2.0):
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            print(f"  [warn] fetch failed ({e}), retry {attempt + 1}/{retries}")
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Could not fetch {url} after {retries} retries")


def parse_catalog_page(soup):
    """Extract rows from one catalog listing page. Returns list of dicts with
    at least name + url + raw test_type_letters, or [] if no rows found
    (used to detect end of pagination)."""
    rows = []

    # Strategy 1: look for a real <table>
    table = soup.find("table")
    if table:
        for tr in table.find_all("tr"):
            cells = tr.find_all("td")
            if not cells:
                continue
            link = tr.find("a", href=True)
            if not link:
                continue
            name = link.get_text(strip=True)
            href = urljoin(BASE, link["href"])
            text_blob = tr.get_text(" ", strip=True)
            # remote testing / adaptive: look for "Yes"/circle glyphs near columns
            remote = _has_positive_marker(cells, keyword_index=1)
            adaptive = _has_positive_marker(cells, keyword_index=2)
            type_letters = _extract_type_letters(text_blob)
            rows.append({
                "name": name,
                "url": href,
                "remote_testing": remote,
                "adaptive_irt": adaptive,
                "test_type_letters": type_letters,
            })
        if rows:
            return rows

    # Strategy 2: div-based "card" or "row" layout fallback
    candidates = soup.select("[class*='catalog'] a[href], [class*='product'] a[href]")
    seen = set()
    for a in candidates:
        href = urljoin(BASE, a["href"])
        name = a.get_text(strip=True)
        if not name or href in seen:
            continue
        if "/product-catalog/view/" not in href:
            continue
        seen.add(href)
        rows.append({
            "name": name,
            "url": href,
            "remote_testing": None,
            "adaptive_irt": None,
            "test_type_letters": [],
        })
    return rows


def _has_positive_marker(cells, keyword_index):
    try:
        cell = cells[keyword_index]
    except IndexError:
        return None
    txt = cell.get_text(strip=True).lower()
    if "yes" in txt or cell.find(class_=re.compile("yes|check|circle|active", re.I)):
        return True
    if "no" in txt:
        return False
    return None


def _extract_type_letters(text_blob):
    # Test type letters are usually rendered as compact badges like "P" or "A B K"
    letters = re.findall(r"\b([ABCDEKPS])\b", text_blob)
    return sorted(set(letters))


def scrape_listing(test_type=1, page_size=12, max_pages=200):
    all_rows = []
    start = 0
    empty_streak = 0
    for _ in tqdm(range(max_pages), desc="Paginating catalog"):
        params = {"start": start, "type": test_type}
        soup = get_soup(CATALOG_URL, params=params)
        rows = parse_catalog_page(soup)
        if not rows:
            empty_streak += 1
            if empty_streak >= 2:
                break
        else:
            empty_streak = 0
            all_rows.extend(rows)
        start += page_size
        time.sleep(0.5)  # be polite
    # de-dup by URL
    dedup = {}
    for r in all_rows:
        dedup[r["url"]] = r
    return list(dedup.values())


def enrich_with_description(row):
    """Visit the detail page to grab the description paragraph + duration +
    job levels + languages if present."""
    try:
        soup = get_soup(row["url"])
    except Exception as e:
        print(f"  [warn] could not fetch detail page for {row['name']}: {e}")
        return row

    text = soup.get_text(" ", strip=True)

    desc = ""
    # Try common description containers first
    for sel in ["[class*='description']", "article", "main"]:
        el = soup.select_one(sel)
        if el:
            candidate = el.get_text(" ", strip=True)
            if len(candidate) > len(desc):
                desc = candidate
    if not desc:
        desc = text[:800]

    row["description"] = desc[:1500]

    # crude duration extraction e.g. "Approximate Completion Time in minutes = 30"
    dur_match = re.search(r"(\d+)\s*minutes", text, re.I)
    row["duration_minutes"] = int(dur_match.group(1)) if dur_match else None

    # job level tags often listed like "Graduate, Manager, Mid-Professional"
    level_match = re.search(
        r"(Graduate|Manager|Mid-Professional|Professional Individual Contributor|"
        r"Supervisor|Entry-Level|Front Line Manager|Executive)[\w,\-\s]*",
        text,
    )
    row["job_levels_raw"] = level_match.group(0)[:200] if level_match else ""

    if row.get("test_type_letters") == []:
        row["test_type_letters"] = _extract_type_letters(text)

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="../data/catalog.json")
    parser.add_argument("--skip-enrich", action="store_true",
                         help="Skip visiting each detail page (faster, less data)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Cap number of assessments (debugging)")
    args = parser.parse_args()

    print("Scraping Individual Test Solutions catalog listing...")
    rows = scrape_listing(test_type=1)
    print(f"Found {len(rows)} raw rows.")

    if not rows:
        print(
            "\n[ERROR] No rows parsed. SHL's HTML structure likely differs from "
            "what this script assumes. Open the catalog URL in a browser, "
            "view-source, and inspect the table/row markup, then patch "
            "parse_catalog_page(). Paste the HTML snippet to Claude for a fast fix."
        )
        sys.exit(1)

    if args.limit:
        rows = rows[: args.limit]

    if not args.skip_enrich:
        print("Enriching each assessment with description text (visits each page)...")
        enriched = []
        for row in tqdm(rows, desc="Fetching detail pages"):
            enriched.append(enrich_with_description(row))
            time.sleep(0.3)
        rows = enriched

    for i, row in enumerate(rows):
        row["id"] = f"shl_{i:04d}"
        row["test_types"] = [
            TEST_TYPE_MAP.get(letter, letter) for letter in row.get("test_type_letters", [])
        ]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(rows)} assessments to {args.out}")


if __name__ == "__main__":
    main()
