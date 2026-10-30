"""Extract verified text from the official government pages listed in the manifest."""
import argparse
import html
import json
import re
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "data" / "circulars"
RETRIEVED = "2026-10-29"

SOURCES = [
    ("pib_2092717", "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2092717&lang=2&reg=48", "Ministry of New and Renewable Energy", "2025-01-14"),
    ("pib_2174715", "https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=2174715&lang=2&reg=48", "Ministry of Mines", "2025-10-04"),
    ("pib_2186639", "https://www.pib.gov.in/PressReleaseIframePage.aspx?PRID=2186639&lang=2&reg=48", "Ministry of Electronics and Information Technology", "2025-11-05"),
    ("pib_2263719", "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2263719&lang=1&reg=20", "Ministry of Education", "2026-05-21"),
    ("pib_2223331", "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2223331&lang=1&reg=3", "Ministry of Road Transport and Highways", "2026-02-04"),
    ("pib_2241731", "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2241731&lang=1&reg=6", "Ministry of Skill Development and Entrepreneurship", "2026-03-18"),
    ("pib_2237650", "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2237650&lang=2&reg=48", "Ministry of Cooperation", "2026-03-10"),
    ("pib_2225505", "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2225505&lang=1&reg=1", "Ministry of Housing and Urban Affairs", "2026-02-09"),
    ("pib_2244104", "https://www.pib.gov.in/PressReleasePage.aspx?PRID=2244104&lang=2&reg=3", "Ministry of Environment, Forest and Climate Change", "2026-03-23"),
]
DIRECTORY = ("moe_circulars", "https://www.education.gov.in/rti-higher-education/circulars-orders-notification", "Ministry of Education", None)
LINKED_INSTRUMENTS = {
    "pib_2174715": "https://mines.gov.in/admin/storage/ckeditor/SCHEME_GUIDELINES_02_1759320252.pdf",
    "pib_2186639": "https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/nov/doc2025115685601.pdf",
    "pib_2225505": "https://pmay-urban.gov.in/PHQ/A_Comprehensive_Framework_to_Promote_Affordable_Housing.pdf",
}


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.fields, self.active, self.title = {}, None, []
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and attrs.get("id") in ("ltrTitlee", "ltrDescriptionn"):
            self.fields[attrs["id"]] = attrs.get("value", "")
        if attrs.get("id") in ("MinistryName", "Titleh2", "PrDateTime"):
            self.active = attrs["id"]
            self.fields.setdefault(self.active, [])
        if tag in ("h1", "title"):
            self.in_title = True

    def handle_endtag(self, tag):
        if tag in ("h1", "title"):
            self.in_title = False
        if tag in ("div", "h2", "span", "p"):
            self.active = None

    def handle_data(self, data):
        if self.active:
            self.fields[self.active].append(data)
        if self.in_title:
            self.title.append(data)


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        elif not self.skip and tag in ("p", "br", "li", "h1", "h2", "h3", "tr"):
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
        elif not self.skip and tag in ("p", "li", "h1", "h2", "h3", "tr"):
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def clean_markup(markup):
    parser = TextParser()
    parser.feed(markup)
    lines = [re.sub(r"\s+", " ", x).strip() for x in "".join(parser.out).splitlines()]
    return "\n".join(x for x in lines if x)


def extract(identifier):
    raw = BASE / "raw" / f"{identifier}.html"
    parser = PageParser()
    parser.feed(raw.read_text(encoding="utf-8", errors="replace"))
    title = html.unescape(parser.fields.get("ltrTitlee", "")).strip()
    source = clean_markup(html.unescape(parser.fields.get("ltrDescriptionn", "")))
    return title, source


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="retrieve source pages before extraction")
    a = ap.parse_args()
    if a.refresh:
        import urllib.request
        for ident, url, _, _ in SOURCES + [DIRECTORY]:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "PrecisAI research benchmark contact: local user"})
                with urllib.request.urlopen(req, timeout=30) as response:
                    (BASE / "raw" / f"{ident}.html").write_bytes(response.read())
                print(f"downloaded {ident}")
            except Exception as exc:
                print(f"failed {ident}: {exc}")
    records = []
    for ident, url, ministry, published in SOURCES:
        try:
            title, text = extract(ident)
            if not title or not text:
                raise ValueError("official page had no extractable title or article body")
            (BASE / "cleaned" / f"{ident}.txt").write_text(text + "\n", encoding="utf-8")
            records.append({"document_id": ident, "title": title, "issuing_ministry": ministry,
                "publication_date": published, "original_url": url, "document_type": "press release",
                "retrieval_date": RETRIEVED, "raw_file": f"raw/{ident}.html", "cleaned_file": f"cleaned/{ident}.txt",
                "extraction_status": "success", "document_completeness": "press release only; underlying full instrument not included unless separately linked and verified",
                "reference_summary_status": "pending", "reference_summary": None,
                "linked_instrument_url": LINKED_INSTRUMENTS.get(ident),
                "linked_instrument_status": "official link identified but full document not downloaded or extracted" if ident in LINKED_INSTRUMENTS else "not found or signed link inaccessible"})
            sample = (f"Document ID: {ident}\nOriginal source URL: {url}\nDocument type: press release\n"
                      f"Reference summary status: pending\nHuman-reviewed reference summary: null\n\nExtracted source text:\n{text}\n")
            if len(records) <= 3:
                (BASE / "samples" / f"{ident}.txt").write_text(sample, encoding="utf-8")
            print(f"extracted {ident}: {len(text.split())} words")
        except Exception as exc:
            records.append({"document_id": ident, "title": None, "issuing_ministry": ministry,
                "publication_date": published, "original_url": url, "document_type": "press release",
                "retrieval_date": RETRIEVED, "raw_file": f"raw/{ident}.html" if (BASE / "raw" / f"{ident}.html").exists() else None,
                "cleaned_file": None, "extraction_status": "failed", "failure_reason": str(exc),
                "reference_summary_status": "pending", "reference_summary": None})
            print(f"failed extraction {ident}: {exc}")
    did, url, ministry, published = DIRECTORY
    directory_raw = BASE / "raw" / f"{did}.html"
    records.append({"document_id": did, "title": "Circulars, Orders and Notifications directory",
        "issuing_ministry": ministry, "publication_date": published, "original_url": url,
        "document_type": "discovery page (not a circular)", "retrieval_date": RETRIEVED,
        "raw_file": f"raw/{did}.html" if directory_raw.exists() else None, "cleaned_file": None,
        "extraction_status": "not_applicable", "failure_reason": "directory returned no extractable entries; retained as a discovery link, not benchmark document",
        "reference_summary_status": "pending", "reference_summary": None})
    (BASE / "manifest.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8")
    print(f"manifest records: {len(records)}; documents extracted: {sum(x['extraction_status']=='success' for x in records)}")


if __name__ == "__main__":
    main()
