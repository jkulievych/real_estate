"""
Author: Yuliia Kuliievych
License: MIT
"""
import csv, json, time, random, re, requests, html
from bs4 import BeautifulSoup

BASE_URL    = "https://www.imobiliare.ro/vanzare-apartamente"
MAX_PAGES   = 333
OUTPUT_FILE = "data/raw/imobiliare.csv"
HEADERS     = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"}
FIELDS      = ["id", "title", "locality", "category", "rooms", "area_m2", "layout_type", "price_eur", "currency", "description", "image_url", "listing_id", "url"]


def get_rooms(title, desc):
    if re.search(r"garsonier", title, re.I): return 1
    m = re.search(r"cu\s+(\d+)\s+camere", title, re.I) or re.search(r"(\d+)\s+camere", desc, re.I)
    return int(m.group(1)) if m else None

def get_area(desc):
    m = re.search(r"(\d{2,4}[.,]\d{1,2})\s*mp\b", desc, re.I) or re.search(r"(\d{2,4})\s*mp\b", desc, re.I)
    return float(m.group(1).replace(",", ".")) if m else None

def get_layout(title, desc):
    m = re.search(r"(semi|ne)?decomandat[aă]?", title + " " + desc, re.I)
    return m.group(0).lower().rstrip("aă") if m else None

def get_locality(title):
    m = re.search(r"\bîn\s+(.+)$", title, re.I)
    return m.group(1).strip() if m else None

def get_id(raw):
    m = re.search(r"item-(\d+)", raw)
    return m.group(1) if m else None

def get_urls(soup):
    urls = {}
    for a in soup.find_all("a", href=True):
        if a["href"].startswith("/oferta/"):
            m = re.search(r"-(\d+)$", a["href"])
            if m:
                urls[m.group(1)] = "https://www.imobiliare.ro" + a["href"]
    return urls

def parse(soup, global_id):
    urls = get_urls(soup)
    for script in soup.find_all("script", type="application/ld+json"):
        try: data = json.loads(script.string or "")
        except: continue
        for node in (data.get("@graph") or [data]):
            items = node.get("mainEntity", {}).get("itemListElement", [])
            if not items: continue
            rows = []
            for item in items:
                p = item.get("item", {})
                if not p: continue
                title  = p.get("name", "")
                desc   = html.unescape(p.get("description", "")).strip()
                ps     = p.get("offers", {}).get("priceSpecification", {})
                img    = p.get("image", {})
                lst_id = get_id(p.get("@id", ""))
                global_id += 1
                rows.append({
                    "id":          global_id,
                    "title":       title,
                    "locality":    get_locality(title),
                    "category":    p.get("category"),
                    "rooms":       get_rooms(title, desc),
                    "area_m2":     get_area(desc),
                    "layout_type": get_layout(title, desc),
                    "price_eur":   ps.get("price"),
                    "currency":    ps.get("priceCurrency"),
                    "description": desc,
                    "image_url":   img.get("url") if isinstance(img, dict) else None,
                    "listing_id":  lst_id,
                    "url":         urls.get(lst_id),
                })
            return rows, global_id
    return [], global_id


global_id = 0
all_rows = []
session = requests.Session()

for page in range(1, MAX_PAGES + 1):
    url = BASE_URL if page == 1 else f"{BASE_URL}?page={page}"
    print(f"Strona {page}/{MAX_PAGES} — {url}")
    try:
        r = session.get(url, headers=HEADERS, timeout=25)
        r.raise_for_status()
    except Exception as e:
        print(f"Bład: {e}"); break
    rows, global_id = parse(BeautifulSoup(r.text, "lxml"), global_id)
    if not rows:
        print("Nie znalieziono ogłoszeń"); break
    all_rows.extend(rows)
    print(f"  +{len(rows)} ogłoszeń (wszystko: {len(all_rows)})")
    if page < MAX_PAGES:
        time.sleep(random.uniform(2, 5))

filtered = [r for r in all_rows if r["area_m2"]]
for i, r in enumerate(filtered, 1): r["id"] = i
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(filtered)

print(f"Zebrano: {len(all_rows)} | Z powierzchnią: {len(filtered)} | Zachowano '{OUTPUT_FILE}'")