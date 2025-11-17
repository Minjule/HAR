import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from zipfile import ZipFile
import zipfile
import cv2
import re

# ----------------------------
# CONFIGURATION
# ----------------------------
BASE_URL = "https://fenix.ur.edu.pl/~mkepski/ds/uf.html"  
DOWNLOAD_DIR = "downloads"
EXTRACT_DIR = "merged_dataset"
MAX_ROWS = 30

def fetch_and_parse(url, timeout=10):
    """Return (response, soup) or (None, None) and print validation info."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
        return None, None

    print(f"[INFO] HTTP {r.status_code}  Content-Type: {r.headers.get('content-type')}")
    print(f"[INFO] Response length: {len(r.text or '')} chars")
    if 'html' not in (r.headers.get('content-type') or '').lower():
        print("[WARN] Content-Type does not look like HTML")

    preview = (r.text or "")[:500].replace("\n", " ")
    print(f"[DEBUG] Response preview: {preview}")

    soup = BeautifulSoup(r.text, "html.parser")
    rows = soup.select("body div table tr")
    print(f"[INFO] Parsed {len(rows)} <tr> rows")

    # show a few sample links / anchors for quick verification
    for i, row in enumerate(rows[:5]):
        a = row.find("a", href=True)
        print(f"  sample row {i}: href={a['href'] if a else '(no <a>)'}")
    return r, soup

# ----------------------------
# SCRAPE LINKS
# ----------------------------
print("[INFO] Scraping dataset links...")
response, soup = fetch_and_parse(BASE_URL)
if soup is None:
    print("[ERROR] Could not fetch or parse the page. Aborting.")
    raise SystemExit(1)

links = []
seen = set()
rows = soup.select("body div table tr")  # navigate to <tr> rows

# match any URL that contains "cam0-rgb" then any chars then ".zip" (case-insensitive)
pattern = re.compile(r"cam0-rgb.*\.zip", flags=re.I)

for i, row in enumerate(rows):
    if i >= MAX_ROWS:
        break
    # Select 4th column <td> of this row
    td = row.select_one("td:nth-child(4) a[href]")
    if td and "cam0-rgb.zip" in td["href"]:
        link = td["href"]
        if not link.startswith("http"):
            link = requests.compat.urljoin(BASE_URL, link)
        links.append(link)

print(f"[INFO] Found {len(links)} matching 'cam0-rgb.zip' links.")


# ----------------------------
# DOWNLOAD FILES
# ----------------------------
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

for url in tqdm(links, desc="Downloading"):
    filename = os.path.join(DOWNLOAD_DIR, os.path.basename(url))
    if os.path.exists(filename):
        print(f"[SKIP] {filename} already exists.")
        continue
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")

# ----------------------------
# EXTRACT ALL INTO ONE FOLDER
# ----------------------------
os.makedirs(EXTRACT_DIR, exist_ok=True)

for zip_file in tqdm(os.listdir(DOWNLOAD_DIR), desc="Extracting"):
    if zip_file.endswith(".zip"):
        zip_path = os.path.join(DOWNLOAD_DIR, zip_file)
        try:
            with ZipFile(zip_path, 'r') as zf:
                zf.extractall(EXTRACT_DIR)
        except Exception as e:
            print(f"[ERROR] Failed to extract {zip_file}: {e}")

print("[✅ DONE] All cam0-rgb.zip files extracted into one folder:", EXTRACT_DIR)
