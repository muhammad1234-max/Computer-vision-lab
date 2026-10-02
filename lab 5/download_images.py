"""
download_images.py
Downloads all images required for CV Lab 5 Tasks 1-10.
"""

import urllib.request
import os
import ssl

# Allow unverified HTTPS (needed on some systems)
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

os.makedirs("images", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

IMAGES = {
    # Task 1 & 2 — Unevenly illuminated document
    "sudoku.png": (
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/sudoku.png"
    ),
    # Task 3, 7, 8 — Object / background coins image
    "coins.png": (
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/coins.png"
    ),
    # Task 4 & 5 — Multi-colour objects
    "smarties.png": (
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/smarties.png"
    ),
    # Task 6 — Brain MRI for region growing
    "brain.png": (
        "https://raw.githubusercontent.com/PacktPublishing/"
        "OpenCV-3-Computer-Vision-with-Python-Cookbook/master/Chapter09/brain.png"
    ),
    # Task 9 & 10 — Complex colour image
    "dog.jpeg": (
        "https://raw.githubusercontent.com/PacktPublishing/"
        "OpenCV-3-Computer-Vision-with-Python-Cookbook/master/Chapter09/dog.jpeg"
    ),
}

# Fallback URLs
FALLBACKS = {
    "brain.png": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/"
        "MRI_brain_sagittal_section.jpg/320px-MRI_brain_sagittal_section.jpg",
    ],
    "dog.jpeg": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/"
        "YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg",
    ],
    "smarties.png": [
        "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/smarties.png",
    ],
}


def download(name, url, dest):
    print(f"  Downloading {name} …", end=" ", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=30) as r, \
                open(dest, "wb") as f:
            f.write(r.read())
        size = os.path.getsize(dest)
        if size < 1000:
            os.remove(dest)
            raise ValueError(f"File too small ({size} bytes) — likely error page")
        print(f"OK ({size // 1024} KB)")
        return True
    except Exception as e:
        print(f"FAILED — {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


for name, url in IMAGES.items():
    dest = os.path.join("images", name)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        print(f"  {name} already present — skipped.")
        continue

    ok = download(name, url, dest)

    if not ok and name in FALLBACKS:
        for fb_url in FALLBACKS[name]:
            print(f"  Trying fallback …", end=" ", flush=True)
            ok = download(name, fb_url, dest)
            if ok:
                break

    if not ok:
        print(f"  !! Could not download {name}. "
              "Please place it manually in the images/ folder.")

print("\nAll downloads complete. Files in images/:")
for f in os.listdir("images"):
    size = os.path.getsize(os.path.join("images", f))
    print(f"  {f:25s}  {size // 1024:>5} KB")
