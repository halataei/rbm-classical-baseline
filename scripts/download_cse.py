r"""
Download the CSE-CIC-IDS2018 "Processed Traffic Data for ML Algorithms" CSV files (about 6.5 GB, 10 files)
from the dataset's official public Amazon S3 bucket over plain HTTPS. No AWS account or AWS software needed.
Run it in VS Code with the "Run Python File" button.

WHERE TO PUT IT
    C:\RBM\download_cse.py     <- this file
The files are saved in C:\RBM\CSE\ . If the download is interrupted, run the script again: finished files are
skipped and a partly downloaded file continues from where it stopped.
Uses only the Python standard library.
"""

BUCKET_URL = "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com"
PREFIX = "Processed Traffic Data for ML Algorithms/"
DEST_FOLDER = "CSE"        # relative to the folder of this script

import os, sys, time, urllib.request, urllib.parse, urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = HERE / DEST_FOLDER


def list_files():
    """Ask S3 for the list of objects under PREFIX (public bucket, anonymous request)."""
    files, token = [], None
    while True:
        q = {"list-type": "2", "prefix": PREFIX}
        if token:
            q["continuation-token"] = token
        url = BUCKET_URL + "/?" + urllib.parse.urlencode(q, quote_via=urllib.parse.quote)
        with urllib.request.urlopen(url, timeout=60) as r:
            root = ET.fromstring(r.read())
        ns = {"s3": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
        p = "s3:" if ns else ""
        for c in root.findall(f"{p}Contents", ns):
            key, size = c.find(f"{p}Key", ns).text, int(c.find(f"{p}Size", ns).text)
            if key.lower().endswith(".csv"):
                files.append((key, size))
        trunc = root.find(f"{p}IsTruncated", ns)
        if trunc is None or trunc.text != "true":
            return files
        token = root.find(f"{p}NextContinuationToken", ns).text


def download(key, size):
    name = key.split("/")[-1]
    out, part = DEST / name, DEST / (name + ".part")
    if out.exists() and out.stat().st_size == size:
        print(f"   {name}: already downloaded, skipped", flush=True)
        return
    start = part.stat().st_size if part.exists() else 0
    url = BUCKET_URL + "/" + urllib.parse.quote(key)
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    t0, last, done = time.time(), 0.0, start
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if start else "wb") as f:
        if start and r.status != 206:              # server ignored the resume request: start again
            f.seek(0); f.truncate(); done = 0
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf); done += len(buf)
            now = time.time()
            if now - last > 1 or done == size:
                speed = (done - start) / max(now - t0, 1e-9) / 1e6
                left = (size - done) / 1e6 / speed if speed > 0 else 0
                print(f"\r   {name}: {done / 1e9:6.2f} / {size / 1e9:.2f} GB  ({100 * done / size:5.1f}%)  "
                      f"{speed:5.1f} MB/s  about {int(left // 60)} min {int(left % 60)} s left   ", end="", flush=True)
                last = now
    print(flush=True)
    if part.stat().st_size != size:
        raise IOError(f"{name}: size {part.stat().st_size} differs from expected {size}; run the script again")
    os.replace(part, out)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Listing files in the official CSE-CIC-IDS2018 bucket ...", flush=True)
    try:
        files = list_files()
    except (urllib.error.URLError, OSError) as e:
        print(f"\nCould not reach Amazon S3: {e}\n"
              "If you are in a region where AWS blocks connections, connect through a VPN and run this script again.",
              flush=True)
        sys.exit(1)
    total = sum(s for _, s in files)
    print(f"{len(files)} files, {total / 1e9:.2f} GB in total, saving to {DEST}", flush=True)
    for i, (key, size) in enumerate(files, 1):
        print(f"[{i}/{len(files)}]", flush=True)
        for attempt in range(1, 6):
            try:
                download(key, size)
                break
            except (urllib.error.URLError, OSError, IOError) as e:
                print(f"\n   interrupted ({e}); retrying in 10 s (attempt {attempt}/5)", flush=True)
                time.sleep(10)
        else:
            print("   giving up on this file for now; run the script again later to resume.", flush=True)
    have = [p for p in DEST.glob("*.csv")]
    print(f"\nDone. {len(have)} of {len(files)} CSV files are complete in {DEST}.", flush=True)


if __name__ == "__main__":
    main()
