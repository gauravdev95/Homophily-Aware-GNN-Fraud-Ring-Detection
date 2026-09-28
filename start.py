#!/usr/bin/env python3
"""One-command project setup.

Usage:
    python start.py

Does everything needed to get this project running, in order:
  1. Checks Python >= 3.10.
  2. Creates a virtualenv at .venv/ (if missing).
  3. Installs requirements.txt into it.
  4. Downloads any missing dataset files into data/raw/elliptic_pp/
     (txs_features.csv comes from the GitHub data-v1 release; Google Drive
     is the automatic fallback).
  5. Runs scripts/verify_data.py to sanity-check the dataset.

Stdlib only — no dependencies needed to run this script itself.
Works on Windows, macOS and Linux.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQS = ROOT / "requirements.txt"
DATA_DIR = ROOT / "data" / "raw" / "elliptic_pp"
VERIFY = ROOT / "scripts" / "verify_data.py"

REPO = "gauravdev95/Homophily-Aware-GNN-Fraud-Ring-Detection"
RELEASE_DL = f"https://github.com/{REPO}/releases/download/data-v1/txs_features.csv"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/main/data/raw/elliptic_pp"
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l"

# Official Google Drive file IDs (fallback download source).
DRIVE_IDS = {
    "txs_features.csv": "19q09IFhfkOOBOXvn_dKhWjILJtjCcsjc",
    "txs_classes.csv": "1DiBxn8TXdbJqoSw58pYUeaqO3oOKhuQO",
    "txs_edgelist.csv": "1Q2yG_CIDvfdGP-fKVPSw979EYgQukjz5",
}

FILES = {
    # name: (expected bytes or None, [urls in priority order])
    "txs_features.csv": (
        694_789_588,
        [RELEASE_DL, f"https://drive.google.com/uc?id={DRIVE_IDS['txs_features.csv']}&export=download"],
    ),
    "txs_classes.csv": (
        2_361_914,
        [f"{RAW_BASE}/txs_classes.csv", f"https://drive.google.com/uc?id={DRIVE_IDS['txs_classes.csv']}&export=download"],
    ),
    "txs_edgelist.csv": (
        4_470_584,
        [f"{RAW_BASE}/txs_edgelist.csv", f"https://drive.google.com/uc?id={DRIVE_IDS['txs_edgelist.csv']}&export=download"],
    ),
}

CHUNK = 1024 * 1024  # 1 MB


# ---------------------------------------------------------------- output helpers

def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


_COLOR = _supports_color()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def step(n: int, total: int, msg: str) -> None:
    print(f"\n{_c('1;36', f'[{n}/{total}]')} {_c('1', msg)}")


def ok(msg: str) -> None:
    print(f"  {_c('32', 'OK')}  {msg}")


def info(msg: str) -> None:
    print(f"  {_c('34', '--')}  {msg}")


def warn(msg: str) -> None:
    print(f"  {_c('33', '!!')}  {msg}")


def fail(msg: str) -> "NoReturn":
    print(f"\n{_c('1;31', 'FAILED:')} {msg}", file=sys.stderr)
    raise SystemExit(1)


# ---------------------------------------------------------------- setup steps

def check_python() -> None:
    if sys.version_info < (3, 10):
        fail(f"Python 3.10+ required, found {sys.version.split()[0]}. "
             "Install a newer Python and re-run.")
    ok(f"Python {sys.version.split()[0]}")


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def ensure_venv() -> Path:
    py = venv_python()
    if py.exists():
        ok(f"virtualenv already exists at {VENV}")
        return py
    info(f"creating virtualenv at {VENV} ...")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    except subprocess.CalledProcessError:
        fail("could not create virtualenv. Make sure 'venv' is available "
             "(Debian/Ubuntu: sudo apt install python3-venv).")
    ok(f"created {VENV}")
    return venv_python()


def install_deps(py: Path, reinstall: bool) -> None:
    if not REQS.exists():
        fail(f"{REQS.name} not found in project root.")
    info("upgrading pip ...")
    subprocess.run([str(py), "-m", "pip", "install", "-q", "--upgrade", "pip"],
                   check=False)
    cmd = [str(py), "-m", "pip", "install", "-r", str(REQS)]
    if reinstall:
        cmd.append("--force-reinstall")
    info(f"installing {REQS.name} (this can take a few minutes) ...")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        fail("pip install failed. Check your network connection and re-run.")
    ok("dependencies installed")


# ---------------------------------------------------------------- downloads

def _drive_confirm_download(url: str, dest: Path, expected: int | None) -> bool:
    """Download a Google Drive file, handling the virus-scan confirm page."""
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    def fetch(u: str):
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
        return opener.open(req, timeout=120)

    try:
        resp = fetch(url)
    except Exception as exc:
        warn(f"drive request failed: {exc}")
        return False

    ctype = resp.headers.get("Content-Type", "")
    if "text/html" in ctype:
        # Interstitial page (virus-scan warning) — extract the real download URL.
        html = resp.read().decode("utf-8", errors="replace")
        m = re.search(r'id="download-form"\s+action="([^"]+)"', html)
        if m:
            real = m.group(1).replace("&amp;", "&")
        else:
            m2 = re.search(r'"confirm"\s*:\s*"([^"]+)"', html) or \
                 re.search(r'confirm=([0-9A-Za-z_-]+)', html)
            if not m2:
                warn("could not parse drive confirm page")
                return False
            sep = "&" if "?" in url else "?"
            real = f"{url}{sep}confirm={m2.group(1)}"
        try:
            resp = fetch(real)
        except Exception as exc:
            warn(f"drive confirm download failed: {exc}")
            return False
        if "text/html" in resp.headers.get("Content-Type", ""):
            warn("drive still returned an HTML page (permission/quota issue?)")
            return False

    return _stream_to_file(resp, dest, expected)


def _stream_to_file(resp, dest: Path, expected: int | None) -> bool:
    total = resp.headers.get("Content-Length")
    total = int(total) if total and total.isdigit() else expected
    tmp = dest.with_suffix(dest.suffix + ".part")
    downloaded = 0
    try:
        with open(tmp, "wb") as fh:
            while True:
                buf = resp.read(CHUNK)
                if not buf:
                    break
                fh.write(buf)
                downloaded += len(buf)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {_c('34', '--')}  {downloaded / 1e6:7.1f} / "
                          f"{total / 1e6:.1f} MB ({pct:5.1f}%)",
                          end="", flush=True)
        print()
    except Exception as exc:
        warn(f"download interrupted: {exc}")
        tmp.unlink(missing_ok=True)
        return False
    if expected and abs(downloaded - expected) / expected > 0.01:
        warn(f"size mismatch: got {downloaded / 1e6:.1f} MB, "
             f"expected {expected / 1e6:.1f} MB")
        tmp.unlink(missing_ok=True)
        return False
    shutil.move(str(tmp), str(dest))
    return True


def download_file(name: str, expected: int | None, urls: list[str]) -> bool:
    dest = DATA_DIR / name
    for url in urls:
        src = "github release" if "releases/download" in url \
            else "github repo" if "raw.githubusercontent" in url else "google drive"
        info(f"downloading {name} from {src} ...")
        try:
            if "drive.google.com" in url:
                good = _drive_confirm_download(url, dest, expected)
            else:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                good = _stream_to_file(urllib.request.urlopen(req, timeout=120),
                                       dest, expected)
        except Exception as exc:
            warn(f"{src} failed: {exc}")
            good = False
        if good and dest.exists():
            ok(f"{name} ({dest.stat().st_size / 1e6:.1f} MB)")
            return True
        warn(f"{src} did not work, trying next source ...")
    return False


def ensure_data(no_download: bool) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    missing = [n for n in FILES if not (DATA_DIR / n).exists()]
    for name in FILES:
        if (DATA_DIR / name).exists():
            ok(f"{name} already present")
    if not missing:
        return
    if no_download:
        fail("missing data files: " + ", ".join(missing) +
             ". Re-run without --no-download to fetch them automatically.")
    for name in missing:
        expected, urls = FILES[name]
        if not download_file(name, expected, urls):
            fail(f"could not download {name} from any source.\n"
                 f"  Download it manually from {DRIVE_FOLDER}\n"
                 f"  and place it at data/raw/elliptic_pp/{name}")
    ok("all dataset files present")


# ---------------------------------------------------------------- verify

def run_verify(py: Path) -> None:
    if not VERIFY.exists():
        warn(f"{VERIFY.name} not found, skipping verification")
        return
    info(f"running {VERIFY.name} ...")
    rc = subprocess.run([str(py), str(VERIFY)]).returncode
    if rc != 0:
        fail("data verification reported problems (see output above).")
    ok("dataset verified")


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(
        description="One-command setup for the Homophily-Aware GNN project.")
    ap.add_argument("--no-download", action="store_true",
                    help="do not download missing dataset files")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the data verification step")
    ap.add_argument("--reinstall", action="store_true",
                    help="force-reinstall dependencies")
    args = ap.parse_args()

    total = 5 if not args.no_verify else 4
    print(_c("1;36", "=" * 60))
    print(_c("1;36", "  Homophily-Aware GNN — project setup"))
    print(_c("1;36", "=" * 60))

    n = 1
    step(n, total, "Checking Python"); n += 1
    check_python()

    step(n, total, "Setting up virtualenv"); n += 1
    py = ensure_venv()

    step(n, total, "Installing dependencies"); n += 1
    install_deps(py, args.reinstall)

    step(n, total, "Checking dataset"); n += 1
    ensure_data(args.no_download)

    if not args.no_verify:
        step(n, total, "Verifying dataset")
        run_verify(py)

    print(f"\n{_c('1;32', 'All set!')}")
    print("  Activate the env :  " + (_c('1', r".venv\Scripts\activate") if os.name == "nt"
                                      else _c('1', "source .venv/bin/activate")))
    print("  Re-verify data   :  " + _c('1', "python scripts/verify_data.py"))
    print("  Explore          :  " + _c('1', "notebooks/01_data_exploration.ipynb"))


if __name__ == "__main__":
    main()
