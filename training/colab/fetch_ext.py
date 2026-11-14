"""Fetch the external test/extra datasets straight onto Colab (run in the background, logs to /content/ext/fetch.log).

DefectDet (CC BY 4.0, Mendeley t9d9zs3bmb) and EPID (CC BY 4.0, Zenodo 16811808): open downloads, checksums
verified against the published values. Real-IAD pcb (CC BY-NC-SA, research only, gated): only if the
HF_TOKEN Colab secret is present; the token is read from the environment and never printed. Real-IAD is
for testing only and is never copied into the repo or used for published weights.
"""
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

EXT = Path("/content/ext")
EXT.mkdir(exist_ok=True)
LOG = open(EXT / "fetch.log", "a", buffering=1)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=LOG)


def digest(p, algo):
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, url, algo=None, expect=None, headers=()):
    out = EXT / name
    if out.exists() and expect and digest(out, algo) == expect:
        log(name, "already present, checksum ok")
        return True
    cmd = ["curl", "-sSL", "--retry", "5", "-o", str(out), url]
    for h in headers:
        cmd[1:1] = ["-H", h]
    r = subprocess.run(cmd)
    if r.returncode != 0 or not out.exists():
        log(name, "download FAILED", r.returncode)
        return False
    if expect:
        got = digest(out, algo)
        log(name, "checksum", "OK" if got == expect else f"MISMATCH {got}")
        return got == expect
    log(name, "downloaded", out.stat().st_size // 2**20, "MB")
    return True


def unzip(name, dest):
    d = EXT / dest
    d.mkdir(exist_ok=True)
    subprocess.run(["unzip", "-q", "-o", str(EXT / name), "-d", str(d)])
    log(name, "unzipped to", d)


what = sys.argv[1:] or ["defectdet", "epid", "realiad"]
if "defectdet" in what and fetch("DefectDet.zip",
        "https://data.mendeley.com/public-files/datasets/t9d9zs3bmb/files/eb82201c-168b-4364-ac2f-7d289728dd44/file_downloaded",
        "sha256", "250fc8f60e8ee40ae20bfcb00ff2b16366b395fe2fc9aac29e79a10793080b31"):
    unzip("DefectDet.zip", "defectdet")
if "epid" in what and fetch("EPID.zip", "https://zenodo.org/api/records/16811808/files/EPID.zip/content",
                            "md5", "e43ac4a858ef7190f0b0920dcd27e0e4"):
    unzip("EPID.zip", "epid")
if "realiad" in what:
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        log("realiad: no HF_TOKEN in the environment; add it as a Colab secret and re-run with 'realiad'")
    elif fetch("realiad_pcb_512.zip",
               "https://huggingface.co/datasets/Real-IAD/Real-IAD/resolve/main/realiad_512/pcb.zip",
               headers=(f"Authorization: Bearer {tok}",)):
        unzip("realiad_pcb_512.zip", "realiad")
log("FETCH_DONE", what)
