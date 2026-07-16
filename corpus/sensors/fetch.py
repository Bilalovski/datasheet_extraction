"""Fetch the real datasheet corpus.

The gold labels in ``gold.json`` are committed; the datasheet PDFs and the text
extracted from them are copyrighted and are not, so this script downloads each
PDF from its official (or a public mirror) URL and extracts the text the loader
reads. Run it once from the repo root:

    python corpus/sensors/fetch.py

Requires ``pdftotext`` (poppler-utils) on PATH. The ``-layout`` flag preserves
the parametric tables the labels were read from — plain extraction reflows them
into unreadable prose.

The eight parts are the "batch 1" corpus: ranging (lidar / ToF / ultrasonic),
plus environmental and magnetic point sensors. All are simple single-supply
devices, chosen so every spec is a single stated number rather than the
multi-rail, application-dependent mess a bare radar SoC datasheet would be.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"

#: doc_id -> datasheet PDF URL. doc_id must match the keys in gold.json.
SOURCES: dict[str, str] = {
    "vl53l1x": "https://www.pololu.com/file/0J1506/vl53l1x.pdf",
    "vl53l0x": "https://www.pololu.com/file/0J1187/VL53L0X.pdf",
    "lidarlite-v3": "https://static.garmin.com/pumac/LIDAR_Lite_v3_Operation_Manual_and_Technical_Specifications.pdf",
    "hc-sr04": "https://cdn.sparkfun.com/datasheets/Sensors/Proximity/HCSR04.pdf",
    "tmp117": "https://www.ti.com/lit/ds/symlink/tmp117.pdf",
    "bmp388": "https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmp388-ds001.pdf",
    "bmm150": "https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmm150-ds001.pdf",
    "bme680": "https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bme680-ds001.pdf",
}


def fetch_pdf(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    if not data.startswith(b"%PDF"):
        raise RuntimeError(f"{url} did not return a PDF (got {data[:16]!r})")
    dest.write_bytes(data)


def main() -> int:
    if shutil.which("pdftotext") is None:
        print("error: pdftotext not found on PATH (install poppler-utils)", file=sys.stderr)
        return 1

    for doc_id, url in SOURCES.items():
        pdf = HERE / f"{doc_id}.pdf"
        txt = HERE / f"{doc_id}.txt"
        try:
            print(f"fetching {doc_id} ...", file=sys.stderr)
            fetch_pdf(url, pdf)
            subprocess.run(["pdftotext", "-layout", str(pdf), str(txt)], check=True)
        except Exception as exc:
            # A mirror can rot; report and continue so one dead link doesn't
            # block the rest. Grab the missing datasheet by hand into <doc_id>.txt.
            print(f"  FAILED {doc_id}: {exc}", file=sys.stderr)
            continue
        finally:
            pdf.unlink(missing_ok=True)  # keep only the text; the PDF isn't needed

    have = sorted(p.stem for p in HERE.glob("*.txt"))
    print(f"\n{len(have)}/{len(SOURCES)} datasheets extracted: {', '.join(have)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
