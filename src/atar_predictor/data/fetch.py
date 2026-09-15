"""Download UAC scaling reports into data/raw (local, private-study use only).

    uv run python -m atar_predictor.data.fetch [YEAR ...]
"""

from __future__ import annotations

import sys
import urllib.request

from .extract import RAW

URL = "https://uac.edu.au/assets/documents/scaling-reports/scaling-report-{year}-nsw-hsc.pdf"
DEFAULT_YEARS = range(2020, 2026)


def main(years: list[int] | range = DEFAULT_YEARS) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for year in years:
        dest = RAW / f"scaling-report-{year}-nsw-hsc.pdf"
        if dest.exists():
            print(f"{year}: already downloaded")
            continue
        urllib.request.urlretrieve(URL.format(year=year), dest)
        print(f"{year}: saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or DEFAULT_YEARS)
