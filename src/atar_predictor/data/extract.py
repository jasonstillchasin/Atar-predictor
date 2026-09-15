"""Extract appendix tables A2, A3, A8, A9 from UAC scaling report PDFs into tidy CSVs.

    uv run python -m atar_predictor.data.extract

Reads data/raw/scaling-report-{YYYY}-nsw-hsc.pdf, writes data/processed/*.csv.
Both directories are gitignored: UAC does not permit redistributing this data.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

A3_STATS = ["mean", "sd", "max", "p99", "p90", "p75", "p50", "p25"]
A2_BANDS = ["pct_b6", "pct_b5", "pct_b4", "pct_b3", "pct_b2"]
NAME_FIXES = {"English EALD": "English EAL/D"}
APPENDIX_START = 35  # 0-based page index; the appendix begins around page 40 in every year


def _num(cell: str | None) -> float:
    s = (cell or "").replace(",", "").replace("–", "").replace("\n", "").strip()
    try:
        return float(s)
    except ValueError:
        return math.nan


def _name(cell: str | None) -> str:
    name = " ".join((cell or "").split())
    return NAME_FIXES.get(name, name)


def _header(table: list[list[str | None]]) -> str:
    return " ".join((c or "") for row in table[:2] for c in row).replace("\n", " ")


def _appendix_tables(pdf: pdfplumber.PDF):
    for page in pdf.pages[APPENDIX_START:]:
        yield from page.extract_tables()


def extract_a3(pdf: pdfplumber.PDF, year: int) -> pd.DataFrame:
    records, course, number = [], None, None
    for table in _appendix_tables(pdf):
        if "Type of mark" not in _header(table):
            continue
        for row in table:
            label = (row[2] or "").strip().lower()
            if label.startswith("hsc"):
                course, number, mark_type = _name(row[0]), _num(row[1]), "hsc"
            elif label.startswith("sca"):
                mark_type = "scaled"  # continues the course from the previous row, even across pages
            else:
                continue
            stats = dict(zip(A3_STATS, (_num(c) for c in row[3:11])))
            records.append({"year": year, "course": course, "number": number, "mark_type": mark_type, **stats})
    return pd.DataFrame(records)


def extract_a2(pdf: pdfplumber.PDF, year: int) -> pd.DataFrame:
    records = []
    for table in _appendix_tables(pdf):
        if "Median HSC" not in _header(table):
            continue
        for row in table:
            number = _num(row[1])
            if math.isnan(number):
                continue
            bands = dict(zip(A2_BANDS, (_num(c) for c in row[4:9])))
            records.append({
                "year": year,
                "course": _name(row[0]),
                "number": number,
                "median_hsc": _num(row[2]),
                "median_band": (row[3] or "").strip(),
                **bands,
            })
    return pd.DataFrame(records)


def extract_a8_a9(pdf: pdfplumber.PDF, report_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    page = [p for p in pdf.pages if "Table A9" in (p.extract_text() or "")][-1]
    a8, a9 = [], []
    for table in page.extract_tables():
        header_idx = next(
            (i for i, r in enumerate(table) if (r[0] or "").strip() in ("Percentile", "ATAR")), None
        )
        if header_idx is None:
            continue
        header = table[header_idx]
        years = [int(re.search(r"\d{4}", c or "").group()) for c in header[1:]]
        for row in table[header_idx + 1 :]:
            key = _num(row[0])
            for y, cell in zip(years, row[1:]):
                if header[0].strip() == "Percentile":
                    a8.append({"report_year": report_year, "year": y, "percentile": key, "atar": _num(cell)})
                else:
                    a9.append({"report_year": report_year, "year": y, "atar": key, "lowest_aggregate": _num(cell)})
    return pd.DataFrame(a8), pd.DataFrame(a9)


def validate_a3(a3: pd.DataFrame) -> list[str]:
    issues = []
    for (year, course), g in a3.groupby(["year", "course"]):
        types = sorted(g["mark_type"])
        if types != ["hsc", "scaled"]:
            issues.append(f"{year} {course}: mark rows {types}")
            continue
        for _, r in g.iterrows():
            vals = r[A3_STATS].astype(float)
            if ((vals < 0) | (vals > 50)).any():
                issues.append(f"{year} {course} {r.mark_type}: value outside 0-50")
            chain = [r["max"], r.p99, r.p90, r.p75, r.p50, r.p25]
            present = [v for v in chain if not math.isnan(v)]
            if any(a < b for a, b in zip(present, present[1:])):
                issues.append(f"{year} {course} {r.mark_type}: percentiles not monotone {present}")
    return issues


def _latest(df: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    """A8/A9 repeat each year across five reports; keep the value from the newest report."""
    return df.sort_values("report_year").drop_duplicates(key, keep="last").sort_values(key)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    a2s, a3s, a8s, a9s = [], [], [], []
    for path in sorted(RAW.glob("scaling-report-*-nsw-hsc.pdf")):
        year = int(re.search(r"(\d{4})", path.name).group(1))
        with pdfplumber.open(path) as pdf:
            a2s.append(extract_a2(pdf, year))
            a3s.append(extract_a3(pdf, year))
            a8, a9 = extract_a8_a9(pdf, year)
            a8s.append(a8)
            a9s.append(a9)
        print(f"{year}: A2 {len(a2s[-1])} courses, A3 {len(a3s[-1]) // 2} courses, A8 {len(a8)} cells, A9 {len(a9)} cells")

    a3 = pd.concat(a3s, ignore_index=True)
    tables = {
        "a2_hsc_bands": pd.concat(a2s, ignore_index=True),
        "a3_mark_stats": a3,
        "a8_atar_percentiles": _latest(pd.concat(a8s), ["year", "percentile"]),
        "a9_atar_aggregates": _latest(pd.concat(a9s), ["year", "atar"]),
    }
    for name, df in tables.items():
        df.to_csv(PROCESSED / f"{name}.csv", index=False)

    issues = validate_a3(a3)
    print(f"A3 validation: {len(issues)} issue(s)")
    for issue in issues:
        print("  ", issue)


if __name__ == "__main__":
    main()
