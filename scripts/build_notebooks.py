"""Build and execute the analysis notebooks.

    uv run python scripts/build_notebooks.py

Notebooks read the locally extracted UAC data, so their outputs contain UAC figures:
keep executed notebooks out of any public repository.
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

SETUP = """\
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from atar_predictor.data.extract import PROCESSED
pd.set_option("display.width", 140)
plt.rcParams["figure.figsize"] = (9, 4.5)
"""

NOTEBOOKS = {
    "01_extraction_qa.ipynb": [
        ("md", "# 01 · Extraction QA\nChecks the tables extracted from UAC scaling reports (A2, A3, A8, A9)."),
        ("code", SETUP),
        ("code", """\
a2 = pd.read_csv(PROCESSED / "a2_hsc_bands.csv")
a3 = pd.read_csv(PROCESSED / "a3_mark_stats.csv")
a8 = pd.read_csv(PROCESSED / "a8_atar_percentiles.csv")
a9 = pd.read_csv(PROCESSED / "a9_atar_aggregates.csv")
pd.DataFrame({
    "A2 courses": a2.groupby("year").size(),
    "A3 courses": a3.groupby("year").size() // 2,
    "A3 without percentiles": a3[a3.p50.isna()].groupby("year").size() // 2,
}).fillna(0).astype(int)"""),
        ("code", """\
from atar_predictor.data.extract import validate_a3
issues = validate_a3(a3)
print(f"{len(issues)} validation issue(s)", *issues, sep="\\n")"""),
        ("md", "## Scaled-mark curves by year\nThe curve used by the simulator passes through every published percentile."),
        ("code", """\
from atar_predictor.scaling import ScalingData
scaling = ScalingData.load()
ps = np.linspace(0, 1, 201)
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, course in zip(axes, ["English Advanced", "Mathematics Advanced", "Business Studies"]):
    for year in scaling.years:
        ax.plot(ps * 100, scaling.curve(course, year)(ps), label=year)
    ax.set_title(course); ax.set_xlabel("Course percentile"); ax.grid(alpha=.3)
axes[0].set_ylabel("Scaled mark per unit"); axes[0].legend()
plt.tight_layout()"""),
        ("md", "## Aggregate → ATAR (Table A9)"),
        ("code", """\
for year, g in a9.groupby("year"):
    plt.plot(g.lowest_aggregate, g.atar, marker=".", label=year)
plt.xlabel("Lowest aggregate"); plt.ylabel("ATAR"); plt.legend(ncol=2); plt.grid(alpha=.3)"""),
    ],
    "02_backtest.ipynb": [
        ("md", "# 02 · Temporal backtest\nBuild from the four previous years, predict the next year, "
               "for 2023, 2024 and 2025. Scaled-mark errors are per unit (out of 50)."),
        ("code", SETUP),
        ("code", """\
from atar_predictor.backtest import scaled_quantile_backtest, atar_curve_backtest, year_drift_backtest
from atar_predictor.scaling import ScalingData
from atar_predictor.atar import AtarConverter
a3 = pd.read_csv(PROCESSED / "a3_mark_stats.csv")
a9 = pd.read_csv(PROCESSED / "a9_atar_aggregates.csv")
scaling, converter = ScalingData.load(), AtarConverter.load()
TESTS = {t: [y for y in range(t - 4, t) if y >= 2020] for t in (2023, 2024, 2025)}"""),
        ("md", "## Scaled-mark quantiles (courses with ≥ 1,000 students)\n"
               "`in_train_range`: inside the min–max of training years (what year sampling covers; "
               "with 3–4 years, ~60–75% is expected even for a perfect model). `in_normal_80`: inside mean ± 1.28 SD."),
        ("code", """\
rows = []
for test, train in TESTS.items():
    q = scaled_quantile_backtest(a3, train, test)
    big = q[q.course.isin(a3[(a3.year == test) & (a3.number >= 1000)].course)]
    rows.append({"test_year": test, "n": len(big), "MAE": big.error.abs().mean(),
                 "bias": big.error.mean(), "in_train_range": big.in_train_range.mean(),
                 "in_normal_80": big.in_normal_80.mean()})
pd.DataFrame(rows).round(3)"""),
        ("code", """\
q = scaled_quantile_backtest(a3, TESTS[2025], 2025)
q.assign(abs_error=q.error.abs()).sort_values("abs_error", ascending=False).head(10).round(2)"""),
        ("md", "## Aggregate → ATAR curve drift"),
        ("code", """\
fig, ax = plt.subplots()
for test, train in TESTS.items():
    c = atar_curve_backtest(a9, train, test)
    ax.plot(c["aggregate"], c["error"], label=f"{test} (MAE {c['error'].abs().mean():.2f})")
ax.axhline(0, color="k", lw=.8); ax.set_xlabel("Aggregate"); ax.set_ylabel("Actual − pooled ATAR")
ax.legend(); ax.grid(alpha=.3)"""),
        ("md", "## End to end: same statewide percentile in five common courses"),
        ("code", """\
COURSES = ["English Advanced", "Mathematics Advanced", "Physics", "Chemistry", "Economics"]
pd.concat({t: year_drift_backtest(scaling, converter, COURSES, [.5, .7, .8, .9, .95, .99], train, t)
           for t, train in TESTS.items()}, names=["test_year"]).round(2)"""),
    ],
    "03_demo_and_sensitivity.ipynb": [
        ("md", "# 03 · Demo and sensitivity\nHow much each assumption moves the example student's ATAR range."),
        ("code", SETUP),
        ("code", """\
from pathlib import Path
from atar_predictor.models import StudentProfile
from atar_predictor.simulate import predict
from atar_predictor.scaling import ScalingData
from atar_predictor.atar import AtarConverter
from atar_predictor.assessment import AssessmentConfig
from atar_predictor.moderation import ModerationConfig
scaling, converter = ScalingData.load(), AtarConverter.load()
profile = StudentProfile.from_yaml(Path("..") / "examples" / "student.yaml")
base = predict(profile, scaling, converter)
print(f"ATAR {base.atar_p50} (80% range {base.atar_p10}–{base.atar_p90}), aggregate {base.aggregate_p50:.1f}")
plt.hist(base.atar_draws, bins=60); plt.xlabel("ATAR"); plt.ylabel("draws")
for v in (base.atar_p10, base.atar_p50, base.atar_p90): plt.axvline(v, color="k", ls="--", lw=.8)"""),
        ("code", """\
def run(label, prof=profile, **kw):
    p = predict(prof, scaling, converter, draws=3000, **kw)
    return {"scenario": label, "p10": p.atar_p10, "p50": p.atar_p50, "p90": p.atar_p90}

rows = [run("baseline (strength unknown)")]
for s in ("below", "average", "above", "selective"):
    rows.append(run(f"cohort_strength={s}", profile.model_copy(update={"cohort_strength": s})))
for r in (0.8, 1.0):
    rows.append(run(f"spread_ratio={r}", mod_cfg=ModerationConfig(spread_ratio=r)))
for s in (0.2, 0.5):
    rows.append(run(f"sigma_school={s}", mod_cfg=ModerationConfig(sigma_school=s)))
for n in (1, 3):
    rows.append(run(f"n_years={n}", n_years=n))
table = pd.DataFrame(rows).set_index("scenario")
table"""),
        ("code", """\
ax = table.p50.plot(kind="barh", xerr=[table.p50 - table.p10, table.p90 - table.p50], capsize=3, color="#7aa6c2")
ax.invert_yaxis(); ax.set_xlabel("ATAR (P50 with P10–P90)"); ax.grid(axis="x", alpha=.3)"""),
        ("md", "## Range narrows as Year 12 tasks are completed\nEach course keeps the same evidence type "
               "(entered rank, task ranks, z-scores or marks only); only the later Year 12 tasks are "
               "progressively marked as not yet sat."),
        ("code", """\
def blank_after(prof, keep):
    courses = []
    for c in prof.courses:
        tasks = [t if i < keep else t.model_copy(update={"mark": None, "rank": None, "rank_of": None,
                                                            "cohort_mean": None, "cohort_sd": None})
                 for i, t in enumerate(c.year12)]
        courses.append(c.model_copy(update={"year12": tasks}))
    return prof.model_copy(update={"courses": courses})

pd.DataFrame([run(f"first {k} Y12 task(s) done", blank_after(profile, k)) for k in (1, 2, 3)]).set_index("scenario")"""),
    ],
}


def build() -> None:
    NB.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src)
            for kind, src in cells
        ]
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(NB)}}).execute()
        nbformat.write(nb, NB / name)
        print(f"executed {name}")


if __name__ == "__main__":
    build()
