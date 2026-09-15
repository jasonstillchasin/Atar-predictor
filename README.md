# nsw-atar-predictor

Estimate an **NSW ATAR range** from a student's Year 11 and Year 12 school assessment results
(each task's mark and weighting, plus cohort rank), using UAC's published scaling statistics.

This is a **simulator of the official pipeline**, not a model trained on student outcomes
(no public dataset pairs school marks with ATARs):

```
tasks + weights ─► weighted internal mark / school percentile      (assessment.py)
                 ─► statewide course percentile                    (moderation.py)
                 ─► scaled mark per unit, from UAC Table A3        (scaling.py)
                 ─► best 2 English units + best 8 others           (aggregate.py)
                 ─► ATAR, from UAC Table A9                        (atar.py)
Monte Carlo over rank uncertainty, school strength and scaling year (simulate.py)
```

## Quick start

```bash
uv sync
uv run python -m atar_predictor.data.fetch      # UAC scaling reports 2020–2025 -> data/raw
uv run python -m atar_predictor.data.extract    # tables A2, A3, A8, A9 -> data/processed
uv run atar-predict examples/student.yaml       # or --json
uv run pytest
uv run python scripts/build_notebooks.py        # QA, backtest, sensitivity notebooks
```

## Web app

```bash
uv run atar-web        # then open http://127.0.0.1:8765
```

A local page to enter each course's Year 11/12 tasks (mark, out of, weighting, optional task
rank and cohort stats), save and switch between students, and see the estimate update as you
type: the ATAR with its likely range, a histogram of simulated outcomes, and per-course school
position, predicted HSC mark/band and scaled mark. Students are saved as YAML in `students/`
(gitignored; same format as below, so `atar-predict students/<id>.yaml` works too). It binds to
localhost only; don't expose it publicly (student data, UAC data terms).

## Student profile

See [examples/student.yaml](examples/student.yaml). Per course:

| Field | Meaning |
|---|---|
| `name` | UAC Table A3 course name, e.g. `Mathematics Advanced` |
| `cohort_size`, `rank` | Current overall course rank, if known (strongest input) |
| `year11`, `year12` | Lists of tasks |

Per task: `name`, `weight` (% of that year), `mark` (+ `out_of`, default 100; omit `mark` for
tasks not yet sat), and optionally `rank`/`rank_of` or `cohort_mean`/`cohort_sd`.

Top level: `cohort_strength: below | average | above | selective` (omit if unsure; the
range widens).

## Averages

`atar_predictor.averages.compute_averages(profile)` (also in the CLI, `--json` output and the
web page's report card) gives weighted school-mark averages:

| Average | How |
|---|---|
| Subject, per year | Weighted mark over that year's completed tasks |
| Subject, overall | Years combined in proportion to how much of each is complete (two finished years count equally) |
| All subjects, per year / overall | Mean across subjects with a mark; unit-weighted versions also returned |

## How the school percentile is chosen

1. An entered overall course `rank` (if given)
2. Otherwise, the weighted average of task ranks
3. Otherwise, the weighted z-score against task cohort mean/SD
4. Otherwise, marks alone against a default school distribution (flagged, wide range)

Year 12 is the main signal (it is what NESA moderates; **Year 11 results do not count toward
the ATAR**). Year 11 carries weight `0.4 × (share of Year 12 weighting not yet completed)`.

## Key assumptions (all configurable; none can be fitted from public data)

| Parameter | Default | Where |
|---|---|---|
| Year 11 prior weight | 0.4 × remaining Y12 | `AssessmentConfig.alpha_y11` |
| SD of an upcoming task mark | 8 points | `AssessmentConfig.task_sd` |
| Within-school spread vs state | 0.9 | `ModerationConfig.spread_ratio` |
| School strength spread (unknown) | 0.35 z | `ModerationConfig.sigma_school` |
| Cohort strength shifts | −0.4 / 0 / +0.4 / +1.0 z | `moderation.STRENGTH_SHIFT` |
| Scaling years sampled | last 5 | `predict(n_years=5)` |

## Validation

- `tests/test_atar.py`: reproduces Table A9 exactly; matches the 2025 report's Table 3.2 and
  the cubic-spline model (section 3.2.9) against Table A8
- `tests/test_scaling.py`: curves pass through published percentiles and are monotone
- `notebooks/02_backtest.ipynb` / `tests/test_backtest.py`: build from the previous four years,
  predict 2023, 2024 and 2025. Scaled-mark quantiles for courses with 1,000+ students were off
  by 0.41–0.47 marks per unit on average; the aggregate→ATAR curve by 0.13–0.48 ATAR points;
  a student at a fixed statewide percentile in five common courses moved by at most 1.1 ATAR
  points. Single-course quantiles fell inside the training years' range only 46–66% of the time,
  so year sampling somewhat understates course-level drift (small next to rank/school uncertainty).
- Extraction was cross-checked against a second PDF engine (8,937 / 8,938 A3 values matched;
  the remaining one is printed without a decimal in the PDF)
- Manual: compare `atar-predict` P50 with UAC's ATAR Compass for a few profiles

## Limitations

- School strength is unknown without school-level data; it dominates the range.
- Table A3 gives percentiles only at P25/50/75/90/99; below P25 the curve is extrapolated.
- Courses with fewer than 40 students use a normal approximation.
- New courses (Enterprise Computing, Software Engineering) borrow predecessor-course history.
# Atar-predictor
