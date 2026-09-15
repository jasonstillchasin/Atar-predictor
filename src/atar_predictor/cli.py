"""atar-predict PROFILE.yaml: print an estimated ATAR range for a student profile."""

from __future__ import annotations

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .averages import averages_from
from .models import StudentProfile
from .report import prediction_to_dict
from .simulate import predict

app = typer.Typer(add_completion=False)
console = Console()


def _atar(x: float) -> str:
    return "30 or less" if x <= 30 else f"{x:.2f}"


def _pct(x: float | None) -> str:
    # Round halves up (79.25 -> 79.3), matching the web page, rather than Python's half-to-even.
    return "–" if x is None else str(Decimal(str(x)).quantize(Decimal("0.1"), ROUND_HALF_UP))


@app.command()
def main(
    profile_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="Student profile YAML"),
    draws: int = typer.Option(5000, help="Monte Carlo draws"),
    seed: int = typer.Option(0),
    years: int = typer.Option(5, help="How many recent UAC scaling years to sample from"),
    as_json: bool = typer.Option(False, "--json", help="Print JSON instead of tables"),
) -> None:
    profile = StudentProfile.from_yaml(profile_path)
    pred = predict(profile, draws=draws, seed=seed, n_years=years)
    if as_json:
        print(json.dumps(prediction_to_dict(pred), indent=2))
        return

    console.print(
        f"\n[bold]Estimated ATAR {_atar(pred.atar_p50)}[/bold]  "
        f"(80% range {_atar(pred.atar_p10)} – {_atar(pred.atar_p90)})   "
        f"aggregate {pred.aggregate_p50:.1f}/500   scaling years {pred.years[0]}–{pred.years[-1]}\n"
    )

    assess = Table(title="School assessment")
    for col in ("Course", "Y11 weighted %", "Y12 weighted %", "Y12 done", "School percentile", "From"):
        assess.add_column(col, justify="left" if col == "Course" else "right")
    for c in pred.courses:
        a = c.assessment
        assess.add_row(
            c.info.name,
            _pct(a.y11 and a.y11.weighted_mark),
            _pct(a.y12 and a.y12.weighted_mark),
            "–" if a.y12 is None else f"{a.y12.completed_fraction:.0%}",
            f"{a.rank_percentile:.0%} ± {a.rank_percentile_sd:.0%}",
            a.rank_source,
        )
    console.print(assess)

    avg = averages_from([(c.info, c.assessment) for c in pred.courses])
    averages = Table(title="Averages (weighted school marks)")
    for col in ("Subject", "Year 11 %", "Year 12 %", "Overall %"):
        averages.add_column(col, justify="left" if col == "Subject" else "right")
    for s in avg.subjects:
        averages.add_row(s.course, _pct(s.year11), _pct(s.year12), _pct(s.overall))
    averages.add_section()
    averages.add_row("[bold]All subjects[/bold]", _pct(avg.year11), _pct(avg.year12), _pct(avg.overall))
    averages.add_row(
        "[dim]weighted by units[/dim]",
        _pct(avg.year11_unit_weighted),
        _pct(avg.year12_unit_weighted),
        _pct(avg.overall_unit_weighted),
    )
    console.print(averages)

    units = {c.course: c.units_used for c in pred.median_breakdown.contributions}
    scaled = Table(title="Predicted HSC and scaling (median draw)")
    for col in ("Course", "Units", "State percentile", "HSC mark", "Band", "Scaled / unit (P10–P90)", "Units counted"):
        scaled.add_column(col, justify="left" if col == "Course" else "right")
    for c in pred.courses:
        scaled.add_row(
            c.info.name + (" *" if c.approximate_curve else ""),
            str(c.info.units),
            f"{c.state_percentile_p50:.0%}",
            str(c.hsc_mark_p50),
            c.band_p50,
            f"{c.scaled_p50:.1f} ({c.scaled_p10:.1f}–{c.scaled_p90:.1f})",
            str(units.get(c.info.name, 0)),
        )
    console.print(scaled)
    if any(c.approximate_curve for c in pred.courses):
        console.print("* small course: no published percentiles, normal approximation used")

    for w in pred.warnings:
        console.print(f"[yellow]! {w}[/yellow]")
    console.print(
        "\n[dim]Estimate from UAC's published scaling statistics under stated assumptions. "
        "Not an official ATAR.[/dim]"
    )


if __name__ == "__main__":
    app()
