import numpy as np
import pytest

from atar_predictor.aggregate import best_ten, best_ten_batch
from atar_predictor.courses import course_info


def infos(names):
    enrolled = set(names)
    return [course_info(n, enrolled) for n in names]


def test_unit_rules():
    assert course_info("Mathematics Extension 1").units == 1
    assert course_info("Mathematics Extension 1", {"Mathematics Extension 2"}).units == 2
    assert course_info("Mathematics Extension 2").units == 2
    assert course_info("English Extension 1").english
    assert course_info("Physics").units == 2
    assert course_info("History Extension").units == 1
    assert course_info("Studies of Religion I").units == 1


def test_english_counted_even_when_lowest():
    names = ["English Standard", "Physics", "Chemistry", "Mathematics Advanced", "Economics", "Legal Studies"]
    scaled = [15, 40, 40, 40, 40, 40]  # 12 units: English must count, one 2-unit course drops
    r = best_ten(infos(names), scaled)
    assert r.aggregate == pytest.approx(2 * 15 + 8 * 40)
    assert {c.course for c in r.contributions} >= {"English Standard"}
    assert r.eligible


def test_english_extension_can_fill_english_requirement():
    names = ["English Advanced", "English Extension 1", "Physics", "Chemistry", "Mathematics Advanced", "Economics"]
    scaled = [30, 45, 40, 40, 40, 40]
    r = best_ten(infos(names), scaled)
    # English slots: Ext1 45 + Adv 30 -> best 2 English = 75; rest best 8 of {Adv 30, 40 x8} = 320
    assert r.aggregate == pytest.approx(75 + 320)


def test_partial_course_inclusion():
    names = ["English Advanced", "Physics", "Chemistry", "Mathematics Advanced", "Mathematics Extension 1", "Economics"]
    scaled = [30, 40, 38, 42, 44, 20]  # 11 units: the last unit counted is one unit of Economics
    r = best_ten(infos(names), scaled)
    econ = next(c for c in r.contributions if c.course == "Economics")
    assert econ.units_used == 1
    assert r.aggregate == pytest.approx(60 + 80 + 76 + 84 + 44 + 20)


def test_batch_matches_scalar():
    names = ["English Advanced", "English Extension 1", "Mathematics Extension 1", "Mathematics Extension 2",
             "Physics", "Chemistry", "Economics"]
    ci = infos(names)
    rng = np.random.default_rng(0)
    draws = rng.uniform(0, 50, size=(200, len(names)))
    batch = best_ten_batch(ci, draws)
    scalar = [best_ten(ci, list(row)).aggregate for row in draws]
    assert batch == pytest.approx(scalar)


def test_ineligible_pattern():
    r = best_ten(infos(["English Advanced", "Physics", "Chemistry"]), [30, 30, 30])
    assert not r.eligible
    assert "fewer than 10 units of ATAR courses" in r.reasons
