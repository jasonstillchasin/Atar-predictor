import pytest

from atar_predictor.data.extract import PROCESSED


def _require_data():
    if not (PROCESSED / "a3_mark_stats.csv").exists():
        pytest.skip("UAC data not extracted (it is not committed): run python -m atar_predictor.data.extract")


@pytest.fixture(scope="session")
def scaling():
    _require_data()
    from atar_predictor.scaling import ScalingData

    return ScalingData.load()


@pytest.fixture(scope="session")
def converter():
    _require_data()
    from atar_predictor.atar import AtarConverter

    return AtarConverter.load()
