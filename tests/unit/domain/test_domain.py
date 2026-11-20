"""Domain modellerinin, sabitlerinin ve hedef kolonlarinin dogrulugunu test eder."""

import pytest

from city_pulse.domain import (
    FEATURE_COLUMNS,
    ISTANBUL_TIMEZONE,
    TARGET_COLUMN,
    TRAFFIC_COLUMNS,
)


@pytest.mark.unit
def test_istanbul_timezone_is_correct() -> None:
    """Zaman diliminin Turkiye saati oldugunu dogrula."""
    assert ISTANBUL_TIMEZONE == "Europe/Istanbul"


@pytest.mark.unit
def test_target_column_is_valid() -> None:
    """Hedef kolonun Avrupa yakasi oldugunu ve hedef listesinde var oldugunu dogrula."""
    assert TARGET_COLUMN == "traffic_index_av"
    assert TARGET_COLUMN in TRAFFIC_COLUMNS.values()


@pytest.mark.unit
def test_features_do_not_contain_target() -> None:
    """Model girdilerinin (features) icine yanlislikla hedef (target) karismis mi?

    Veri sizintisini (Data Leakage) onlemek icin hedef degisken girdilerde OLMAMALIDIR!
    """
    assert TARGET_COLUMN not in FEATURE_COLUMNS
