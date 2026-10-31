"""Domain modellerinin ve hedef kolonların doğruluğunu test eder."""

from city_pulse.domain import (
    FEATURE_COLUMNS,
    ISTANBUL_TIMEZONE,
    TARGET_COLUMN,
    TRAFFIC_COLUMNS,
)


def test_istanbul_timezone_is_correct() -> None:
    """Zaman diliminin Türkiye saati olduğunu doğrula."""
    assert ISTANBUL_TIMEZONE == "Europe/Istanbul"


def test_target_column_is_valid() -> None:
    """Hedef kolonun Avrupa yakası olduğunu ve hedef listesinde var olduğunu doğrula."""
    assert TARGET_COLUMN == "traffic_index_av"
    assert TARGET_COLUMN in TRAFFIC_COLUMNS.values()  


def test_features_do_not_contain_target() -> None:
    """Model girdilerinin (features) içine yanlışlıkla tahmin edilecek hedef (target) karışmış mı?"""
    # Veri sızıntısını (Data Leakage) önlemek için hedef değişken girdilerde OLMAMALIDIR!
    assert TARGET_COLUMN not in FEATURE_COLUMNS