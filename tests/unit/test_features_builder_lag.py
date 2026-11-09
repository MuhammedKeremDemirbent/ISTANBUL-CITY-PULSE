"""_attach_lag_features fonksiyonu için birim testler.

Zaman anahtarlı eşlemenin neden satır kaydırmadan (shift) daha doğru olduğu:
shift(N) "N satır önceki değeri al" demektir; tabloda saat boşlukları varsa
boşluktan sonraki satır yanlış zamandaki değeri taşır. Zaman anahtarlı
join ise "timestamp - N saat değerine sahip satır varsa onun değerini al,
yoksa NULL bırak" demektir — veri gerçeğini bozmuş olmaz.
"""

from datetime import datetime, timedelta

import polars as pl
import pytest

from city_pulse.features.builder import FeatureSchemaError, _attach_lag_features
from city_pulse.features.columns import TARGET_COLUMN, TIMESTAMP_COLUMN

# ---------------------------------------------------------------------------
# Yardımcı fonksiyon
# ---------------------------------------------------------------------------


def _make_frame(hours: list[int]) -> pl.DataFrame:
    """Verilen saat listesinden saatlik trafik tablosu üretir.

    Hedef değer saat indeksine eşit seçilir; böylece lag'in hangi saate
    baktığı doğrudan sayıdan okunabilir (ör. saat 7 → değer 7.0).
    Başlangıç: 2024-01-01 00:00 + N saat.
    """
    base = datetime(2024, 1, 1, 0, 0)
    timestamps = [base + timedelta(hours=h) for h in hours]
    return pl.DataFrame(
        {
            TIMESTAMP_COLUMN: timestamps,
            TARGET_COLUMN: [float(h) for h in hours],
        }
    ).with_columns(pl.col(TIMESTAMP_COLUMN).cast(pl.Datetime))


# ---------------------------------------------------------------------------
# Test: lag_1h — önceki saat tabloda yoksa NULL
# ---------------------------------------------------------------------------


def test_lag_1h_is_null_when_previous_hour_missing() -> None:
    """Tabloda saat boşluğu olduğunda lag_1h NULL olmalı.

    Saatler: [0, 1, 2, 5]
    Beklenen lag_1h: [None, 0.0, 1.0, None]
      - saat 0: 1 saat öncesi (-1) tabloda yok → None
      - saat 1: 1 saat öncesi (0) tabloda var  → 0.0
      - saat 2: 1 saat öncesi (1) tabloda var  → 1.0
      - saat 5: 1 saat öncesi (4) tabloda yok  → None
    """
    frame = _make_frame([0, 1, 2, 5])
    result = _attach_lag_features(frame).sort(TIMESTAMP_COLUMN)

    lag_1h = result["traffic_index_av_lag_1h"].to_list()
    assert lag_1h[0] is None, "saat 0: öncesi yok → None bekleniyor"
    assert lag_1h[1] == pytest.approx(0.0), "saat 1: öncesi saat 0 → 0.0"
    assert lag_1h[2] == pytest.approx(1.0), "saat 2: öncesi saat 1 → 1.0"
    assert lag_1h[3] is None, "saat 5: öncesi (saat 4) tabloda yok → None"


# ---------------------------------------------------------------------------
# Test: lag_24h — satır ofseti değil gerçek zaman farkı esas alınmalı
# ---------------------------------------------------------------------------


def test_lag_24h_uses_real_time_not_row_offset() -> None:
    """lag_24h satır kaydırmasına değil gerçek zaman farkına göre hesaplanmalı.

    Tablo: 0..35 saatleri, ancak saat 10 eksik.
    shift(24) saati 10'u çıkardığı için saat 25 için lag'i saat 1'den alır (yanlış).
    Zaman join'i ise saat 25 için 25-24=1 saatini arar: tabloda var → 1.0 döner.

    Saat 34 için: 34-24=10, tabloda yok → None bekleniyor.
    """
    all_hours = list(range(36))
    all_hours.remove(10)  # saat 10'u çıkar
    frame = _make_frame(all_hours)
    result = _attach_lag_features(frame).sort(TIMESTAMP_COLUMN)

    # saat 24 → lag_24h = değer(0) = 0.0
    row_24 = result.filter(
        pl.col(TIMESTAMP_COLUMN) == datetime(2024, 1, 1, 0, 0) + timedelta(hours=24)
    )
    assert row_24["traffic_index_av_lag_24h"][0] == pytest.approx(0.0), (
        "saat 24: lag_24h = değer(saat 0) = 0.0"
    )

    # saat 25 → lag_24h = değer(1) = 1.0
    row_25 = result.filter(
        pl.col(TIMESTAMP_COLUMN) == datetime(2024, 1, 1, 0, 0) + timedelta(hours=25)
    )
    assert row_25["traffic_index_av_lag_24h"][0] == pytest.approx(1.0), (
        "saat 25: lag_24h = değer(saat 1) = 1.0"
    )

    # saat 34 → lag_24h = değer(10), saat 10 tabloda yok → None
    row_34 = result.filter(
        pl.col(TIMESTAMP_COLUMN) == datetime(2024, 1, 1, 0, 0) + timedelta(hours=34)
    )
    assert row_34["traffic_index_av_lag_24h"][0] is None, (
        "saat 34: lag_24h = değer(saat 10), tabloda yok → None"
    )


# ---------------------------------------------------------------------------
# Test: satır sayısı değişmemeli
# ---------------------------------------------------------------------------


def test_row_count_unchanged() -> None:
    """_attach_lag_features satır eklememeli veya silmemeli."""
    frame = _make_frame([0, 1, 2, 5, 10, 24, 25])
    result = _attach_lag_features(frame)
    assert result.height == frame.height


# ---------------------------------------------------------------------------
# Test: tekrar eden timestamp FeatureSchemaError fırlatmalı
# ---------------------------------------------------------------------------


def test_duplicate_timestamp_raises() -> None:
    """Aynı timestamp iki kez geçiyorsa join satır çoğaltır; hata fırlatılmalı."""
    base = datetime(2024, 1, 1, 0, 0)
    frame = pl.DataFrame(
        {
            TIMESTAMP_COLUMN: [base, base, base + timedelta(hours=1)],
            TARGET_COLUMN: [1.0, 2.0, 3.0],
        }
    ).with_columns(pl.col(TIMESTAMP_COLUMN).cast(pl.Datetime))

    with pytest.raises(FeatureSchemaError, match="tekrar"):
        _attach_lag_features(frame)
