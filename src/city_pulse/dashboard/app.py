import sys
from datetime import datetime
from pathlib import Path

# Proje kökündeki 'src' dizinini Python yoluna ekle
SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import polars as pl
import streamlit as st

from city_pulse.dashboard.charts import (
    plot_heatmap,
    plot_timeseries,
    plot_weather_impact,
)
from city_pulse.dashboard.data import (
    DAY_NAMES,
    compute_heatmap_matrix,
    compute_kpis,
    compute_weather_impact,
    filter_by_day,
    load_latest_forecast,
)

# 1. Sayfa Ayarları
st.set_page_config(
    page_title="Istanbul City Pulse | Traffic Forecast",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Veri Yükleme
try:
    df, filename = load_latest_forecast()
except FileNotFoundError:
    st.info(
        "⏳ **Henüz Canlı Tahmin Üretilmedi.**\n\n"
        "Sistem Airflow zamanlayıcısına bırakıldı. İlk tahmin bu gece **03:00**'te "
        "otomatik olarak üretilip buraya yansıyacaktır."
    )
    st.stop()

# 3. Sidebar (Filtreler ve Kontrol)
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Istanbul_Metropolitan_Municipality_logo.svg/512px-Istanbul_Metropolitan_Municipality_logo.svg.png",
        width=130,
    )
    st.markdown("### ⚙️ Filtreler ve Ayarlar")
    st.caption(f"📁 **Aktif Veri Dosyası:**\n`{filename}`")

    day_options = ["Tüm Hafta"] + list(DAY_NAMES.values())
    selected_day = st.selectbox("📅 Gün Filtresi", day_options)

    traffic_threshold = float(
        st.slider(
            "🚨 Kritik Yoğunluk Eşiği",
            min_value=30,
            max_value=80,
            value=50,
            step=5,
            help="Bu değerin üzerindeki saatler grafiklerde kritik olarak vurgulanır.",
        )
    )

    st.markdown("---")
    st.markdown(
        """
        **Mimari Katmanları:**
        - 📥 **Ingestion:** İBB Açık Veri & Open-Meteo
        - ⚙️ **Dönüşüm:** dbt & DuckDB
        - 🤖 **MLOps:** LightGBM & MLflow
        - ⏱️ **Orkestrasyon:** Apache Airflow
        """
    )

# 4. Veriyi Filtrele ve Metrikleri Hesapla
filtered_df = filter_by_day(df, selected_day)
kpis = compute_kpis(filtered_df, traffic_threshold)

# 5. Başlık ve KPI Kartları
st.title("🚗 Istanbul City Pulse — Canlı Trafik Tahmin Paneli")
st.markdown(
    "Open-Meteo hava tahminleri ve LightGBM makine öğrenimi modeliyle üretilen **168 saatlik (7 gün)** şehir içi trafik projeksiyonu."
)

# Ferah Metrik Kutuları
with st.container():
    c1, c2, c3, c4 = st.columns(4)
    with c1, st.container(border=True):
        st.metric(
            "🔥 En Yoğun Saat",
            f"% {kpis.max_traffic_val:.1f}",
            delta=f"{kpis.max_traffic_time.strftime('%d %b %H:%M')}",
            delta_color="inverse",
        )
    with c2, st.container(border=True):
        st.metric(
            "📊 Beklenen Ortalama Trafik",
            f"% {kpis.avg_traffic_val:.1f}",
        )
    with c3, st.container(border=True):
        st.metric(
            "🌧️ Yağışlı Saat Sayısı",
            f"{kpis.rain_hours} Saat",
            delta=f"{kpis.total_rain_mm:.1f} mm yağış",
        )
    with c4, st.container(border=True):
        st.metric(
            f"🚨 Kritik Saatler (≥ %{traffic_threshold:.0f})",
            f"{kpis.critical_hours_count} Saat",
            delta="Riskli" if kpis.critical_hours_count > 12 else "Normal",
            delta_color="inverse" if kpis.critical_hours_count > 12 else "normal",
        )

st.write("")

# 6. Sekmeli Görünüm (Modern SaaS Yapısı)
tab_trend, tab_heatmap, tab_weather, tab_data = st.tabs(
    [
        "📈 168 Saatlik Trafik Eğrisi",
        "🗓️ Gün & Saat Isı Haritası (Heatmap)",
        "🌧️ Meteorolojik Etki & Pik Saatler",
        "🔍 Ham Veri & Dışa Aktar",
    ]
)

# TAB 1: ZAMAN SERİSİ
with tab_trend:
    st.markdown("#### 7 Günlük Kesintisiz Trafik ve Yağış Projeksiyonu")
    st.caption(
        "Alttaki aralık çubuğunu (Range Slider) sürükleyerek dilediğiniz gün ve saat dilimine yakınlaşabilirsiniz."
    )
    fig_ts = plot_timeseries(filtered_df.to_pandas(), traffic_threshold)
    st.plotly_chart(fig_ts, use_container_width=True)

# TAB 2: ISI HARİTASI
with tab_heatmap:
    st.markdown("#### Haftalık Yoğunluk Matrisi (00:00 — 23:00)")
    st.caption(
        "Kutuların içindeki sayılar o saatteki ortalama trafik yoğunluk yüzdesini gösterir."
    )
    fig_heat = plot_heatmap(compute_heatmap_matrix(df))
    st.plotly_chart(fig_heat, use_container_width=True)

# TAB 3: HAVA DURUMU & PİK SAATLER
with tab_weather:
    col_w1, col_w2 = st.columns([5, 5])
    with col_w1:
        st.markdown("#### 🌧️ Yağışın Trafik Üzerindeki Etkisi")
        st.plotly_chart(
            plot_weather_impact(compute_weather_impact(df)),
            use_container_width=True,
        )

    with col_w2:
        st.markdown("#### 🚨 Önümüzdeki En Yoğun 5 Pik Saat")
        top_peaks = (
            df.sort("predicted_traffic_index_av", descending=True)
            .head(5)
            .select(
                [
                    pl.col("timestamp").alias("Zaman"),
                    pl.col("predicted_traffic_index_av").alias("Trafik İndeksi (%)"),
                    pl.col("precipitation").alias("Yağış (mm)"),
                    pl.col("temperature_2m").alias("Sıcaklık (°C)"),
                ]
            )
            .to_pandas()
        )
        st.dataframe(top_peaks, use_container_width=True, hide_index=True)

# TAB 4: HAM VERİ VE İNDİRME
with tab_data:
    st.markdown("#### Tahmin Veri Seti Detayı")
    cols = [
        "timestamp",
        "predicted_traffic_index_av",
        "temperature_2m",
        "precipitation",
        "wind_speed_10m",
        "is_weekend",
        "is_holiday",
    ]
    table_df = filtered_df.select(cols).to_pandas()
    st.dataframe(table_df, use_container_width=True)

    csv_data = table_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Filtrelenmiş Veriyi CSV Olarak İndir",
        csv_data,
        f"istanbul_traffic_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv",
    )
