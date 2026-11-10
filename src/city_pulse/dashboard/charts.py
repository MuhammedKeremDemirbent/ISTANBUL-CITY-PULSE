import pandas as pd  # type: ignore[import-untyped]  # pandas-stubs kurulu değil
import plotly.express as px  # type: ignore[import-untyped]  # plotly tip stub'ı bulunmuyor
import plotly.graph_objects as go  # type: ignore[import-untyped]  # plotly tip stub'ı bulunmuyor


def plot_timeseries(pdf: pd.DataFrame, threshold: float) -> go.Figure:
    """168 saatlik trafik ve yağış eğrisini ferah ve kaydırılabilir çizer."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pdf["timestamp"],
            y=pdf["predicted_traffic_index_av"],
            mode="lines",
            name="Tahmini Trafik Yoğunluğu (%)",
            line={"color": "#2563EB", "width": 3},
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.08)",
            hovertemplate="<b>Tarih/Saat:</b> %{x|%d %b %H:%M}<br><b>Yoğunluk:</b> %{y:.1f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=pdf["timestamp"],
            y=pdf["precipitation"],
            name="Beklenen Yağış (mm)",
            yaxis="y2",
            opacity=0.35,
            marker_color="#0284C7",
            hovertemplate="<b>Yağış:</b> %{y:.1f} mm<extra></extra>",
        )
    )

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text=f"Kritik Yoğunluk Eşiği (%{threshold:.0f})",
        annotation_position="top right",
    )

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin={"l": 40, "r": 40, "t": 30, "b": 30},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 12},
        },
        yaxis={
            "title": "Trafik İndeksi (0 - 100)",
            "range": [0, 100],
            "gridcolor": "#F1F5F9",
        },
        yaxis2={
            "title": "Yağış (mm)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "range": [0, max(pdf["precipitation"].max() * 3.5, 6)],
        },
        xaxis={
            "title": "Zaman Akışı (İstediğiniz aralığı alttaki kaydırıcıdan seçebilirsiniz)",
            "showgrid": True,
            "gridcolor": "#F1F5F9",
            "rangeslider": {"visible": True, "thickness": 0.06},
        },
    )
    return fig


def plot_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """Tüm ekranı ferah kullanan Gün & Saat ısı haritası."""
    fig = px.imshow(
        matrix,
        labels={"x": "Günün Saati", "y": "Gün", "color": "Trafik (%)"},
        x=[f"{h:02d}:00" for h in range(24)],
        y=matrix.index.to_list(),
        color_continuous_scale="Reds",
        aspect="auto",
        text_auto=".0f",
    )
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis={"tickangle": 0, "side": "bottom", "gridcolor": "#E2E8F0"},
        yaxis={"gridcolor": "#E2E8F0"},
        coloraxis_colorbar={"title": "Yoğunluk (%)", "thickness": 15},
    )
    return fig


def plot_weather_impact(impact_df: pd.DataFrame) -> go.Figure:
    """Yağışlı vs Kuru havanın trafik etkisi çubuk grafiği."""
    fig = px.bar(
        impact_df,
        x="condition",
        y="avg_traffic",
        color="condition",
        color_discrete_map={
            "Yağmurlu Hava": "#0284C7",
            "Açık/Kuru Hava": "#F59E0B",
        },
        text=impact_df["avg_traffic"].apply(lambda v: f"%{v:.1f}"),
        labels={"condition": "Hava Koşulu", "avg_traffic": "Ortalama İndeks"},
    )
    fig.update_layout(
        template="plotly_white",
        height=380,
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        yaxis={
            "range": [0, 100],
            "title": "Ortalama Trafik İndeksi (%)",
            "gridcolor": "#F1F5F9",
        },
    )
    return fig
