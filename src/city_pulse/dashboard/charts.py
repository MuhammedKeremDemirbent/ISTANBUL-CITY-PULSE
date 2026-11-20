import pandas as pd  # type: ignore[import-untyped]
import plotly.express as px  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]


def plot_timeseries(pdf: pd.DataFrame, threshold: float) -> go.Figure:
    """Plot 168-hour traffic and precipitation interactive time-series."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=pdf["timestamp"],
            y=pdf["predicted_traffic_index_av"],
            mode="lines",
            name="Predicted Traffic Index (%)",
            line={"color": "#2563EB", "width": 3},
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.08)",
            hovertemplate="<b>Date/Time:</b> %{x|%d %b %H:%M}<br><b>Index:</b> %{y:.1f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=pdf["timestamp"],
            y=pdf["precipitation"],
            name="Expected Precipitation (mm)",
            yaxis="y2",
            opacity=0.35,
            marker_color="#0284C7",
            hovertemplate="<b>Precipitation:</b> %{y:.1f} mm<extra></extra>",
        )
    )

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text=f"Critical Threshold ({threshold:.0f}%)",
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
            "title": "Traffic Index (0 - 100)",
            "range": [0, 100],
            "gridcolor": "#F1F5F9",
        },
        yaxis2={
            "title": "Precipitation (mm)",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "range": [0, max(pdf["precipitation"].max() * 3.5, 6)],
        },
        xaxis={
            "title": "Timeline (Select range using slider below)",
            "showgrid": True,
            "gridcolor": "#F1F5F9",
            "rangeslider": {"visible": True, "thickness": 0.06},
        },
    )
    return fig


def plot_heatmap(matrix: pd.DataFrame) -> go.Figure:
    """Plot Day & Hour traffic intensity heatmap."""
    fig = px.imshow(
        matrix,
        labels={"x": "Hour of Day", "y": "Day", "color": "Traffic (%)"},
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
        coloraxis_colorbar={"title": "Index (%)", "thickness": 15},
    )
    return fig


def plot_weather_impact(impact_df: pd.DataFrame) -> go.Figure:
    """Plot bar chart comparing traffic during rainy vs dry weather."""
    fig = px.bar(
        impact_df,
        x="condition",
        y="avg_traffic",
        color="condition",
        color_discrete_map={
            "Rainy Weather": "#0284C7",
            "Clear/Dry Weather": "#F59E0B",
        },
        text=impact_df["avg_traffic"].apply(lambda v: f"%{v:.1f}"),
        labels={"condition": "Weather Condition", "avg_traffic": "Average Index"},
    )
    fig.update_layout(
        template="plotly_white",
        height=380,
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        yaxis={
            "range": [0, 100],
            "title": "Average Traffic Index (%)",
            "gridcolor": "#F1F5F9",
        },
    )
    return fig
