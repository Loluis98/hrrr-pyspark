"""
HRRR Gold Layer — Interactive Weather Dashboard
================================================
A Dash app that reads directly from the gold/daily_capitals Delta table
(via pyarrow + parquet) and exposes a polished, dynamic UI for your team.

Controls:
  • City dropdown  – pick any US state capital in the gold table
  • Date range slider – narrow the window being visualised
  • Metric tabs – Temperature · Moisture · Wind · Atmosphere

Run from the project root (use the databricks environment):

    conda run -n databricks python -m dashboard.app
    # then open  http://127.0.0.1:8050

Or simply:

    conda activate databricks
    python -m dashboard.app
"""

import os
import sys

# Make project root importable regardless of how the script is invoked
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html
from plotly.subplots import make_subplots

from utils.spark import build_spark_explore
# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_gold_daily() -> pd.DataFrame:
    """
    Read the ``hrrr_gold_capitals_daily`` Delta table via Spark and return a
    pandas DataFrame.

    Uses ``build_spark_explore()`` — the same lightweight session used by the
    EDA notebooks — which registers all Delta tables in the SQL catalog so we
    can query by name rather than by raw path.
    """
    spark = build_spark_explore()
    try:
        df_spark = spark.read.table("hrrr_gold_capitals_daily")
        df = df_spark.toPandas()
    finally:
        spark.stop()

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    return df

DF = load_gold_daily()
ALL_CITIES = sorted(DF["city"].unique().tolist())
DATE_MIN = DF["date"].min()
DATE_MAX = DF["date"].max()
ALL_DATES = sorted(DF["date"].unique().tolist())
DATE_MARKS = {
    i: {"label": d.strftime("%b %d"), "style": {"color": "#cbd5e1", "fontSize": "11px", "fontWeight": "500"}}
    for i, d in enumerate(ALL_DATES)
    if i % max(1, len(ALL_DATES) // 6) == 0 or i == len(ALL_DATES) - 1
}

# ---------------------------------------------------------------------------
# Metric groups
# ---------------------------------------------------------------------------

METRIC_GROUPS = {
    "🌡️ Temperature": {
        "metrics": [
            ("avg_t_2m", "Avg Temp (K)", "#f97316"),
            ("max_t_2m", "Max Temp (K)", "#ef4444"),
            ("min_t_2m", "Min Temp (K)", "#3b82f6"),
        ],
        "y_label": "Temperature (K)",
        "description": "2-metre air temperature — daily avg / max / min. Values in Kelvin (273 K ≈ 0 °C).",
    },
    "💧 Moisture": {
        "metrics": [
            ("avg_dpt", "Dew Point (K)", "#0ea5e9"),
            ("avg_vis", "Visibility (m)", "#8b5cf6"),
            ("avg_cape", "CAPE (J/kg)", "#10b981"),
        ],
        "y_label": "Value",
        "description": "Dew-point temperature, surface visibility, and Convective Available Potential Energy (CAPE — higher = more storm energy).",
    },
    "💨 Wind": {
        "metrics": [
            ("avg_gust", "Avg Gust (m/s)", "#f59e0b"),
            ("max_gust", "Max Gust (m/s)", "#ef4444"),
            ("avg_u",   "U-wind (m/s)",   "#3b82f6"),
            ("avg_v",   "V-wind (m/s)",   "#6366f1"),
        ],
        "y_label": "Wind speed (m/s)",
        "description": "Surface wind gusts (daily avg & max) and 10-metre U/V wind components.",
    },
    "🌬️ Atmosphere": {
        "metrics": [
            ("avg_sp",   "Avg Pressure (Pa)",   "#64748b"),
            ("avg_blh",  "Boundary Layer H (m)", "#f97316"),
            ("avg_ishf", "Sensible Heat Flux",   "#ec4899"),
            ("avg_slhtf","Latent Heat Flux",     "#10b981"),
        ],
        "y_label": "Value",
        "description": "Surface pressure, boundary layer height, and turbulent heat fluxes.",
    },
}

# ---------------------------------------------------------------------------
# Colour palette & theme tokens
# ---------------------------------------------------------------------------

BG_DARK   = "#0f172a"
BG_CARD   = "#1e293b"
BG_CARD2  = "#162032"
BORDER    = "#334155"
TEXT_PRI  = "#f1f5f9"
TEXT_SEC  = "#94a3b8"
ACCENT    = "#6366f1"
ACCENT2   = "#818cf8"
FONT      = "'Inter', 'Segoe UI', sans-serif"

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT, color=TEXT_SEC, size=12),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(
        bgcolor="rgba(30,41,59,0.8)",
        bordercolor=BORDER,
        borderwidth=1,
        font=dict(color=TEXT_PRI, size=11),
    ),
    xaxis=dict(
        gridcolor=BORDER, zerolinecolor=BORDER,
        tickfont=dict(color=TEXT_SEC),
    ),
    yaxis=dict(
        gridcolor=BORDER, zerolinecolor=BORDER,
        tickfont=dict(color=TEXT_SEC),
    ),
    hovermode="x unified",
)

# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

app = Dash(
    __name__,
    title="HRRR Gold · Weather Dashboard",
    assets_folder=_ASSETS_DIR,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    ],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)


app.layout = html.Div(
    style={
        "backgroundColor": BG_DARK,
        "minHeight": "100vh",
        "fontFamily": FONT,
        "color": TEXT_PRI,
        "padding": "0",
    },
    children=[
        # ── Header ──────────────────────────────────────────────────────
        html.Div(
            style={
                "background": f"linear-gradient(135deg, {BG_CARD} 0%, #1a1f35 100%)",
                "borderBottom": f"1px solid {BORDER}",
                "padding": "20px 36px",
                "display": "flex",
                "alignItems": "center",
                "gap": "16px",
            },
            children=[
                html.Div("⛈️", style={"fontSize": "36px"}),
                html.Div([
                    html.H1(
                        "HRRR Weather Intelligence",
                        style={"margin": "0", "fontSize": "22px", "fontWeight": "700",
                               "letterSpacing": "-0.3px"},
                    ),
                    html.P(
                        "Gold Layer · Daily Aggregations by US State Capital",
                        style={"margin": "2px 0 0", "fontSize": "13px",
                               "color": TEXT_SEC, "fontWeight": "400"},
                    ),
                ]),
                html.Div(
                    style={"marginLeft": "auto", "display": "flex", "gap": "8px",
                           "alignItems": "center"},
                    children=[
                        html.Span("🟢 Live", style={"fontSize": "12px", "color": "#22c55e",
                                                     "fontWeight": "600"}),
                        html.Span(
                            f"{len(DF)} records · {DF['city'].nunique()} cities",
                            style={"fontSize": "12px", "color": TEXT_SEC,
                                   "backgroundColor": BG_CARD2, "padding": "4px 10px",
                                   "borderRadius": "12px", "border": f"1px solid {BORDER}"},
                        ),
                    ],
                ),
            ],
        ),

        # ── Controls bar ────────────────────────────────────────────────
        html.Div(
            className="controls-bar",
            style={
                "display": "flex",
                "gap": "24px",
                "alignItems": "flex-end",
                "flexWrap": "wrap",
                "padding": "20px 36px",
                "backgroundColor": BG_CARD2,
                "borderBottom": f"1px solid {BORDER}",
            },
            children=[
                # City selector
                html.Div([
                    html.Label("City", className="control-label",
                               style={"fontSize": "11px", "fontWeight": "700",
                                      "color": "#cbd5e1", "textTransform": "uppercase",
                                      "letterSpacing": "0.8px", "marginBottom": "6px",
                                      "display": "block"}),
                    dcc.Dropdown(
                        id="city-dropdown",
                        options=[{"label": c, "value": c} for c in ALL_CITIES],
                        value=ALL_CITIES[0],
                        clearable=False,
                        style={
                            "width": "240px",
                            "fontSize": "13px",
                        },
                        className="dash-dropdown-dark",
                    ),
                ]),

                # Compare city (optional)
                html.Div([
                    html.Label("Compare city (optional)", className="control-label",
                               style={"fontSize": "11px", "fontWeight": "700",
                                      "color": "#cbd5e1", "textTransform": "uppercase",
                                      "letterSpacing": "0.8px", "marginBottom": "6px",
                                      "display": "block"}),
                    dcc.Dropdown(
                        id="city-compare-dropdown",
                        options=[{"label": c, "value": c} for c in ALL_CITIES],
                        value=None,
                        clearable=True,
                        placeholder="None",
                        style={
                            "width": "240px",
                            "fontSize": "13px",
                        },
                        className="dash-dropdown-dark",
                    ),
                ]),

                # Date range
                html.Div(
                    style={"flex": "1", "minWidth": "260px"},
                    children=[
                        html.Label("Date range", className="control-label",
                                   style={"fontSize": "11px", "fontWeight": "700",
                                          "color": "#cbd5e1", "textTransform": "uppercase",
                                          "letterSpacing": "0.8px", "marginBottom": "6px",
                                          "display": "block"}),
                        html.Div(id="date-range-label",
                                 style={"fontSize": "13px", "color": "#f8fafc",
                                        "fontWeight": "600",
                                        "marginBottom": "8px"}),
                        dcc.RangeSlider(
                            id="date-range-slider",
                            min=0,
                            max=len(ALL_DATES) - 1,
                            step=1,
                            value=[0, len(ALL_DATES) - 1],
                            marks=DATE_MARKS,
                            className="dash-range-slider-dark",
                        ),
                    ],
                ),
            ],
        ),

        # ── KPI cards ───────────────────────────────────────────────────
        html.Div(id="kpi-cards",
                 style={"padding": "20px 36px"}),

        # ── Metric tabs + charts ─────────────────────────────────────────
        html.Div(
            style={"padding": "0 36px 36px"},
            children=[
                dcc.Tabs(
                    id="metric-tabs",
                    value=list(METRIC_GROUPS.keys())[0],
                    children=[
                        dcc.Tab(label=name, value=name,
                                style={"backgroundColor": BG_CARD2, "color": TEXT_SEC,
                                       "border": f"1px solid {BORDER}",
                                       "borderBottom": "none", "padding": "10px 18px",
                                       "fontSize": "13px", "fontFamily": FONT},
                                selected_style={"backgroundColor": BG_CARD,
                                                "color": TEXT_PRI,
                                                "border": f"1px solid {BORDER}",
                                                "borderBottom": f"2px solid {ACCENT}",
                                                "padding": "10px 18px",
                                                "fontSize": "13px", "fontFamily": FONT,
                                                "fontWeight": "600"})
                        for name in METRIC_GROUPS
                    ],
                    style={"marginBottom": "0"},
                ),

                html.Div(
                    style={
                        "backgroundColor": BG_CARD,
                        "border": f"1px solid {BORDER}",
                        "borderTop": "none",
                        "borderRadius": "0 0 12px 12px",
                        "padding": "24px",
                    },
                    children=[
                        html.P(id="metric-description",
                               style={"color": TEXT_SEC, "fontSize": "13px",
                                      "marginTop": "0", "marginBottom": "16px"}),
                        dcc.Graph(id="main-chart",
                                  config={"displayModeBar": False},
                                  style={"height": "380px"}),
                    ],
                ),
            ],
        ),

        # ── Data table preview ──────────────────────────────────────────
        html.Div(
            style={"padding": "0 36px 48px"},
            children=[
                html.H3(id="data-table-title",
                        style={"color": TEXT_SEC, "fontSize": "13px",
                               "fontWeight": "600", "textTransform": "uppercase",
                               "letterSpacing": "0.8px", "marginBottom": "12px"}),
                html.Div(id="data-table",
                         style={"overflowX": "auto"}),
            ],
        ),

    ],
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("date-range-label", "children"),
    Input("date-range-slider", "value"),
)
def update_date_label(slider_range):
    start = ALL_DATES[slider_range[0]]
    end   = ALL_DATES[slider_range[1]]
    days  = (end - start).days + 1
    return f"{start.strftime('%b %d, %Y')}  →  {end.strftime('%b %d, %Y')}  ({days} days)"


@callback(
    Output("kpi-cards", "children"),
    Input("city-dropdown",         "value"),
    Input("date-range-slider",     "value"),
)
def update_kpis(city, slider_range):
    start_date = ALL_DATES[slider_range[0]]
    end_date   = ALL_DATES[slider_range[1]]

    mask = (
        (DF["city"] == city)
        & (DF["date"] >= pd.Timestamp(start_date))
        & (DF["date"] <= pd.Timestamp(end_date))
    )
    sub = DF[mask]

    if sub.empty:
        return [html.P("No data for selected filters.", style={"color": TEXT_SEC})]

    avg_t = float(sub["avg_t_2m"].mean())
    t_c   = round(avg_t - 273.15, 1)
    t_f   = round(t_c * 9/5 + 32, 1)
    min_t_c = round(float(sub["min_t_2m"].min()) - 273.15, 1)
    max_t_c = round(float(sub["max_t_2m"].max()) - 273.15, 1)

    # Dew point
    if "avg_dpt" in sub.columns:
        dpt_c = round(float(sub["avg_dpt"].mean()) - 273.15, 1)
    else:
        dpt_c = 0.0

    # Moisture (Relative Humidity via Magnus formula from T and Dew Point)
    try:
        e_sat = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
        e_act = 6.112 * np.exp((17.67 * dpt_c) / (dpt_c + 243.5))
        rh = min(100.0, max(0.0, round(float((e_act / e_sat) * 100), 0)))
    except Exception:
        rh = 0.0

    # Wind speed
    if "avg_u" in sub.columns and "avg_v" in sub.columns:
        avg_wind_speed = round(float(np.sqrt(sub["avg_u"]**2 + sub["avg_v"]**2).mean()), 1)
    else:
        avg_wind_speed = round(float(sub.get("avg_gust", pd.Series([0.0])).mean()), 1)

    max_gust = round(float(sub["max_gust"].max()), 1) if "max_gust" in sub.columns else 0.0
    avg_vis  = round(float(sub["avg_vis"].mean()) / 1000, 1) if "avg_vis" in sub.columns else 0.0
    avg_cape = round(float(sub["avg_cape"].mean()), 0) if "avg_cape" in sub.columns else 0.0

    # ── Big Prominent Hero Card for Average Temperature ──
    hero_temp_card = html.Div(
        style={
            "backgroundColor": BG_CARD,
            "background": "linear-gradient(145deg, #1e293b 0%, #1a2336 100%)",
            "border": "1px solid rgba(249, 115, 22, 0.45)",
            "borderRadius": "16px",
            "padding": "24px 28px",
            "width": "500px",
            "minWidth": "300px",
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "space-between",
            "boxShadow": "0 10px 25px -4px rgba(249, 115, 22, 0.15)",
        },
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"},
                children=[
                    html.Div("🌡️", style={"fontSize": "36px"}),
                    html.Span(
                        f"{city}",
                        style={
                            "fontSize": "11px",
                            "fontWeight": "700",
                            "letterSpacing": "0.8px",
                            "color": "#fdba74",
                            "backgroundColor": "rgba(249, 115, 22, 0.15)",
                            "padding": "4px 10px",
                            "borderRadius": "12px",
                            "border": "1px solid rgba(249, 115, 22, 0.3)",
                        },
                    ),
                ],
            ),
            html.Div(
                style={"margin": "14px 0"},
                children=[
                    html.Div(
                        "Average Temperature",
                        style={
                            "fontSize": "12px",
                            "color": "#cbd5e1",
                            "textTransform": "uppercase",
                            "letterSpacing": "0.8px",
                            "fontWeight": "700",
                            "marginBottom": "4px",
                        },
                    ),
                    html.Div(
                        style={"display": "flex", "alignItems": "baseline", "gap": "6px"},
                        children=[
                            html.Span(
                                f"{t_c}",
                                style={"fontSize": "56px", "fontWeight": "800", "color": "#f97316", "lineHeight": "1"},
                            ),
                            html.Span(
                                "°C",
                                style={"fontSize": "24px", "fontWeight": "600", "color": "#fdba74"},
                            ),
                            html.Span(
                                f"({t_f}°F)",
                                style={"fontSize": "14px", "color": TEXT_SEC, "marginLeft": "6px"},
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                style={
                    "paddingTop": "12px",
                    "borderTop": f"1px solid {BORDER}",
                    "display": "flex",
                    "justifyContent": "space-between",
                    "fontSize": "12px",
                    "color": TEXT_SEC,
                },
                children=[
                    html.Span(f"Min: {min_t_c}°C"),
                    html.Span(f"Max: {max_t_c}°C"),
                ],
            ),
        ],
    )

    # ── 3x2 Companion Metrics Grid (6 cards) ──
    def _companion(icon, label, value, unit="", color=ACCENT2):
        return html.Div(
            style={
                "backgroundColor": BG_CARD,
                "border": f"1px solid {BORDER}",
                "borderRadius": "14px",
                "padding": "16px 20px",
                "display": "flex",
                "flexDirection": "column",
                "justifyContent": "space-between",
                "minHeight": "85px",
            },
            children=[
                html.Div(
                    style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"},
                    children=[
                        html.Div(
                            label,
                            style={
                                "fontSize": "11px",
                                "color": "#cbd5e1",
                                "textTransform": "uppercase",
                                "letterSpacing": "0.8px",
                                "fontWeight": "700",
                            },
                        ),
                        html.Div(icon, style={"fontSize": "20px"}),
                    ],
                ),
                html.Div(
                    style={"marginTop": "8px"},
                    children=[
                        html.Span(
                            value,
                            style={"fontSize": "24px", "fontWeight": "700", "color": color},
                        ),
                        html.Span(
                            f" {unit}" if unit else "",
                            style={"fontSize": "12px", "color": TEXT_SEC, "fontWeight": "500"},
                        ),
                    ],
                ),
            ],
        )

    companion_grid = html.Div(
        style={
            "flex": "1",
            "minWidth": "450px",
            "display": "grid",
            "gridTemplateColumns": "repeat(3, 1fr)",
            "gap": "14px",
        },
        children=[
            _companion("🌪️", "Peak Gust", f"{max_gust:.1f}", "m/s", color="#f59e0b"),
            _companion("💨", "Avg Wind Speed", f"{avg_wind_speed:.1f}", "m/s", color="#3b82f6"),
            _companion("💧", "Avg Dew Point", f"{dpt_c:.1f}", "°C", color="#0ea5e9"),
            _companion("🌧️", "Moisture (RH)", f"{rh:.0f}", "%", color="#38bdf8"),
            _companion("👁️", "Avg Visibility", f"{avg_vis:.1f}", "km", color="#8b5cf6"),
            _companion("⚡", "Avg CAPE", f"{avg_cape:.0f}", "J/kg", color="#10b981"),
        ],
    )

    return html.Div(
        style={
            "display": "flex",
            "gap": "18px",
            "alignItems": "stretch",
            "flexWrap": "wrap",
        },
        children=[hero_temp_card, companion_grid],
    )


@callback(
    Output("main-chart",         "figure"),
    Output("metric-description", "children"),
    Input("city-dropdown",         "value"),
    Input("city-compare-dropdown", "value"),
    Input("date-range-slider",     "value"),
    Input("metric-tabs",           "value"),
)
def update_chart(city, city2, slider_range, tab):
    start_date = ALL_DATES[slider_range[0]]
    end_date   = ALL_DATES[slider_range[1]]

    group = METRIC_GROUPS[tab]
    metrics = group["metrics"]
    description = group["description"]

    def _filter(c):
        return DF[
            (DF["city"] == c)
            & (DF["date"] >= pd.Timestamp(start_date))
            & (DF["date"] <= pd.Timestamp(end_date))
        ].sort_values("date")

    sub1 = _filter(city)

    fig = go.Figure()

    # Primary city traces
    for col, label, color in metrics:
        if col not in sub1.columns:
            continue
        fig.add_trace(go.Scatter(
            x=sub1["date"], y=sub1[col],
            name=f"{city} · {label}",
            line=dict(color=color, width=2),
            mode="lines+markers",
            marker=dict(size=5, symbol="circle"),
            hovertemplate=f"<b>{city}</b><br>{label}: %{{y:.2f}}<extra></extra>",
        ))

    # Compare city traces (dashed)
    if city2 and city2 != city:
        sub2 = _filter(city2)
        for col, label, color in metrics:
            if col not in sub2.columns:
                continue
            fig.add_trace(go.Scatter(
                x=sub2["date"], y=sub2[col],
                name=f"{city2} · {label}",
                line=dict(color=color, width=2, dash="dash"),
                mode="lines+markers",
                marker=dict(size=4, symbol="diamond"),
                opacity=0.75,
                hovertemplate=f"<b>{city2}</b><br>{label}: %{{y:.2f}}<extra></extra>",
            ))

    layout = dict(
        **PLOT_LAYOUT,
        yaxis_title=group["y_label"],
        title=dict(
            text=f"<b>{city}</b>" + (f" vs <b>{city2}</b>" if city2 and city2 != city else ""),
            font=dict(color=TEXT_PRI, size=15),
            x=0,
        ),
    )
    fig.update_layout(**layout)
    return fig, description


@callback(
    Output("data-table", "children"),
    Output("data-table-title", "children"),
    Input("city-dropdown",     "value"),
    Input("date-range-slider", "value"),
)
def update_table(city, slider_range):
    start_date = ALL_DATES[slider_range[0]]
    end_date   = ALL_DATES[slider_range[1]]

    city_display = city if city else "Selected City"
    title_children = [
        html.Span("Raw gold data · latest 10 records for "),
        html.Span(city_display, style={"color": ACCENT2, "fontWeight": "700"}),
    ]

    sub = DF[
        (DF["city"] == city)
        & (DF["date"] >= pd.Timestamp(start_date))
        & (DF["date"] <= pd.Timestamp(end_date))
    ].sort_values("date", ascending=False).head(10)

    if sub.empty:
        return html.P("No data.", style={"color": TEXT_SEC}), title_children

    # Show a curated subset of columns
    show_cols = ["date", "avg_t_2m", "max_t_2m", "min_t_2m",
                 "avg_dpt", "avg_gust", "max_gust",
                 "avg_vis", "avg_cape", "avg_sp"]
    show_cols = [c for c in show_cols if c in sub.columns]

    header_style = {
        "padding": "10px 14px", "textAlign": "left",
        "fontSize": "11px", "fontWeight": "600",
        "color": TEXT_SEC, "textTransform": "uppercase",
        "letterSpacing": "0.8px", "backgroundColor": BG_CARD2,
        "borderBottom": f"1px solid {BORDER}",
    }
    cell_style = {
        "padding": "9px 14px", "fontSize": "12px",
        "color": TEXT_PRI, "borderBottom": f"1px solid {BORDER}",
        "fontVariantNumeric": "tabular-nums",
    }

    rows = []
    for _, row in sub.iterrows():
        cells = []
        for c in show_cols:
            val = row[c]
            if c == "date":
                display = str(val)[:10]
            elif isinstance(val, float):
                display = f"{val:.2f}"
            else:
                display = str(val)
            cells.append(html.Td(display, style=cell_style))
        rows.append(html.Tr(cells, style={"transition": "background 0.15s"}))

    table = html.Table(
        style={
            "width": "100%", "borderCollapse": "collapse",
            "backgroundColor": BG_CARD,
            "border": f"1px solid {BORDER}",
            "borderRadius": "10px", "overflow": "hidden",
        },
        children=[
            html.Thead(html.Tr([
                html.Th(c.replace("_", " ").title(), style=header_style)
                for c in show_cols
            ])),
            html.Tbody(rows),
        ],
    )
    return table, title_children


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import webbrowser, threading, time

    def _open():
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8050")

    threading.Thread(target=_open, daemon=True).start()
    app.run(host="0.0.0.0", port=8050, debug=False)
