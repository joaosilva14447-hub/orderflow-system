"""
Footprint chart renderer — Plotly implementation.
Shows buy/sell volume per price level per bar with imbalance highlighting.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from core.footprint import FootprintBar


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v/1_000:.1f}K"
    return f"{v:.0f}"

def _rgba(r, g, b, a): return f"rgba({r},{g},{b},{a})"

GREEN      = (38,  166, 154)
RED        = (239, 83,  80)
YELLOW     = (255, 214, 10)
IMBALANCE  = (255, 165,  0)   # orange highlight for imbalances


# ── Main chart ────────────────────────────────────────────────────────────────

def render_footprint(
    footprints:  list[FootprintBar],
    imb_thresh:  float = 3.0,
    show_nums:   bool  = True,
    max_bars:    int   = 30,
) -> go.Figure:
    """
    Renders a footprint chart for the last `max_bars` bars.
    Each bar shows buy (green) | sell (red) volume per tick level.
    Imbalances are highlighted in orange. POC is highlighted in yellow.
    """
    fps = footprints[-max_bars:]
    if not fps:
        return go.Figure()

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.82, 0.18],
        shared_xaxes=True,
        vertical_spacing=0.03,
    )

    # Collect all price levels across selected bars for consistent Y-axis
    all_prices = set()
    for fp in fps:
        all_prices.update(fp.levels.keys())

    if not all_prices:
        return go.Figure()

    y_min = min(all_prices)
    y_max = max(all_prices)
    tick_size = fps[0].tick_size

    # Max total volume across all levels (for color scaling)
    max_vol = max(
        (lv.total for fp in fps for lv in fp.levels.values()),
        default=1.0
    )

    # ── Heatmap cells + text ──────────────────────────────────────────────────
    shapes      = []
    annotations = []

    bar_width = 0.4  # fraction of bar spacing (0–1)

    for i, fp in enumerate(fps):
        ts  = fp.bar.timestamp / 1000
        poc = fp.poc
        imbalances = {p: side for p, side in fp.get_imbalances(imb_thresh)}

        sorted_levels = sorted(fp.levels.items())

        for price, lv in sorted_levels:
            if lv.total == 0:
                continue

            intensity = min(lv.total / max_vol, 1.0)

            # Cell color
            if price == poc:
                fill = _rgba(*YELLOW, 0.75)
                border = _rgba(*YELLOW, 1.0)
            elif price in imbalances:
                fill = _rgba(*IMBALANCE, 0.5 + intensity * 0.4)
                border = _rgba(*IMBALANCE, 1.0)
            elif lv.delta >= 0:
                fill = _rgba(*GREEN, 0.15 + intensity * 0.55)
                border = _rgba(*GREEN, 0.6)
            else:
                fill = _rgba(*RED, 0.15 + intensity * 0.55)
                border = _rgba(*RED, 0.6)

            # Rectangle cell
            shapes.append(dict(
                type="rect",
                x0=i - bar_width,
                x1=i + bar_width,
                y0=price - tick_size * 0.48,
                y1=price + tick_size * 0.48,
                fillcolor=fill,
                line=dict(color=border, width=0.5),
                xref="x", yref="y",
                layer="below",
            ))

            # Text: "buy | sell"
            if show_nums:
                label = f"{_fmt(lv.buy)} | {_fmt(lv.sell)}"
                txt_color = "white" if price != poc else "black"
                annotations.append(dict(
                    x=i, y=price,
                    text=label,
                    showarrow=False,
                    font=dict(size=8, color=txt_color, family="monospace"),
                    xref="x", yref="y",
                ))

        # ── OHLC candle outline ───────────────────────────────────────────────
        bar   = fp.bar
        color = _rgba(*GREEN, 1) if bar.close >= bar.open else _rgba(*RED, 1)
        # Wick
        shapes.append(dict(
            type="line",
            x0=i, x1=i,
            y0=bar.low, y1=bar.high,
            line=dict(color=color, width=1),
            xref="x", yref="y",
        ))
        # Body
        shapes.append(dict(
            type="rect",
            x0=i - bar_width * 0.9,
            x1=i + bar_width * 0.9,
            y0=min(bar.open, bar.close),
            y1=max(bar.open, bar.close),
            fillcolor=color,
            line=dict(color=color, width=1),
            xref="x", yref="y",
        ))

    # ── Invisible scatter to set axis range ───────────────────────────────────
    x_vals = list(range(len(fps)))
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=[fp.bar.close for fp in fps],
            mode="markers",
            marker=dict(opacity=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1, col=1,
    )

    # ── Delta bar chart ───────────────────────────────────────────────────────
    deltas      = [fp.delta for fp in fps]
    delta_cols  = [_rgba(*GREEN, 0.85) if d >= 0 else _rgba(*RED, 0.85) for d in deltas]
    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=deltas,
            marker_color=delta_cols,
            name="Delta",
            showlegend=False,
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_color="gray", line_width=0.8, row=2, col=1)

    # ── X-axis labels (timestamps) ────────────────────────────────────────────
    tick_labels = [
        datetime.fromtimestamp(fp.bar.timestamp / 1000).strftime("%H:%M")
        for fp in fps
    ]

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        shapes      = shapes,
        annotations = annotations,
        template    = "plotly_dark",
        height      = 750,
        margin      = dict(l=60, r=10, t=40, b=40),
        title       = (
            f"Footprint Chart · {fps[0].bar.symbol} · {fps[0].bar.timeframe} "
            f"· last {len(fps)} bars"
        ),
        xaxis=dict(
            tickmode  = "array",
            tickvals  = x_vals,
            ticktext  = tick_labels,
            tickangle = -45,
            showgrid  = False,
        ),
        yaxis=dict(
            title    = "Price",
            showgrid = True,
            gridcolor= "rgba(80,80,80,0.3)",
            range    = [y_min - tick_size * 2, y_max + tick_size * 2],
        ),
        xaxis2=dict(showgrid=False),
        yaxis2=dict(title="Delta", showgrid=True, gridcolor="rgba(80,80,80,0.3)"),
        bargap=0.1,
    )

    return fig


def render_footprint_summary(footprints: list[FootprintBar]) -> go.Figure:
    """
    Delta profile across bars — cumulative buy/sell pressure over time.
    Useful for seeing macro order flow direction.
    """
    fps = footprints
    if not fps:
        return go.Figure()

    timestamps = [
        datetime.fromtimestamp(fp.bar.timestamp / 1000) for fp in fps
    ]
    cum_buys  = np.cumsum([fp.cum_buy  for fp in fps])
    cum_sells = np.cumsum([fp.cum_sell for fp in fps])
    cum_delta = cum_buys - cum_sells

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=cum_buys,
        name="Cumulative Buy",
        line=dict(color=_rgba(*GREEN, 1), width=1.5),
        fill="tozeroy", fillcolor=_rgba(*GREEN, 0.1),
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=cum_sells,
        name="Cumulative Sell",
        line=dict(color=_rgba(*RED, 1), width=1.5),
        fill="tozeroy", fillcolor=_rgba(*RED, 0.1),
    ))
    fig.update_layout(
        title    = "Cumulative Buy vs Sell Pressure",
        template = "plotly_dark",
        height   = 220,
        margin   = dict(l=0, r=0, t=40, b=0),
    )
    return fig
