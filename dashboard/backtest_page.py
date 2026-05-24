"""
Backtest dashboard page — signal engine + performance metrics visualisation.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pandas as pd

from core.backtest import BacktestResult, WalkForwardResult, TradeResult


GREEN = "#26a69a"
RED   = "#ef5350"
GRAY  = "#888888"


def _ts(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")


def render_equity_curve(result: BacktestResult, title: str = "Equity Curve") -> go.Figure:
    eq  = result.equity_curve
    idx = list(range(len(eq)))

    peak = [max(eq[:i+1]) for i in range(len(eq))]
    dd   = [(eq[i] - peak[i]) / peak[i] * 100 for i in range(len(eq))]

    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                        shared_xaxes=True, vertical_spacing=0.05)

    fig.add_trace(go.Scatter(
        x=idx, y=eq, name="Equity",
        line=dict(color=GREEN, width=2),
        fill="tozeroy", fillcolor="rgba(38,166,154,0.08)",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=idx, y=peak, name="Peak",
        line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"),
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=idx, y=dd, name="Drawdown %",
        marker_color=RED, opacity=0.7,
    ), row=2, col=1)

    fig.update_layout(
        title=title, template="plotly_dark",
        height=420, margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
    )
    fig.update_yaxes(title_text="Capital (€)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    return fig


def render_metrics_table(report: dict) -> go.Figure:
    """Colour-coded metrics table."""
    if "error" in report:
        return go.Figure()

    rows = [
        ("Total PnL",          f"€{report['total_pnl']:,.2f}",      report['total_pnl'] > 0),
        ("Total Return",       f"{report['total_return_pct']:.2f}%", report['total_return_pct'] > 0),
        ("Trades",             str(report['n_trades']),               True),
        ("Win Rate",           f"{report['win_rate']*100:.1f}%",     report['win_rate'] > 0.5),
        ("Profit Factor",      f"{report['profit_factor']:.3f}",     report['profit_factor'] > 1.5),
        ("Expectancy/trade",   f"€{report['expectancy']:.2f}",       report['expectancy'] > 0),
        ("Sharpe Ratio",       f"{report['sharpe']:.3f}",            report['sharpe'] > 1.0),
        ("Sortino Ratio",      f"{report['sortino']:.3f}",           report['sortino'] > 1.5),
        ("Omega Ratio",        f"{report['omega']:.3f}",             report['omega'] > 1.5),
        ("Calmar Ratio",       f"{report['calmar']:.3f}",            report['calmar'] > 3.0),
        ("Max Drawdown",       f"{report['max_drawdown_pct']:.2f}%", report['max_drawdown_pct'] > -20),
        ("DD Duration (bars)", str(report['max_dd_duration']),        report['max_dd_duration'] < 50),
        ("Avg Win",            f"€{report['avg_win']:.2f}",          True),
        ("Avg Loss",           f"€{report['avg_loss']:.2f}",         True),
        ("Best Trade",         f"€{report['best_trade']:.2f}",       True),
        ("Worst Trade",        f"€{report['worst_trade']:.2f}",      True),
        ("Kelly (Full)",       f"{report['kelly_full']*100:.1f}%",   report['kelly_full'] > 0),
        ("Kelly (Half)",       f"{report['kelly_half']*100:.1f}%",   report['kelly_half'] > 0),
    ]

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [GREEN if r[2] else RED for r in rows]

    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Metric</b>", "<b>Value</b>"],
            fill_color="rgba(50,50,50,0.8)",
            font=dict(color="white", size=13),
            align="left",
        ),
        cells=dict(
            values=[labels, values],
            fill_color=["rgba(30,30,30,0.8)"] * len(labels),
            font=dict(color=[["white"] * len(labels), colors], size=12),
            align=["left", "right"],
            height=28,
        ),
    ))
    fig.update_layout(
        template="plotly_dark", height=560,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    return fig


def render_trades_scatter(result: BacktestResult, bars) -> go.Figure:
    """PnL scatter per trade coloured by signal type."""
    if not result.trades:
        return go.Figure()

    ts_all  = [datetime.fromtimestamp(b.timestamp / 1000) for b in bars]
    type_colors = {
        "CVD_DIV_BULL":       GREEN,
        "CVD_DIV_BEAR":       RED,
        "VWAP_CROSS_BULL":    "#64b4ff",
        "VWAP_CROSS_BEAR":    "#ff6496",
        "POC_RECLAIM_BULL":   "#ffd700",
        "POC_RECLAIM_BEAR":   "#ff8c00",
        "VA_BREAKOUT_BULL":   "#00fa9a",
        "VA_BREAKOUT_BEAR":   "#ff4500",
        "VWAP_BOUNCE_BULL":   "#adff2f",
        "VWAP_BOUNCE_BEAR":   "#da70d6",
    }

    fig = go.Figure()
    by_type = result.by_type()

    for stype, trades in by_type.items():
        fig.add_trace(go.Scatter(
            x=[datetime.fromtimestamp(t.signal.timestamp / 1000) for t in trades],
            y=[t.pnl for t in trades],
            mode="markers",
            marker=dict(
                color=type_colors.get(stype, GRAY),
                size=10,
                symbol=["circle" if t.is_win else "x" for t in trades],
                line=dict(color="white", width=0.5),
            ),
            name=stype,
            text=[
                f"{stype}<br>{'WIN' if t.is_win else 'LOSS'}<br>"
                f"PnL: €{t.pnl:.2f}<br>R: {t.r_multiple:.2f}<br>"
                f"Exit: {t.exit_reason}"
                for t in trades
            ],
            hoverinfo="text",
        ))

    fig.add_hline(y=0, line_color=GRAY, line_width=1)
    fig.update_layout(
        title="Trade PnL by Signal Type",
        template="plotly_dark", height=320,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def render_signal_breakdown(result: BacktestResult) -> go.Figure:
    """Win rate + avg PnL per signal type."""
    by_type = result.by_type()
    if not by_type:
        return go.Figure()

    stypes  = list(by_type.keys())
    wr_vals = [
        sum(1 for t in ts if t.is_win) / len(ts) * 100
        for ts in by_type.values()
    ]
    avg_pnl = [
        sum(t.pnl for t in ts) / len(ts)
        for ts in by_type.values()
    ]
    n_trades = [len(ts) for ts in by_type.values()]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Win Rate % by Signal", "Avg PnL by Signal"])

    fig.add_trace(go.Bar(
        x=stypes, y=wr_vals, name="Win Rate %",
        marker_color=[GREEN if w > 50 else RED for w in wr_vals],
        text=[f"{w:.0f}% ({n})" for w, n in zip(wr_vals, n_trades)],
        textposition="outside",
    ), row=1, col=1)

    fig.add_hline(y=50, line_dash="dash", line_color=GRAY, row=1, col=1)

    fig.add_trace(go.Bar(
        x=stypes, y=avg_pnl, name="Avg PnL",
        marker_color=[GREEN if p > 0 else RED for p in avg_pnl],
        text=[f"€{p:.1f}" for p in avg_pnl],
        textposition="outside",
    ), row=1, col=2)

    fig.update_layout(
        template="plotly_dark", height=340,
        margin=dict(l=0, r=0, t=50, b=60),
        showlegend=False,
    )
    return fig


def render_walk_forward(wf: WalkForwardResult) -> go.Figure:
    """IS vs OOS equity curves side by side."""
    is_eq  = wf.is_result.equity_curve
    oos_eq = wf.oos_result.equity_curve

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[
                            f"In-Sample ({int(wf.is_pct*100)}%)",
                            f"Out-of-Sample ({int((1-wf.is_pct)*100)}%)",
                        ])

    fig.add_trace(go.Scatter(y=is_eq,  name="IS Equity",
                             line=dict(color="#64b4ff", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(y=oos_eq, name="OOS Equity",
                             line=dict(color=GREEN if wf.has_edge else RED, width=2)), row=1, col=2)

    fig.update_layout(
        template="plotly_dark", height=320,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig
