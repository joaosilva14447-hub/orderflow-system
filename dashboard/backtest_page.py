"""
Backtest dashboard page — signal engine + performance metrics visualisation.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from collections import defaultdict
import numpy as np
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


# ── Historical backtest — additional charts ───────────────────────────────────

def render_monthly_pnl(result: BacktestResult) -> go.Figure:
    """
    Monthly PnL bar chart — shows how the strategy performed in each calendar month.
    Useful for spotting seasonality or regime sensitivity.
    """
    if not result.trades:
        return go.Figure()

    monthly: dict[str, float] = defaultdict(float)
    for t in result.trades:
        month = datetime.fromtimestamp(t.signal.timestamp / 1000).strftime("%Y-%m")
        monthly[month] += t.pnl

    months = sorted(monthly.keys())
    pnls   = [monthly[m] for m in months]
    colors = [GREEN if p > 0 else RED for p in pnls]

    fig = go.Figure(go.Bar(
        x=months, y=pnls,
        marker_color=colors,
        text=[f"€{p:+,.0f}" for p in pnls],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.add_hline(y=0, line_color=GRAY, line_width=1)

    cum = 0.0
    cum_y = []
    for p in pnls:
        cum += p
        cum_y.append(cum)

    fig.add_trace(go.Scatter(
        x=months, y=cum_y, name="Cumulative PnL",
        line=dict(color="rgba(255,255,255,0.6)", width=1.5, dash="dot"),
        yaxis="y2",
    ))

    fig.update_layout(
        title="Monthly PnL (bars) + Cumulative PnL (line)",
        template="plotly_dark", height=320,
        margin=dict(l=0, r=0, t=40, b=60),
        xaxis=dict(tickangle=-45),
        yaxis=dict(title="PnL (€)"),
        yaxis2=dict(title="Cumulative (€)", overlaying="y", side="right",
                    showgrid=False),
        showlegend=True,
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def render_rolling_winrate(result: BacktestResult, window: int = 20) -> go.Figure:
    """
    Rolling win rate over the last N trades.
    Reveals if signal quality improved or degraded over time — regime detection.
    """
    if len(result.trades) < window:
        return go.Figure()

    wins  = [1.0 if t.is_win else 0.0 for t in result.trades]
    dates = [datetime.fromtimestamp(t.signal.timestamp / 1000) for t in result.trades]

    rolling_wr  = []
    rolling_pf  = []   # rolling profit factor
    for i in range(window - 1, len(wins)):
        slice_      = result.trades[i - window + 1 : i + 1]
        wr          = sum(1 for t in slice_ if t.is_win) / window * 100
        gross_win   = sum(t.pnl for t in slice_ if t.pnl > 0)
        gross_loss  = abs(sum(t.pnl for t in slice_ if t.pnl < 0))
        pf          = gross_win / gross_loss if gross_loss > 0 else 2.0
        rolling_wr.append(wr)
        rolling_pf.append(min(pf, 4.0))   # cap at 4 for scale

    x = dates[window - 1:]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.06,
                        subplot_titles=[f"Rolling {window}-Trade Win Rate",
                                        f"Rolling {window}-Trade Profit Factor"])

    # Win rate
    fig.add_trace(go.Scatter(
        x=x, y=rolling_wr,
        name="Win Rate %",
        line=dict(color="#64b4ff", width=2),
        fill="tozeroy", fillcolor="rgba(100,180,255,0.08)",
    ), row=1, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color=GRAY,
                  annotation_text="50% (neutral)", row=1, col=1)
    fig.add_hrect(y0=33, y1=100, fillcolor="rgba(38,166,154,0.04)",
                  line_width=0, row=1, col=1)

    # Profit factor
    pf_colors = [GREEN if v >= 1.0 else RED for v in rolling_pf]
    fig.add_trace(go.Bar(
        x=x, y=rolling_pf,
        name="Profit Factor",
        marker_color=pf_colors, opacity=0.75,
    ), row=2, col=1)
    fig.add_hline(y=1.0, line_dash="dash", line_color=GRAY,
                  annotation_text="Breakeven", row=2, col=1)

    fig.update_layout(
        template="plotly_dark", height=400,
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
    )
    fig.update_yaxes(title_text="Win Rate %", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="Profit Factor", range=[0, 4.1],  row=2, col=1)
    return fig


def render_pnl_distribution(result: BacktestResult) -> go.Figure:
    """
    PnL distribution histogram — shows the shape of the return distribution.
    A profitable system should have a right-skewed distribution.
    """
    if not result.trades:
        return go.Figure()

    pnls   = [t.pnl for t in result.trades]
    mean_p = float(np.mean(pnls))
    med_p  = float(np.median(pnls))

    win_pnls  = [p for p in pnls if p > 0]
    loss_pnls = [p for p in pnls if p < 0]

    fig = go.Figure()
    if win_pnls:
        fig.add_trace(go.Histogram(
            x=win_pnls, name="Wins",
            marker_color=GREEN, opacity=0.75,
            xbins=dict(size=max(1.0, (max(win_pnls) - min(win_pnls)) / 20)),
        ))
    if loss_pnls:
        fig.add_trace(go.Histogram(
            x=loss_pnls, name="Losses",
            marker_color=RED, opacity=0.75,
            xbins=dict(size=max(1.0, (max(loss_pnls) - min(loss_pnls)) / 20)),
        ))

    fig.add_vline(x=0,      line_color="white",  line_width=1.5)
    fig.add_vline(x=mean_p, line_color="yellow", line_dash="dash",
                  annotation_text=f"Mean €{mean_p:+.1f}", annotation_position="top right")
    fig.add_vline(x=med_p,  line_color="orange", line_dash="dot",
                  annotation_text=f"Median €{med_p:+.1f}", annotation_position="top left")

    fig.update_layout(
        title="PnL Distribution — Trade Returns",
        template="plotly_dark", height=300, barmode="overlay",
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_title="PnL (€)", yaxis_title="# Trades",
        legend=dict(orientation="h", y=1.06),
    )
    return fig


def render_r_multiple_chart(result: BacktestResult) -> go.Figure:
    """
    R-multiple scatter over time — each trade's P&L expressed as multiples of 1R.
    Shows consistency: a good system has wins clustering around +2R, losses around -1R.
    """
    if not result.trades:
        return go.Figure()

    dates = [datetime.fromtimestamp(t.signal.timestamp / 1000) for t in result.trades]
    rmult = [t.r_multiple for t in result.trades]
    colors = [GREEN if r > 0 else RED for r in rmult]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=rmult,
        mode="markers",
        marker=dict(color=colors, size=8,
                    symbol=["circle" if r > 0 else "x" for r in rmult],
                    line=dict(color="white", width=0.5)),
        name="R-Multiple",
        text=[f"{'WIN' if r > 0 else 'LOSS'}: {r:+.2f}R  (€{t.pnl:+.0f})"
              for r, t in zip(rmult, result.trades)],
        hoverinfo="text",
    ))

    # Target bands
    fig.add_hline(y=0,  line_color=GRAY,    line_width=1)
    fig.add_hline(y=2,  line_color=GREEN,   line_dash="dot",
                  annotation_text="+2R target", annotation_position="left")
    fig.add_hline(y=-1, line_color=RED,     line_dash="dot",
                  annotation_text="-1R stop",   annotation_position="left")

    # Running average R
    cum_r = [sum(rmult[:i+1]) / (i+1) for i in range(len(rmult))]
    fig.add_trace(go.Scatter(
        x=dates, y=cum_r,
        name="Avg R (running)",
        line=dict(color="rgba(255,255,255,0.5)", width=1.5, dash="longdash"),
    ))

    fig.update_layout(
        title="R-Multiple per Trade",
        template="plotly_dark", height=320,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis_title="R-Multiple",
        legend=dict(orientation="h", y=1.06),
    )
    return fig


def trade_streak_stats(result: BacktestResult) -> dict:
    """
    Win/loss streak analysis.
    Returns: max_win_streak, max_loss_streak, avg_win_streak, avg_loss_streak,
             current_streak (positive = wins, negative = losses).
    """
    if not result.trades:
        return {}

    wins = [t.is_win for t in result.trades]

    max_win = max_loss = cur = 0
    win_streaks:  list[int] = []
    loss_streaks: list[int] = []

    i = 0
    while i < len(wins):
        if wins[i]:
            run = 0
            while i < len(wins) and wins[i]:
                run += 1; i += 1
            win_streaks.append(run)
            max_win = max(max_win, run)
        else:
            run = 0
            while i < len(wins) and not wins[i]:
                run += 1; i += 1
            loss_streaks.append(run)
            max_loss = max(max_loss, run)

    # Current streak (from the end)
    cur_streak = 0
    last_result = wins[-1] if wins else None
    for w in reversed(wins):
        if w == last_result:
            cur_streak += 1
        else:
            break
    if last_result is False:
        cur_streak = -cur_streak   # negative = losing streak

    return {
        "max_win_streak":  max_win,
        "max_loss_streak": max_loss,
        "avg_win_streak":  round(float(np.mean(win_streaks)),  1) if win_streaks  else 0.0,
        "avg_loss_streak": round(float(np.mean(loss_streaks)), 1) if loss_streaks else 0.0,
        "current_streak":  cur_streak,
    }
