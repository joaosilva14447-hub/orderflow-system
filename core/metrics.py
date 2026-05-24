"""
Performance metrics — institutional grade.
Sharpe · Sortino · Omega · Kelly · Calmar · Profit Factor · Expectancy
"""

import numpy as np
from typing import Optional


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_arr(x) -> np.ndarray:
    return np.array(x, dtype=float)


def annualised(r: float, periods: int) -> float:
    return r * periods


# ── Core metrics ──────────────────────────────────────────────────────────────

def sharpe(returns, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Risk-adjusted return: excess return per unit of total volatility."""
    r = _to_arr(returns)
    if len(r) < 2:
        return 0.0
    excess = r - risk_free / periods_per_year
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def sortino(returns, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """Like Sharpe but only penalises downside volatility."""
    r = _to_arr(returns)
    if len(r) < 2:
        return 0.0
    excess   = r - risk_free / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = np.std(downside, ddof=1)
    if downside_std == 0:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))


def omega(returns, threshold: float = 0.0) -> float:
    """
    Omega ratio — probability-weighted ratio of gains to losses above threshold.
    > 1 = more gains than losses; > 2 = institutional quality.
    """
    r = _to_arr(returns)
    gains  = np.sum(np.maximum(r - threshold, 0))
    losses = np.sum(np.maximum(threshold - r, 0))
    if losses == 0:
        return float("inf")
    return float(gains / losses)


def kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Full Kelly fraction — optimal bet size.
    Returns fraction of capital to risk per trade.
    Half-Kelly (result / 2) recommended in practice.
    """
    if avg_loss == 0 or win_rate <= 0:
        return 0.0
    b = avg_win / avg_loss       # win/loss ratio
    q = 1 - win_rate
    k = (b * win_rate - q) / b
    return max(round(float(k), 4), 0.0)


def max_drawdown(equity_curve) -> float:
    """Maximum peak-to-trough decline as a fraction (e.g. 0.15 = 15%)."""
    eq = _to_arr(equity_curve)
    if len(eq) < 2:
        return 0.0
    peak   = np.maximum.accumulate(eq)
    dd     = (eq - peak) / peak
    return float(np.min(dd))


def max_drawdown_duration(equity_curve) -> int:
    """Number of bars spent in drawdown (longest streak)."""
    eq   = _to_arr(equity_curve)
    peak = np.maximum.accumulate(eq)
    in_dd = (eq < peak).astype(int)
    max_dur, cur = 0, 0
    for v in in_dd:
        cur = cur + 1 if v else 0
        max_dur = max(max_dur, cur)
    return max_dur


def profit_factor(returns) -> float:
    """Gross profit / gross loss. > 1.5 = good; > 2.0 = excellent."""
    r = _to_arr(returns)
    gross_win  = np.sum(r[r > 0])
    gross_loss = np.abs(np.sum(r[r < 0]))
    if gross_loss == 0:
        return float("inf")
    return float(gross_win / gross_loss)


def calmar(returns, equity_curve, periods_per_year: int = 252) -> float:
    """Annual return / |max drawdown|. > 3 = institutional grade."""
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return float("inf")
    r   = _to_arr(returns)
    ann = annualised(np.mean(r), periods_per_year)
    return float(ann / mdd)


def expectancy(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Expected return per trade in currency units."""
    return win_rate * avg_win - (1 - win_rate) * avg_loss


def win_rate(returns) -> float:
    r = _to_arr(returns)
    if len(r) == 0:
        return 0.0
    return float(np.sum(r > 0) / len(r))


# ── Master report ─────────────────────────────────────────────────────────────

def full_report(
    trade_returns:  list[float],     # PnL per trade in currency
    equity_curve:   list[float],     # running equity
    initial_capital: float = 10_000,
    periods_per_year: int  = 252,
    risk_free:       float = 0.0,
) -> dict:
    """Compute the complete performance report."""
    r   = _to_arr(trade_returns)
    eq  = _to_arr(equity_curve)

    if len(r) == 0:
        return {"error": "No trades"}

    wins  = r[r > 0]
    losses= r[r < 0]
    wr    = float(len(wins) / len(r)) if len(r) > 0 else 0.0
    aw    = float(np.mean(wins))   if len(wins)   > 0 else 0.0
    al    = float(np.mean(np.abs(losses))) if len(losses) > 0 else 0.0

    # Percentage returns for ratio metrics
    pct_r = r / initial_capital

    total_pnl  = float(np.sum(r))
    total_ret  = total_pnl / initial_capital

    k     = kelly(wr, aw, al)
    mdd   = max_drawdown(eq)

    return {
        # Returns
        "total_pnl":         round(total_pnl, 2),
        "total_return_pct":  round(total_ret * 100, 2),
        "avg_trade_pnl":     round(float(np.mean(r)), 2),

        # Trade stats
        "n_trades":          len(r),
        "n_wins":            int(len(wins)),
        "n_losses":          int(len(losses)),
        "win_rate":          round(wr, 4),
        "avg_win":           round(aw, 2),
        "avg_loss":          round(al, 2),
        "best_trade":        round(float(np.max(r)), 2),
        "worst_trade":       round(float(np.min(r)), 2),

        # Risk-adjusted
        "sharpe":            round(sharpe(pct_r, risk_free, periods_per_year), 3),
        "sortino":           round(sortino(pct_r, risk_free, periods_per_year), 3),
        "omega":             round(omega(pct_r), 3),
        "calmar":            round(calmar(pct_r, eq, periods_per_year), 3),
        "profit_factor":     round(profit_factor(r), 3),
        "expectancy":        round(expectancy(wr, aw, al), 2),

        # Drawdown
        "max_drawdown_pct":  round(mdd * 100, 2),
        "max_dd_duration":   max_drawdown_duration(eq),

        # Sizing
        "kelly_full":        round(k, 4),
        "kelly_half":        round(k / 2, 4),
    }
