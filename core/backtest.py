"""
Vectorized backtesting engine — zero lookahead bias.
Supports walk-forward validation, per-signal breakdown, equity curve.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from core.models import Bar
from core.signals import Signal
from core.metrics import full_report


# ── Trade result ──────────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    signal:       Signal
    entry_price:  float
    exit_price:   float
    exit_bar_idx: int
    exit_reason:  str          # 'tp' | 'sl' | 'timeout'
    pnl:          float        # currency
    pnl_pct:      float
    r_multiple:   float        # how many R did we make/lose

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class BacktestResult:
    trades:        list[TradeResult]
    equity_curve:  list[float]
    initial_capital: float
    report:        dict
    signals_used:  int
    signals_total: int

    def by_type(self) -> dict[str, list[TradeResult]]:
        out: dict[str, list] = {}
        for t in self.trades:
            out.setdefault(t.signal.signal_type, []).append(t)
        return out


# ── Engine ────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Event-driven simulation on bar data.
    One trade at a time (no pyramiding).
    Entry at signal bar close + 1 tick slippage.
    Exit at TP/SL or max_bars_in_trade timeout.
    """

    def __init__(
        self,
        initial_capital:   float = 10_000.0,
        risk_per_trade:    float = 0.01,       # 1% risk per trade
        max_bars_in_trade: int   = 20,
        commission_pct:    float = 0.0005,     # 0.05% per side (Bybit taker)
        slippage_pct:      float = 0.0002,     # 0.02% entry slippage
    ):
        self.initial_capital   = initial_capital
        self.risk_per_trade    = risk_per_trade
        self.max_bars_in_trade = max_bars_in_trade
        self.commission_pct    = commission_pct
        self.slippage_pct      = slippage_pct

    def _position_size(self, capital: float, entry: float, stop: float) -> float:
        """Risk-based position sizing: risk_pct * capital / distance_to_stop."""
        risk_amount = capital * self.risk_per_trade
        distance    = abs(entry - stop)
        if distance == 0:
            return 0.0
        return risk_amount / distance

    def run(
        self,
        bars:    list[Bar],
        signals: list[Signal],
        is_pct:  float = 1.0,      # 1.0 = use all bars (for walk-forward pass 0.7)
    ) -> BacktestResult:

        cutoff    = int(len(bars) * is_pct)
        is_sigs   = [s for s in signals if s.bar_idx < cutoff]
        sig_map   = {s.bar_idx: s for s in is_sigs}

        capital      = self.initial_capital
        equity       = [capital]
        trades: list[TradeResult] = []

        in_trade  = False
        trade_sig = None
        trade_entry = 0.0
        bars_held = 0

        for i, bar in enumerate(bars[:cutoff]):

            if in_trade and trade_sig is not None:
                bars_held += 1
                sl = trade_sig.stop_loss
                tp = trade_sig.take_profit
                direction = trade_sig.direction

                hit_sl = hit_tp = timeout = False

                if direction == "long":
                    hit_tp = bar.high >= tp
                    hit_sl = bar.low  <= sl
                else:
                    hit_tp = bar.low  <= tp
                    hit_sl = bar.high >= sl

                timeout = bars_held >= self.max_bars_in_trade

                if hit_tp or hit_sl or timeout:
                    exit_price  = tp if hit_tp else sl if hit_sl else bar.close
                    exit_reason = "tp" if hit_tp else "sl" if hit_sl else "timeout"

                    size = self._position_size(capital, trade_entry, sl)
                    if size == 0:
                        in_trade = False
                        continue

                    raw_pnl  = (exit_price - trade_entry) * size * (1 if direction == "long" else -1)
                    costs    = (trade_entry + exit_price) * size * (self.commission_pct + self.slippage_pct / 2)
                    pnl      = raw_pnl - costs
                    pnl_pct  = pnl / capital
                    r_mult   = pnl / (capital * self.risk_per_trade) if capital > 0 else 0.0

                    capital  = max(capital + pnl, 0.0)
                    equity.append(capital)

                    trades.append(TradeResult(
                        signal       = trade_sig,
                        entry_price  = trade_entry,
                        exit_price   = exit_price,
                        exit_bar_idx = i,
                        exit_reason  = exit_reason,
                        pnl          = pnl,
                        pnl_pct      = pnl_pct,
                        r_multiple   = r_mult,
                    ))
                    in_trade = False

            # New signal — only enter if not in trade
            if not in_trade and i in sig_map:
                s             = sig_map[i]
                trade_sig     = s
                trade_entry   = s.price * (1 + self.slippage_pct * (1 if s.direction == "long" else -1))
                in_trade      = True
                bars_held     = 0

        pnls   = [t.pnl for t in trades]
        report = full_report(pnls, equity, self.initial_capital) if trades else {"error": "No trades"}

        return BacktestResult(
            trades         = trades,
            equity_curve   = equity,
            initial_capital= self.initial_capital,
            report         = report,
            signals_used   = len(is_sigs),
            signals_total  = len(signals),
        )


# ── Walk-Forward Validation ───────────────────────────────────────────────────

@dataclass
class WalkForwardResult:
    is_result:  BacktestResult     # In-sample
    oos_result: BacktestResult     # Out-of-sample
    is_pct:     float
    degradation: float             # OOS Sharpe / IS Sharpe (1.0 = no degradation)

    @property
    def has_edge(self) -> bool:
        """OOS Sharpe > 0.5 AND profit factor > 1.2 → genuine edge."""
        oos = self.oos_result.report
        return (
            oos.get("sharpe", 0)        > 0.5
            and oos.get("profit_factor", 0) > 1.2
            and oos.get("n_trades", 0)  >= 10
        )


def walk_forward(
    bars:       list[Bar],
    signals:    list[Signal],
    is_pct:     float = 0.70,
    engine_cfg: dict  = None,
) -> WalkForwardResult:
    """
    Split data: first is_pct for in-sample optimisation,
    remaining for out-of-sample validation.
    """
    cfg    = engine_cfg or {}
    engine = BacktestEngine(**cfg)

    cutoff   = int(len(bars) * is_pct)
    is_sigs  = [s for s in signals if s.bar_idx <  cutoff]
    oos_sigs = [s for s in signals if s.bar_idx >= cutoff]

    # Remap OOS bar indices to start from 0 for OOS slice
    oos_bars = bars[cutoff:]
    oos_sigs_remapped = []
    for s in oos_sigs:
        import copy
        s2 = copy.copy(s)
        s2.bar_idx = s.bar_idx - cutoff
        oos_sigs_remapped.append(s2)

    is_result  = engine.run(bars,     is_sigs,  is_pct=1.0)
    oos_result = engine.run(oos_bars, oos_sigs_remapped, is_pct=1.0)

    is_sharpe  = is_result.report.get("sharpe",  0.0)
    oos_sharpe = oos_result.report.get("sharpe", 0.0)
    degradation = oos_sharpe / is_sharpe if is_sharpe != 0 else 0.0

    return WalkForwardResult(
        is_result   = is_result,
        oos_result  = oos_result,
        is_pct      = is_pct,
        degradation = round(degradation, 3),
    )
