from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from strategy import MembershipMap, StrategyConfig, backtest


ProgressCallback = Callable[[int, int, str], None]


def build_membership_map(universe: pd.DataFrame) -> MembershipMap | None:
    """Build point-in-time membership windows from optional CSV columns.

    Expected optional columns: active_from, active_to (YYYY-MM-DD).
    If no actual date is supplied, return None so the caller can explicitly
    report that the backtest still uses today's universe.
    """
    required = {"ticker", "active_from", "active_to"}
    if not required.issubset(universe.columns):
        return None

    starts = pd.to_datetime(universe["active_from"], errors="coerce")
    ends = pd.to_datetime(universe["active_to"], errors="coerce")
    if not starts.notna().any() and not ends.notna().any():
        return None

    membership: MembershipMap = {}
    for ticker, start, end in zip(universe["ticker"], starts, ends):
        key = str(ticker).upper()
        window = (
            None if pd.isna(start) else pd.Timestamp(start).tz_localize(None),
            None if pd.isna(end) else pd.Timestamp(end).tz_localize(None),
        )
        # Duplicate rows are allowed so a security can have multiple index
        # membership windows (exit then re-entry).
        membership.setdefault(key, [])
        membership[key].append(window)  # type: ignore[union-attr]
    return membership


def config_label(cfg: StrategyConfig) -> str:
    mf = "MF on" if cfg.market_filter else "MF off"
    return f"L{cfg.lookback_months}-1 | N{cfg.top_n} | SL{cfg.stop_loss:.0%} | {mf}"


def generate_configs(
    base_cfg: StrategyConfig,
    lookbacks: Iterable[int],
    top_ns: Iterable[int],
    stop_losses: Iterable[float],
    market_filters: Iterable[bool],
) -> list[StrategyConfig]:
    configs = []
    for lookback, top_n, stop_loss, market_filter in product(
        sorted(set(int(x) for x in lookbacks)),
        sorted(set(int(x) for x in top_ns)),
        sorted(set(float(x) for x in stop_losses)),
        list(dict.fromkeys(bool(x) for x in market_filters)),
    ):
        configs.append(
            replace(
                base_cfg,
                lookback_months=lookback,
                top_n=top_n,
                stop_loss=stop_loss,
                market_filter=market_filter,
            )
        )
    return configs


def _stats_from_returns(returns: pd.Series) -> Dict[str, float]:
    r = returns.dropna().astype(float)
    if len(r) < 2:
        return {
            "Total return": np.nan,
            "CAGR": np.nan,
            "Volatility": np.nan,
            "Sharpe (rf=0)": np.nan,
            "Max drawdown": np.nan,
        }

    growth = (1.0 + r).cumprod()
    total_return = float(growth.iloc[-1] - 1.0)
    days = max((r.index[-1] - r.index[0]).days, 1)
    years = days / 365.25
    cagr = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1 else -1.0
    vol = float(r.std(ddof=0) * np.sqrt(252))
    sharpe = float((r.mean() * 252) / vol) if vol > 0 else np.nan

    # Include the starting capital (1.0) so an immediate loss is a drawdown.
    equity = np.concatenate([[1.0], growth.to_numpy()])
    running_max = np.maximum.accumulate(equity)
    dd = equity / running_max - 1.0

    return {
        "Total return": total_return,
        "CAGR": float(cagr),
        "Volatility": vol,
        "Sharpe (rf=0)": sharpe,
        "Max drawdown": float(dd.min()),
    }


def chronological_split(start: pd.Timestamp, end: pd.Timestamp, train_fraction: float) -> pd.Timestamp:
    train_fraction = float(np.clip(train_fraction, 0.50, 0.90))
    span = end - start
    return (start + span * train_fraction).normalize()


def run_parameter_sweep(
    prices: pd.DataFrame,
    sectors: Dict[str, str],
    benchmark: pd.Series | None,
    base_cfg: StrategyConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_fraction: float,
    lookbacks: Iterable[int],
    top_ns: Iterable[int],
    stop_losses: Iterable[float],
    market_filters: Iterable[bool],
    initial_capital: float = 100_000.0,
    membership: MembershipMap | None = None,
    progress: ProgressCallback | None = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.Timestamp]:
    """Test a parameter grid with a strict chronological train/OOS split.

    The configuration ranking is based only on the training Sharpe. The OOS
    metrics are reported afterwards and are never used to choose the winner.
    """
    start = pd.Timestamp(start).tz_localize(None)
    end = pd.Timestamp(end).tz_localize(None)
    split_date = chronological_split(start, end, train_fraction)
    test_start = split_date + pd.Timedelta(days=1)

    configs = generate_configs(base_cfg, lookbacks, top_ns, stop_losses, market_filters)
    rows: list[dict] = []
    curves: Dict[str, pd.DataFrame] = {}

    total = len(configs)
    for i, cfg in enumerate(configs, start=1):
        label = config_label(cfg)
        if progress:
            progress(i, total, label)

        result, trades = backtest(
            prices=prices,
            sectors=sectors,
            cfg=cfg,
            benchmark=benchmark,
            initial_capital=initial_capital,
            membership=membership,
        )
        result = result.loc[start:end].copy()
        curves[label] = result

        train_ret = result.loc[start:split_date, "strategy_return"] if not result.empty else pd.Series(dtype=float)
        test_ret = result.loc[test_start:end, "strategy_return"] if not result.empty else pd.Series(dtype=float)
        train_stats = _stats_from_returns(train_ret)
        test_stats = _stats_from_returns(test_ret)

        if "benchmark_return" in result:
            bench_test_stats = _stats_from_returns(result.loc[test_start:end, "benchmark_return"])
        else:
            bench_test_stats = {"CAGR": np.nan, "Sharpe (rf=0)": np.nan, "Max drawdown": np.nan}

        n_rebalances = int(trades["signal_date"].nunique()) if not trades.empty else 0
        avg_positions = float(len(trades) / n_rebalances) if n_rebalances else 0.0
        stop_rate = float(trades["stopped"].mean()) if not trades.empty and "stopped" in trades else np.nan

        rows.append({
            "Config": label,
            "Lookback": cfg.lookback_months,
            "Positions": cfg.top_n,
            "Stop": cfg.stop_loss,
            "Market filter": cfg.market_filter,
            "Train CAGR": train_stats["CAGR"],
            "Train Sharpe": train_stats["Sharpe (rf=0)"],
            "Train Max DD": train_stats["Max drawdown"],
            "Test CAGR": test_stats["CAGR"],
            "Test Sharpe": test_stats["Sharpe (rf=0)"],
            "Test Max DD": test_stats["Max drawdown"],
            "Benchmark Test CAGR": bench_test_stats["CAGR"],
            "Benchmark Test Sharpe": bench_test_stats["Sharpe (rf=0)"],
            "OOS excess CAGR": test_stats["CAGR"] - bench_test_stats["CAGR"] if pd.notna(bench_test_stats["CAGR"]) else np.nan,
            "Sharpe degradation": test_stats["Sharpe (rf=0)"] - train_stats["Sharpe (rf=0)"],
            "Avg positions": avg_positions,
            "Stop rate": stop_rate,
        })

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary, curves, split_date

    summary["Train rank"] = summary["Train Sharpe"].rank(ascending=False, method="min")
    summary = summary.sort_values(["Train rank", "Train CAGR"], ascending=[True, False]).reset_index(drop=True)
    return summary, curves, split_date


def oos_equity_curves(
    summary: pd.DataFrame,
    curves: Dict[str, pd.DataFrame],
    split_date: pd.Timestamp,
    initial_capital: float,
    top_k: int = 3,
) -> pd.DataFrame:
    """Return rebased OOS equity curves for the best training configurations."""
    if summary.empty:
        return pd.DataFrame()

    test_start = split_date + pd.Timedelta(days=1)
    out: Dict[str, pd.Series] = {}
    for label in summary.head(top_k)["Config"]:
        result = curves.get(label)
        if result is None or result.empty:
            continue
        r = result.loc[test_start:, "strategy_return"].dropna()
        if r.empty:
            continue
        out[label] = initial_capital * (1.0 + r).cumprod()

    # Benchmark is identical for every configuration; take it from the first curve.
    if curves:
        first = next(iter(curves.values()))
        if "benchmark_return" in first:
            br = first.loc[test_start:, "benchmark_return"].dropna()
            if not br.empty:
                out["Benchmark"] = initial_capital * (1.0 + br).cumprod()

    return pd.DataFrame(out)
