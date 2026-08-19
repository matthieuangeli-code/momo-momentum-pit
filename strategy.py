from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yfinance as yf


MembershipWindow = Tuple[pd.Timestamp | None, pd.Timestamp | None]
MembershipValue = Union[MembershipWindow, Sequence[MembershipWindow]]
MembershipMap = Dict[str, MembershipValue]


@dataclass
class StrategyConfig:
    lookback_months: int = 12
    skip_months: int = 1
    sma_days: int = 200
    top_n: int = 6
    top_percentile: float = 0.20
    max_per_sector: int = 2
    stop_loss: float = 0.12
    market_filter: bool = True
    defensive_exposure: float = 0.50
    transaction_cost_bps: float = 10.0


def _normalize_close(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Return a Date x Ticker close-price DataFrame from yfinance output."""
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        if "Close" in lvl0:
            close = raw["Close"].copy()
        elif "Adj Close" in lvl0:
            close = raw["Adj Close"].copy()
        else:
            raise ValueError("Impossible de trouver les colonnes Close dans les données Yahoo.")
    else:
        col = "Close" if "Close" in raw.columns else "Adj Close"
        close = raw[[col]].copy()
        close.columns = [tickers[0]]

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    close = close.sort_index().dropna(axis=1, how="all")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close


def download_prices(
    tickers: Iterable[str],
    start: str,
    end: str | None = None,
    batch_size: int = 80,
) -> pd.DataFrame:
    """Download adjusted daily close prices in batches.

    yfinance auto_adjust=True means Close is adjusted for splits/dividends.
    """
    clean = list(dict.fromkeys(t.strip().upper() for t in tickers if str(t).strip()))
    frames: List[pd.DataFrame] = []
    for i in range(0, len(clean), batch_size):
        batch = clean[i : i + batch_size]
        raw = yf.download(
            batch,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
            timeout=20,
        )
        if not raw.empty:
            frames.append(_normalize_close(raw, batch))

    if not frames:
        return pd.DataFrame()
    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices.sort_index()


def latest_signals(prices: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Compute 12-1 momentum + SMA signal from daily adjusted closes."""
    if prices.empty:
        return pd.DataFrame()

    monthly = prices.resample("ME").last()
    # Classic 12-1: price one month ago divided by price 12 months ago.
    mom = monthly.shift(cfg.skip_months) / monthly.shift(cfg.lookback_months) - 1.0
    sma = prices.rolling(cfg.sma_days, min_periods=cfg.sma_days).mean()

    last_px = prices.ffill().iloc[-1]
    last_sma = sma.ffill().iloc[-1]
    last_mom = mom.ffill().iloc[-1]

    out = pd.DataFrame({
        "price": last_px,
        "momentum_12_1": last_mom,
        "sma200": last_sma,
    })
    out["above_sma"] = out["price"] > out["sma200"]
    out["rank"] = out["momentum_12_1"].rank(ascending=False, method="min")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["momentum_12_1", "sma200"])
    # yfinance often leaves the columns index named "Ticker". Normalize the
    # signal index name so Streamlit reset_index() always creates `ticker`.
    out.index.name = "ticker"
    return out.sort_values("momentum_12_1", ascending=False)


def select_positions(
    signals: pd.DataFrame,
    sectors: Dict[str, str],
    cfg: StrategyConfig,
) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()

    eligible = signals[signals["above_sma"]].copy()
    if eligible.empty:
        return eligible

    n_top_bucket = max(cfg.top_n, int(np.ceil(len(signals) * cfg.top_percentile)))
    eligible = eligible.sort_values("momentum_12_1", ascending=False).head(n_top_bucket)
    eligible["sector"] = [sectors.get(t, "Unknown") for t in eligible.index]

    chosen = []
    sector_counts: Dict[str, int] = {}
    for ticker, row in eligible.iterrows():
        sector = row["sector"]
        # Historical/delisted names may not have sector metadata. Treat each
        # unknown ticker as its own bucket instead of accidentally allowing
        # only `max_per_sector` unknown names in the whole portfolio.
        sector_key = f"Unknown::{ticker}" if sector == "Unknown" else sector
        if sector_counts.get(sector_key, 0) >= cfg.max_per_sector:
            continue
        chosen.append(ticker)
        sector_counts[sector_key] = sector_counts.get(sector_key, 0) + 1
        if len(chosen) >= cfg.top_n:
            break

    result = eligible.loc[chosen].copy() if chosen else eligible.iloc[0:0].copy()
    if not result.empty:
        result["weight"] = 1.0 / len(result)
        result["stop_price"] = result["price"] * (1.0 - cfg.stop_loss)
    return result


def _month_end_dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
    # Last actual trading date in each calendar month.
    s = pd.Series(prices.index, index=prices.index)
    return pd.DatetimeIndex(s.groupby(prices.index.to_period("M")).last().values)


def _membership_windows(value: MembershipValue | None) -> list[MembershipWindow]:
    if value is None:
        return []
    # Backward-compatible single tuple: (start, end).
    if isinstance(value, tuple) and len(value) == 2 and not isinstance(value[0], (tuple, list)):
        return [value]  # type: ignore[list-item]
    return list(value)  # type: ignore[arg-type]


def is_active_on(
    ticker: str,
    date: pd.Timestamp,
    membership: MembershipMap | None,
) -> bool:
    if not membership:
        return True
    if ticker not in membership:
        return False
    for start, end in _membership_windows(membership[ticker]):
        if (start is None or date >= start) and (end is None or date <= end):
            return True
    return False


def _filter_by_membership(
    signals: pd.DataFrame,
    date: pd.Timestamp,
    membership: MembershipMap | None,
) -> pd.DataFrame:
    """Filter signals to securities active in a point-in-time universe.

    A ticker may have multiple membership windows (exit then re-entry). When a
    membership map is supplied, tickers absent from it are deliberately
    excluded; with membership=None, the current-universe behaviour is kept.
    """
    if signals.empty or not membership:
        return signals
    keep = [ticker for ticker in signals.index if is_active_on(ticker, date, membership)]
    return signals.loc[keep]

def _select_on_date(
    date: pd.Timestamp,
    prices: pd.DataFrame,
    monthly_close: pd.DataFrame,
    sma: pd.DataFrame,
    sectors: Dict[str, str],
    cfg: StrategyConfig,
    membership: MembershipMap | None = None,
) -> List[str]:
    month_period = date.to_period("M")
    month_periods = monthly_close.index.to_period("M")
    positions = np.where(month_periods == month_period)[0]
    if len(positions) == 0:
        return []
    i = int(positions[-1])
    if i < cfg.lookback_months:
        return []

    # At month-end t, 12-1 = month t-1 / month t-12.
    p1 = monthly_close.iloc[i - cfg.skip_months]
    p12 = monthly_close.iloc[i - cfg.lookback_months]
    mom = p1 / p12 - 1.0
    px = prices.loc[:date].ffill().iloc[-1]
    sma_now = sma.loc[:date].ffill().iloc[-1]
    signals = pd.DataFrame({"price": px, "momentum_12_1": mom, "sma200": sma_now})
    signals["above_sma"] = signals["price"] > signals["sma200"]
    signals = signals.replace([np.inf, -np.inf], np.nan).dropna(subset=["momentum_12_1", "sma200", "price"])
    signals = _filter_by_membership(signals, date, membership)
    selected = select_positions(signals, sectors, cfg)
    return selected.index.tolist()


def backtest(
    prices: pd.DataFrame,
    sectors: Dict[str, str],
    cfg: StrategyConfig,
    benchmark: pd.Series | None = None,
    initial_capital: float = 100_000.0,
    membership: MembershipMap | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly-rebalanced equal-weight momentum backtest.

    Stop loss is applied approximately using daily adjusted closes. If a close falls below
    the entry stop during the month, the position goes to cash for the rest of that month.
    Market filter: benchmark below its SMA200 reduces next month's total exposure to
    cfg.defensive_exposure; otherwise exposure is 100%.
    """
    if prices.empty:
        return pd.DataFrame(), pd.DataFrame()

    prices = prices.sort_index().ffill(limit=5)
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    monthly_close = prices.resample("ME").last()
    sma = prices.rolling(cfg.sma_days, min_periods=cfg.sma_days).mean()
    rebalance_dates = _month_end_dates(prices)

    benchmark = benchmark.reindex(prices.index).ffill() if benchmark is not None else None
    bench_sma = benchmark.rolling(cfg.sma_days, min_periods=cfg.sma_days).mean() if benchmark is not None else None

    portfolio_returns = pd.Series(0.0, index=prices.index, name="strategy_return")
    trade_rows = []
    prev_weights: Dict[str, float] = {}
    cost_rate = cfg.transaction_cost_bps / 10_000.0

    for k in range(len(rebalance_dates) - 1):
        signal_date = rebalance_dates[k]
        next_signal_date = rebalance_dates[k + 1]
        selected = _select_on_date(signal_date, prices, monthly_close, sma, sectors, cfg, membership)

        exposure = 1.0
        if cfg.market_filter and benchmark is not None and bench_sma is not None:
            b = benchmark.loc[:signal_date].dropna()
            bs = bench_sma.loc[:signal_date].dropna()
            if not b.empty and not bs.empty and b.iloc[-1] < bs.iloc[-1]:
                exposure = cfg.defensive_exposure

        hold_idx = prices.index[(prices.index > signal_date) & (prices.index <= next_signal_date)]
        if len(hold_idx) == 0:
            continue

        target_weights = ({t: exposure / len(selected) for t in selected} if selected else {})
        all_names = set(prev_weights) | set(target_weights)
        turnover = sum(abs(target_weights.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in all_names)
        rebalance_cost = turnover * cost_rate

        month_ret = pd.Series(0.0, index=hold_idx)
        if rebalance_cost:
            month_ret.iloc[0] -= rebalance_cost

        if not selected:
            portfolio_returns.loc[hold_idx] = month_ret
            prev_weights = {}
            continue

        end_weights: Dict[str, float] = {}
        for ticker in selected:
            w = target_weights[ticker]
            entry_price = prices.loc[:signal_date, ticker].dropna().iloc[-1]
            stop_price = entry_price * (1.0 - cfg.stop_loss)
            ticker_rets = returns.loc[hold_idx, ticker].fillna(0.0).copy()
            ticker_px = prices.loc[hold_idx, ticker]

            stopped = ticker_px <= stop_price
            stop_date = stopped.idxmax() if stopped.any() else None

            membership_exit_date = None
            if membership:
                inactive = pd.Series(
                    [not is_active_on(ticker, d, membership) for d in hold_idx],
                    index=hold_idx,
                )
                membership_exit_date = inactive.idxmax() if inactive.any() else None

            exit_dates = [d for d in (stop_date, membership_exit_date) if d is not None]
            effective_exit = min(exit_dates) if exit_dates else None
            if effective_exit is not None:
                # Keep returns through the last active/stop observation, then cash.
                # For a membership exit we zero the first inactive day itself;
                # for a price stop we include that day's close-to-close move.
                if membership_exit_date is not None and effective_exit == membership_exit_date:
                    ticker_rets.loc[ticker_rets.index >= effective_exit] = 0.0
                else:
                    ticker_rets.loc[ticker_rets.index > effective_exit] = 0.0
                ticker_rets.loc[effective_exit] -= cost_rate
                end_weights[ticker] = 0.0
            else:
                end_weights[ticker] = w

            month_ret = month_ret.add(w * ticker_rets, fill_value=0.0)
            trade_rows.append({
                "signal_date": signal_date,
                "ticker": ticker,
                "entry_price": float(entry_price),
                "stop_price": float(stop_price),
                "stop_date": stop_date,
                "stopped": stop_date is not None,
                "membership_exit_date": membership_exit_date,
                "exposure": exposure,
            })

        portfolio_returns.loc[hold_idx] = month_ret
        prev_weights = {t: w for t, w in end_weights.items() if w > 0}

    equity = initial_capital * (1.0 + portfolio_returns).cumprod()
    result = pd.DataFrame({
        "strategy_return": portfolio_returns,
        "strategy_equity": equity,
    })

    if benchmark is not None:
        bench_ret = benchmark.pct_change(fill_method=None).fillna(0.0)
        # Align benchmark start to first meaningful strategy observation.
        result["benchmark_return"] = bench_ret
        result["benchmark_equity"] = initial_capital * (1.0 + bench_ret).cumprod()

    trades = pd.DataFrame(trade_rows)
    return result, trades


def performance_stats(equity: pd.Series, returns: pd.Series) -> Dict[str, float]:
    x = equity.dropna()
    r = returns.reindex(x.index).fillna(0.0)
    if len(x) < 2:
        return {}

    days = max((x.index[-1] - x.index[0]).days, 1)
    years = days / 365.25
    total_return = x.iloc[-1] / x.iloc[0] - 1.0
    cagr = (x.iloc[-1] / x.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    vol = r.std(ddof=0) * np.sqrt(252)
    sharpe = (r.mean() * 252) / vol if vol > 0 else np.nan
    dd = x / x.cummax() - 1.0

    return {
        "Total return": float(total_return),
        "CAGR": float(cagr),
        "Volatility": float(vol),
        "Sharpe (rf=0)": float(sharpe),
        "Max drawdown": float(dd.min()),
    }
