import unittest

import sys
import types

# The unit tests exercise only the pure strategy/research logic. Avoid requiring
# the optional network dependency yfinance in the test environment.
if "yfinance" not in sys.modules:
    yf_stub = types.ModuleType("yfinance")
    yf_stub.download = lambda *args, **kwargs: None
    sys.modules["yfinance"] = yf_stub

import numpy as np
import pandas as pd

from research import build_membership_map, generate_configs, oos_equity_curves, run_parameter_sweep
from strategy import StrategyConfig, backtest, is_active_on, latest_signals, select_positions
from pit_data import _infer_membership_windows, merge_pit_with_yahoo


class MomentumCoreTests(unittest.TestCase):
    @staticmethod
    def synthetic_prices():
        idx = pd.bdate_range("2020-01-02", "2026-01-30")
        rng = np.random.default_rng(42)
        data = {}
        drifts = {"AAA": 0.00055, "BBB": 0.00035, "CCC": 0.00010, "DDD": -0.00005}
        for ticker, drift in drifts.items():
            shocks = rng.normal(drift, 0.009, len(idx))
            data[ticker] = 100 * np.exp(np.cumsum(shocks))
        return pd.DataFrame(data, index=idx)

    def test_scanner_selects_positions(self):
        prices = self.synthetic_prices()
        cfg = StrategyConfig(lookback_months=6, sma_days=100, top_n=2, max_per_sector=1)
        signals = latest_signals(prices, cfg)
        sectors = {"AAA": "A", "BBB": "B", "CCC": "C", "DDD": "D"}
        picks = select_positions(signals, sectors, cfg)
        self.assertLessEqual(len(picks), 2)
        self.assertTrue(set(picks.index).issubset(set(prices.columns)))

    def test_scanner_normalizes_yfinance_ticker_index_name(self):
        prices = self.synthetic_prices()
        prices.columns.name = "Ticker"
        cfg = StrategyConfig(lookback_months=6, sma_days=100, top_n=2, max_per_sector=1)
        signals = latest_signals(prices, cfg)
        self.assertEqual(signals.index.name, "ticker")
        top = signals.head(30).copy().reset_index().rename(columns={"index": "ticker"})
        self.assertIn("ticker", top.columns)
        sectors = {t: t for t in prices.columns}
        top["sector"] = top["ticker"].map(sectors).fillna("Unknown")
        display = top[["ticker", "sector", "momentum_12_1", "above_sma", "price", "sma200"]]
        self.assertFalse(display.empty)

    def test_backtest_and_membership(self):
        prices = self.synthetic_prices()
        benchmark = prices.mean(axis=1)
        sectors = {t: t for t in prices.columns}
        cfg = StrategyConfig(lookback_months=6, sma_days=100, top_n=2, max_per_sector=1)
        membership = {
            "AAA": (None, pd.Timestamp("2022-12-31")),
            "BBB": (None, None),
            "CCC": (None, None),
            "DDD": (None, None),
        }
        result, trades = backtest(prices, sectors, cfg, benchmark=benchmark, membership=membership)
        self.assertFalse(result.empty)
        self.assertIn("strategy_equity", result.columns)
        late_aaa = trades[(trades["ticker"] == "AAA") & (trades["signal_date"] > pd.Timestamp("2022-12-31"))]
        self.assertTrue(late_aaa.empty)

    def test_membership_csv_detection(self):
        df = pd.DataFrame({
            "ticker": ["AAA", "BBB"],
            "active_from": ["2021-01-01", ""],
            "active_to": ["", "2024-12-31"],
        })
        membership = build_membership_map(df)
        self.assertIsNotNone(membership)
        self.assertEqual(membership["AAA"][0][0], pd.Timestamp("2021-01-01"))
        self.assertEqual(membership["BBB"][0][1], pd.Timestamp("2024-12-31"))


    def test_multiple_membership_windows(self):
        membership = {
            "AAA": [
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
                (pd.Timestamp("2023-01-01"), None),
            ]
        }
        self.assertTrue(is_active_on("AAA", pd.Timestamp("2021-06-01"), membership))
        self.assertFalse(is_active_on("AAA", pd.Timestamp("2022-06-01"), membership))
        self.assertTrue(is_active_on("AAA", pd.Timestamp("2024-06-01"), membership))
        self.assertFalse(is_active_on("MISSING", pd.Timestamp("2024-06-01"), membership))

    def test_infer_pit_windows_from_index_calendar(self):
        index_dates = pd.bdate_range("2020-01-01", "2020-06-30")
        first = index_dates[:35]
        second = index_dates[80:120]
        member_dates = first.append(second)
        windows = _infer_membership_windows(member_dates, index_dates, max_missing_index_days=20)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0][0], first[0])
        self.assertEqual(windows[0][1], first[-1])
        self.assertEqual(windows[1][0], second[0])

    def test_pit_prices_override_yahoo_but_yahoo_fills_warmup(self):
        dates = pd.date_range("2020-01-01", periods=4, freq="D")
        pit = pd.DataFrame({"AAA": [np.nan, 101.0, 102.0, np.nan]}, index=dates)
        yahoo = pd.DataFrame({"AAA": [90.0, 99.0, 100.0, 103.0]}, index=dates)
        merged = merge_pit_with_yahoo(pit, yahoo)
        self.assertEqual(merged.loc[dates[0], "AAA"], 90.0)
        self.assertEqual(merged.loc[dates[1], "AAA"], 101.0)
        self.assertEqual(merged.loc[dates[2], "AAA"], 102.0)
        self.assertEqual(merged.loc[dates[3], "AAA"], 103.0)

    def test_duplicate_csv_rows_build_multiple_windows(self):
        df = pd.DataFrame({
            "ticker": ["AAA", "AAA"],
            "active_from": ["2020-01-01", "2023-01-01"],
            "active_to": ["2021-12-31", ""],
        })
        membership = build_membership_map(df)
        self.assertEqual(len(membership["AAA"]), 2)
        self.assertFalse(is_active_on("AAA", pd.Timestamp("2022-06-01"), membership))

    def test_parameter_sweep_has_oos_metrics(self):
        prices = self.synthetic_prices()
        benchmark = prices.mean(axis=1)
        sectors = {t: t for t in prices.columns}
        cfg = StrategyConfig(sma_days=100, max_per_sector=1, transaction_cost_bps=5)
        summary, curves, split_date = run_parameter_sweep(
            prices=prices,
            sectors=sectors,
            benchmark=benchmark,
            base_cfg=cfg,
            start=pd.Timestamp("2021-01-01"),
            end=pd.Timestamp("2025-12-31"),
            train_fraction=0.70,
            lookbacks=[6, 9],
            top_ns=[2, 3],
            stop_losses=[0.10],
            market_filters=[False],
        )
        self.assertEqual(len(summary), 4)
        self.assertEqual(len(curves), 4)
        self.assertIn("Test CAGR", summary.columns)
        self.assertIn("OOS excess CAGR", summary.columns)
        self.assertTrue(pd.Timestamp("2024-01-01") < split_date < pd.Timestamp("2025-01-01"))
        oos = oos_equity_curves(summary, curves, split_date, 100_000, top_k=2)
        self.assertFalse(oos.empty)
        self.assertIn("Benchmark", oos.columns)

    def test_default_grid_count(self):
        cfg = StrategyConfig()
        configs = generate_configs(cfg, [6, 9, 12], [4, 6, 8], [0.08, 0.12, 0.15], [True])
        self.assertEqual(len(configs), 27)


if __name__ == "__main__":
    unittest.main()
