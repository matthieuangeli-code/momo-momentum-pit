from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from strategy import MembershipMap, MembershipWindow

# Reproducible snapshot: commit that added the EURO STOXX 50 history.
PIT_REPOSITORY = "AndyLongest/HistoricalIndexPrices"
PIT_COMMIT = "bdb5c01084b314a94edfad155547d9373d0d8191"
PIT_INDEX = "eurostoxx50"
PIT_START = pd.Timestamp("2014-10-31")
PIT_END = pd.Timestamp("2025-08-22")
PIT_EXPECTED_SECURITIES = 66
PIT_CONFIDENCE = "medium-high"

_API_DIR = (
    f"https://api.github.com/repos/{PIT_REPOSITORY}/contents/"
    f"data/{PIT_INDEX}/constituents?ref={PIT_COMMIT}"
)
_RAW_BASE = (
    f"https://raw.githubusercontent.com/{PIT_REPOSITORY}/{PIT_COMMIT}/"
    f"data/{PIT_INDEX}"
)

ProgressCallback = Callable[[int, int, str], None]


@dataclass
class PitDataset:
    prices: pd.DataFrame
    membership: MembershipMap
    tickers: list[str]
    index_prices: pd.Series
    start: pd.Timestamp
    end: pd.Timestamp
    cache_dir: Path
    source_commit: str = PIT_COMMIT


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "MOMO-Momentum/1.0 research-only",
            "Accept": "application/vnd.github+json",
        }
    )
    return session


def _download_text(session: requests.Session, url: str, timeout: int = 45) -> str:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def _safe_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _validate_price_csv(path: Path) -> None:
    header = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    required = {"date", "symbol", "normalized_adj_close"}
    cols = {x.strip().strip('"') for x in header.split(",")}
    if not required.issubset(cols):
        raise ValueError(f"Fichier PIT invalide: {path.name}")


def cache_path(base_dir: Path) -> Path:
    return Path(base_dir) / "data" / "pit" / PIT_INDEX / PIT_COMMIT[:12]


def cache_status(base_dir: Path) -> dict[str, object]:
    root = cache_path(base_dir)
    manifest = root / "manifest.json"
    constituents = root / "constituents"
    count = len(list(constituents.glob("*/prices_daily.csv"))) if constituents.exists() else 0
    total_bytes = sum(p.stat().st_size for p in root.rglob("*.csv")) if root.exists() else 0
    return {
        "ready": manifest.exists() and count >= PIT_EXPECTED_SECURITIES,
        "files": count,
        "expected": PIT_EXPECTED_SECURITIES,
        "bytes": total_bytes,
        "path": root,
    }


def ensure_eurostoxx50_pit(
    base_dir: Path,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> Path:
    """Download and cache the pinned EURO STOXX 50 point-in-time dataset.

    Only CSV files are downloaded (not Parquet) to keep the cache compact.
    A single GitHub API request lists the 66 historical symbols; raw files are
    then fetched from the pinned commit for reproducibility.
    """
    root = cache_path(base_dir)
    manifest_path = root / "manifest.json"
    status = cache_status(base_dir)
    if status["ready"] and not force:
        return root

    root.mkdir(parents=True, exist_ok=True)
    session = _session()

    api_response = session.get(_API_DIR, timeout=45)
    api_response.raise_for_status()
    listing = api_response.json()
    tickers = sorted(
        str(item["name"]).upper()
        for item in listing
        if item.get("type") == "dir" and item.get("name")
    )
    if len(tickers) < PIT_EXPECTED_SECURITIES:
        raise RuntimeError(
            f"Liste PIT incomplète: {len(tickers)} titres trouvés, "
            f"{PIT_EXPECTED_SECURITIES} attendus."
        )

    index_path = root / "index_prices_daily.csv"
    if force or not index_path.exists():
        _safe_write_text(index_path, _download_text(session, f"{_RAW_BASE}/index_prices_daily.csv"))

    def fetch_one(ticker: str) -> tuple[str, Path]:
        target = root / "constituents" / ticker / "prices_daily.csv"
        if target.exists() and target.stat().st_size > 100 and not force:
            _validate_price_csv(target)
            return ticker, target
        url = f"{_RAW_BASE}/constituents/{ticker}/prices_daily.csv"
        # Use one short-lived session per worker; requests.Session is not guaranteed
        # to be thread-safe for arbitrary concurrent mutation.
        worker_session = _session()
        text = _download_text(worker_session, url)
        _safe_write_text(target, text)
        _validate_price_csv(target)
        return ticker, target

    total = len(tickers)
    done = 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch_one, ticker): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            future.result()
            done += 1
            if progress:
                progress(done, total, ticker)

    metadata = {
        "repository": PIT_REPOSITORY,
        "commit": PIT_COMMIT,
        "index": PIT_INDEX,
        "coverage_start": PIT_START.date().isoformat(),
        "coverage_end": PIT_END.date().isoformat(),
        "expected_securities": PIT_EXPECTED_SECURITIES,
        "membership_confidence": PIT_CONFIDENCE,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "tickers": tickers,
        "note": (
            "Historical membership/prices reconstructed by the upstream dataset from public "
            "STOXX announcements; official monthly STOXX archive requires login."
        ),
    }
    _safe_write_text(manifest_path, json.dumps(metadata, indent=2, ensure_ascii=False))
    return root


def _infer_membership_windows(
    member_dates: pd.DatetimeIndex,
    index_dates: pd.DatetimeIndex,
    max_missing_index_days: int = 20,
) -> list[MembershipWindow]:
    """Infer contiguous membership windows from member-only price rows.

    The upstream files contain prices only while the security belongs to the
    index. We split a window only when more than `max_missing_index_days`
    index trading dates are absent between two member observations. This avoids
    treating short suspensions/local holidays as index exits.
    """
    if len(member_dates) == 0:
        return []

    mdates = pd.DatetimeIndex(pd.to_datetime(member_dates)).tz_localize(None).sort_values().unique()
    idates = pd.DatetimeIndex(pd.to_datetime(index_dates)).tz_localize(None).sort_values().unique()
    if len(idates) == 0:
        return [(pd.Timestamp(mdates[0]), pd.Timestamp(mdates[-1]))]

    positions = idates.searchsorted(mdates)
    breaks = [0]
    for i in range(1, len(mdates)):
        # Number of index observations strictly between consecutive member rows.
        missing = int(positions[i] - positions[i - 1] - 1)
        if missing > max_missing_index_days:
            breaks.append(i)
    breaks.append(len(mdates))

    windows: list[MembershipWindow] = []
    for a, b in zip(breaks[:-1], breaks[1:]):
        windows.append((pd.Timestamp(mdates[a]), pd.Timestamp(mdates[b - 1])))
    return windows


def load_eurostoxx50_pit(base_dir: Path) -> PitDataset:
    root = cache_path(base_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Historique PIT non téléchargé. Lance d'abord ensure_eurostoxx50_pit().")

    index_df = pd.read_csv(root / "index_prices_daily.csv", parse_dates=["date"])
    index_df["date"] = pd.to_datetime(index_df["date"]).dt.tz_localize(None)
    index_df = index_df.sort_values("date")
    index_dates = pd.DatetimeIndex(index_df["date"])
    index_close_col = "normalized_adj_close" if "normalized_adj_close" in index_df.columns else "close"
    index_prices = pd.Series(
        pd.to_numeric(index_df[index_close_col], errors="coerce").to_numpy(),
        index=index_dates,
        name="EURO_STOXX_50",
    ).dropna()

    frames: list[pd.Series] = []
    membership: MembershipMap = {}
    tickers: list[str] = []

    for path in sorted((root / "constituents").glob("*/prices_daily.csv")):
        ticker = path.parent.name.upper()
        df = pd.read_csv(path, usecols=["date", "normalized_adj_close"], parse_dates=["date"])
        if df.empty:
            continue
        dates = pd.DatetimeIndex(pd.to_datetime(df["date"])).tz_localize(None)
        values = pd.to_numeric(df["normalized_adj_close"], errors="coerce")
        s = pd.Series(values.to_numpy(), index=dates, name=ticker).sort_index().dropna()
        if s.empty:
            continue
        frames.append(s)
        membership[ticker] = _infer_membership_windows(s.index, index_dates)
        tickers.append(ticker)

    prices = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    return PitDataset(
        prices=prices,
        membership=membership,
        tickers=sorted(tickers),
        index_prices=index_prices,
        start=PIT_START,
        end=PIT_END,
        cache_dir=root,
    )


def merge_pit_with_yahoo(pit_prices: pd.DataFrame, yahoo_prices: pd.DataFrame) -> pd.DataFrame:
    """Use authoritative PIT rows inside membership periods, Yahoo elsewhere.

    Yahoo data is useful for the pre-entry 6–12 month momentum lookback. The PIT
    dataset remains preferred wherever it has an observation. Membership
    windows still control eligibility, so Yahoo rows after an index exit do not
    make a security selectable.
    """
    if pit_prices.empty:
        return yahoo_prices.sort_index()
    if yahoo_prices.empty:
        return pit_prices.sort_index()
    left = pit_prices.copy()
    left.index = pd.to_datetime(left.index).tz_localize(None)
    right = yahoo_prices.copy()
    right.index = pd.to_datetime(right.index).tz_localize(None)
    return left.combine_first(right).sort_index()
