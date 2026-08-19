from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from research import build_membership_map, oos_equity_curves, run_parameter_sweep
from pit_data import (
    PIT_COMMIT,
    PIT_END,
    PIT_EXPECTED_SECURITIES,
    PIT_START,
    cache_status,
    ensure_eurostoxx50_pit,
    load_eurostoxx50_pit,
    merge_pit_with_yahoo,
)

from strategy import (
    StrategyConfig,
    backtest,
    download_prices,
    latest_signals,
    performance_stats,
    select_positions,
)

BASE_DIR = Path(__file__).resolve().parent
UNIVERSE_FILE = BASE_DIR / "universe.csv"

st.set_page_config(
    page_title="MOMO Momentum",
    page_icon=":material/trending_up:",
    layout="wide",
)
st.title("MOMO Momentum")
st.caption("Scanner + backtest + recherche robuste d’une stratégie momentum avec filtre MM200")


@st.cache_data(ttl=3600, show_spinner=False)
def load_universe() -> pd.DataFrame:
    return pd.read_csv(UNIVERSE_FILE)


@st.cache_data(ttl="1h", max_entries=32, show_spinner=False)
def get_prices(tickers: tuple[str, ...], start: str, end: str | None) -> pd.DataFrame:
    return download_prices(tickers, start=start, end=end)


def fmt_pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:.1%}"


def pit_status_caption() -> str:
    status = cache_status(BASE_DIR)
    mb = float(status["bytes"]) / (1024 * 1024)
    state = "prêt" if status["ready"] else "à télécharger"
    return (
        f"Historique PIT EURO STOXX 50 : **{state}** — "
        f"{status['files']}/{status['expected']} titres en cache, {mb:.1f} Mo. "
        f"Couverture {PIT_START.date()} → {PIT_END.date()}."
    )


def prepare_pit_inputs(
    requested_start: date,
    requested_end: date,
    benchmark_ticker: str | None = None,
    sector_lookup: dict[str, str] | None = None,
):
    """Prepare point-in-time prices, membership and benchmark for a run."""
    effective_start = max(pd.Timestamp(requested_start), PIT_START)
    effective_end = min(pd.Timestamp(requested_end), PIT_END)
    if effective_start >= effective_end:
        raise ValueError(
            f"La période PIT doit recouper {PIT_START.date()} → {PIT_END.date()}."
        )

    dl_bar = st.progress(0.0, text="Vérification de l’historique point-in-time...")

    def _pit_progress(done: int, total: int, ticker: str) -> None:
        dl_bar.progress(done / total, text=f"Historique PIT {done}/{total} — {ticker}")

    ensure_eurostoxx50_pit(BASE_DIR, progress=_pit_progress)
    dl_bar.empty()
    pit = load_eurostoxx50_pit(BASE_DIR)

    # Yahoo is only a complement: it provides pre-entry history needed for
    # 6/9/12-month momentum. PIT observations take precedence when available.
    start_download = (effective_start.date() - timedelta(days=430)).isoformat()
    end_download = (effective_end.date() + timedelta(days=2)).isoformat()
    yahoo = get_prices(tuple(pit.tickers), start_download, end_download)
    prices = merge_pit_with_yahoo(pit.prices, yahoo)
    prices = prices.loc[pd.Timestamp(start_download):pd.Timestamp(end_download)].copy()

    sector_lookup = sector_lookup or {}
    sectors = {t: sector_lookup.get(t, "Unknown") for t in pit.tickers}

    bench = None
    bench_name = "EURO STOXX 50 (price index)"
    if benchmark_ticker:
        bench_df = get_prices((benchmark_ticker.strip().upper(),), start_download, end_download)
        if not bench_df.empty:
            bench = bench_df.iloc[:, 0]
            bench_name = benchmark_ticker.strip().upper()
    if bench is None:
        bench = pit.index_prices

    return prices, sectors, pit.membership, bench, bench_name, effective_start, effective_end


universe = load_universe()
membership_map = build_membership_map(universe)
pit_cache = cache_status(BASE_DIR)

with st.container(horizontal=True):
    st.metric("Univers actuel", f"{len(universe)} actions", border=True)
    st.metric(
        "Historique PIT",
        f"{pit_cache['files']}/{pit_cache['expected']} titres",
        border=True,
    )
    st.metric("Observations vérifiées", "134 148", border=True)
    st.metric("Couverture PIT", "2014–2025", border=True)

with st.sidebar:
    st.header("Réglages")
    region_options = ["Tous"] + sorted(universe["region"].dropna().unique().tolist())
    regions = st.multiselect("Régions", region_options, default=["Tous"])

    if "Tous" not in regions and regions:
        selected_universe = universe[universe["region"].isin(regions)].copy()
    else:
        selected_universe = universe.copy()

    custom = st.text_area(
        "Tickers supplémentaires (Yahoo, séparés par virgule)",
        placeholder="Ex: AIR.PA, EQNR.OL, SAP.DE",
    )
    custom_tickers = [x.strip().upper() for x in custom.split(",") if x.strip()]

    top_n = st.slider("Nombre de positions", 3, 15, 6)
    max_sector = st.slider("Maximum par secteur", 1, 5, 2)
    stop_loss = st.slider("Stop catastrophe", 0.05, 0.25, 0.12, 0.01)
    market_filter = st.checkbox("Filtre marché MM200", value=True)
    defensive = st.slider("Exposition si marché < MM200", 0.0, 1.0, 0.50, 0.05)
    cost_bps = st.slider("Coût par transaction (bps)", 0, 50, 10, 1)
    initial_capital = st.number_input("Capital test (NOK)", 10_000, 10_000_000, 100_000, 10_000)

cfg = StrategyConfig(
    top_n=top_n,
    max_per_sector=max_sector,
    stop_loss=stop_loss,
    market_filter=market_filter,
    defensive_exposure=defensive,
    transaction_cost_bps=float(cost_bps),
)

all_tickers = list(dict.fromkeys(selected_universe["ticker"].tolist() + custom_tickers))
sector_map = dict(zip(universe["ticker"], universe["sector"]))
for t in custom_tickers:
    sector_map.setdefault(t, "Custom")

scan_tab, bt_tab, research_tab, methodology_tab = st.tabs(["🔎 Scanner actuel", "🧪 Backtest", "🔬 Recherche robuste", "ℹ️ Méthode"])

with scan_tab:
    st.subheader("Sélection actuelle")
    st.write(f"Univers chargé : **{len(all_tickers)} actions**")

    if st.button("Lancer le scanner", type="primary", width="stretch"):
        start = (date.today() - timedelta(days=520)).isoformat()
        with st.spinner("Téléchargement des cours et calcul des signaux..."):
            prices = get_prices(tuple(all_tickers), start, None)

        if prices.empty:
            st.error("Aucune donnée téléchargée. Vérifie la connexion internet et les tickers Yahoo Finance.")
        else:
            signals = latest_signals(prices, cfg)
            picks = select_positions(signals, sector_map, cfg)

            missing = sorted(set(all_tickers) - set(prices.columns))
            c1, c2, c3 = st.columns(3)
            c1.metric("Actions avec données", len(prices.columns))
            c2.metric("Candidats > MM200", int(signals["above_sma"].sum()) if not signals.empty else 0)
            c3.metric("Sélection", len(picks))

            if missing:
                with st.expander(f"{len(missing)} tickers sans données"):
                    st.write(", ".join(missing))

            if picks.empty:
                st.warning("Aucune action ne satisfait actuellement les règles.")
            else:
                display = picks.reset_index().rename(columns={"index": "ticker"})
                display["momentum_12_1"] = display["momentum_12_1"].map(lambda x: f"{x:.1%}")
                display["weight"] = display["weight"].map(lambda x: f"{x:.1%}")
                display["price"] = display["price"].round(2)
                display["sma200"] = display["sma200"].round(2)
                display["stop_price"] = display["stop_price"].round(2)
                st.dataframe(
                    display[["ticker", "sector", "momentum_12_1", "price", "sma200", "weight", "stop_price"]],
                    hide_index=True,
                    width="stretch",
                )

                st.info(
                    "Le stop affiché est un niveau de contrôle du risque, pas un ordre garanti : "
                    "un gap baissier peut exécuter plus bas."
                )

            st.subheader("Top momentum de l'univers")
            top = signals.head(30).copy().reset_index().rename(columns={"index": "ticker"})
            top["sector"] = top["ticker"].map(sector_map).fillna("Unknown")
            top["momentum_12_1"] = top["momentum_12_1"].map(lambda x: f"{x:.1%}")
            st.dataframe(
                top[["ticker", "sector", "momentum_12_1", "above_sma", "price", "sma200"]],
                hide_index=True,
                width="stretch",
            )

with bt_tab:
    st.subheader("Backtest mensuel")
    bt_source = st.radio(
        "Univers du backtest",
        ["Univers actuel", "EURO STOXX 50 historique (point-in-time)"],
        horizontal=True,
        key="bt_source",
    )
    use_pit_bt = bt_source.startswith("EURO STOXX")
    if use_pit_bt:
        st.info(pit_status_caption())
        st.caption(
            "Mode recommandé pour valider la stratégie : 66 titres historiques, entrées/sorties d’indice appliquées à chaque date. "
            f"Source figée au commit {PIT_COMMIT[:12]}."
        )

    col1, col2, col3 = st.columns(3)
    default_start = date.today().replace(year=date.today().year - 10)
    bt_start = col1.date_input("Début", value=default_start, key="bt_start")
    bt_end = col2.date_input("Fin", value=date.today(), key="bt_end")
    benchmark_default = "^STOXX50E" if use_pit_bt else "URTH"
    benchmark_ticker = col3.text_input("Benchmark Yahoo", value=benchmark_default, key=f"bt_benchmark_{'pit' if use_pit_bt else 'current'}")

    if st.button("Lancer le backtest", width="stretch"):
        try:
            if use_pit_bt:
                with st.spinner("Préparation de l’univers historique + complément Yahoo pour le lookback..."):
                    prices, bt_sectors, bt_membership, bench, bench_name, eff_start, eff_end = prepare_pit_inputs(
                        bt_start, bt_end, benchmark_ticker, sector_map
                    )
                if pd.Timestamp(bt_start) < eff_start or pd.Timestamp(bt_end) > eff_end:
                    st.warning(f"Période ajustée à la couverture PIT : {eff_start.date()} → {eff_end.date()}.")
            else:
                eff_start, eff_end = pd.Timestamp(bt_start), pd.Timestamp(bt_end)
                start_download = (bt_start - timedelta(days=430)).isoformat()
                end_download = (bt_end + timedelta(days=2)).isoformat()
                with st.spinner("Téléchargement historique et backtest..."):
                    prices = get_prices(tuple(all_tickers), start_download, end_download)
                    bench_df = get_prices((benchmark_ticker.strip().upper(),), start_download, end_download)
                bt_sectors = sector_map
                bt_membership = membership_map
                bench = bench_df.iloc[:, 0] if not bench_df.empty else None
                bench_name = benchmark_ticker.strip().upper()

            if prices.empty:
                st.error("Impossible de préparer les données de l'univers.")
            else:
                result, trades = backtest(
                    prices=prices,
                    sectors=bt_sectors,
                    cfg=cfg,
                    benchmark=bench,
                    initial_capital=float(initial_capital),
                    membership=bt_membership,
                )
                result = result.loc[eff_start:eff_end]

                if result.empty:
                    st.error("Pas assez de données pour cette période.")
                else:
                    strategy_stats = performance_stats(result["strategy_equity"], result["strategy_return"])
                    benchmark_stats = (
                        performance_stats(result["benchmark_equity"], result["benchmark_return"])
                        if "benchmark_equity" in result
                        else {}
                    )

                    metrics = st.columns(5)
                    labels = ["CAGR", "Total return", "Max drawdown", "Volatility", "Sharpe (rf=0)"]
                    for c, label in zip(metrics, labels):
                        value = strategy_stats.get(label, float("nan"))
                        c.metric(label, "—" if pd.isna(value) else (f"{value:.2f}" if label == "Sharpe (rf=0)" else f"{value:.1%}"))

                    fig = go.Figure()
                    s_eq = result["strategy_equity"].dropna()
                    if not s_eq.empty:
                        s_rebased = float(initial_capital) * s_eq / s_eq.iloc[0]
                        fig.add_trace(go.Scatter(x=s_rebased.index, y=s_rebased, name="MOMO Momentum"))
                    if "benchmark_equity" in result:
                        b = result["benchmark_equity"].dropna()
                        if not b.empty:
                            rebased = float(initial_capital) * b / b.iloc[0]
                            fig.add_trace(go.Scatter(x=rebased.index, y=rebased, name=bench_name))
                    fig.update_layout(xaxis_title="Date", yaxis_title="Capital simulé (NOK)", hovermode="x unified")
                    st.plotly_chart(fig, width="stretch")

                    if benchmark_stats:
                        comparison = pd.DataFrame({"MOMO Momentum": strategy_stats, bench_name: benchmark_stats}).T
                        for col in ["Total return", "CAGR", "Volatility", "Max drawdown"]:
                            comparison[col] = comparison[col].map(lambda x: f"{x:.1%}")
                        comparison["Sharpe (rf=0)"] = comparison["Sharpe (rf=0)"].map(lambda x: f"{x:.2f}")
                        st.subheader("Comparaison")
                        st.dataframe(comparison, width="stretch")

                    if not trades.empty:
                        st.subheader("Journal des positions mensuelles")
                        st.dataframe(
                            trades.sort_values("signal_date", ascending=False).head(200),
                            hide_index=True,
                            width="stretch",
                        )
                        csv = trades.to_csv(index=False).encode("utf-8")
                        st.download_button("Télécharger le journal CSV", csv, "momo_momentum_trades.csv", "text/csv")

                    if use_pit_bt:
                        st.success(
                            "Biais de survivance fortement réduit : le portefeuille ne peut sélectionner un titre que pendant ses fenêtres "
                            "historiques d’appartenance à l’EURO STOXX 50. Les observations PIT ont priorité sur Yahoo."
                        )
                    else:
                        st.warning(
                            "Univers actuel : biais de survivance encore présent. Utilise le mode EURO STOXX 50 historique pour la validation sérieuse."
                        )
        except Exception as exc:
            st.error(f"Erreur pendant le backtest : {exc}")

with research_tab:
    st.subheader("Recherche multi-paramètres + validation hors échantillon")
    st.write(
        "On choisit les paramètres uniquement sur la première partie de l’historique (train), "
        "puis on mesure leur performance sur la fin de période (out-of-sample)."
    )
    research_source = st.radio(
        "Univers de recherche",
        ["EURO STOXX 50 historique (point-in-time)", "Univers actuel"],
        horizontal=True,
        key="research_source",
    )
    use_pit_research = research_source.startswith("EURO STOXX")
    if use_pit_research:
        st.info(pit_status_caption())
        st.caption(
            "Le mode PIT est le mode de validation recommandé. Les 66 dossiers historiques sont téléchargés une fois, puis conservés en cache local."
        )

    r1, r2, r3 = st.columns(3)
    research_start = r1.date_input(
        "Début recherche",
        value=date.today().replace(year=date.today().year - 10),
        key="research_start",
    )
    research_end = r2.date_input("Fin recherche", value=date.today(), key="research_end")
    research_benchmark = r3.text_input("Benchmark recherche", value="^STOXX50E" if use_pit_research else "URTH", key=f"research_benchmark_{'pit' if use_pit_research else 'current'}")

    train_pct = st.slider(
        "Part de l’historique utilisée pour choisir les paramètres (%)",
        min_value=55,
        max_value=85,
        value=70,
        step=5,
        help="70 % train / 30 % out-of-sample par défaut.",
    )
    train_fraction = train_pct / 100.0

    g1, g2, g3 = st.columns(3)
    lookbacks = g1.multiselect("Lookback momentum (mois)", [6, 9, 12], default=[6, 9, 12])
    research_top_ns = g2.multiselect("Nombre de positions", [4, 6, 8, 10, 12], default=[4, 6, 8])
    stop_pcts = g3.multiselect("Stops (%)", [8, 10, 12, 15, 20], default=[8, 12, 15])

    compare_market_filter = st.checkbox(
        "Tester aussi filtre marché ON et OFF",
        value=False,
        help="Désactivé par défaut pour garder 27 variantes. Activé, la grille double.",
    )
    research_market_filters = [False, True] if compare_market_filter else [market_filter]
    variant_count = len(lookbacks) * len(research_top_ns) * len(stop_pcts) * len(research_market_filters)
    st.caption(f"Grille actuelle : **{variant_count} variantes**.")

    if use_pit_research:
        st.success(
            f"Validation point-in-time activée : {PIT_EXPECTED_SECURITIES} titres historiques, fenêtres multiples d’entrée/sortie, "
            f"couverture {PIT_START.date()} → {PIT_END.date()}."
        )
    elif membership_map is None:
        st.warning("Univers actuel : biais de survivance présent. Préfère le mode EURO STOXX 50 historique pour conclure sur l’alpha.")
    else:
        st.success("Fenêtres active_from / active_to détectées dans universe.csv.")

    invalid_grid = not lookbacks or not research_top_ns or not stop_pcts or variant_count == 0
    too_many = variant_count > 60
    if too_many:
        st.error("Grille limitée à 60 variantes pour garder l’application réactive. Réduis un des ensembles de paramètres.")

    if st.button(
        "Lancer l’étude robuste",
        type="primary",
        width="stretch",
        disabled=invalid_grid or too_many,
    ):
        try:
            if use_pit_research:
                with st.spinner("Préparation de l’univers historique + complément Yahoo pour les lookbacks..."):
                    research_prices, research_sectors, research_membership, research_bench, benchmark_name, eff_start, eff_end = prepare_pit_inputs(
                        research_start, research_end, research_benchmark, sector_map
                    )
                if pd.Timestamp(research_start) < eff_start or pd.Timestamp(research_end) > eff_end:
                    st.warning(f"Période de recherche ajustée à la couverture PIT : {eff_start.date()} → {eff_end.date()}.")
            else:
                eff_start, eff_end = pd.Timestamp(research_start), pd.Timestamp(research_end)
                start_download = (research_start - timedelta(days=430)).isoformat()
                end_download = (research_end + timedelta(days=2)).isoformat()
                with st.spinner("Téléchargement des prix une seule fois pour toute la grille..."):
                    research_prices = get_prices(tuple(all_tickers), start_download, end_download)
                    research_bench_df = get_prices((research_benchmark.strip().upper(),), start_download, end_download)
                research_sectors = sector_map
                research_membership = membership_map
                research_bench = research_bench_df.iloc[:, 0] if not research_bench_df.empty else None
                benchmark_name = research_benchmark.strip().upper()

            if research_prices.empty:
                st.error("Impossible de préparer les données de l’univers.")
            else:
                bar = st.progress(0.0, text="Préparation de la grille...")

                def _progress(done: int, total: int, label: str) -> None:
                    bar.progress(done / total, text=f"{done}/{total} — {label}")

                summary, curves, split_date = run_parameter_sweep(
                    prices=research_prices,
                    sectors=research_sectors,
                    benchmark=research_bench,
                    base_cfg=cfg,
                    start=eff_start,
                    end=eff_end,
                    train_fraction=train_fraction,
                    lookbacks=lookbacks,
                    top_ns=research_top_ns,
                    stop_losses=[x / 100.0 for x in stop_pcts],
                    market_filters=research_market_filters,
                    initial_capital=float(initial_capital),
                    membership=research_membership,
                    progress=_progress,
                )
                bar.empty()
                st.session_state["research_payload"] = {
                    "summary": summary,
                    "curves": curves,
                    "split_date": split_date,
                    "benchmark": benchmark_name,
                    "capital": float(initial_capital),
                    "source": research_source,
                }
        except Exception as exc:
            st.error(f"Erreur pendant l’étude robuste : {exc}")

    payload = st.session_state.get("research_payload")
    if payload:
        summary = payload["summary"]
        curves = payload["curves"]
        split_date = payload["split_date"]
        benchmark_name = payload["benchmark"]
        capital = payload["capital"]

        if summary.empty:
            st.error("Pas assez de données pour comparer les variantes.")
        else:
            best = summary.iloc[0]
            beat_rate = (summary["OOS excess CAGR"] > 0).mean()
            st.info(
                f"Séparation chronologique : **train jusqu’au {split_date.date()}**, puis **out-of-sample à partir du {(split_date + pd.Timedelta(days=1)).date()}**. "
                "Le classement ci-dessous est basé uniquement sur le Sharpe du train."
            )

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Variantes", len(summary))
            m2.metric("Meilleur train", best["Config"])
            m3.metric("CAGR OOS", fmt_pct(best["Test CAGR"]))
            m4.metric(f"{benchmark_name} OOS", fmt_pct(best["Benchmark Test CAGR"]))
            m5.metric("Variantes > benchmark OOS", f"{beat_rate:.0%}")

            st.subheader("Résultats — sélection sur train, contrôle sur OOS")
            table = summary.copy()
            pct_cols = [
                "Stop", "Train CAGR", "Train Max DD", "Test CAGR", "Test Max DD",
                "Benchmark Test CAGR", "OOS excess CAGR", "Stop rate",
            ]
            for col in pct_cols:
                table[col] = table[col].map(lambda x: "—" if pd.isna(x) else f"{x:.1%}")
            for col in ["Train Sharpe", "Test Sharpe", "Benchmark Test Sharpe", "Sharpe degradation"]:
                table[col] = table[col].map(lambda x: "—" if pd.isna(x) else f"{x:.2f}")
            table["Avg positions"] = table["Avg positions"].map(lambda x: f"{x:.1f}")
            st.dataframe(
                table[[
                    "Train rank", "Config", "Train CAGR", "Train Sharpe", "Train Max DD",
                    "Test CAGR", "Test Sharpe", "Test Max DD", "OOS excess CAGR",
                    "Sharpe degradation", "Avg positions", "Stop rate",
                ]],
                hide_index=True,
                width="stretch",
            )

            scatter = go.Figure()
            scatter.add_trace(
                go.Scatter(
                    x=summary["Train Sharpe"],
                    y=summary["Test Sharpe"],
                    mode="markers",
                    text=summary["Config"],
                    hovertemplate="%{text}<br>Train Sharpe=%{x:.2f}<br>OOS Sharpe=%{y:.2f}<extra></extra>",
                    name="Variantes",
                )
            )
            scatter.update_layout(
                title="Stabilité : Sharpe train vs out-of-sample",
                xaxis_title="Sharpe train",
                yaxis_title="Sharpe out-of-sample",
            )
            st.plotly_chart(scatter, width="stretch")

            oos = oos_equity_curves(summary, curves, split_date, capital, top_k=3)
            if not oos.empty:
                fig = go.Figure()
                for col in oos.columns:
                    fig.add_trace(go.Scatter(x=oos.index, y=oos[col], name=col))
                fig.update_layout(
                    title="Out-of-sample : top 3 choisis sur le train",
                    xaxis_title="Date",
                    yaxis_title="Capital simulé (NOK)",
                    hovermode="x unified",
                )
                st.plotly_chart(fig, width="stretch")

            best_csv = summary.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger les résultats de la grille (CSV)",
                best_csv,
                "momo_momentum_parameter_sweep.csv",
                "text/csv",
            )

            st.caption(
                "Une bonne variante n’est pas celle qui a le meilleur chiffre absolu, mais celle dont les performances restent correctes "
                "hors échantillon et qui n’est pas isolée au milieu de paramètres voisins médiocres."
            )

with methodology_tab:
    st.subheader("Règles implémentées")
    st.markdown(
        """
- **Momentum 12–1** : cours de fin du mois précédent / cours d'il y a 12 mois − 1.
- **Filtre tendance** : cours supérieur à la moyenne mobile 200 jours.
- **Sélection** : meilleurs scores de momentum, avec limite par secteur.
- **Pondération** : poids égaux.
- **Rebalancement** : fin de chaque mois.
- **Stop catastrophe** : sortie approximative lorsqu'une clôture quotidienne passe sous le seuil configuré.
- **Filtre de marché** : si le benchmark est sous sa MM200, l'exposition du mois suivant est réduite.
- **Pas de levier, pas de short, pas d'options.**
- **Recherche robuste** : grille multi-paramètres classée sur la période train, puis contrôle sur une période out-of-sample séparée chronologiquement.
        """
    )
    st.subheader("Historique point-in-time")
    st.markdown(
        f"""
- **EURO STOXX 50 : 31/10/2014 → 22/08/2025**, 66 titres historiques.
- Source publique : `AndyLongest/HistoricalIndexPrices`, snapshot `{PIT_COMMIT[:12]}`.
- Les fichiers de chaque titre ne contiennent que les observations pendant ses périodes d’appartenance à l’indice.
- MOMO détecte aussi les sorties puis ré-entrées et applique ces fenêtres à chaque rebalancement.
- Yahoo complète le pré-historique nécessaire au calcul du momentum, mais les prix PIT ont priorité pendant l’appartenance.
        """
    )
    st.subheader("Limites importantes")
    st.markdown(
        """
1. Données Yahoo Finance : très pratiques pour tester, pas conçues comme flux institutionnel d'exécution.
2. Le stop est simulé sur les clôtures quotidiennes et ne modélise pas précisément les gaps/intraday.
3. Les coûts sont une approximation en points de base : le spread/slippage réel peut être différent.
4. Le mode **EURO STOXX 50 historique** utilise un univers point-in-time 2014–2025 et supporte plusieurs fenêtres d’entrée/sortie par titre. Le mode « univers actuel » conserve un biais de survivance.
5. Une séparation train/OOS réduit le risque de sur-ajustement mais ne le supprime pas.
6. Les cours PIT sont prioritaires pendant l’appartenance à l’indice ; Yahoo complète surtout le pré-historique nécessaire au momentum avant une entrée. Les anciens tickers indisponibles chez Yahoo peuvent donc avoir un warm-up plus conservateur.
7. Ce programme est un outil de recherche personnelle, pas un conseil d'investissement ni un système d'ordre automatique.
        """
    )
