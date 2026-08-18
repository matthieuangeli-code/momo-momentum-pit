from pathlib import Path

from pit_data import cache_status, ensure_eurostoxx50_pit, load_eurostoxx50_pit

BASE_DIR = Path(__file__).resolve().parent


def progress(done: int, total: int, ticker: str) -> None:
    print(f"[{done:02d}/{total:02d}] {ticker}")


if __name__ == "__main__":
    print("Téléchargement/vérification de l'historique PIT EURO STOXX 50...")
    ensure_eurostoxx50_pit(BASE_DIR, progress=progress)
    ds = load_eurostoxx50_pit(BASE_DIR)
    status = cache_status(BASE_DIR)
    window_count = sum(len(w) if isinstance(w, list) else 1 for w in ds.membership.values())
    print()
    print(f"Titres historiques : {len(ds.tickers)}")
    print(f"Fenêtres d'appartenance détectées : {window_count}")
    print(f"Couverture : {ds.start.date()} -> {ds.end.date()}")
    print(f"Cache : {status['path']}")
    print(f"Taille CSV : {status['bytes'] / (1024 * 1024):.1f} Mo")
