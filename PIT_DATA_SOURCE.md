# Provenance des données PIT

Source utilisée pour le mode historique :

- Repository: `AndyLongest/HistoricalIndexPrices`
- Snapshot: `bdb5c01084b314a94edfad155547d9373d0d8191`
- Dataset: `data/eurostoxx50`
- Coverage: `2014-10-31` to `2025-08-22`
- Expected historical security folders: `66`

Le projet source documente que les fichiers `constituents/<symbol>/prices_daily.csv` contiennent les prix nettoyés seulement pendant les périodes où le titre appartient à l'indice. Pour l'EURO STOXX 50, les changements principaux et temporaires ont été reconstruits à partir d'annonces STOXX et recoupés avec un historique public. La source qualifie la confiance de `medium-high`, l'archive mensuelle officielle STOXX nécessitant une connexion.

MOMO ne versionne pas une copie du dataset. `pit_data.py` télécharge les CSV depuis ce commit précis et les met en cache local dans `data/pit/`.
