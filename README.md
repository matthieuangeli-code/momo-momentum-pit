# MOMO Momentum

Application Streamlit pour tester une stratégie de swing/momentum systématique avec un mode de validation **point-in-time**.

## Règles

- momentum 12–1 (grille 6/9/12 mois dans la recherche robuste) ;
- cours > moyenne mobile 200 jours ;
- top N actions ;
- maximum N actions par secteur quand le secteur est connu ;
- pondération égale ;
- rebalancement mensuel ;
- stop catastrophe approximé sur les clôtures quotidiennes ;
- filtre optionnel du marché via MM200 du benchmark ;
- coûts de transaction paramétrables.

## Installation Windows

Ouvre PowerShell dans ce dossier :

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Ou double-clique sur `run_windows.bat` si Python est déjà installé.

## Accès depuis n'importe où — Streamlit Community Cloud

Le dépôt est prêt pour Streamlit Community Cloud :

- point d'entrée : `app.py` ;
- branche : `main` ;
- dépendances : `requirements.txt` ;
- configuration : `.streamlit/config.toml` ;
- version Python recommandée au déploiement : **3.12**.

Déploiement :

1. ouvre `https://share.streamlit.io` et connecte ton compte GitHub ;
2. autorise Streamlit Community Cloud à accéder au dépôt privé `matthieuangeli-code/momo-momentum-pit` ;
3. clique sur **Create app** puis **Yup, I have an app** ;
4. sélectionne le dépôt `matthieuangeli-code/momo-momentum-pit` ;
5. branche `main` ;
6. fichier `app.py` ;
7. dans **Advanced settings**, choisis Python **3.12** ;
8. clique sur **Deploy**.

L'application reçoit alors une URL stable en `*.streamlit.app`, accessible depuis un téléphone ou n'importe quel ordinateur. Comme le dépôt GitHub est privé, l'application peut rester privée dans les réglages de partage Streamlit. Les futurs changements poussés sur GitHub sont détectés automatiquement par Community Cloud et redéployés.

Au premier backtest PIT d'une nouvelle instance, l'application télécharge les 66 historiques depuis le commit source figé. Le cache local n'est volontairement pas versionné ; cela évite de republier un jeu de données tiers sans licence explicite. Sur Community Cloud, ce cache peut être reconstruit après un redémarrage de l'instance.

## Historique point-in-time EURO STOXX 50

Le mode recommandé pour valider la stratégie utilise le jeu de données public :

- dépôt : `AndyLongest/HistoricalIndexPrices` ;
- commit figé : `bdb5c01084b314a94edfad155547d9373d0d8191` ;
- couverture : **2014-10-31 → 2025-08-22** ;
- **66 titres historiques** ;
- 134 148 observations de constituants documentées par le projet source ;
- les fichiers d'un titre ne contiennent que les observations où il appartient à l'indice.

Au premier lancement d'un backtest/recherche en mode PIT, MOMO télécharge uniquement les CSV EURO STOXX 50 depuis le commit figé et les garde dans `data/pit/`. Ce cache est exclu de Git avec `.gitignore`.

Tu peux aussi précharger les données sans lancer Streamlit :

```powershell
python fetch_pit_history.py
```

Le moteur reconstruit plusieurs fenêtres d'appartenance lorsqu'un titre sort puis réintègre l'indice. Un titre absent de l'indice à une date donnée ne peut pas être sélectionné. S'il quitte l'indice pendant un mois déjà détenu, la simulation le sort vers le cash.

### Pré-historique momentum

Le dataset PIT contient volontairement les prix seulement pendant l'appartenance. Or un signal 12–1 a besoin d'environ un an de prix avant une entrée dans l'indice. MOMO télécharge donc aussi les prix Yahoo disponibles et les utilise comme **complément avant l'entrée**. Dès qu'une observation PIT existe, elle a priorité sur Yahoo. Les anciens tickers que Yahoo ne sait plus servir restent utilisables avec leurs données PIT, mais peuvent avoir un warm-up plus conservateur.

## Recherche robuste

L'onglet **Recherche robuste** teste par défaut 27 variantes :

- lookback : 6 / 9 / 12 mois ;
- positions : 4 / 6 / 8 ;
- stop : 8 / 12 / 15 %.

La période est séparée chronologiquement en train / out-of-sample. Le classement est fait uniquement sur le train, puis les résultats OOS sont affichés sans servir au choix du gagnant.

## Scanner actuel

Le scanner reste basé sur `universe.csv` et Yahoo Finance pour rechercher les signaux actuels. Les tickers sont au format Yahoo, par exemple :

- `EQNR.OL`
- `AIR.PA`
- `SAP.DE`
- `NOVO-B.CO`

## Limites

- La source EURO STOXX 50 publique indique une confiance **medium-high** : les changements ont été reconstruits depuis les annonces STOXX et recoupés avec un historique public, car l'archive mensuelle officielle complète nécessite une connexion STOXX.
- Le benchmark `^STOXX50E` est un **price index** ; il ne faut donc pas interpréter naïvement chaque point de CAGR excédentaire face à des cours actions ajustés des dividendes.
- Le stop est simulé sur les clôtures, pas en intraday.
- Les coûts/slippage restent une approximation.
- Le mode « univers actuel » conserve un biais de survivance ; utilise le mode PIT pour la validation sérieuse.
- Le projet est un outil de recherche personnelle. Il ne passe aucun ordre et ne constitue pas un conseil d'investissement.
