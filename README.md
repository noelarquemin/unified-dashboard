# 📈 UnifiedDash v5 — Streamlit

Version Streamlit de ton script d'analyse actions (fondamental + quantitatif + technique,
Monte Carlo, DCF, facteurs explicatifs). Au lieu de générer un fichier HTML dans Colab,
l'app tourne en continu : tu tapes un ticker, elle affiche le rapport directement dans le
navigateur, et tu peux aussi le télécharger en `.html`.

## 📁 Structure du dépôt

```
.
├── app.py                  # L'application Streamlit (tout le script adapté)
├── requirements.txt        # Dépendances Python
├── .streamlit/
│   └── config.toml         # Thème sombre assorti au dashboard
└── README.md
```

Garde cette structure exacte dans ton dépôt GitHub — Streamlit Cloud a besoin de trouver
`requirements.txt` à la racine et `app.py` au chemin que tu indiques au déploiement.

## 🚀 Déployer sur Streamlit Community Cloud (gratuit)

1. Crée un dépôt GitHub (public ou privé) et pousse ces 3 éléments (`app.py`,
   `requirements.txt`, `.streamlit/config.toml`) tels quels :
   ```bash
   git init
   git add .
   git commit -m "UnifiedDash v5 - Streamlit"
   git branch -M main
   git remote add origin https://github.com/<ton-user>/<ton-repo>.git
   git push -u origin main
   ```
2. Va sur **https://share.streamlit.io** et connecte-toi avec ton compte GitHub.
3. Clique sur **"New app"**.
4. Sélectionne ton dépôt, la branche `main`, et indique `app.py` comme fichier principal.
5. Clique sur **Deploy**. Streamlit Cloud installe automatiquement `requirements.txt` puis
   lance l'app. Le premier déploiement prend 1 à 3 minutes.
6. Tu obtiens une URL du type `https://<ton-app>.streamlit.app` que tu peux partager.

Toute modification poussée sur `main` redéploie automatiquement l'app.

## 💻 Lancer en local

```bash
git clone https://github.com/<ton-user>/<ton-repo>.git
cd <ton-repo>
python -m venv .venv && source .venv/bin/activate   # optionnel mais recommandé
pip install -r requirements.txt
streamlit run app.py
```

L'app s'ouvre sur `http://localhost:8501`.

## ⚙️ Ce que fait l'app

- Barre latérale : ticker à analyser, hypothèses DCF (bear/base/bull, croissance
  terminale, horizon), nombre de simulations Monte Carlo, historique utilisé pour les
  bandes ±σ, fenêtre du graphique technique.
- Bouton **"Lancer l'analyse"** : télécharge les données via `yfinance` + `yahooquery`,
  calcule les scores (croissance, rentabilité, bilan, marges, valorisation, allocation du
  capital), le DCF, le Monte Carlo Ornstein-Uhlenbeck, le scénario cyclique (FFT), et
  génère le rapport HTML complet (identique à la version Colab), affiché directement dans
  la page.
- Bouton de téléchargement pour récupérer le rapport en `.html` autonome (utilisable hors
  ligne, envoyable par email, etc.).

## ⏱️ Performance & limites

- Une analyse complète prend généralement 20 à 90 secondes selon le ticker (plusieurs
  appels réseau à Yahoo Finance + Monte Carlo + génération de ~10 graphiques
  matplotlib).
- Le tier gratuit de Streamlit Community Cloud a des ressources limitées (~1 Go de RAM).
  Avec les valeurs par défaut (1000 simulations, 100 ans d'historique pour les bandes),
  ça passe sans problème. Si tu observes des ralentissements avec plusieurs utilisateurs
  simultanés, réduis le nombre de simulations Monte Carlo dans la barre latérale.
- Les données proviennent de Yahoo Finance (non officiel) : certains tickers, notamment
  hors marché américain, peuvent nécessiter un suffixe (`.PA` pour Paris, `.L` pour
  Londres, etc.) et certaines données financières (bilan, cash-flow) peuvent être
  manquantes ou incomplètes selon la couverture de Yahoo pour ce titre.

## ⚠️ Avertissement

Ce dashboard est un outil d'analyse automatisé à but informatif. Il ne constitue pas un
conseil en investissement. Les scores, le DCF et les projections Monte Carlo reposent sur
des hypothèses simplificatrices et des données publiques qui peuvent être incomplètes ou
erronées.
