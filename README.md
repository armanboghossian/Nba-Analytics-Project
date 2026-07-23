# 🏀 NBA Player & Game Performance Analysis

An end-to-end exploratory data analysis and machine learning project built on NBA game-log data. The notebook takes raw box-score data, cleans it, visualizes scoring/efficiency/defensive trends, and builds several ML models to predict wins, identify MVP/DPOY candidates, cluster player archetypes, and find "similar players" using cosine similarity.


## Key Questions Answered

This project explores a season's worth of NBA game logs to answer questions like:

- Who are the league's top scorers, playmakers, and defenders?
- What separates a win from a loss statistically (points, shooting %, plus/minus)?
- Do players perform better at home or on the road?
- Can we predict game outcomes from box-score stats?
- Which players cluster into similar "archetypes" (PCA + K-Means)?
- Who are this season's data-driven MVP / DPOY / All-Star candidates?
- Which players play most similarly to a given player (e.g. LeBron James)?

Everything is contained in a single Jupyter notebook, `nba.ipynb`, organized as a linear analysis pipeline: **load → clean → explore → visualize → engineer features → model → interpret**.

---

##  Features

- **Data cleaning** — handles missing values, merges player IDs with full names via the `nba_api`
- **Exploratory Data Analysis (EDA)** — distributions, correlation heatmaps, summary statistics
- **Visualizations** — static (Matplotlib/Seaborn) and interactive (Plotly) charts
- **Feature engineering** — True Shooting % (TS%), efficiency rating, defensive impact score, home/away flag
- **Leaderboards** — top scorers, assist leaders, steal leaders, defensive impact, efficiency
- **Advanced stats** — triple-average players (10+/10+/10+), win vs. loss performance breakdown
- **Machine learning**
  - PCA for dimensionality reduction & player-performance projection
  - K-Means clustering for player archetypes
  - Linear Regression to predict scoring from peripheral stats
  - Random Forest / XGBoost classifiers to predict wins and "elite scorer" status
  - SHAP for explainable AI — what actually drives a win
- **Player similarity engine** — cosine similarity over standardized player profiles to find comparable players
- **Data-driven awards** — MVP probability, DPOY probability, and an algorithmically selected All-Star roster

---

##  Dataset

The notebook expects a CSV file named **`nba.csv`** in the project's root directory, containing per-game player box-score statistics with (at least) the following columns:

| Column | Description |
|---|---|
| `Player_ID` | NBA.com player identifier |
| `GAME_DATE` | Date of the game |
| `MATCHUP` | Matchup string (contains `vs.` for home games, `@` for away) |
| `WL` | Game result (`W`/`L`) |
| `PTS`, `REB`, `AST`, `STL`, `BLK`, `TOV`, `PF` | Core box-score stats |
| `FGM`, `FGA`, `FG_PCT` | Field goal makes / attempts / percentage |
| `FG3_PCT` | 3-point percentage |
| `FTM`, `FTA`, `FT_PCT` | Free throw makes / attempts / percentage |
| `DREB` | Defensive rebounds |
| `PLUS_MINUS` | Plus/minus for the game |

Player full names are pulled in separately via the [`nba_api`](https://github.com/swar/nba_api) package and merged in on `Player_ID`.

> ⚠️ This dataset is not included in the repo. Provide your own `nba.csv` (e.g. exported from [stats.nba.com](https://www.nba.com/stats) or the `nba_api` package) and place it alongside the notebook.

---

## 🛠 Tech Stack

| Category | Libraries |
|---|---|
| Data handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| NBA data access | `nba_api` |
| Machine learning | `scikit-learn` (PCA, K-Means, Linear Regression, Random Forest), `xgboost` |
| Model explainability | `shap` |

---



### Prerequisites

- Python 3.9+
- Jupyter Notebook or JupyterLab (or VS Code with the Jupyter extension)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies**
   ```bash
   pip install "numpy<2" pandas matplotlib seaborn plotly scikit-learn xgboost shap nba_api jupyter
   ```

4. **Add your dataset**

   Place your `nba.csv` file in the same directory as `nba.ipynb`.

---

## Usage

1. Launch Jupyter:
   ```bash
   jupyter notebook nba.ipynb
   ```
2. Run the cells from top to bottom (`Cell → Run All`). The notebook is designed as a sequential pipeline — later sections (modeling, clustering, similarity search) depend on data structures created earlier (e.g. `df`, `player_profile`, `features`).
3. Explore the generated charts inline, and tweak parameters — e.g.:
   - Change the `stars` list in the rolling scoring average section to track different players
   - Change `player_name = "LeBron James"` in the similarity section to compare a different player
   - Adjust the `n_clusters` value in the K-Means section to explore different archetype groupings
   - Adjust the `Games >= 20` filters to change the minimum sample size for "qualified" players

---


## Analysis Breakdown

The notebook flows through the following stages:

1. **Setup & Loading** — imports, load `nba.csv`, initial `head()` / `info()` / `describe()` checks
2. **Cleaning** — drop rows with missing `WL`, check for NaNs
3. **Player Name Mapping** — merge `Player_ID` with full names via `nba_api`
4. **EDA & Distributions** — points histogram, correlation heatmap, win/loss box plots
5. **Leaderboards** — top scorers, assists, steals (filtered to players with 20+ games)
6. **Win vs. Loss Analysis** — average points/FG%/assists/rebounds/plus-minus by result
7. **Relationships** — points vs. plus-minus, assists vs. turnovers (regression plots), interactive Plotly scatter/heatmaps
8. **Home vs. Away Performance** — league-wide splits on points, TS%, and plus-minus
9. **Rolling Trends** — 10-game rolling scoring average for select stars
10. **Efficiency & Triple-Average Players** — custom efficiency formula, 10/10/10 finders
11. **Dimensionality Reduction & Clustering** — PCA projection, K-Means player archetypes
12. **Predictive Modeling**
    - Linear Regression: predicting points from peripheral stats
    - Random Forest: predicting "elite scorer" status
    - XGBoost + SHAP: explaining what drives wins
13. **Awards & Team Building** — formula-based MVP/DPOY probability scores, data-driven All-Star team
14. **Defensive Impact** — custom defensive score (blocks, steals, defensive rebounds, fouls)
15. **Player Similarity Search** — cosine similarity over standardized player profiles

---

## Notes & Limitations

- Several formulas (Efficiency, TS%, Defensive Impact, MVP/DPOY Probability, All-Star Score) are **custom heuristics** built for this analysis — they are not official NBA advanced stats and are meant for exploratory purposes, not authoritative rankings.
- The MVP/DPOY/All-Star sections use simple weighted scoring rather than validated statistical models; treat the outputs as illustrative rather than predictive.
- The notebook installs `numpy<2` for compatibility with some ML/plotting libraries — run the first cell before anything else.
- Some cells (e.g. clustering, XGBoost setup) reference variables created earlier in the notebook, so cells must be run in order.

