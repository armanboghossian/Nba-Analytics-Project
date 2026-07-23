import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="NBA Dashboard", page_icon="🏀", layout="wide")
st.markdown("""<style>
div[data-testid="stMetric"]{background:#202A38;border-left:3px solid #FF6B35;border-radius:8px;padding:10px 14px;}
</style>""", unsafe_allow_html=True)

PALETTE = ["#FF6B35", "#2EC4B6", "#4C72B0", "#C44E52", "#F2C14E", "#8E7CC3"]


@st.cache_data(show_spinner="Loading game logs...")
def load_data():
    df = pd.read_csv("nba.csv").dropna(subset=["WL"]).copy()
    from nba_api.stats.static import players as nba_players
    names = pd.DataFrame(nba_players.get_players())[["id", "full_name"]]
    df = df.merge(names, left_on="Player_ID", right_on="id", how="left")
    df["full_name"] = df["full_name"].fillna("Player " + df["Player_ID"].astype(str))
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df["HOME"] = df["MATCHUP"].astype(str).str.contains(r"vs\.")
    denom = 2 * (df["FGA"] + 0.44 * df["FTA"])
    df["TS_PCT"] = np.where(denom > 0, df["PTS"] / denom, np.nan)
    return df


df = load_data()
st.title("🏀 NBA Game Log Dashboard")

# ---- Best Scorers ----
st.header("Best Scorers")
top_n = st.slider("Show top N", 5, 25, 10)
scorers = df.groupby("full_name")["PTS"].mean().sort_values(ascending=False).head(top_n)
fig = px.bar(scorers.iloc[::-1], orientation="h", color_discrete_sequence=[PALETTE[0]],
             labels={"value": "PPG", "full_name": ""}, title="Top Scorers (PPG)")
st.plotly_chart(fig, use_container_width=True)

# ---- Archetypes ----
st.header("Player Archetypes")
k = st.slider("Clusters", 2, 8, 4)
profile = df.groupby("full_name")[["PTS", "REB", "AST", "STL", "BLK", "TOV"]].mean()
profile = profile[df.groupby("full_name").size() >= 20]
X = StandardScaler().fit_transform(profile)
clusters = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
coords = PCA(n_components=2, random_state=42).fit_transform(X)

label_pool = {"PTS": "Scorer", "REB": "Rebounder", "AST": "Playmaker", "STL": "Defender", "BLK": "Rim Protector"}
z = pd.DataFrame(X, columns=profile.columns, index=profile.index)
z["Cluster"] = clusters
centers = z.groupby("Cluster")[list(label_pool)].mean()
top_feat = centers.idxmax(axis=1)
names, seen = {}, {}
for c, feat in top_feat.items():
    base = label_pool[feat]
    seen[base] = seen.get(base, 0) + 1
    names[c] = base if seen[base] == 1 else f"{base} {seen[base]}"

plot_df = profile.reset_index()
plot_df["Archetype"] = pd.Series(clusters, index=profile.index).map(names).values
plot_df["PC1"], plot_df["PC2"] = coords[:, 0], coords[:, 1]
fig = px.scatter(plot_df, x="PC1", y="PC2", color="Archetype", hover_name="full_name",
                  color_discrete_sequence=PALETTE, title="Player Archetypes (PCA)")
st.plotly_chart(fig, use_container_width=True)

# ---- Rolling Average: Big Stars ----
st.header("10-Game Rolling Average — Top 5 Stars")
stat = st.selectbox("Stat", ["PTS", "REB", "AST"], index=0)
stars = df.groupby("full_name")[stat].mean().sort_values(ascending=False).head(5).index
trend = df[df["full_name"].isin(stars)].sort_values(["full_name", "GAME_DATE"]).copy()
trend["Rolling"] = trend.groupby("full_name")[stat].transform(lambda s: s.rolling(10, min_periods=1).mean())
fig = px.line(trend, x="GAME_DATE", y="Rolling", color="full_name", color_discrete_sequence=PALETTE,
              title=f"10-Game Rolling {stat} - Top 5 Stars", labels={"GAME_DATE": "", "Rolling": stat})
st.plotly_chart(fig, use_container_width=True)

# ---- Home vs Away ----
st.header("Home vs Away Performance")
metrics = ["PTS", "TS_PCT", "PLUS_MINUS"]
labels = ["Points", "True Shooting %", "Plus/Minus"]
means = df.groupby("HOME")[metrics].mean()
fig = make_subplots(rows=1, cols=3, subplot_titles=labels)
for i, m in enumerate(metrics, start=1):
    vals = [means.loc[False, m], means.loc[True, m]]
    fig.add_trace(go.Bar(x=["Away", "Home"], y=vals, marker_color=["#C44E52", "#4C72B0"],
                          text=[f"{v:.2f}" for v in vals], textposition="outside", showlegend=False), row=1, col=i)
fig.update_layout(title_text="Home vs Away Performance (League-wide)", height=380)
st.plotly_chart(fig, use_container_width=True)