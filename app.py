import streamlit as st
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
import seaborn as sns

# st.set_page_config(layout="wide")

### Data Import ###
df = pd.read_csv("./data/all_seasons.csv")
df.rename(columns={'Unnamed: 0': 'index_player_season'}, inplace=True)

players_draft = df.groupby('player_name').first()
players_draft_josema = players_draft.copy()
players_draft_josema = players_draft_josema[['player_height', 'player_weight', 'season', 'pts']]
players_draft_josema['player_imc'] = players_draft_josema.apply(lambda row: row.player_weight / ((0.01 * row.player_height) ** 2), axis = 1)

players_draft_imc_year = players_draft_josema.groupby('season').agg({'player_height':'mean', 'player_weight':'mean', 'player_imc':'mean', 'pts': 'mean'})

st.title('Análisis de datos deportivos sobre jugadores de la NBA 🏀')

st.dataframe(players_draft_imc_year)


## Altura/Peso por temporada
st.header("Análisis altura/peso a lo largo de las temporadas")

seasons = players_draft_josema.season.unique()

seasons_order = seasons.sort()
fig, ax = plt.subplots()
season_selected = st.select_slider("¿Qué temporada desea analizar?", options=seasons, key='1')
players_draft_season = players_draft_josema[players_draft_josema.season == season_selected]

with st.expander("Añadir temporada"):
    season_selected2 = st.select_slider("¿Con qué temporada desea comparar?", options=seasons, key='2')
    players_draft_season2 = players_draft_josema[players_draft_josema.season == season_selected2]

st.subheader('Estadísticas temporada 1')
col1, col2, col3 = st.columns(3)
media_season_altura = np.mean(players_draft_season.player_height)
media_season_peso = np.mean(players_draft_season.player_weight)
total_season_jugadores = len(players_draft_season)
col1.metric("Total jugadores", total_season_jugadores)
col2.metric("Media altura", np.round(media_season_altura,2))
col3.metric("Media peso", np.round(media_season_peso,2))

st.subheader('Estadísticas temporada 2')
col1, col2, col3 = st.columns(3)
media_season_altura2 = np.mean(players_draft_season2.player_height)
media_season_peso2 = np.mean(players_draft_season2.player_weight)
total_season_jugadores2 = len(players_draft_season2)
col1.metric("Total jugadores", total_season_jugadores2)
col2.metric("Media altura", np.round(media_season_altura2,2))
col3.metric("Media peso", np.round(media_season_peso2,2))


ax = sns.scatterplot(data=players_draft_season, x='player_weight', y='player_height')
ax2 = sns.scatterplot(data=players_draft_season2, x='player_weight', y='player_height')
st.pyplot(fig)

fig34, ax34 = plt.subplots()
two_seasons = players_draft_josema[(players_draft_josema.season == season_selected) | (players_draft_josema.season == season_selected2) ]
ax34 = sns.boxplot(data=two_seasons, x='season', y='player_height')
st.pyplot(fig34)

fig56, ax56 = plt.subplots()
ax56 = sns.boxplot(data=two_seasons, x='season', y='player_weight')
st.pyplot(fig56)