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

st.write('#TEST MARKDOWN')

st.dataframe(players_draft_imc_year)


## Altura/Peso por temporada


seasons = players_draft_josema.season.unique()
season = seasons[1]

fig, ax = plt.subplots()
season_selected = st.selectbox ("¿Qué temporada desea analizar?", seasons, key = 'attribute_season')
players_draft_season = players_draft_josema[players_draft_josema.season == season_selected]
col1, col2, col3 = st.columns(3)
media_season_altura = np.mean(players_draft_season.player_height)
media_season_peso = np.mean(players_draft_season.player_weight)
total_season_jugadores = len(players_draft_season)
col1.metric("Total jugadores", total_season_jugadores)
col2.metric("Media altura", np.round(media_season_altura,2))
col3.metric("Media peso", np.round(media_season_peso,2))
ax = sns.scatterplot(data=players_draft_season, x='player_weight', y='player_height')
st.pyplot(fig)