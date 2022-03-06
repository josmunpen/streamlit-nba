import streamlit as st
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

### Data Import ###
df = pd.read_csv("./data/all_seasons.csv")
df.rename(columns={'Unnamed: 0': 'index_player_season'}, inplace=True)

players_draft = df.groupby('player_name').first()
players_draft_josema = players_draft.copy()
players_draft_josema = players_draft_josema[['player_height', 'player_weight', 'season', 'pts']]
players_draft_josema['player_imc'] = players_draft_josema.apply(lambda row: row.player_weight / ((0.01 * row.player_height) ** 2), axis = 1)

players_draft_imc_year = players_draft_josema.groupby('season').agg({'player_height':'mean', 'player_weight':'mean', 'player_imc':'mean', 'pts': 'mean'})

st.markdown('#TEST MARKDOWN')

