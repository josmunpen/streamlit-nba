import streamlit as st
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
import seaborn as sns
import joblib

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

st.write('A continuación se muestra el DataFrame que recoge las estadísticas principales de los jugadores drafteados cada temporada.')

st.dataframe(players_draft_imc_year)


## Altura/Peso por temporada
st.header("Análisis altura/peso a lo largo de las temporadas 📈")

seasons = players_draft_josema.season.unique()

seasons_order = seasons.sort()
fig, ax = plt.subplots()
season_selected = st.select_slider("¿Qué temporada desea analizar?", options=seasons, key='1')
players_draft_season = players_draft_josema[players_draft_josema.season == season_selected]



st.subheader('Estadísticas temporada 1: {}'.format(season_selected))
col1, col2, col3 = st.columns(3)
media_season_altura = np.mean(players_draft_season.player_height)
media_season_peso = np.mean(players_draft_season.player_weight)
total_season_jugadores = len(players_draft_season)
col1.metric("Total jugadores", total_season_jugadores)
col2.metric("Media altura", np.round(media_season_altura,2))
col3.metric("Media peso", np.round(media_season_peso,2))

with st.expander("Añadir temporada"):
    season_selected2 = st.select_slider("¿Con qué temporada desea comparar?", options=seasons, key='2')
    players_draft_season2 = players_draft_josema[players_draft_josema.season == season_selected2]
    st.subheader('Estadísticas temporada 2: {}'.format(season_selected2))
    col1, col2, col3 = st.columns(3)
    media_season_altura2 = np.mean(players_draft_season2.player_height)
    media_season_peso2 = np.mean(players_draft_season2.player_weight)
    total_season_jugadores2 = len(players_draft_season2)
    col1.metric("Total jugadores", total_season_jugadores2)
    col2.metric("Media altura", np.round(media_season_altura2,2))
    col3.metric("Media peso", np.round(media_season_peso2,2))


ax = sns.scatterplot(data=players_draft_season, x='player_weight', y='player_height')
ax2 = sns.scatterplot(data=players_draft_season2, x='player_weight', y='player_height')
ax.set(xlabel="Altura (cm)", ylabel = "Peso (kg)")
st.pyplot(fig)

with st.expander("Distribución de altura y peso"):
    fig34, ax34 = plt.subplots()
    two_seasons = players_draft_josema[(players_draft_josema.season == season_selected) | (players_draft_josema.season == season_selected2) ]
    ax34 = sns.boxplot(data=two_seasons, x='season', y='player_height')
    ax34.set(ylabel="Altura", xlabel='Temporada')
    st.pyplot(fig34)

    fig56, ax56 = plt.subplots()
    ax56 = sns.boxplot(data=two_seasons, x='season', y='player_weight')
    ax56.set(ylabel="Peso", xlabel='Temporada')
    st.pyplot(fig56)


players_draft_josema['player_imc'] = players_draft_josema.apply(lambda row: row.player_weight / ((0.01 * row.player_height) ** 2), axis = 1)
with st.expander('Análisis media móvil por temporada'):
    players_draft_imc_year = players_draft_josema.groupby('season').agg({'player_height':'mean', 'player_weight':'mean', 'player_imc':'mean', 'pts': 'mean'})


    fig79, ax79 = plt.subplots()
    ax79 = sns.lineplot(x=players_draft_imc_year.index, y=players_draft_imc_year.player_height.rolling(6).mean()).set(xlabel='Temporada', ylabel="Altura (cm)")
    plt.xticks(rotation=45,horizontalalignment="right")
    st.pyplot(fig79)


    fig8 = plt.figure()
    sns.lineplot(x=players_draft_imc_year.index, y=players_draft_imc_year.player_weight.rolling(6).mean()).set(xlabel='Temporada', ylabel="Peso (kg)")
    plt.xticks(rotation=45,horizontalalignment="right")
    st.pyplot(fig8)

    fig9 = plt.figure()
    sns.lineplot(x=players_draft_imc_year.index, y=players_draft_imc_year.player_imc.rolling(6).mean()).set(xlabel='Temporada', ylabel="IMC")
    plt.xticks(rotation=45,horizontalalignment="right")
    st.pyplot(fig9)


st.write('_______________________________________________')
## Predicción
st.header('Predicción de puntos de un jugador en la siguiente temporada 🔮')

st.write('Tras realizar una comparativa mediante validación cruzada con 14 modelos se ha seleccionado el **regresor de bosques aleatorios** (Random Forest Regressor) con una configuración de 100 estimadores como **mejor modelo**. ')

res_pts = pd.read_csv("./data/res_pts.csv")
res_pts.drop(columns=['Unnamed: 0'], inplace=True)

with st.expander("Comparativa de modelos"):
    st.dataframe(res_pts)
    
st.subheader('Métricas del modelo seleccionado')
col1, col2, col3 = st.columns(3)
col1.metric("MAE", '1.57')
col2.metric("RMSE", '2.21')
col3.metric("Tiempo empleado en entrenamiento", '5.51s')

regressor = joblib.load('./data/rf_joblib.pkl')

def predict_pts(age, height, weight, draft_round, draft_number, reb, ast):
    prediction=regressor.predict([[age, height, weight, draft_round, draft_number, reb, ast]]) #predictions using our model
    return prediction 

height = st.text_input("Altura (m)", placeholder=2.06)
weight = st.text_input("Peso (kg)", placeholder=102.06)
age = st.text_input("Edad", placeholder=33) 
draft_round = st.text_input("Ronda de draft (1-7)", placeholder=1) 
draft_number = st.text_input("Posición de draft (1-41)", placeholder=23) 
reb = st.text_input("Media de rebotes", placeholder=7.9) 
ast = st.text_input("Media de asistencias", placeholder=0.8) 
result=""

if st.button("Predicción"):
    result=predict_pts(age, height, weight, draft_round, draft_number, reb, ast) 
    st.success("El jugador realizará una media de {} puntos por partido la siguiente temporada.".format(result))
