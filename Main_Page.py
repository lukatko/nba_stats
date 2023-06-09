from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import datetime as dt
from geopy.geocoders import Nominatim
import plotly.express as px
import plotly.graph_objects as go
from sklearn import preprocessing
import xgboost as xgb

st.set_page_config(
    page_title="NBA Statistics",
    page_icon="🏀",
)

st.write("# NBA Statistics")

st.sidebar.success("Select a team above")

@st.cache_data
def load_data():
    data = pd.read_csv("csv/game.csv")
    data["game_date"] = pd.to_datetime(data["game_date"], format='%Y-%m-%d %H:%M:%S')
    data = data.loc[data["game_date"].dt.year >= 1990]
    return data

data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text('')

teams = data["team_name_home"].unique()

st.write("### Locations of NBA teams")
lats = []
longs = []
geolocator = Nominatim(user_agent="NBA stats")

map_loading = st.text('Loading map...')

@st.cache_data
def load_map():
    team_names = []
    for i in data["team_name_home"].unique():
        if (i == "New Orleans/Oklahoma City Hornets" or i == "Golden State Warriors"): continue
        loc = geolocator.geocode(" ".join(i.split()[:-1]))
        lats.append(loc.latitude)
        longs.append(loc.longitude)
        team_names.append(i)

    loc = geolocator.geocode("San Francisco")
    lats.append(loc.latitude)
    longs.append(loc.longitude)
    team_names.append("Golden State Warriors")

    df = pd.DataFrame()
    df["Team Name"] = team_names
    df["latitude"] = lats
    df["longitude"] = longs
    return df

df = load_map()
st.map(df)
map_loading.text('')

st.write('### Evolution of 3 point shots over time')
threes_attempted = []
share_on_fgs = []
percentage_made = []
seasons = []

for i in data["season_id"].unique():
    temp = data.loc[data["season_id"] == i]
    threes_attempted.append((temp["fg3a_home"] + temp["fg3a_away"]).sum() / (len(temp) * 2))
    share_on_fgs.append(threes_attempted[-1] / ((temp["fga_home"] + temp["fga_away"]).sum() / (len(temp) * 2)) * 100)
    percentage_made.append(((temp["fg3m_home"] + temp["fg3m_away"]).sum() / (len(temp) * 2)) / threes_attempted[-1] * 100)
    seasons.append(f'{str(i - 20000)}/{str(i - 20000 + 1)}')

df1 = pd.DataFrame()
df1["3pts attempted"] = threes_attempted
df1["Share of 3pts on field goals"] = share_on_fgs
df1["Percentage of 3pts made"] = percentage_made

lines = st.multiselect("Select what you want to plot", df1.columns)

fig = go.Figure()
for line in lines:
    fig = fig.add_trace(go.Scatter(x=seasons, y=df1[line].to_list(), name=line))

st.plotly_chart(fig)
st.markdown(
    '''
    ### We can see:
    - Number of 3 pointers attempted has really started growing around year 2011 that might be because of the GSW team.
    - Field goals attempted have stayed basically the same over the years because the number of 3 pointers has grown almost the same way as their share on total field goals.
    - The accuracy from three has stayed the same over the years, only their number has grown.
    - The growth of threes has slowed over past years, so maybe it will stay the same for the years to come.
    '''
    )

st.markdown(
    '''
    ### Logistic regression:
    - We've trained our model on some variables, now you can write stats of some player and we will tell you which position he most likely plays.
    '''
)

@st.cache_data
def load_players():
    player_data = pd.read_csv("csv/2021-2022 NBA Player Stats - Playoffs.csv", sep=';', encoding='latin-1')
    return player_data.loc[(player_data["G"] > 5) & (player_data["MP"] > 3)]

player_data = load_players()
player_data = player_data[["Pos", "PTS", "FGA", "FG%", "3PA", "3P%", "FTA", "FT%", "TRB", "AST", "BLK", "STL", "TOV"]]
le = preprocessing.LabelEncoder()
X = player_data.drop("Pos", axis=1)
le.fit(player_data["Pos"])
y = le.transform(player_data["Pos"])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.15)
tr = st.text("Training...")
xgb_clf = xgb.XGBClassifier(objective='multi:softmax', 
                            num_class=5, 
                            early_stopping_rounds=10, 
                            seed=42,
                            learning_rate=0.05,
                            colsample_bytree=0.8)
xgb_clf.fit(X_train, 
            y_train,
            verbose=1,
            eval_set=[(X_train, y_train), (X_test, y_test)])

y_pred = xgb_clf.predict(X_test)
tr.text('')
st.write("Accuracy on test set is:", accuracy_score(y_test, y_pred))

PTS = st.number_input('Points  per game:', min_value=0.0, max_value=50.0, value=10.0)
FGA = st.number_input('FGA  per game:', min_value=0.0, max_value=50.0, value=7.0)
FGM = st.number_input('FG made % per game:', min_value=0.0, max_value=100.0, value=52.0)
PA = st.number_input('3 Points attempted  per game:', min_value=0.0, max_value=50.0, value=3.0)
PM = st.number_input('3 Points made % per game:', min_value=0.0, max_value=100.0, value=33.0)
FTA = st.number_input('FTA  per game:', min_value=0.0, max_value=50.0, value=10.0)
FTM = st.number_input('FTM % per game:', min_value=0.0, max_value=100.0, value=24.0)
TRB = st.number_input('Total rebounds  per game:', min_value=0.0, max_value=25.0, value=5.0)
AST = st.number_input('Assists  per game:', min_value=0.0, max_value=20.0, value=11.0)
BLK = st.number_input('Blocks  per game:', min_value=0.0, max_value=10.0, value=2.5)
STL = st.number_input('Steals  per game:', min_value=0.0, max_value=10.0, value=2.0)
TOV = st.number_input('Turnovers  per game:', min_value=0.0, max_value=15.0, value=4.0)


if st.button('Predict Position'):
    pos = xgb_clf.predict(np.array([PTS, FGA, FGM, PA, PM, FTA, FTM, TRB, AST, BLK, STL, TOV]).reshape(1, -1))
    st.success(f'The predicted position of the player is {le.inverse_transform(pos)[0]}')