def daj_stranku(team):
    import streamlit as st
    import pandas as pd
    import plotly.express as px

    st.set_page_config(page_title=team, page_icon="🏀")

    def load_data():
        data = pd.read_csv("csv/game.csv")
        data["game_date"] = pd.to_datetime(data["game_date"], format='%Y-%m-%d %H:%M:%S')
        data = data.loc[data["game_date"].dt.year >= 1990]
        data = data.loc[(data["team_abbreviation_home"] == team) 
                        | (data["team_abbreviation_away"] == team)]
        return data

    data_load_state = st.text('Loading data...')
    data = load_data()
    data_load_state.text('')

    home = data["team_abbreviation_home"].to_list()[0] == team
    full_name = ""
    if (home): full_name = data["team_name_home"].to_list()[0]
    else: full_name = data["team_name_away"].to_list()[0]

    st.markdown(f'# {full_name} basic stats')
    st.sidebar.header(team)

    percentages = []
    seasons = []
    threes_attemped = []
    fg_attempted = []
    threes_made = []
    fg_made = []
    ft_attempted = []
    ft_made = []

    for i in data["season_id"].unique():
        temp = data.loc[data["season_id"] == i]
        games_home = temp.loc[temp["team_abbreviation_home"] == team]
        games_away = temp.loc[temp["team_abbreviation_away"] == team]
        suma = (games_home["wl_home"] == 'W').sum() + (games_away["wl_away"] == 'W').sum()
        percentages.append(round(suma / len(temp) * 100, 2))
        seasons.append(f'{str(i - 20000)}/{str(i - 20000 + 1)}')
        threes_attemped.append((games_home["fg3a_home"].sum() + games_away["fg3a_away"].sum()) / len(temp))
        threes_made.append((games_home["fg3m_home"].sum() + games_away["fg3m_away"].sum()) / len(temp))
        fg_attempted.append((games_away["fga_away"].sum() + games_home["fga_home"].sum()) / len(temp))
        fg_made.append((games_away["fgm_away"].sum() + games_home["fgm_home"].sum()) / len(temp))
        ft_attempted.append((games_away["fta_away"].sum() + games_home["fta_home"].sum()) / len(temp))
        ft_made.append((games_away["ftm_away"].sum() + games_home["ftm_home"].sum()) / len(temp))

    st.write('### Percentage of games won by season')
    df = pd.DataFrame()
    df["Seasons"] = seasons
    df["Percentage"] = percentages

    st.line_chart(data=df, x="Seasons", y="Percentage")

    st.write('### 3 points attempted per game next to three points made per game')

    df1 = pd.DataFrame()
    df1["Seasons"] = seasons * 2
    df1["3 pt shots"] = threes_attemped + threes_made
    df1["type"] = len(seasons) * ["attempted"] + len(seasons) * ["made"]
    fig = px.bar(df1, x="Seasons", y="3 pt shots", color="type", barmode="group")

    st.plotly_chart(fig)

    st.write('### Share of 3 pointers on total field goals attempted per game')

    df1["Field goals"] = [x - i for x, i in zip(fg_attempted, threes_attemped)] + threes_attemped
    df1["type of shot"] = len(seasons) * ["2 pointer"] + len(seasons) * ["3 pointer"]

    fig = px.bar(df1, x="Seasons", y="Field goals", color="type of shot")
    st.plotly_chart(fig)

    st.write('### Number of free throws attempted and made per game')
    df1["free throws"] = ft_attempted + ft_made

    fig = px.bar(df1, x="Seasons", y="free throws", color="type", barmode="group")
    st.plotly_chart(fig)

