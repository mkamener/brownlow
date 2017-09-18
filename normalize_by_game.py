# Process the raw AFL_stats data and include game normalised data

import pandas as pd
import numpy as np
from sklearn import preprocessing

# Columns of interest
column_names = ["Player", "Team", "Opponent", "Round", "Kicks", "Marks", "Hand Balls", "Disp", "Goals", "Behinds", "Hit Outs",
"Tackles", "Rebounds", "Inside 50", "Clearances", "Clangers", "Frees For", "Frees Against",
"Uncontested Possessions", "Contested Possessions", "Contested Marks", "Marks Inside 50",
"One Percenters", "Bounces", "Brownlow"]

# 23 rounds starting from Round 1
round_list = []
for i in range(1, 24):
    round_list.append(str(i))

# Loop through years of interest
for year in range(1999, 2017):

    # Create filename
    fname = "AFL_stats/" + str(year) + "_stats.txt"

    # Load data
    data = pd.read_csv(fname)

    # Extract columns of interest
    data = data.loc[:, column_names]

    # Ignore finals rows
    data = data[data.Round.isin(round_list)]

    # Generate list of team names
    team_names = data.Team.unique()

    # Create new team total stat headings
    game_col_names = column_names[4:24]
    for i in range(len(game_col_names)):
        game_col_names[i] = game_col_names[i] + " G"

    # Create empty DataFrame to house all team normalized stats
    game_norm = pd.DataFrame(index=data.index, columns=game_col_names)
    game_norm = game_norm.fillna(value=0)

    # Loop through each game for each team and calculate team normalized stats
    for round_num in round_list:
        print(str(year) + " Round " + round_num)

        for team in team_names:

            # Extract the data for one game in one round
            team_playing = (data.Team == team) | (data.Opponent == team)
            indices = (data.Round == round_num) & team_playing

            # Check if team does not have a bye
            # This also prevents zero array error in sklearn MinMaxScaler()
            if sum(indices) > 0:
                norm_data = data[indices]

                # Crop to values of interest
                # 24 columns, last is brownlow
                norm_data = norm_data.iloc[:, 4:24]

                # Perform min-max scaling by game
                game_array = norm_data.values #returns a numpy array
                scaler = preprocessing.MinMaxScaler()
                game_norm_array = scaler.fit_transform(game_array)

                # Add normalized array to game normalized DataFRame
                game_norm[indices] = game_norm_array

    # Create new year series
    year_series = pd.Series(index=data.index, data=year, name="Year")

    # Join data tables together ignorning string data
    output_data = pd.concat([data.Player, year_series, data.iloc[:, 4:24], game_norm, data.Brownlow], axis=1)

    # Save data as CSV file
    fname = "stats_team/" + str(year) + "_stats_normal.csv"
    output_data.to_csv(fname, index=False)