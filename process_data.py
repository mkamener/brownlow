# Process the raw AFL_stats data and include game normalised data

import pandas as pd
import numpy as np

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
    team_col_names = column_names[4:24]
    for i in range(len(team_col_names)):
        team_col_names[i] = team_col_names[i] + " T"

    # Create empty DataFrame to house all team normalized stats
    team_sum = pd.DataFrame(index=data.index, columns=team_col_names)
    team_sum = team_sum.fillna(value=0)

    # Loop through each game for each team and calculate team normalized stats
    for round_num in round_list:
        print(str(year) + " Round " + round_num)
        for team in team_names:
            # Extract the data for one team in one round
            indices = (data.Round == round_num) & (data.Team == team)
            team_data = data[indices]

            # Crop to values of interest
            team_data = team_data.iloc[:, 4:24]  # 24 columns, last is brownlow

            # Repeat team sum values for each player on team
            team_sum_np = np.tile(team_data.sum().values, (indices.sum(), 1))

            # Add new data into pandas DataFrame in relevant position, so each player has a columns relating to their
            # team totals
            team_sum[indices] = team_sum_np

            # TODO: Add in opponent team totals
            # TODO: Add in win/loss flag

    # Create new year series
    year_series = pd.Series(index=data.index, data=year, name="Year")

    # Join data tables together ignorning string data
    output_data = pd.concat([data.Player, year_series, data.iloc[:, 4:24], team_sum, data.Brownlow], axis=1)

    # Save data as CSV file
    fname = "stats_team/" + str(year) + "_stats_team.csv"
    output_data.to_csv(fname, index=False)
