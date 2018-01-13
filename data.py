import pandas as pd
import numpy as np
from sklearn import preprocessing

### Column processing helper lists ###

# Info columns
def get_info_list():
    info = ["Year", "Round", "Game", "Team", "Opponent", "Player"]
    info = [col.lower() for col in info]
    return info

def get_stats_list():
    stats = ['Kicks', 'Marks', 'Hand Balls', 'Disp', 'Goals',
    'Behinds', 'Hit Outs', 'Tackles', 'Rebounds', 'Inside 50',
    'Clearances', 'Clangers', 'Frees For', 'Frees Against',
    'Uncontested Possessions', 'Contested Possessions',
    'Contested Marks', 'Marks Inside 50', 'One Percenters',
    'Bounces', 'Goal Assists', '% Time Played', 'Brownlow']
    stats = [s.lower() for s in stats]
    stats.remove('brownlow')
    stats.extend(['score', 'team win'])
    return stats

def get_gamenorm_list():
    stats = get_stats_list()
    gn_cols = ['gn ' + col for col in stats]
    return gn_cols

def get_x_cols(df):
    info = get_info_list()
    gn_cols = get_gamenorm_list()
    x_cols = info + gn_cols
    return df[x_cols]

def get_y_cols(df):
    y_cols = get_info_list()
    y_cols.extend([col for col in df.columns if 'vote' in col])
    y_cols.append('brownlow')
    return df[y_cols]

### Total data loading function ###

def load_and_normalize(norm_type = 'std'):

    print('Loading csv files'.ljust(52, '.'), end = ' ')
    df = create_total_dataframe()
    df = convert_round_to_numeric(df)
    print('Complete.')

    print('Dropping finals'.ljust(52, '.'), end = ' ')
    df = drop_finals(df)
    print('Complete.')

    print('Adding game column'.ljust(52, '.'), end = ' ')
    df = add_game_column(df)
    print('Complete.')

    print('Lowering column names'.ljust(52, '.'), end = ' ')
    df = lower_column_names(df)
    print('Complete.')

    print('Adding score columns'.ljust(52, '.'), end = ' ')
    df = add_score_column(df)
    df = add_team_score_column(df)
    print('Complete.')

    print('Adding win column'.ljust(52, '.'), end = ' ')
    df = add_team_win_column(df)
    print('Complete.')

    print('Reordering columns'.ljust(52, '.'), end = ' ')
    df = reorder_columns(df)
    print('Complete.')

    print('Initializing game-normalized columns'.ljust(52, '.'), end = ' ')
    df = add_game_normalized_cols(df)
    print('Complete.')

    print('Calculating and filling game-normalized columns'.ljust(52, '.'), end = ' ')
    df = calc_game_normalized_cols(df, norm_type)
    print('Complete.')

    print('Moving Brownlow to end and creating vote columns'.ljust(52, '.'), end = ' ')
    df = move_brownlow_to_end(df)
    df = add_vote_columns(df)
    print('Complete.')

    return df


### Helper functions ###

def create_total_dataframe():
    df_dict = {}
    for year in range(1965, 2017):
        fname = "AFL_stats/{}_stats.txt".format(year)
        df_dict[year] = pd.read_csv(fname)
        df_dict[year]["Year"] = year
    df = pd.concat(df_dict, ignore_index = True)
    return df

def convert_round_to_numeric(df):
    df['Round'] = pd.to_numeric(df['Round'], errors =  'coerce')
    return df

def drop_finals(df):
    df = df.dropna(subset = ['Round'])
    return df

def add_game_column(df):
    # Creates string containing the set of teams playing for each game and puts in Game column
    df["Game"] = ['-'.join(sorted(game)) for game in zip(df["Team"], df["Opponent"])]
    return df

def lower_column_names(df):
    df.columns = [col.lower() for col in df.columns]
    return df

def add_score_column(df):
    df['score'] = 6*df['goals'] + df['behinds']
    return df

def add_team_score_column(df):
    df['game score'] = df.groupby(['year', 'round', 'game'])['score'].transform('sum')
    df['team score'] = df.groupby(['year', 'round', 'team'])['score'].transform('sum')
    df['opponent score'] = df['game score'] - df['team score']
    df['score margin'] = df['team score'] - df['opponent score']
    return df

def add_team_win_column(df):
    df['team win'] = (df['team score'] > df['opponent score']).astype(int)
    return df

def reorder_columns(df):
    cols = df.columns.tolist()
    info = get_info_list()
    for col in reversed(info):
        cols.insert(0, cols.pop(cols.index(col)))
    df = df[cols]
    return df

def add_game_normalized_cols(df):
    stats = get_stats_list()
    gn_cols = ['gn ' + col for col in stats]
    for col in gn_cols:
        df[col] = 0
    return df

def calc_game_normalized_cols(df, norm_type = 'std'):
    stats = get_stats_list()
    gn_cols = ['gn ' + col for col in stats]
    if norm_type == 'std':
        trans_func = lambda x: (x - x.mean()) / x.std()
    elif norm_type == 'minmax':
        trans_func = lambda x: (x - x.min()) / (x.max() - x.min())
    df_norm = df.groupby(['year', 'round', 'game'])[stats].transform(trans_func).fillna(0)
    df[gn_cols] = df_norm
    return df

def move_brownlow_to_end(df):
    cols = df.columns.tolist()
    cols.append('brownlow')
    cols.remove('brownlow')
    df = df[cols]
    return df

def add_vote_columns(df):
    df['no vote'] = (df['brownlow'] == 0).astype(int)
    df['any vote'] = (df['brownlow'] > 0).astype(int)
    df['1 vote'] = (df['brownlow'] == 1).astype(int)
    df['2 vote'] = (df['brownlow'] == 2).astype(int)
    df['3 vote'] = (df['brownlow'] == 3).astype(int)
    return df