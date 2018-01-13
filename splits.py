### Packages ###

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.utils import shuffle

from data import get_info_list, get_stats_list, get_gamenorm_list, get_x_cols, get_y_cols

### Main function ###

def split_all(df, start_year, end_year, test_years, val_size):

    df = keep_years(df, start_year, end_year)
    df_trainval, df_test = split_trainval_test_years(df, test_years)
    df_train, df_val = train_val_game_split(df_trainval, val_size)

    x_train = get_x_cols(df_train)
    x_val = get_x_cols(df_val)
    x_test = get_x_cols(df_test)

    y_train = get_y_cols(df_train)
    y_val = get_y_cols(df_val)
    y_test = get_y_cols(df_test)

    return x_train, x_val, x_test, y_train, y_val, y_test

### Helper functions ###

def keep_years(df, start_year, end_year):
    return df[(df['year'] >= start_year) & (df['year'] <= end_year)]

def split_trainval_test_years(df, test_years):
    # test_years as list
    df_trainval = df[~df['year'].isin(test_years)]
    df_test = df[df['year'].isin(test_years)]
    return df_trainval, df_test

def train_val_game_split(df, val_size, random_state = 0):
    groupby_game = df.groupby(['year', 'round', 'game']).size()
    unique_games = groupby_game.reset_index()[['year', 'round', 'game']]
    unique_games = unique_games.sample(frac = 1)
    games = unique_games.shape[0]
    val_rows = int(val_size * games)
    val_games = unique_games[:val_rows]
    train_games = unique_games[val_rows:]
    df_val = df.merge(val_games, how = 'inner')
    df_train = df.merge(train_games, how = 'inner')
    return df_train, df_val

def balanced_undersample(x, y):
    minority = int((sum(y == 0) > sum(y == 1)))
    y_min = y[y == minority]
    y_maj = y[y != minority]
    x_min = x[y == minority]
    x_maj = x[y != minority]
    n_min = y_min.shape[0]

    x_maj_sample, y_maj_sample = resample(*[x_maj, y_maj], n_samples = n_min)
    x_balanced = pd.concat([x_min, x_maj_sample])
    y_balanced = pd.concat([y_min, y_maj_sample])
    x_balanced, y_balanced = shuffle(*[x_balanced, y_balanced])

    return x_balanced, y_balanced

def keep_gamenorm_only(df_x):
    gn_cols = get_gamenorm_list()
    return df_x[gn_cols]