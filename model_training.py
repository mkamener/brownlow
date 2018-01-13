### Packages ###

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import resample
from sklearn.utils import shuffle

from data import get_info_list, get_stats_list, get_gamenorm_list, get_x_cols, get_y_cols
import splits

# Todo
# Calculate average and stdev
# Allow model flexibility

### Main function ###

def train_and_predict(df, balanced_bags = 10):

    # Generate column reference lists
    y_cols = get_y_cols(df)
    gn_cols = get_gamenorm_list()
    info = get_info_list()
    params = get_best_params()

    print('Splitting into train, val and test sets'.ljust(52, '.'), end = ' ')
    x_train, x_val, x_test, y_train, y_val, y_test = \
        splits.split_all(df, 2002, 2015, [2015], 0.25)
    print('Complete.')

    # Todo: add margin (only normalize after all splits)

    for i in range(balanced_bags):

        print((' '*2 + 'Creating bag {}').format(i).ljust(52, '.'), end = ' ')
        y = y_train['3 vote']
        x = x_train[gn_cols]
        x_bal, y_bal = splits.balanced_undersample(x, y)
        print('Complete.')

        print((' '*4 + 'Training'.format(i)).ljust(52, '.'), end = ' ')
        forest = train_forest(x_bal, y_bal, params)
        print('Complete.')

        print((' '*4 + 'Inferring 3 vote probabilities').ljust(52, '.'), end = ' ')
        add_vote_prob_to_y(forest, x_train[gn_cols], y_train, i)
        add_vote_prob_to_y(forest, x_val[gn_cols], y_val, i)
        #add_vote_prob_to_y(forest, x_test, y_test, bag_num)
        print('Complete.')

        print((' '*4 + 'Inferring game votes from game rank').ljust(52, '.'), end = ' ')
        add_game_votes_to_y(y_train, i)
        add_game_votes_to_y(y_val, i)
        #add_game_votes_to_y(y_test, bag_num)
        print('Complete.')

    #reorder
    y_train = move_votes_to_end_of_y(y_train)
    y_val = move_votes_to_end_of_y(y_val)

    return x_train, x_val, y_train, y_val #add test set later


### Helper functions ###

def train_forest(x, y, params):
    forest = RandomForestClassifier(**params)
    forest.fit(x,y)
    return forest

def add_vote_prob_to_y(forest, df_x, df_y, bag_num):
    # df_x needs to have lower case info columns attached
    # does not return anything, only modifies the df_y
    gn_cols = get_gamenorm_list()
    x = df_x[gn_cols]
    df_y['bag {} prob'.format(bag_num)] = forest.predict_proba(x)[:,1]

def add_game_votes_to_y(df_y, bag_num):
    group_by_game = df_y.groupby(['year', 'round', 'game'])
    rank_by_game = group_by_game.rank(ascending = False)['bag {} prob'.format(bag_num)]
    label = 'bag {} vote'.format(bag_num)
    df_y[label] = rank_by_game.map({1:3, 2:2, 3:1}).fillna(0)

def move_votes_to_end_of_y(df_y):
    cols = df_y.columns.tolist()
    bag_votes = [col for col in cols if all(word in col for word in ['vote','bag'])]
    for col in bag_votes:
        cols.append(cols.pop(cols.index(col)))
    df_y = df_y[cols]
    return df_y

def get_best_params():
    best_params = { 'n_estimators': 100,
                    'max_depth': 30,
                    'min_samples_split': 10,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'bootstrap': True}
    return best_params

def paramater_grid():
    parameter_grid = {
                 'max_depth' : [3,10, 30],
                 'n_estimators': [10, 30, 100],
                 'max_features': ['sqrt'],
                 'min_samples_split': [2, 10, 30],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False]}
    return parameter_grid

def grid_search(x, y, parameter_grid, n_folds = 5, metric = 'f1', verbose = 1):
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(y, n_folds)

    grid_search = GridSearchCV( forest,
                                scoring = metric,
                                param_grid = paramater_grid,
                                cv = cross_validation,
                                verbose = verbose)
    grid_search.fit(x,y)
    return grid_search