# ---------------------- MACHINE LEARNING --------------------------------
"""
# Function to run entrie program

def logisticClassifier(home, away):
    placeholer = []
"""

# Import libraries
import pandas as pd
import numpy as np
import turicreate as tc
import os.path as path
from sklearn import preprocessing
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.simplefilter(action = 'ignore', category = UserWarning)

# Import dataset
allTeamsSeasons = pd.read_csv('~/desktop/data_science/NHL/Datasets/all_teams.csv')

# Replacing duplicate acronyms
allTeamsSeasons['opposingTeam'].replace({'S.J': 'SJS'}, inplace = True)
allTeamsSeasons['opposingTeam'].replace({'N.J': 'NJD'}, inplace = True)
allTeamsSeasons['opposingTeam'].replace({'T.B': 'TBL'}, inplace = True)
allTeamsSeasons['opposingTeam'].replace({'L.A': 'LAK'}, inplace = True)
allTeamsSeasons['opposingTeam'].replace({'ATL': 'WPG'}, inplace = True)

allTeamsSeasons['team'].replace({'S.J': 'SJS'}, inplace = True)
allTeamsSeasons['team'].replace({'N.J': 'NJD'}, inplace = True)
allTeamsSeasons['team'].replace({'T.B': 'TBL'}, inplace = True)
allTeamsSeasons['team'].replace({'L.A': 'LAK'}, inplace = True)
allTeamsSeasons['team'].replace({'ATL': 'WPG'}, inplace = True)

allTeamsSeasons['name'].replace({'S.J': 'SJS'}, inplace = True)
allTeamsSeasons['name'].replace({'N.J': 'NJD'}, inplace = True)
allTeamsSeasons['name'].replace({'T.B': 'TBL'}, inplace = True)
allTeamsSeasons['name'].replace({'L.A': 'LAK'}, inplace = True)
allTeamsSeasons['name'].replace({'ATL': 'WPG'}, inplace = True)

allTeamsSeasons['playerTeam'].replace({'S.J': 'SJS'}, inplace = True)
allTeamsSeasons['playerTeam'].replace({'N.J': 'NJD'}, inplace = True)
allTeamsSeasons['playerTeam'].replace({'T.B': 'TBL'}, inplace = True)
allTeamsSeasons['playerTeam'].replace({'L.A': 'LAK'}, inplace = True)
allTeamsSeasons['playerTeam'].replace({'ATL': 'WPG'}, inplace = True)

# Adding columns
shootout_game = np.where((allTeamsSeasons['situation'] == 'all') & (allTeamsSeasons['goalsFor'] == allTeamsSeasons['goalsAgainst']), 1, 0)
allTeamsSeasons.insert(loc = 6, column = 'Shootout Game', value = shootout_game)

ot_game = np.where(allTeamsSeasons['iceTime'] > 3600.0, 1, 0)
allTeamsSeasons.insert(loc = 7, column = 'OT Game', value = ot_game)

win = np.where(allTeamsSeasons['goalsFor'] > allTeamsSeasons['goalsAgainst'], 1, 0)
allTeamsSeasons.insert(loc = 8, column = 'Win', value = win)

loss = np.where((allTeamsSeasons['OT Game'] == 0) & (allTeamsSeasons['goalsFor'] < allTeamsSeasons['goalsAgainst']), 1, 0)
allTeamsSeasons.insert(loc = 9, column = 'Loss', value = loss)

# ot_loss = np.where((allTeamsSeasons['OT Game'] == 1) & (allTeamsSeasons['goalsFor'] < allTeamsSeasons['goalsAgainst']), 1, 0)
# allTeamsSeasons.insert(loc = 10, column = 'OT Loss', value = ot_loss)

# Adding date columns
allTeamsSeasons['gameDate'] = pd.to_datetime(allTeamsSeasons['gameDate'],format='%Y%m%d')
allTeamsSeasons['year'] = pd.DatetimeIndex(allTeamsSeasons['gameDate']).year
allTeamsSeasons['month'] = pd.DatetimeIndex(allTeamsSeasons['gameDate']).month
allTeamsSeasons['day'] = pd.DatetimeIndex(allTeamsSeasons['gameDate']).day

# Changing home_or_away column to be numerical
le = preprocessing.LabelEncoder()
allTeamsSeasons['home_or_away'] = le.fit_transform(allTeamsSeasons['home_or_away'])

# Slicing the data
allTeamsSeasons = allTeamsSeasons[allTeamsSeasons['situation'] == 'all']
allTeamsSeasons = allTeamsSeasons[allTeamsSeasons['season'] >= 2019]
allTeamsSeasons = allTeamsSeasons[allTeamsSeasons['playoffGame'] == 0]
allTeamsSeasons = allTeamsSeasons[allTeamsSeasons['Shootout Game'] == 0]

# New DataFrame for model
df = allTeamsSeasons[['gameId', 'team', 'opposingTeam', 'goalsFor', 'goalsAgainst', 'OT Game', 'Win', 'home_or_away'
                      , 'flurryScoreVenueAdjustedxGoalsFor', 'flurryScoreVenueAdjustedxGoalsAgainst', 'year', 'month', 'day']]

# Converting DataFrame to SFrame
wpgSF = tc.SFrame(df)

# Split the data into train and test data
seed = 100
train_data, test_data = wpgSF.random_split(0.775, seed = seed)

# Create a model
model = tc.logistic_classifier.create(train_data
                                     , target='Win'
                                     , features = ['flurryScoreVenueAdjustedxGoalsFor'
                                                  ,'flurryScoreVenueAdjustedxGoalsAgainst'
                                                  , 'home_or_away'
                                                  , 'opposingTeam'
                                                  , 'year', 'month', 'day']
                                     , l1_penalty = 0.5
                                     , l2_penalty = 1.4
                                     , max_iterations = 50
                                     , seed = seed
                                     , validation_set= 'auto'
                                     )

"""
# Making predictions using Tkinter input
pred = tc.SFrame({'team': home_team.get(),
                  'opposingTeam': away_team.get()
                  })

# Prediction results
pred_class = model.predict(pred, output_type = "class")
pred_probability = model.predict(pred, output_type = 'probability')
pred_margin = model.predict(pred, output_type = "margin")
"""

"""
# Making predictions using new function
"""

import pickle
pickle.dump(model, open('turicreate.pkl', 'wb'))
