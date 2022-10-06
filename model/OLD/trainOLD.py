# Import libraries
import pandas as pd
import numpy as np
import turicreate as tc
import os.path as path
from sklearn import preprocessing
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.simplefilter(action = 'ignore', category = UserWarning)
import pickle

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
allTeamsSeasons['team'] = le.fit_transform(allTeamsSeasons['team'])
allTeamsSeasons['opposingTeam'] = le.fit_transform(allTeamsSeasons['opposingTeam'])

# Slicing the data
allTeamsSeasons = allTeamsSeasons[allTeamsSeasons['situation'] == 'all']
allTeamsSeasons = allTeamsSeasons[allTeamsSeasons['season'] >= 2019]
allTeamsSeasons = allTeamsSeasons[allTeamsSeasons['playoffGame'] == 0]
allTeamsSeasons = allTeamsSeasons[allTeamsSeasons['Shootout Game'] == 0]

# New DataFrame for model
df = allTeamsSeasons[['team', 'opposingTeam', 'Win', 'home_or_away', 'year', 'month', 'day'
                      , 'flurryScoreVenueAdjustedxGoalsFor', 'flurryScoreVenueAdjustedxGoalsAgainst']]

# Split the data - Train, Test, Split
from sklearn.model_selection import train_test_split

X = df.drop(labels = "Win", axis = 1)
y = df["Win"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4, random_state=43)

# Train a LinearRegression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

lm = LogisticRegression(random_state = 16, max_iter=2000)
#lm.fit(X_train.values,y_train)
lm.fit(X_train, y_train)

"""
# Get the train and test accuracy scores
print()
print(f"Train Score: {lm.score(X_train, y_train)}")
print(f"Test Score: {lm.score(X_test, y_test)}")

print()
# Get the train and test logloss results
print(f"Train LogLoss: {log_loss(y_train, lm.predict_proba(X_train))}")
print(f"Test LogLoss: {log_loss(y_test, lm.predict_proba(X_test))}")
"""

# Saving the model
pickle.dump(lm, open('/Users/nathananderson/Desktop/NHL_Predictor/NHL_Predictor/models/nhl.pkl', 'wb'))
