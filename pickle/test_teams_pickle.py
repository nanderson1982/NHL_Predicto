import numpy as np
import pandas as pd
import pickle

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  

# Read teams pickle file as dataframe
df2 = pd.read_pickle('/Users/nathananderson/Desktop/NHL_Predictor/pickle/teams.pkl')
#print(df2)

# Read ML model pickfile 
model = pickle.load(open('/Users/nathananderson/Desktop/NHL_Predictor/pickle/nhl.pkl', 'rb'))

# Team variables for testing
list = ['WPG', 'EDM']

h = list[1]
a = list[0]

# Find the corresponding team number -> needed for prediction
home_num = df2[df2['team'] == h]['team#'].values[0]
away_num = df2[df2['opposingTeam'] == a]['opposingTeam#'].values[0]
print('home team number:', home_num)

# Making predictions
pred = model.predict([[home_num, away_num,5,5,5,5,5,5]]) 
print('The predicted class:', pred[0])

prob = model.predict_proba([[home_num, away_num,5,5,5,5,5,5]])
# class 0 probability
print(prob[0][0])

# ------------------------------------------------------------------------
# Change column type
#df2['team'] = df2['team'].astype(str)

# Checking column types
#d = df2.dtypes
#print(d)

# Pulling 1st team name
#a = df2['team'][0]
#print(a)

"""
a=list[1]
print("h is:", h)
print(type(h))
"""

#homeTeamNumber = df2.query('team = h')['team#']
#print(homeTeamNumber)
