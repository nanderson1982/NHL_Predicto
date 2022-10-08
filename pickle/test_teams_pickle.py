import numpy as np
import pandas as pd
import pickle

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  

# read pickle file as dataframe
df2 = pd.read_pickle('/Users/nathananderson/Desktop/NHL_Predictor/pickle/teams.pkl')
# print the dataframe
#print(df2)
list = ['WPG', 'EDM']
nlist = (','.join(str(a)for a in list))

h = list[0]
a = list[1]

home_num = df2[df2['team'] == h]['team#']
away_num = df2[df2['opposingTeam'] == a]['opposingTeam#']

model = pickle.load(open('/Users/nathananderson/Desktop/NHL_Predictor/pickle/nhl.pkl', 'rb'))

pred = model.predict([[home_num, away_num,5,5,5,5,5,5]]) ### Worked but I changed the features in the model above
print('The number is:', pred)

prob = model.predict_proba([[home_num, away_num,5,5,5,5,5,5]]) ### Worked but I changed the features in the model above
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
