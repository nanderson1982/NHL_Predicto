import pandas as pd
# read pickle file as dataframe
df2 = pd.read_pickle('/Users/nathananderson/Desktop/NHL_Predictor/pickle/teams.pkl')
# print the dataframe
#print(df2)
list = ['JETS', 'OILERS']

a = df2['team'][0]
print(a)

h=list[0]
a=list[1]

homeTeamNumber = df2.query('team = h')['team#']
print(homeTeamNumber)
