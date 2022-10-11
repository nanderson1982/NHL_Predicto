# Import libraries
import numpy as np
import pandas as pd
import api
import data_cleaning as dc
import feature_engineering as fe

# Importing the most recent data
url = 'https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv'
rawData = api.api_call(url)
print("** The raw data has been successfully downloaded.")

# Cleaning the rawData
cleanData = dc.clean(rawData)
print("** The raw data has been successfully cleaned.")

# Feature Engineering using the cleanData
df = fe.fengine(cleanData)
print("** Feature Engineering has been successfully completed.")

# Creating team DataFrame
teamsData = df[['team', 'team#', 'opposingTeam', 'opposingTeam#']].copy()
teamsData = teamsData.drop_duplicates(subset= 'team#', keep= 'first')
teamsData = teamsData.sort_values('team#').reset_index(drop = True)
print("** DataFrame created.")

# Creating pickle of teamsData DataFrame - save file
teamsData.to_pickle('/Users/nathananderson/Desktop/NHL_Predictor/pickle/teams.pkl')
print('DataFrame saved to Pickle file.')
