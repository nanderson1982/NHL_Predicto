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
teamsData = df['team', 'opposingTeam', 'home'].copy()
#new = old[['A', 'C', 'D']].copy()

