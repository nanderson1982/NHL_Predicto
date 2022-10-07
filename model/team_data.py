# Import libraries
import numpy as np
import pandas as pd
import api
import data_cleaning as dc

# Importing the most recent data
url = 'https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv'
rawData = api.api_call(url)
print("** The raw data has been successfully downloaded.")

# Cleaning the rawData
cleanData = dc.clean(rawData)
print("** The raw data has been successfully cleaned.")

# Creating team DataFrame
homeTeam = cleanData['team'].copy()
print(homeTeam)

