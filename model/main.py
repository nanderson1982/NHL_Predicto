"""
This file will import and clean the data.
Train a ML Model.
Save a pickle file to be used in Web App.
"""

# Import custom functions
import api
import data_cleaning as dc
import feature_engineering as fe
import train as tr

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

# Training the Logistic Classifier and saving the Pickle file
print("** Training the ML model")
tr.model(df)

# Display all steps complete
print("** All processes complete.")