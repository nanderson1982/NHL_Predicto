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

filtered = df[['team',
 'home_or_away',
 'xGoalsPercentage',
 'corsiPercentage',
 'fenwickPercentage',
 'iceTime',
 'xOnGoalFor',
 'xGoalsFor',
 'xReboundsFor',
 'xFreezeFor',
 'xPlayStoppedFor',
 'xPlayContinuedInZoneFor',
 'xPlayContinuedOutsideZoneFor',
 'flurryAdjustedxGoalsFor',
 'scoreVenueAdjustedxGoalsFor',
 'flurryScoreVenueAdjustedxGoalsFor',
 'shotsOnGoalFor',
 'missedShotsFor',
 'blockedShotAttemptsFor',
 'shotAttemptsFor',
 'goalsFor',
 'reboundsFor',
 'reboundGoalsFor',
 'freezeFor',
 'playStoppedFor',
 'playContinuedInZoneFor',
 'playContinuedOutsideZoneFor',
 'savedShotsOnGoalFor',
 'savedUnblockedShotAttemptsFor',
 'penaltiesFor',
 'penalityMinutesFor',
 'faceOffsWonFor',
 'hitsFor',
 'takeawaysFor',
 'giveawaysFor',
 'lowDangerShotsFor',
 'mediumDangerShotsFor',
 'highDangerShotsFor',
 'lowDangerxGoalsFor',
 'mediumDangerxGoalsFor',
 'highDangerxGoalsFor',
 'lowDangerGoalsFor',
 'mediumDangerGoalsFor',
 'highDangerGoalsFor',
 'scoreAdjustedShotsAttemptsFor',
 'unblockedShotAttemptsFor',
 'scoreAdjustedUnblockedShotAttemptsFor',
 'dZoneGiveawaysFor',
 'xGoalsFromxReboundsOfShotsFor',
 'xGoalsFromActualReboundsOfShotsFor',
 'reboundxGoalsFor',
 'totalShotCreditFor',
 'scoreAdjustedTotalShotCreditFor',
 'scoreFlurryAdjustedTotalShotCreditFor',
 'xOnGoalAgainst',
 'xGoalsAgainst',
 'xReboundsAgainst',
 'xFreezeAgainst',
 'xPlayStoppedAgainst',
 'xPlayContinuedInZoneAgainst',
 'xPlayContinuedOutsideZoneAgainst',
 'flurryAdjustedxGoalsAgainst',
 'scoreVenueAdjustedxGoalsAgainst',
 'flurryScoreVenueAdjustedxGoalsAgainst',
 'shotsOnGoalAgainst',
 'missedShotsAgainst',
 'blockedShotAttemptsAgainst',
 'shotAttemptsAgainst',
 'goalsAgainst',
 'reboundsAgainst',
 'reboundGoalsAgainst',
 'freezeAgainst',
 'playStoppedAgainst',
 'playContinuedInZoneAgainst',
 'playContinuedOutsideZoneAgainst',
 'savedShotsOnGoalAgainst',
 'savedUnblockedShotAttemptsAgainst',
 'penaltiesAgainst',
 'penalityMinutesAgainst',
 'faceOffsWonAgainst',
 'hitsAgainst',
 'takeawaysAgainst',
 'giveawaysAgainst',
 'lowDangerShotsAgainst',
 'mediumDangerShotsAgainst',
 'highDangerShotsAgainst',
 'lowDangerxGoalsAgainst',
 'mediumDangerxGoalsAgainst',
 'highDangerxGoalsAgainst',
 'lowDangerGoalsAgainst',
 'mediumDangerGoalsAgainst',
 'highDangerGoalsAgainst',
 'scoreAdjustedShotsAttemptsAgainst',
 'unblockedShotAttemptsAgainst',
 'scoreAdjustedUnblockedShotAttemptsAgainst',
 'dZoneGiveawaysAgainst',
 'xGoalsFromxReboundsOfShotsAgainst',
 'xGoalsFromActualReboundsOfShotsAgainst',
 'reboundxGoalsAgainst',
 'totalShotCreditAgainst',
 'scoreAdjustedTotalShotCreditAgainst',
 'scoreFlurryAdjustedTotalShotCreditAgainst']]

# Saving the label columns
labelsdf = filtered[['team', 'home_or_away']]

# Dropping the label columns
filtered = filtered.drop(labels = ["team", "home_or_away"], axis = 1)

# Saving column headers
column_headers = filtered.columns.values.tolist()

# Normalizing the data
import pandas as pd
from sklearn import preprocessing

#normalized = normalized.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(filtered)
normalized_df = pd.DataFrame(x_scaled, columns= column_headers)