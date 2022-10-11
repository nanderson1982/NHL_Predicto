def fengine(df):
    # Import libraries
    import numpy as np
    import pandas as pd
    from sklearn import preprocessing

    # Adding columns
    shootout_game = np.where((df['situation'] == 'all') & (df['goalsFor'] == df['goalsAgainst']), 1, 0)
    df.insert(loc = 6, column = 'Shootout Game', value = shootout_game)

    ot_game = np.where(df['iceTime'] > 3600.0, 1, 0)
    df.insert(loc = 7, column = 'OT Game', value = ot_game)

    win = np.where(df['goalsFor'] > df['goalsAgainst'], 1, 0)
    df.insert(loc = 8, column = 'Win', value = win)

    loss = np.where((df['OT Game'] == 0) & (df['goalsFor'] < df['goalsAgainst']), 1, 0)
    df.insert(loc = 9, column = 'Loss', value = loss)

    # ot_loss = np.where((df['OT Game'] == 1) & (df['goalsFor'] < df['goalsAgainst']), 1, 0)
    # df.insert(loc = 10, column = 'OT Loss', value = ot_loss)

    # Adding date columns
    df['gameDate'] = pd.to_datetime(df['gameDate'],format='%Y%m%d')
    df['year'] = pd.DatetimeIndex(df['gameDate']).year
    df['month'] = pd.DatetimeIndex(df['gameDate']).month
    df['day'] = pd.DatetimeIndex(df['gameDate']).day

    # Adding columns to be numerical
    le = preprocessing.LabelEncoder()
    df['home_or_away#'] = le.fit_transform(df['home_or_away'])
    df['team#'] = le.fit_transform(df['team'])
    df['opposingTeam#'] = le.fit_transform(df['opposingTeam'])

    # Slicing the data
    df = df[df['situation'] == 'all']
    df = df[df['season'] >= 2020]
    df = df[df['playoffGame'] == 0]
    df = df[df['Shootout Game'] == 0]

    return df