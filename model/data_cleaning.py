def clean(df):

    # Replacing duplicate team name acronyms
    df['opposingTeam'].replace({'S.J': 'SJS'}, inplace = True)
    df['opposingTeam'].replace({'N.J': 'NJD'}, inplace = True)
    df['opposingTeam'].replace({'T.B': 'TBL'}, inplace = True)
    df['opposingTeam'].replace({'L.A': 'LAK'}, inplace = True)
    df['opposingTeam'].replace({'ATL': 'WPG'}, inplace = True)

    df['team'].replace({'S.J': 'SJS'}, inplace = True)
    df['team'].replace({'N.J': 'NJD'}, inplace = True)
    df['team'].replace({'T.B': 'TBL'}, inplace = True)
    df['team'].replace({'L.A': 'LAK'}, inplace = True)
    df['team'].replace({'ATL': 'WPG'}, inplace = True)

    df['name'].replace({'S.J': 'SJS'}, inplace = True)
    df['name'].replace({'N.J': 'NJD'}, inplace = True)
    df['name'].replace({'T.B': 'TBL'}, inplace = True)
    df['name'].replace({'L.A': 'LAK'}, inplace = True)
    df['name'].replace({'ATL': 'WPG'}, inplace = True)

    df['playerTeam'].replace({'S.J': 'SJS'}, inplace = True)
    df['playerTeam'].replace({'N.J': 'NJD'}, inplace = True)
    df['playerTeam'].replace({'T.B': 'TBL'}, inplace = True)
    df['playerTeam'].replace({'L.A': 'LAK'}, inplace = True)
    df['playerTeam'].replace({'ATL': 'WPG'}, inplace = True)
    
    cleanData = df
    
    return cleanData