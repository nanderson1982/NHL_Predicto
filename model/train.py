def model(df):

    # Import warning libraries
    import warnings
    warnings.simplefilter(action = 'ignore', category = FutureWarning)
    
    # Split the data - Train, Test, Split
    from sklearn.model_selection import train_test_split
    
    # New DataFrame for model
    df = df[['team#', 'opposingTeam#', 'Win', 'home_or_away#', 'year', 'month', 'day'
                        , 'flurryScoreVenueAdjustedxGoalsFor', 'flurryScoreVenueAdjustedxGoalsAgainst']]

    X = df.drop(labels = "Win", axis = 1)
    y = df["Win"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4, random_state=10)

    # Train a LinearRegression model
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    lm = LogisticRegression(random_state = 16, max_iter=2000)
    #lm.fit(X_train.values,y_train)
    lm.fit(X_train, y_train)
    print("** Logistic Regression model has been trained.")
    
    # Get the train and test accuracy scores
    print(f"** Training Score: {lm.score(X_train, y_train)}")
    print(f"** Testing Score: {lm.score(X_test, y_test)}")

    # Get the train and test logloss results
    print(f"** Training LogLoss: {log_loss(y_train, lm.predict_proba(X_train))}")
    print(f"** Testing LogLoss: {log_loss(y_test, lm.predict_proba(X_test))}")
    
    # Saving the model
    import pickle
    pickle.dump(lm, open('/Users/nathananderson/Desktop/NHL_Predictor/pickle/nhl.pkl', 'wb'))
    print("** Model successfully saved as a Pickle file.")