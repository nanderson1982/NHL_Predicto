import pickle

model = pickle.load(open('/Users/nathananderson/Desktop/NHL_Predictor/models/nhl.pkl', 'rb'))
print('Home Team Win(1) - Home Team Lose(0)', model.predict([[1,2,1,2022,10,3,4,4]]))