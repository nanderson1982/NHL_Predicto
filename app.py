# Import libraries
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  


# Create an app object using the Flask class
app = Flask(__name__)

# Load the trained model - Pickle File
model = pickle.load(open('/Users/nathananderson/Desktop/NHL_Predictor/pickle/nhl.pkl', 'rb'))

teams = pd.read_pickle('/Users/nathananderson/Desktop/NHL_Predictor/pickle/teams.pkl')
                        
# Define the route to be home. Here, home function is with '/', our root directory. 
# The decorator below links the relative route of the URL to the function it is decorating.
# Running the app sends us to index.html.
# Note that render_template means it looks for the file in the templates folder. 

# use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')

# You can use the methods argument of the route() decorator to handle different HTTP methods.
# GET: A GET message is send, and the server returns data
# POST: Used to send HTML form data to the server.
# Add Post method to the decorator to allow for form submission. 
# Redirect to /predict page with the output

#-------------------------------------------------------------------------
@app.route('/predict', methods = ['POST'])
def predict():

    # Capturing user input for 'team' and 'opposingTeam' from index.html
    userInput = []
    for i in request.form.values():
        userInput.append(i.upper())
        
    # Home team user input
    homeTeam = userInput[0]    
    homeNum = teams[teams['team'] == userInput[0]]['team#'].values[0]
    
    # Away team user input
    awayTeam = userInput[1]
    awayNum = teams[teams['opposingTeam'] == awayTeam]['opposingTeam#'].values[0]

    # Making the prediction 
    pred = model.predict([[homeNum, 
                           awayNum,
                           5,5,5,5,5,5]])
    
    # Saving the Winner
    winner = []
    if pred[0] == 1:
        winner = userInput[0]
    else:
        winner = userInput[1] 
    
    # Returning info back to index,html
    #return render_template('index.html', prediction_text = userInput[0], test_text = pred[0]) 
    return render_template('index.html', 
                           prediction_text = ('The winner is:', winner), 
                           test_text = pred[0],
                           input_length = homeTeam)
    
#-------------------------------------------------------------------------

#    features = [np.array(int_features)]  #Convert to the form [[a, b]] for input to the model
#    prediction = model.predict(features)  # features Must be in the form [[a, b]]

#When the Python interpreter reads a source file, it first defines a few special variables. 
#For now, we care about the __name__ variable.
#If we execute our code in the main program, like in our case here, it assigns
# __main__ as the name (__name__). 
#So if we want to run our code right here, we can check if __name__ == __main__
#if so, execute it here. 
#If we import this file (module) to another file then __name__ == app 
#(which is the name of this python file).
#-------------------------------------------------------------------------

if __name__ == "__main__":
    app.run()