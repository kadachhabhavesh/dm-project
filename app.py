from flask import Flask, request, jsonify
#from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

# Load your trained model
model = joblib.load('ipl.pkl')
#CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get JSON data sent in the POST request

    
    df = pd.DataFrame([data])  
    
    demo2 = {}
    for column in ['batting_team', 'bowling_team', 'city']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        demo2[column] = le  # Save the encoder for future use

    prediction = model.predict(df)

    return jsonify({'prediction': prediction.tolist()}) 

#if __name__ == '__main__':
    app.run()
