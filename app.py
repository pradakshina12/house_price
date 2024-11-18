from flask import Flask, request, render_template
import pandas as pd
import pickle

# Load the trained model for housing price prediction
with open('log_reg_crt.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

# Encoding mappings for categorical variables
mainroad_mapping = {'Yes': 1, 'No': 0}
guestroom_mapping = {'Yes': 1, 'No': 0}
basement_mapping = {'Yes': 1, 'No': 0}
airconditioning_mapping = {'Yes': 1, 'No': 0}
furnishingstatus_mapping = {'Furnished': 1, 'Semi-Furnished': 2, 'Unfurnished': 0}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Map input values to match model's expected feature names
    input_data = {
        'area': float(request.form['area']),
        'bedrooms': int(request.form['bedrooms']),
        'bathrooms': float(request.form['bathrooms']),
        'stories': int(request.form['stories']),
        'mainroad': request.form['mainroad'],
        'guestroom': request.form['guestroom'],
        'basement': request.form['basement'],
        'airconditioning': request.form['airconditioning'],
        'parking': int(request.form['parking']),
        'furnishingstatus': request.form['furnishingstatus'],
    }

    # Convert categorical variables using mappings
    input_data['mainroad'] = mainroad_mapping.get(input_data['mainroad'], -1)
    input_data['guestroom'] = guestroom_mapping.get(input_data['guestroom'], -1)
    input_data['basement'] = basement_mapping.get(input_data['basement'], -1)
    input_data['airconditioning'] = airconditioning_mapping.get(input_data['airconditioning'], -1)
    input_data['furnishingstatus'] = furnishingstatus_mapping.get(input_data['furnishingstatus'], -1)

    # Convert input data to DataFrame with only the selected features
    input_df = pd.DataFrame([input_data])

    # Predict the housing price
    prediction = model.predict(input_df)
    prediction_text = f'Predicted House Price: ${prediction[0]:,.2f}'
    
    # Display result on the page
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
