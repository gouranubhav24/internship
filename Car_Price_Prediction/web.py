from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__, static_url_path='/static')

model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form['name']
        company = request.form['company']
        year = int(request.form['year'])
        kms_driven = int(request.form['kms_driven'])
        fuel_type = request.form['fuel_type']

        # Create a DataFrame with the user input
        input_data = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                   data=[[name, company, year, kms_driven, fuel_type]])

        # Predict the car price using the model
        predicted_price = model.predict(input_data)
        predicted_price = np.round(predicted_price, 2)

        return render_template('result.html', predicted_price=predicted_price[0])

    except Exception as e:
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
