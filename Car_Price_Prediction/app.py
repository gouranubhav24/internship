import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form.to_dict()
        input_data = list(input_data.values())
        input_data = list(map(str, input_data))
        price_prediction = model.predict([input_data])
        return render_template('result.html', prediction=price_prediction[0])
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)


