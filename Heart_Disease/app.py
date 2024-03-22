from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the pre-trained model and scaler
lr = LogisticRegression(solver='liblinear')
sc = StandardScaler()

# Load the pre-trained model and scaler
df = pd.read_csv('D:\InternPe\Heart_disease\heart.csv')
categorical_val = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
dataset = pd.get_dummies(df, columns=categorical_val)
col = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[col] = sc.fit_transform(dataset[col])

X = dataset.drop('target', axis=1)
y = dataset.target
lr.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = {}
        for feature in ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                        'slope', 'ca', 'thal']:
            features[feature] = float(request.form[feature]) if feature in request.form else 0.0

        user_data = pd.DataFrame([features])
        user_data_scaled = sc.transform(user_data)
        prediction = lr.predict(user_data_scaled)

        if prediction[0] == 1:
            result = "The model predicts that the person has heart disease."
        else:
            result = "The model predicts that the person does not have heart disease."

        return redirect(url_for('result', prediction_result=result))

@app.route('/result')
def result():
    prediction_result = request.args.get('prediction_result', '')
    return render_template('result.html', prediction_result=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)
