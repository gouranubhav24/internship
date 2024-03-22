from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import sklearn.datasets

app = Flask(__name__)

breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

sc = StandardScaler()

X_train, _ = sklearn.datasets.load_breast_cancer(return_X_y=True)
sc.fit(X_train)
@app.route('/')
def index():
    feature_names = breast_cancer_dataset.feature_names
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = []
        for feature_name in breast_cancer_dataset.feature_names:
            value = float(request.form[feature_name])
            input_data.append(value)

        input_as_numpy_array = np.asarray(input_data)
        input_reshaped = input_as_numpy_array.reshape(1, -1)
        input_std = sc.transform(input_reshaped)

        prediction = model.predict(input_std)
        prediction_label = np.argmax(prediction)

        result = 'Malignant' if prediction_label == 0 else 'Benign'

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
