from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, template_folder='template')

pipe = pickle.load(open('pipe.pkl', 'rb'))

teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

@app.route('/')
def home():
    return render_template('index.html', teams=teams)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        batting_team = request.form['batting_team']
        bowling_team = request.form['bowling_team']
        host_city = request.form['host_city']
        target = int(request.form['target'])
        current_score = int(request.form['current_score'])
        overs_completed = int(request.form['overs_completed'])
        wickets_lost = int(request.form['wickets_lost'])

        input_data = {
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [host_city],
            'runs_left': [target - current_score],
            'balls_left': (20 - overs_completed) * 6,
            'wickets': 10 - wickets_lost,
            'total_runs_x': [target],
            'crr': current_score * 6 / ((20 - overs_completed) * 6),
            'rrr': (target - current_score) * 6 / ((20 - overs_completed) * 6)
        }

        input_df = pd.DataFrame(input_data)
        input_transformed = pipe.named_steps['step1'].transform(input_df)


        label_encoder = LabelEncoder()
        for column in ['batting_team', 'bowling_team', 'city']:
            input_df[column] = label_encoder.fit_transform(input_df[column])

        result = pipe.named_steps['step2'].predict_proba(input_transformed)
        lose_percentage = round(result[0][0] * 100, 1)
        win_percentage = round(result[0][1] * 100, 1)

        return render_template('result.html',
                               win_percentage=win_percentage,
                               lose_percentage=lose_percentage,
                               batting_team=batting_team,
                               bowling_team=bowling_team)

if __name__ == '__main__':
    app.run(debug=True)
