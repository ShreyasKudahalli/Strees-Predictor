from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model once on startup
model_path = 'model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError("Model not found! Run train_and_save.py first.")
model = joblib.load(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        features = {
            'Avg_Working_Hours_Per_Day': float(request.form['working_hours']),
            'Work_From': request.form['work_from'],
            'Work_Pressure': int(request.form['work_pressure']),
            'Manager_Support': int(request.form['manager_support']),
            'Sleeping_Habit': int(request.form['sleeping_habit']),
            'Exercise_Habit': int(request.form['exercise_habit']),
            'Job_Satisfaction': int(request.form['job_satisfaction']),
            'Work_Life_Balance': request.form['work_life_balance'],
            'Social_Person': int(request.form['social_person']),
            'Lives_With_Family': request.form['lives_with_family'],
            'Working_State': request.form['working_state']
        }

        input_df = pd.DataFrame([features])

        # Add engineered features exactly like training
        input_df['Workload_Index'] = input_df['Avg_Working_Hours_Per_Day'] * input_df['Work_Pressure'] / 8
        input_df['Support_Deficit'] = (5 - input_df['Manager_Support']) * input_df['Work_Pressure']
        input_df['Health_Score'] = (input_df['Sleeping_Habit'] + input_df['Exercise_Habit']) / 2

        prediction = model.predict(input_df)[0]

    return render_template('index.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
