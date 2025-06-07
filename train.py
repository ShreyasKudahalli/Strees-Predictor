import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save():
    df = pd.read_csv('train.csv')

    # Feature engineering example (optional)
    df['Workload_Index'] = df['Avg_Working_Hours_Per_Day'] * df['Work_Pressure'] / 8
    df['Support_Deficit'] = (5 - df['Manager_Support']) * df['Work_Pressure']
    df['Health_Score'] = (df['Sleeping_Habit'] + df['Exercise_Habit']) / 2

    X = df.drop(['Employee_Id', 'Stress_Level'], axis=1)
    y = df['Stress_Level'].astype(int)

    cat_cols = ['Work_From', 'Work_Life_Balance', 'Social_Person', 'Lives_With_Family', 'Working_State']
    num_cols = [col for col in X.columns if col not in cat_cols]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, 'model.pkl')
    print("Model trained and saved to model.pkl")

if __name__ == "__main__":
    train_and_save()
