
import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np
from datetime import datetime

# Database connection parameters
DB_PARAMS = {
    'database': 'employee_db',
    'user': 'root',
    'password': '8445728696',
    'host': 'localhost',
    'port': '3306'
}

def connect_to_db():
    """Connect to MySQL database"""
    try:
        conn = mysql.connector.connect(**DB_PARAMS)
        return conn
    except mysql.connector.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def fetch_data():
    """Fetch employee performance data from MySQL"""
    conn = connect_to_db()
    if not conn:
        return None
    
    query = """
    SELECT 
        employee_id,
        tasks_completed,
        quality_score,
        deadline_met,
        hours_worked,
        task_type,
        selected_for_project
    FROM employee_performance
    """
    
    try:
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except mysql.connector.Error as e:
        print(f"Error fetching data: {e}")
        conn.close()
        return None

def preprocess_data(df):
    """Preprocess the data for model training"""
    # Handle missing values
    df = df.dropna()
    
    # Convert deadline_met to binary (1 for met, 0 for not met)
    df['deadline_met'] = df['deadline_met'].map({'Yes': 1, 'No': 0, True: 1, False: 0})
    
    # Encode task_type
    le = LabelEncoder()
    df['task_type_encoded'] = le.fit_transform(df['task_type'])
    
    # Features and target
    features = ['tasks_completed', 'quality_score', 'deadline_met', 
                'hours_worked', 'task_type_encoded']
    X = df[features]
    y = df['selected_for_project']
    
    # Scale numerical features
    scaler = StandardScaler()
    X[['tasks_completed', 'quality_score', 'hours_worked']] = scaler.fit_transform(
        X[['tasks_completed', 'quality_score', 'hours_worked']]
    )
    
    return X, y, le, scaler

def train_model(X, y):
    """Train Random Forest model"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print("Model Performance:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model

def save_model(model, le, scaler, filename='employee_selection_model.pkl'):
    """Save the model and preprocessing objects"""
    joblib.dump({
        'model': model,
        'label_encoder': le,
        'scaler': scaler
    }, filename)
    print(f"Model saved to {filename}")

def predict_employee_selection(model_data, employee_data):
    """Predict if an employee should be selected"""
    model = model_data['model']
    le = model_data['label_encoder']
    scaler = model_data['scaler']
    
    # Prepare input data
    df = pd.DataFrame([employee_data])
    df['deadline_met'] = df['deadline_met'].map({'Yes': 1, 'No': 0, True: 1, False: 0})
    df['task_type_encoded'] = le.transform(df['task_type'])
    
    # Scale numerical features
    df[['tasks_completed', 'quality_score', 'hours_worked']] = scaler.transform(
        df[['tasks_completed', 'quality_score', 'hours_worked']]
    )
    
    features = ['tasks_completed', 'quality_score', 'deadline_met', 
                'hours_worked', 'task_type_encoded']
    X = df[features]
    
    # Predict
    prediction = model.predict(X)
    probability = model.predict_proba(X)[0][1]
    
    return {
        'should_select': bool(prediction[0]),
        'confidence': float(probability)
    }

def main():
    # Fetch and preprocess data
    df = fetch_data()
    if df is None:
        return
    
    X, y, le, scaler = preprocess_data(df)
    
    # Train model
    model = train_model(X, y)
    
    # Save model
    save_model(model, le, scaler)
    
    # Example prediction
    new_employee = {
        'tasks_completed':4,
        'quality_score': 45.5,
        'deadline_met': 'Yes',
        'hours_worked': 10,
        'task_type': 'development'
    }
    
    model_data = joblib.load('employee_selection_model.pkl')
    prediction = predict_employee_selection(model_data, new_employee)
    
    print("\nPrediction for new employee:")
    print(f"Should select: {prediction['should_select']}")
    print(f"Confidence: {prediction['confidence']:.2%}")

if __name__ == "__main__":
    main()