import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import gender_guesser.detector as gender
import datetime
import random
from faker import Faker

# Global variables
scaler = None
svm_model = None
nn_model = None
metrics = {}  # Metrics including confusion matrix
fake = Faker()

def read_datasets():
    genuine_users = pd.read_csv("dataset/users.csv")
    fake_users = pd.read_csv("dataset/fusers.csv")
    x = pd.concat([genuine_users, fake_users])
    y = len(fake_users) * [0] + len(genuine_users) * [1]
    return x, y

def extract_features(x):
    x['created_at'] = pd.to_datetime(x['created_at'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce').dt.tz_localize(None)
    current_time = datetime.datetime.now()
    x['created_at_age'] = (current_time - x['created_at']).dt.days.fillna(0)
    x['has_url'] = x['url'].notna().astype(int)
    x['has_description'] = x['description'].notna().astype(int)
    
    d = gender.Detector()
    first_names = x['name'].str.split(' ').str.get(0)
    sex = first_names.apply(d.get_gender)
    sex_dict = {'female': -2, 'mostly_female': -1, 'unknown': 0, 'mostly_male': 1, 'male': 2}
    x['gender_code'] = sex.map(sex_dict).fillna(0).astype(int)
    
    x['followers_to_statuses_ratio'] = x['followers_count'] / (x['statuses_count'] + 1)
    x['followers_to_friends_ratio'] = x['followers_count'] / (x['friends_count'] + 1)
    
    features = [
        'statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count',
        'created_at_age', 'has_url', 'has_description', 'followers_to_statuses_ratio',
        'followers_to_friends_ratio', 'gender_code'
    ]
    return x[features].fillna(0)

def train_model():
    global scaler, svm_model, nn_model, metrics
    x, y = read_datasets()
    x = extract_features(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    
    scaler = preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svm_model = SVC(kernel='rbf', C=1, gamma=0.1, probability=True)
    svm_model.fit(X_train_scaled, y_train)
    decision_values_train = svm_model.decision_function(X_train_scaled)
    decision_values_test = svm_model.decision_function(X_test_scaled)
    
    X_train_svm_output = np.column_stack((X_train_scaled, decision_values_train))
    X_test_svm_output = np.column_stack((X_test_scaled, decision_values_test))
    
    nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, alpha=0.01)
    nn_model.fit(X_train_svm_output, y_train)
    
    # Calculate metrics
    y_pred = nn_model.predict(X_test_svm_output)
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()  # Convert to list for JSON compatibility
    print("Metrics:", metrics)

def predict_profile(features):
    features_df = pd.DataFrame([features], columns=[
        'name', 'statuses_count', 'followers_count', 'friends_count', 'favourites_count',
        'listed_count', 'created_at', 'url', 'description'
    ])
    features_extracted = extract_features(features_df)
    features_scaled = scaler.transform(features_extracted)
    svm_decision = svm_model.decision_function(features_scaled)
    features_svm_output = np.column_stack((features_scaled, svm_decision))
    
    probas = nn_model.predict_proba(features_svm_output)[0]
    fake_prob, genuine_prob = probas[0], probas[1]

    result = "Genuine" if genuine_prob > fake_prob else "Fake"
    confidence = max(fake_prob, genuine_prob)
    print(f"Prediction: {result}, Fake Prob: {fake_prob}, Genuine Prob: {genuine_prob}")
    return result, confidence

def simulate_x_profile(username=None):
    is_fake = random.choice([True, False])
    name = username if username else fake.name()
    created_at = fake.date_time_between(start_date="-5y", end_date="now").strftime("%a %b %d %H:%M:%S +0000 %Y")
    
    if is_fake:
        statuses_count = random.randint(10, 100)
        followers_count = random.randint(5, 50)
        friends_count = random.randint(100, 500)
        favourites_count = random.randint(0, 20)
        listed_count = random.randint(0, 2)
        url = None
        description = random.choice([None, "Hi I'm new here!", "Follow me!"])
    else:
        statuses_count = random.randint(50, 1000)
        followers_count = random.randint(100, 2000)
        friends_count = random.randint(50, 1000)
        favourites_count = random.randint(20, 500)
        listed_count = random.randint(5, 50)
        url = fake.url()
        description = fake.sentence()
    
    return {
        'name': name, 'statuses_count': statuses_count, 'followers_count': followers_count,
        'friends_count': friends_count, 'favourites_count': favourites_count, 'listed_count': listed_count,
        'created_at': created_at, 'url': url, 'description': description
    }

if __name__ == "__main__":
    train_model()