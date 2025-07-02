from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import datetime
from model import train_model, predict_profile, simulate_x_profile, extract_features, metrics

app = Flask(__name__)

history = []
dataset = pd.concat([pd.read_csv("dataset/users.csv"), pd.read_csv("dataset/fusers.csv")])
dataset_features = extract_features(dataset.copy())
genuine_users = pd.read_csv("dataset/users.csv")
fake_users = pd.read_csv("dataset/fusers.csv")

train_model()

def generate_visualizations():
    os.makedirs('static/visualizations', exist_ok=True)
    
    genuine_users['name_length'] = genuine_users['name'].str.len()
    fake_users['name_length'] = fake_users['name'].str.len()
    plt.figure(figsize=(10, 6))
    sns.histplot(genuine_users['name_length'], color='green', label='Genuine', alpha=0.5)
    sns.histplot(fake_users['name_length'], color='red', label='Fake', alpha=0.5)
    plt.title('Name Length Distribution')
    plt.xlabel('Name Length')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('static/visualizations/name_length.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(genuine_users['friends_count'], genuine_users['followers_count'], color='green', label='Genuine', alpha=0.5)
    plt.scatter(fake_users['friends_count'], fake_users['followers_count'], color='red', label='Fake', alpha=0.5)
    plt.title('Friends vs Followers')
    plt.xlabel('Friends Count')
    plt.ylabel('Followers Count')
    plt.legend()
    plt.savefig('static/visualizations/friends_followers.png')
    plt.close()
    
    for data, name in [(genuine_users, 'corr_genuine'), (fake_users, 'corr_fake')]:
        plt.figure(figsize=(10, 8))
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr = data[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'Correlation Matrix - {name.split("_")[1].capitalize()}')
        plt.savefig(f'static/visualizations/{name}.png')
        plt.close()
    
    plt.figure(figsize=(12, 6))
    features = ['statuses_count', 'followers_count', 'friends_count', 'favourites_count']
    genuine_data = genuine_users[features].melt()
    genuine_data['Type'] = 'Genuine'
    fake_data = fake_users[features].melt()
    fake_data['Type'] = 'Fake'
    box_data = pd.concat([genuine_data, fake_data])
    sns.boxplot(x='variable', y='value', hue='Type', data=box_data)
    plt.title('Boxplot of Profile Features')
    plt.xlabel('Feature')
    plt.ylabel('Value')
    plt.savefig('static/visualizations/boxplot.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(genuine_users['followers_count'].dropna(), color='green', label='Genuine', alpha=0.5, bins=20)
    sns.histplot(fake_users['followers_count'].dropna(), color='red', label='Fake', alpha=0.5, bins=20)
    plt.title('Followers Count Distribution')
    plt.xlabel('Followers Count')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('static/visualizations/followers_dist.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(genuine_users['statuses_count'].dropna(), color='green', label='Genuine', alpha=0.5, bins=20)
    sns.histplot(fake_users['statuses_count'].dropna(), color='red', label='Fake', alpha=0.5, bins=20)
    plt.title('Statuses Count Distribution')
    plt.xlabel('Statuses Count')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('static/visualizations/statuses_dist.png')
    plt.close()

generate_visualizations()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'simulate' in request.form and request.form['simulate'] == 'true':
        features = simulate_x_profile()
    elif 'x_username' in request.form and request.form['x_username']:
        features = simulate_x_profile(request.form['x_username'])
    else:
        features = {
            'name': request.form['name'],
            'statuses_count': float(request.form['statuses_count']),
            'followers_count': float(request.form['followers_count']),
            'friends_count': float(request.form['friends_count']),
            'favourites_count': float(request.form['favourites_count']),
            'listed_count': float(request.form['listed_count']),
            'created_at': request.form['created_at'],
            'url': request.form['url'] if request.form['url'] else None,
            'description': request.form['description'] if request.form['description'] else None
        }
    
    result, confidence = predict_profile(features)
    entry = {
        "features": features,
        "result": result,
        "confidence": confidence,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    history.append(entry)
    
    return jsonify({"prediction": result, "confidence": confidence, "entry": entry})

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/history')
def history_page():
    return render_template('history.html', history=history)

@app.route('/dataset')
def dataset_page():
    original_sample = dataset.head(5).to_dict(orient='records')
    columns = dataset.columns.tolist()
    dtypes = dataset.dtypes.to_dict()
    stats = dataset_features.describe().to_dict()
    dataset_size = {
        'total': len(dataset),
        'genuine': len(genuine_users),
        'fake': len(fake_users)
    }
    missing_values = dataset.isnull().sum().to_dict()
    confusion_matrix = metrics.get('confusion_matrix', [[0, 0], [0, 0]])  # Default if not computed
    return render_template('dataset.html', columns=columns, dtypes=dtypes, stats=stats, 
                          sample_data=original_sample, dataset_size=dataset_size, 
                          missing_values=missing_values, confusion_matrix=confusion_matrix)

@app.route('/visualizations')
def visualizations_page():
    return render_template('visualizations.html')

@app.route('/metrics')
def metrics_page():
    return render_template('metrics.html', metrics=metrics)

@app.route('/get_history', methods=['GET'])
def get_history():
    return jsonify(history)

@app.route('/download/users')
def download_users():
    return send_file('dataset/users.csv', as_attachment=True)

@app.route('/download/fusers')
def download_fusers():
    return send_file('dataset/fusers.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)