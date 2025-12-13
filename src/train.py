import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import os

print("Loading data")
data = pd.read_csv('data/dataset.csv')

X = data.drop('target', axis=1)
y = data['target']

print("Training model")
model = LogisticRegression(max_iter=200)
model.fit(X, y)
print("Training complete.")

os.makedirs('models', exist_ok=True)
output_path = 'models/model.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(model, f)
print(f"Model saved to {output_path}")