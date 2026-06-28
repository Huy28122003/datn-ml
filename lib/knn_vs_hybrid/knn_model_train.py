import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, 'lib'))
DATASET_PATH = os.path.join(BASE_DIR, 'data_set', 'hybrid.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'knn_vs_hybrid')
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'knn_model.pkl')
TRAINING_SUMMARY_PATH = os.path.join(OUTPUT_DIR, 'knn_model_training_summary.json')

def main():
    if not os.path.exists(DATASET_PATH):
        sys.exit(1)
    df = pd.read_csv(DATASET_PATH)
    meta_cols = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title', 'label']
    available_meta_cols = [c for c in meta_cols if c in df.columns]
    X = df.drop(columns=available_meta_cols, errors='ignore')
    features_list = X.columns.tolist()
    if X.isnull().any().any():
        X = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    knn_model = NearestNeighbors(n_neighbors=5, metric='euclidean', algorithm='auto', n_jobs=-1)
    knn_model.fit(X_scaled)
    model_data = {'knn': knn_model, 'scaler': scaler, 'features': features_list, 'dataset_shape': df.shape}
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    summary_data = {'dataset_rows': df.shape[0], 'dataset_cols': df.shape[1], 'features_count': len(features_list), 'metric': 'euclidean', 'algorithm': 'auto', 'model_file_size_bytes': os.path.getsize(MODEL_SAVE_PATH)}
    with open(TRAINING_SUMMARY_PATH, 'w') as f:
        json.dump(summary_data, f, indent=4)
if __name__ == '__main__':
    main()
