import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'data_set', 'dataset_full.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'xgb_vs_dsfull')
os.makedirs(OUTPUT_DIR, exist_ok=True)
SELECTED_FEATURES_PATH = os.path.join(OUTPUT_DIR, 'xgb_selected_features.json')
BEST_HYPERPARAMS_PATH = os.path.join(OUTPUT_DIR, 'xgb_best_hyperparameters.json')

def main():
    if not os.path.exists(SELECTED_FEATURES_PATH):
        sys.exit(1)
    with open(SELECTED_FEATURES_PATH, 'r') as f:
        selected_features = json.load(f)
    if not os.path.exists(DATASET_PATH):
        sys.exit(1)
    df = pd.read_csv(DATASET_PATH)
    X = df[selected_features]
    y = df['phishing']
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
    n_estimators_range = [50, 100, 150, 200, 250]
    train_accuracies = []
    test_accuracies = []
    for n in n_estimators_range:
        xgb = XGBClassifier(n_estimators=n, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss')
        xgb.fit(X_train, y_train)
        train_accuracies.append(xgb.score(X_train, y_train))
        test_accuracies.append(xgb.score(X_test, y_test))
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, train_accuracies, 'o-', label='Độ chính xác Train', color='#1f77b4', linewidth=2)
    plt.plot(n_estimators_range, test_accuracies, 's-', label='Độ chính xác Test', color='#ff7f0e', linewidth=2)
    plt.title('XGBoost - Khảo sát n_estimators vs Độ chính xác', fontsize=14, pad=15)
    plt.xlabel('Số lượng cây (n_estimators)', fontsize=12)
    plt.ylabel('Độ chính xác (Accuracy)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'xgb_n_estimators_vs_accuracy.png'), dpi=150)
    plt.close()
    optimal_n = 150
    param_grid = {'max_depth': [4, 6, 8], 'learning_rate': [0.05, 0.1, 0.2]}
    xgb_base = XGBClassifier(n_estimators=optimal_n, random_state=42, n_jobs=-1, eval_metric='logloss')
    grid_search = GridSearchCV(estimator=xgb_base, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    results = pd.DataFrame(grid_search.cv_results_)
    scores_matrix = results.pivot(index='param_max_depth', columns='param_learning_rate', values='mean_test_score')
    plt.figure(figsize=(8, 6))
    sns.heatmap(scores_matrix, annot=True, fmt='.4f', cmap='YlGnBu', cbar=True)
    plt.title('XGBoost - Độ chính xác Cross-Validation theo tham số', fontsize=12, pad=12)
    plt.xlabel('learning_rate', fontsize=10)
    plt.ylabel('max_depth', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'xgb_hyperparameter_grid_search.png'), dpi=150)
    plt.close()
    hyperparams = {'n_estimators': optimal_n, 'max_depth': int(best_params['max_depth']), 'learning_rate': float(best_params['learning_rate']), 'random_state': 42, 'eval_metric': 'logloss'}
    with open(BEST_HYPERPARAMS_PATH, 'w') as f:
        json.dump(hyperparams, f, indent=4)
if __name__ == '__main__':
    main()
