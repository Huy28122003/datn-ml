import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xgb_url_extract_test_features import extract_features_from_url
DATASET_PATH = os.path.join(BASE_DIR, 'data_set', 'dataset_full.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'xgb_vs_dsfull')
os.makedirs(OUTPUT_DIR, exist_ok=True)
SELECTED_FEATURES_PATH = os.path.join(OUTPUT_DIR, 'xgb_selected_features.json')
IMPORTANCE_CSV_PATH = os.path.join(OUTPUT_DIR, 'xgb_feature_importance_scores.csv')

def main():
    if not os.path.exists(DATASET_PATH):
        sys.exit(1)
    df = pd.read_csv(DATASET_PATH)
    sample_url = 'https://example.com/test'
    all_pure_features = list(extract_features_from_url(sample_url).keys())
    available_features = [feat for feat in all_pure_features if feat in df.columns]
    X = df[available_features]
    y = df['phishing']
    xgb_selector = XGBClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1, eval_metric='logloss')
    xgb_selector.fit(X, y)
    importances = xgb_selector.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_imp_df = pd.DataFrame({'Feature': [available_features[i] for i in indices], 'Importance': [importances[i] for i in indices]})
    feat_imp_df.to_csv(IMPORTANCE_CSV_PATH, index=False)
    threshold = 0.001
    selected_features_df = feat_imp_df[feat_imp_df['Importance'] > threshold]
    selected_features = selected_features_df['Feature'].tolist()
    with open(SELECTED_FEATURES_PATH, 'w') as f:
        json.dump(selected_features, f, indent=4)
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(30), palette='viridis')
    plt.title('Top 30 Đặc trưng Pure URL quan trọng nhất (XGBoost)', fontsize=14, pad=15)
    plt.xlabel('Độ quan trọng (Weight Importance)', fontsize=12)
    plt.ylabel('Tên đặc trưng', fontsize=12)
    plt.tight_layout()
    plot_path_top30 = os.path.join(OUTPUT_DIR, 'xgb_feature_importance_top30.png')
    plt.savefig(plot_path_top30, dpi=150)
    plt.close()
    plt.figure(figsize=(12, 16))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='magma')
    plt.title('Bảng xếp hạng toàn bộ đặc trưng Pure URL (XGBoost)', fontsize=14, pad=15)
    plt.xlabel('Độ quan trọng', fontsize=12)
    plt.ylabel('Tên đặc trưng', fontsize=12)
    plt.tight_layout()
    plot_path_all = os.path.join(OUTPUT_DIR, 'xgb_feature_importance_all.png')
    plt.savefig(plot_path_all, dpi=150)
    plt.close()
if __name__ == '__main__':
    main()
