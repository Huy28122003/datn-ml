import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, 'data_set', 'hybrid.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'rf_vs_hybrid')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SELECTED_FEATURES_PATH = os.path.join(OUTPUT_DIR, 'rf_selected_features.json')
IMPORTANCE_CSV_PATH = os.path.join(OUTPUT_DIR, 'rf_feature_importance_scores.csv')


def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Loi dataset: {DATASET_PATH}")
        sys.exit(1)

    print("Doc dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Rows: {df.shape[0]}, Cols: {df.shape[1]}")

    print("Loc cot du thua...")
    if 'FILENAME' in df.columns:
        df = df.drop(columns=['FILENAME'])
        print("Xoa FILENAME.")
        
    text_cols = ['URL', 'Domain', 'TLD', 'Title']
    available_text_cols = [c for c in text_cols if c in df.columns]
    if available_text_cols:
        df = df.drop(columns=available_text_cols)
        print("Xoa cot text.")

    if 'label' not in df.columns:
        print("Loi: Thieu cot label!")
        sys.exit(1)

    X = df.drop(columns=['label'])
    y = df['label']

    # huyny - loc leakage
    biased_cols = [
        'URLSimilarityIndex', 'DomainTitleMatchScore', 'URLTitleMatchScore',
        'IsHTTPS', 'IsDomainIP', 'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio',
        'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL'
    ]
    existing_biased_cols = [c for c in biased_cols if c in X.columns]
    if existing_biased_cols:
        X = X.drop(columns=existing_biased_cols)
        print(f"Xoa {len(existing_biased_cols)} cot bias.")

    print(f"So dac trung goc: {X.shape[1]}")

    rf_selector = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf_selector.fit(X, y)
    print("Fit selector xong.")

    importances = rf_selector.feature_importances_
    indices = np.argsort(importances)[::-1]

    feat_imp_df = pd.DataFrame({
        'Feature': [X.columns[i] for i in indices],
        'Importance': [importances[i] for i in indices]
    })

    feat_imp_df.to_csv(IMPORTANCE_CSV_PATH, index=False)
    print(f"Luu ranking: {IMPORTANCE_CSV_PATH}")

    threshold = 0.001
    selected_features_df = feat_imp_df[feat_imp_df['Importance'] > threshold]
    selected_features = selected_features_df['Feature'].tolist()

    print(f"Giu lai {len(selected_features)} dac trung (score > {threshold})")
    print(f"Tong score tich luy: {selected_features_df['Importance'].sum():.2%}")

    # huyny - luu json dac trung
    with open(SELECTED_FEATURES_PATH, 'w') as f:
        json.dump(selected_features, f, indent=4)
    print(f"Luu dac trung: {SELECTED_FEATURES_PATH}")

    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feat_imp_df.head(30),
        palette='viridis'
    )
    plt.title('Top 30 Dac trung Hybrid quan trong nhat (Random Forest)', fontsize=14, pad=15)
    plt.xlabel('Do quan trong (Gini Importance)', fontsize=12)
    plt.ylabel('Ten dac trung', fontsize=12)
    plt.tight_layout()
    plot_path_top30 = os.path.join(OUTPUT_DIR, 'rf_feature_importance_top30.png')
    plt.savefig(plot_path_top30, dpi=150)
    plt.close()

    plt.figure(figsize=(12, 12))
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feat_imp_df,
        palette='magma'
    )
    plt.title('Bang xep hang toan bo dac trung Hybrid', fontsize=14, pad=15)
    plt.xlabel('Do quan trong', fontsize=12)
    plt.ylabel('Ten dac trung', fontsize=12)
    plt.tight_layout()
    plot_path_all = os.path.join(OUTPUT_DIR, 'rf_feature_importance_all.png')
    plt.savefig(plot_path_all, dpi=150)
    plt.close()

    print(f"Luu anh: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
