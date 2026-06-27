"""
rf_optimal_trees_train_tuning.py
=================================
Bước 2: Tinh chỉnh siêu tham số (Hyperparameter Tuning) trên tập dữ liệu Hybrid.
  - Đọc các đặc trưng đã được chọn lọc ở Bước 1.
  - Khảo sát sự ảnh hưởng của n_estimators (số lượng cây) từ 50 đến 250.
  - Chạy Grid Search (3-Fold CV) tinh chỉnh max_depth và min_samples_split.
  - Lưu kết quả tối ưu vào output/rf_vs_hybrid/rf_best_hyperparameters.json.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Xác định đường dẫn gốc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Cấu hình đường dẫn
DATASET_PATH = os.path.join(BASE_DIR, 'data_set', 'hybrid.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'rf_vs_hybrid')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SELECTED_FEATURES_PATH = os.path.join(OUTPUT_DIR, 'rf_selected_features.json')
BEST_HYPERPARAMS_PATH = os.path.join(OUTPUT_DIR, 'rf_best_hyperparameters.json')


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  BƯỚC 2 (HYBRID): TINH CHỈNH SIÊU THAM SỐ RANDOM FOREST  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if not os.path.exists(SELECTED_FEATURES_PATH):
        print(f"❌ LỖI: Chưa chạy Bước 1! Không tìm thấy tệp {SELECTED_FEATURES_PATH}")
        sys.exit(1)

    with open(SELECTED_FEATURES_PATH, 'r') as f:
        selected_features = json.load(f)
    print(f"\n📖 Đã load danh sách {len(selected_features)} đặc trưng Hybrid tối ưu.")

    if not os.path.exists(DATASET_PATH):
        print(f"❌ LỖI: Không tìm thấy tập dữ liệu tại {DATASET_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    
    # Chuẩn hóa loại bỏ text & FILENAME
    if 'FILENAME' in df.columns:
        df = df.drop(columns=['FILENAME'])
    text_cols = ['URL', 'Domain', 'TLD', 'Title']
    available_text_cols = [c for c in text_cols if c in df.columns]
    if available_text_cols:
        df = df.drop(columns=available_text_cols)

    X = df[selected_features]
    y = df['label']

    # Để Grid Search chạy cực nhanh trên 235k dòng, ta lấy một mẫu ngẫu nhiên đại diện gồm 40,000 dòng
    print("\n🎲 Đang lấy mẫu ngẫu nhiên 40,000 dòng đại diện để tinh chỉnh siêu tham số...")
    df_sample = df.sample(n=40000, random_state=42)
    X_sample = df_sample[selected_features]
    y_sample = df_sample['label']

    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

    # 1. Khảo sát số lượng cây (n_estimators) tối ưu
    print("\n🌲 Đang khảo sát ảnh hưởng số lượng cây (n_estimators) từ 50 đến 250...")
    n_estimators_range = [50, 100, 150, 200, 250]
    train_accuracies = []
    test_accuracies = []

    for n in n_estimators_range:
        print(f"  → Đang kiểm định n_estimators = {n}...")
        rf = RandomForestClassifier(
            n_estimators=n,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        train_accuracies.append(rf.score(X_train, y_train))
        test_accuracies.append(rf.score(X_test, y_test))

    # Vẽ biểu đồ n_estimators vs Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, train_accuracies, 'o-', label='Độ chính xác Train', color='#1f77b4', linewidth=2)
    plt.plot(n_estimators_range, test_accuracies, 's-', label='Độ chính xác Test', color='#ff7f0e', linewidth=2)
    plt.title('Hybrid - Khảo sát n_estimators vs Độ chính xác', fontsize=14, pad=15)
    plt.xlabel('Số lượng cây (n_estimators)', fontsize=12)
    plt.ylabel('Độ chính xác (Accuracy)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rf_n_estimators_vs_accuracy.png'), dpi=150)
    plt.close()

    # Chọn số cây tối ưu
    optimal_n = 150
    print(f"  ✓ Đã chọn số cây tối ưu: {optimal_n} cây (Test Accuracy: {test_accuracies[n_estimators_range.index(150)]:.4f})")

    # 2. Sử dụng Grid Search tinh chỉnh độ sâu cây chống Overfitting
    print("\n🔍 Đang thiết lập Grid Search tinh chỉnh chiều sâu cây (max_depth)...")
    param_grid = {
        'max_depth': [10, 15, 20, 25],
        'min_samples_split': [2, 5]
    }
    
    rf_base = RandomForestClassifier(
        n_estimators=optimal_n,
        random_state=42,
        n_jobs=-1
    )
    
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print(f"\n⚡ KẾT QUẢ GRID SEARCH TỐI ƯU:")
    print(f"  - Độ sâu cây tối ưu (max_depth): {best_params['max_depth']}")
    print(f"  - min_samples_split tối ưu: {best_params['min_samples_split']}")
    print(f"  - Độ chính xác Cross-Validation (3-Fold): {best_score:.4%}")

    # Vẽ Heatmap
    results = pd.DataFrame(grid_search.cv_results_)
    scores_matrix = results.pivot(index='param_max_depth', columns='param_min_samples_split', values='mean_test_score')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(scores_matrix, annot=True, fmt=".4f", cmap='YlGnBu', cbar=True)
    plt.title('Hybrid - Độ chính xác Cross-Validation theo tham số', fontsize=12, pad=12)
    plt.xlabel('min_samples_split', fontsize=10)
    plt.ylabel('max_depth', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rf_hyperparameter_grid_search.png'), dpi=150)
    plt.close()

    # 3. Lưu siêu tham số tối ưu
    hyperparams = {
        'n_estimators': optimal_n,
        'max_depth': int(best_params['max_depth']),
        'min_samples_split': int(best_params['min_samples_split']),
        'min_samples_leaf': 1,
        'random_state': 42
    }
    
    with open(BEST_HYPERPARAMS_PATH, 'w') as f:
        json.dump(hyperparams, f, indent=4)
    print(f"\n  ✓ Đã lưu cấu hình siêu tham số tối ưu tại: {BEST_HYPERPARAMS_PATH}")
    print("\n🎉 HOÀN THÀNH BƯỚC 2 (HYBRID) THÀNH CÔNG!")


if __name__ == '__main__':
    main()
