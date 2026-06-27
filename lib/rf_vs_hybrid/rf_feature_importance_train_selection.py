"""
rf_feature_importance_train_selection.py
=========================================
Bước 1: Chọn lọc các đặc trưng quan trọng nhất trên tập dữ liệu Hybrid.
  - Đọc dữ liệu từ dataset hybrid.csv (235,794 hàng, 56 cột).
  - Loại bỏ hoàn toàn cột FILENAME (như yêu cầu) và các cột phi số khác.
  - Huấn luyện Random Forest nhanh để tính toán Feature Importance.
  - Lọc ra các đặc trưng quan trọng nhất (> 0.001) để lưu trữ.
  - Lưu kết quả vào output/rf_vs_hybrid/rf_selected_features.json.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Xác định đường dẫn gốc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Cấu hình đường dẫn
DATASET_PATH = os.path.join(BASE_DIR, 'data_set', 'hybrid.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'rf_vs_hybrid')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SELECTED_FEATURES_PATH = os.path.join(OUTPUT_DIR, 'rf_selected_features.json')
IMPORTANCE_CSV_PATH = os.path.join(OUTPUT_DIR, 'rf_feature_importance_scores.csv')


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  BƯỚC 1 (HYBRID): LỌC & CHỌN LỌC ĐẶC TRƯNG HYBRID        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if not os.path.exists(DATASET_PATH):
        print(f"❌ LỖI: Không tìm thấy tập dữ liệu tại {DATASET_PATH}")
        sys.exit(1)

    print("\n📖 Đang đọc tập dữ liệu Hybrid...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  ✓ Đọc thành công: {df.shape[0]:,} dòng, {df.shape[1]} cột.")

    # 1. Loại bỏ cột đầu tiên FILENAME trước khi chọn cột và xử lý
    print("\n🧹 Đang loại bỏ cột 'FILENAME' (theo yêu cầu) và các cột text...")
    
    # Loại bỏ FILENAME
    if 'FILENAME' in df.columns:
        df = df.drop(columns=['FILENAME'])
        print("  ✓ Đã loại bỏ cột 'FILENAME' thành công.")
        
    # Lấy các cột kiểu số làm đặc trưng đầu vào (loại bỏ URL, Domain, TLD, Title nếu còn)
    text_cols = ['URL', 'Domain', 'TLD', 'Title']
    available_text_cols = [c for c in text_cols if c in df.columns]
    if available_text_cols:
        df = df.drop(columns=available_text_cols)
        print(f"  ✓ Đã loại bỏ các cột text phi số: {available_text_cols}")

    # Tách X và y
    if 'label' not in df.columns:
        print("❌ LỖI: Không tìm thấy cột nhãn 'label' trong tập dữ liệu!")
        sys.exit(1)

    X = df.drop(columns=['label'])
    y = df['label']

    # 1b. Loại bỏ các đặc trưng rò rỉ dữ liệu (Data Leakage) và dễ bị bypass
    biased_cols = [
        'URLSimilarityIndex', 'DomainTitleMatchScore', 'URLTitleMatchScore',
        'IsHTTPS', 'IsDomainIP', 'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio',
        'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL'
    ]
    existing_biased_cols = [c for c in biased_cols if c in X.columns]
    if existing_biased_cols:
        X = X.drop(columns=existing_biased_cols)
        print(f"  🛡️  Đã loại bỏ {len(existing_biased_cols)} đặc trưng gây rò rỉ dữ liệu và bias: {existing_biased_cols}")

    print(f"  ✓ Tổng số cột đặc trưng số đầu vào robust: {X.shape[1]}")

    # 2. Huấn luyện Random Forest nhanh để tính Feature Importance
    print("\n🌲 Đang huấn luyện Random Forest tính toán độ quan trọng đặc trưng...")
    rf_selector = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf_selector.fit(X, y)
    print("  ✓ Huấn luyện bộ lọc hoàn tất!")

    # 3. Tính điểm Feature Importance
    importances = rf_selector.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Tạo dataframe kết quả
    feat_imp_df = pd.DataFrame({
        'Feature': [X.columns[i] for i in indices],
        'Importance': [importances[i] for i in indices]
    })

    # Lưu danh sách điểm số ra file CSV
    feat_imp_df.to_csv(IMPORTANCE_CSV_PATH, index=False)
    print(f"  ✓ Đã lưu bảng xếp hạng đặc trưng tại: {IMPORTANCE_CSV_PATH}")

    # Lọc các đặc trưng có độ quan trọng > 0.001
    threshold = 0.001
    selected_features_df = feat_imp_df[feat_imp_df['Importance'] > threshold]
    selected_features = selected_features_df['Feature'].tolist()

    print(f"\n⚡ KẾT QUẢ CHỌN LỌC ĐẶC TRƯNG HYBRID:")
    print(f"  - Số đặc trưng ban đầu: {X.shape[1]}")
    print(f"  - Số đặc trưng giữ lại (Importance > {threshold}): {len(selected_features)}")
    print(f"  - Tổng điểm quan trọng tích lũy của đặc trưng giữ lại: {selected_features_df['Importance'].sum():.2%}")

    # Lưu cấu hình các đặc trưng đã chọn dưới dạng JSON
    with open(SELECTED_FEATURES_PATH, 'w') as f:
        json.dump(selected_features, f, indent=4)
    print(f"  ✓ Đã lưu cấu hình danh sách đặc trưng chọn lọc tại: {SELECTED_FEATURES_PATH}")

    # 4. Vẽ đồ thị Top 30 đặc trưng quan trọng nhất
    print("\n📊 Đang vẽ biểu đồ trực quan hóa đặc trưng quan trọng...")
    sns.set_theme(style="whitegrid")
    
    # Đồ thị Top 30
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feat_imp_df.head(30),
        palette='viridis'
    )
    plt.title('Top 30 Đặc trưng Hybrid quan trọng nhất (Random Forest)', fontsize=14, pad=15)
    plt.xlabel('Độ quan trọng (Gini Importance)', fontsize=12)
    plt.ylabel('Tên đặc trưng', fontsize=12)
    plt.tight_layout()
    plot_path_top30 = os.path.join(OUTPUT_DIR, 'rf_feature_importance_top30.png')
    plt.savefig(plot_path_top30, dpi=150)
    plt.close()

    # Đồ thị Tất cả đặc trưng
    plt.figure(figsize=(12, 12))
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feat_imp_df,
        palette='magma'
    )
    plt.title('Bảng xếp hạng toàn bộ đặc trưng Hybrid', fontsize=14, pad=15)
    plt.xlabel('Độ quan trọng', fontsize=12)
    plt.ylabel('Tên đặc trưng', fontsize=12)
    plt.tight_layout()
    plot_path_all = os.path.join(OUTPUT_DIR, 'rf_feature_importance_all.png')
    plt.savefig(plot_path_all, dpi=150)
    plt.close()

    print(f"  ✓ Đã xuất các đồ thị phân tích tại thư mục: {OUTPUT_DIR}")
    print("\n🎉 HOÀN THÀNH BƯỚC 1 (HYBRID) THÀNH CÔNG!")


if __name__ == '__main__':
    main()
