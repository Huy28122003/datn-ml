"""
xgb_feature_importance_train_selection.py
=========================================
Bước 1: Chọn lọc các đặc trưng quan trọng nhất cho XGBoost.
  - Trích xuất 98 đặc trưng Pure URL (thuần Lexical).
  - Huấn luyện mô hình XGBoost nhanh để tính toán Feature Importance.
  - Lọc ra các đặc trưng đóng góp tích lũy lớn (> 0.001) để huấn luyện mô hình chính thức.
  - Lưu kết quả vào output/xgb_vs_dsfull/xgb_selected_features.json.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# Xác định đường dẫn gốc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xgb_url_extract_test_features import extract_features_from_url

# Cấu hình đường dẫn
DATASET_PATH = os.path.join(BASE_DIR, 'data_set', 'dataset_full.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'xgb_vs_dsfull')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SELECTED_FEATURES_PATH = os.path.join(OUTPUT_DIR, 'xgb_selected_features.json')
IMPORTANCE_CSV_PATH = os.path.join(OUTPUT_DIR, 'xgb_feature_importance_scores.csv')


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  BƯỚC 1 (XGB): XẾP HẠNG & CHỌN LỌC ĐẶC TRƯNG PURE URL    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if not os.path.exists(DATASET_PATH):
        print(f"❌ LỖI: Không tìm thấy tập dữ liệu tại {DATASET_PATH}")
        sys.exit(1)

    print("\n📖 Đang đọc tập dữ liệu gốc...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  ✓ Đọc thành công: {df.shape[0]:,} dòng, {df.shape[1]} cột.")

    # 1. Trích xuất danh sách đặc trưng Pure URL (Pure URL)
    print("\n🔍 Đang lọc danh sách đặc trưng Pure URL (Lexical)...")
    sample_url = "https://example.com/test"
    all_pure_features = list(extract_features_from_url(sample_url).keys())
    
    # Chỉ giữ lại các cột thực sự có mặt trong dataset gốc
    available_features = [feat for feat in all_pure_features if feat in df.columns]
    print(f"  ✓ Tìm thấy {len(available_features)} đặc trưng Pure URL có trong dataset gốc.")

    X = df[available_features]
    y = df['phishing']

    # 2. Huấn luyện XGBoost nhanh để tính Feature Importance
    print("\n⚡ Đang huấn luyện XGBoost tính toán độ quan trọng của đặc trưng...")
    xgb_selector = XGBClassifier(
        n_estimators=50,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    xgb_selector.fit(X, y)
    print("  ✓ Huấn luyện hoàn tất!")

    # 3. Tính điểm Feature Importance
    importances = xgb_selector.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Tạo dataframe kết quả
    feat_imp_df = pd.DataFrame({
        'Feature': [available_features[i] for i in indices],
        'Importance': [importances[i] for i in indices]
    })

    # Lưu danh sách điểm số ra file CSV
    feat_imp_df.to_csv(IMPORTANCE_CSV_PATH, index=False)
    print(f"  ✓ Đã lưu bảng xếp hạng đặc trưng tại: {IMPORTANCE_CSV_PATH}")

    # Lọc các đặc trưng có độ quan trọng > 0.001
    threshold = 0.001
    selected_features_df = feat_imp_df[feat_imp_df['Importance'] > threshold]
    selected_features = selected_features_df['Feature'].tolist()

    print(f"\n⚡ KẾT QUẢ CHỌN LỌC ĐẶC TRƯNG:")
    print(f"  - Số đặc trưng Pure URL ban đầu: {len(available_features)}")
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
    plt.title('Top 30 Đặc trưng Pure URL quan trọng nhất (XGBoost)', fontsize=14, pad=15)
    plt.xlabel('Độ quan trọng (Weight Importance)', fontsize=12)
    plt.ylabel('Tên đặc trưng', fontsize=12)
    plt.tight_layout()
    plot_path_top30 = os.path.join(OUTPUT_DIR, 'xgb_feature_importance_top30.png')
    plt.savefig(plot_path_top30, dpi=150)
    plt.close()

    # Đồ thị Tất cả đặc trưng Pure URL
    plt.figure(figsize=(12, 16))
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feat_imp_df,
        palette='magma'
    )
    plt.title('Bảng xếp hạng toàn bộ đặc trưng Pure URL (XGBoost)', fontsize=14, pad=15)
    plt.xlabel('Độ quan trọng', fontsize=12)
    plt.ylabel('Tên đặc trưng', fontsize=12)
    plt.tight_layout()
    plot_path_all = os.path.join(OUTPUT_DIR, 'xgb_feature_importance_all.png')
    plt.savefig(plot_path_all, dpi=150)
    plt.close()

    print(f"  ✓ Đã xuất các đồ thị phân tích tại thư mục: {OUTPUT_DIR}")
    print("\n🎉 HOÀN THÀNH BƯỚC 1 (XGB) THÀNH CÔNG!")


if __name__ == '__main__':
    main()
