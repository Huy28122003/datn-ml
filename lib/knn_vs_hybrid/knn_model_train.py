"""
knn_model_train.py
====================
Huấn luyện mô hình KNN (K-Nearest Neighbors) để tìm bản ghi tương đồng nhất trên tập dữ liệu Hybrid.
  - Đọc dữ liệu từ dataset hybrid.csv.
  - Loại bỏ các cột định danh phi số (FILENAME, URL, Domain, TLD, Title) và cột label.
  - Chuẩn hóa đặc trưng bằng StandardScaler để đưa về cùng một phân phối tỷ lệ.
  - Huấn luyện mô hình NearestNeighbors (mặc định tìm K=5 láng giềng gần nhất bằng khoảng cách Euclidean).
  - Lưu trữ Scaler, mô hình KNN và danh sách tên đặc trưng vào output/knn_vs_hybrid/knn_model.pkl.
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Xác định đường dẫn gốc (workspace root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, 'lib'))

# Cấu hình đường dẫn
DATASET_PATH = os.path.join(BASE_DIR, 'data_set', 'hybrid.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'knn_vs_hybrid')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'knn_model.pkl')
TRAINING_SUMMARY_PATH = os.path.join(OUTPUT_DIR, 'knn_model_training_summary.json')


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        HUẤN LUYỆN MÔ HÌNH KNN TÌM BẢN GHI TƯƠNG ĐỒNG     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if not os.path.exists(DATASET_PATH):
        print(f"❌ LỖI: Không tìm thấy tập dữ liệu tại {DATASET_PATH}")
        sys.exit(1)

    print("\n📖 Đang đọc tập dữ liệu Hybrid...")
    df = pd.read_csv(DATASET_PATH)
    print(f"  ✓ Đọc thành công: {df.shape[0]:,} dòng, {df.shape[1]} cột.")

    # 1. Tiền xử lý: Tách cột đặc trưng và cột định danh
    print("\n🧹 Đang phân tách đặc trưng và cột định danh...")
    
    # Danh sách các cột text phi số và cột label
    meta_cols = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title', 'label']
    available_meta_cols = [c for c in meta_cols if c in df.columns]
    
    # Các đặc trưng số dùng để tính khoảng cách
    X = df.drop(columns=available_meta_cols, errors='ignore')
    features_list = X.columns.tolist()
    
    print(f"  ✓ Số đặc trưng số dùng để tính khoảng cách: {len(features_list)}")
    print(f"  ✓ Mẫu cột đặc trưng số đầu tiên: {features_list[:5]}...")

    # Xử lý các giá trị NaN/Null nếu có (trong hybrid.csv thường đã được làm sạch)
    if X.isnull().any().any():
        print("  ⚠️ Phát hiện giá trị thiếu (NaN). Đang tự động điền bằng giá trị trung vị...")
        X = X.fillna(X.median())

    # 2. Chuẩn hóa đặc trưng (Scaling)
    print("\n⚖️ Đang thực hiện chuẩn hóa đặc trưng (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("  ✓ Chuẩn hóa hoàn tất!")

    # 3. Khởi tạo và khớp mô hình NearestNeighbors
    print("\n🎯 Đang khớp mô hình láng giềng gần nhất (NearestNeighbors)...")
    # Sử dụng khoảng cách Euclidean và thuật toán tự động chọn (KDTree/BallTree/Brute-force)
    knn_model = NearestNeighbors(n_neighbors=5, metric='euclidean', algorithm='auto', n_jobs=-1)
    knn_model.fit(X_scaled)
    print("  ✓ Khớp mô hình hoàn tất!")

    # 4. Lưu trữ mô hình và bộ chuẩn hóa
    print("\n📦 Đang lưu trữ tệp mô hình KNN...")
    model_data = {
        'knn': knn_model,
        'scaler': scaler,
        'features': features_list,
        'dataset_shape': df.shape
    }

    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"  ✓ Đã lưu file mô hình tại: {MODEL_SAVE_PATH}")

    # 5. Lưu báo cáo tóm tắt dưới dạng JSON
    summary_data = {
        'dataset_rows': df.shape[0],
        'dataset_cols': df.shape[1],
        'features_count': len(features_list),
        'metric': 'euclidean',
        'algorithm': 'auto',
        'model_file_size_bytes': os.path.getsize(MODEL_SAVE_PATH)
    }
    with open(TRAINING_SUMMARY_PATH, 'w') as f:
        json.dump(summary_data, f, indent=4)
    print(f"  ✓ Đã lưu tóm tắt báo cáo huấn luyện tại: {TRAINING_SUMMARY_PATH}")

    print("\n🎉 HUẦN LUYỆN MÔ HÌNH KNN THÀNH CÔNG!")


if __name__ == '__main__':
    main()
