"""
train_knn.py
============
Huấn luyện mô hình KNN (K-Nearest Neighbors) để tìm tên miền chính thống tương đồng nhất.
Sử dụng đặc trưng TfidfVectorizer ở cấp độ ký tự (character n-grams).
"""

import os
import csv
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOP1M_PATH = os.path.join(BASE_DIR, 'data_set', 'top-1m.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'knn_search')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'knn_model.pkl')

def clean_domain(domain_str):
    """Làm sạch và chuẩn hóa tên miền"""
    d = domain_str.strip().lower()
    if d.startswith('www.'):
        d = d[4:]
    return d

def train():
    print("================================================================")
    print("🚀 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH KNN TÌM TÊN MIỀN TƯƠNG ĐỒNG")
    print("================================================================")
    
    # 1. Đọc dữ liệu từ top-1m.csv
    if not os.path.exists(TOP1M_PATH):
        print(f"❌ LỖI: Không tìm thấy file dữ liệu tại {TOP1M_PATH}")
        return
        
    print(f"📖 Đang đọc dữ liệu từ {TOP1M_PATH}...")
    t_start = time.time()
    
    domains = []
    with open(TOP1M_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            # Dòng tiêu đề có thể chứa "domain", chúng ta bỏ qua
            if row[1].lower() == 'domain':
                continue
            cleaned = clean_domain(row[1])
            if cleaned:
                domains.append(cleaned)
                
    print(f"✓ Đọc thành công {len(domains):,} tên miền trong {time.time() - t_start:.2f} giây.")

    # 2. Trích xuất đặc trưng dạng Character N-grams TF-IDF
    print("\n✍️ Đang chuyển đổi tên miền sang ma trận đặc trưng TF-IDF (Character N-grams)...")
    t_feat = time.time()
    # Sử dụng n-gram từ 2 đến 3 ký tự, giới hạn 15,000 đặc trưng để tối ưu RAM và tốc độ
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 3),
        max_features=15000,
        sublinear_tf=True
    )
    
    X = vectorizer.fit_transform(domains)
    print(f"✓ Trích xuất đặc trưng hoàn tất trong {time.time() - t_feat:.2f} giây.")
    print(f"  ➔ Kích thước ma trận đặc trưng: {X.shape[0]:,} x {X.shape[1]:,}")

    # 3. Huấn luyện mô hình NearestNeighbors
    print("\n🧠 Đang khởi tạo và huấn luyện mô hình KNN (Cosine metric)...")
    t_train = time.time()
    # Dùng thuật toán 'brute' tối ưu cho ma trận thưa (sparse matrix) cực lớn
    knn = NearestNeighbors(
        n_neighbors=1,
        metric='cosine',
        algorithm='brute',
        n_jobs=-1
    )
    knn.fit(X)
    print(f"✓ Huấn luyện KNN hoàn tất trong {time.time() - t_train:.2f} giây.")

    # 4. Lưu mô hình và dữ liệu đi kèm
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    print(f"\n💾 Đang lưu mô hình vào: {MODEL_PATH}...")
    t_save = time.time()
    
    model_data = {
        'vectorizer': vectorizer,
        'knn': knn,
        'domains': domains
    }
    
    # Sử dụng pickle protocol cao nhất để ghi nhanh và tiết kiệm dung lượng
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"✓ Lưu mô hình thành công trong {time.time() - t_save:.2f} giây.")
    file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"  ➔ Dung lượng file mô hình: {file_size_mb:.2f} MB")
    print("================================================================")
    print("🎉 QUÁ TRÌNH HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG!")
    print("================================================================")

if __name__ == '__main__':
    train()
