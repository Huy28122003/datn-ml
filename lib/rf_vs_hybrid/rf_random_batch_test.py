"""
rf_random_batch_test.py
========================
Bộ kiểm thử ngẫu nhiên từ tập dữ liệu Hybrid.
  - Tải mô hình Random Forest Hybrid đã được huấn luyện.
  - Chọn ngẫu nhiên 5 mẫu từ hybrid.csv.
  - Dự đoán và hiển thị chi tiết vector đặc trưng cấu trúc/HTML.
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np

# Xác định đường dẫn gốc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Cấu hình đường dẫn
DATASET_PATH = os.path.join(BASE_DIR, 'data_set', 'hybrid.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'rf_vs_hybrid')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'rf_phishing_model.pkl')


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ LỖI: Không tìm thấy mô hình tại {MODEL_PATH}")
        sys.exit(1)
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def main():
    print("🚀 Đang khởi động bộ kiểm thử ngẫu nhiên Hybrid (Random Forest)...")
    
    # 1. Load mô hình
    model_data = load_model()
    model = model_data['model']
    features = model_data['features']
    
    # 2. Đọc dataset
    if not os.path.exists(DATASET_PATH):
        print(f"❌ LỖI: Không tìm thấy tập dữ liệu tại {DATASET_PATH}")
        sys.exit(1)
        
    df = pd.read_csv(DATASET_PATH)
    
    # 3. Lấy 5 mẫu ngẫu nhiên
    print(f"✓ Đã load mô hình ({len(features)} đặc trưng).")
    print(f"🎲 Đang lấy 5 mẫu ngẫu nhiên từ tập dữ liệu hybrid.csv...")
    
    sampled_df = df.sample(n=5, random_state=np.random.randint(1, 100000))
    
    print("=" * 80)
    
    for idx, (index, row) in enumerate(sampled_df.iterrows(), 1):
        url = row['URL']
        actual_label = int(row['label'])
        filename = row.get('FILENAME', 'N/A')
        
        # Lọc đặc trưng theo thứ tự mô hình yêu cầu
        feature_vector = [row[name] for name in features]
        
        # Dự đoán
        X = np.array([feature_vector])
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        
        label_text = "🚨 PHISHING" if pred == 1 else "✅ LEGITIMATE"
        actual_text = "PHISHING" if actual_label == 1 else "LEGITIMATE"
        
        print(f"\n🚀 [URL {idx}/5]: {url}")
        print(f"   - File gốc:    {filename}")
        print(f"   - Nhãn thực tế: {actual_text}")
        print(f"   - Dự đoán:     {label_text} (Xác suất: {probs[pred]:.2%})")
        print(f"   - Trạng thái:  {'🟢 CHÍNH XÁC' if pred == actual_label else '🔴 SAI LỆCH'}")
        
        print("   📊 Chi tiết 10 đặc trưng Hybrid quan trọng nhất của mẫu này:")
        print("   " + "-" * 55)
        for i, name in enumerate(features[:10], 1):
            val = row[name]
            print(f"     {i:>2}. {name:<30} = {val}")
        print("=" * 80)
        
    print("\n✅ Hoàn thành quy trình kiểm thử ngẫu nhiên Hybrid!")


if __name__ == '__main__':
    main()
