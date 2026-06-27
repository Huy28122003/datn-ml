"""
rf_test_from_csv.py
===================
Kiểm thử mô hình Random Forest Hybrid trên các URL từ phishing_url.csv.
- Tải mô hình Random Forest đã được huấn luyện.
- Chọn các URL cụ thể được chỉ định (dòng 887 - 896).
- Chọn ngẫu nhiên một số URL khác từ phishing_url.csv.
- Dự đoán nhãn (Phishing/Legitimate) bằng cách trích xuất đặc trưng động (Lexical & HTML Scraping).
- Đánh giá tỷ lệ phát hiện (Detection Rate) trên mẫu kiểm thử.
"""

import os
import sys
import pandas as pd
import numpy as np
import random
import argparse

# Xác định thư mục gốc của project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Thêm thư mục lib vào sys.path để import
sys.path.insert(0, os.path.join(BASE_DIR, 'lib'))
# Thêm thư mục rf_vs_hybrid để import rf_phishing_test_predict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rf_phishing_test_predict as rfp

CSV_PATH = os.path.join(BASE_DIR, 'data_set', 'phishing_url.csv')

def load_urls_from_csv(csv_path, num_random=5):
    """
    Đọc dữ liệu từ csv_path.
    Lấy các URL từ dòng 887 đến 896 (1-indexed, tương ứng chỉ số dòng trong file).
    Và lấy thêm `num_random` URL ngẫu nhiên khác.
    """
    if not os.path.exists(csv_path):
        print(f"❌ LỖI: Không tìm thấy file dữ liệu tại {csv_path}")
        sys.exit(1)
        
    print(f"📖 Đang đọc tập dữ liệu từ {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✓ Đã tải {len(df)} dòng dữ liệu.")
    
    # Dòng 887-896 (1-indexed): 
    # Trong pandas, dòng đầu tiên sau header (dòng 2 trong file) có index là 0.
    # Dòng 887 trong file csv sẽ có index là 887 - 2 = 885.
    # Dòng 896 trong file csv sẽ có index là 896 - 2 = 894.
    specified_indices = list(range(885, min(895, len(df))))
    specified_df = df.iloc[specified_indices]
    
    # Loại bỏ các dòng đã lấy để chọn ngẫu nhiên các dòng còn lại
    remaining_df = df.drop(index=specified_indices)
    
    # Lấy ngẫu nhiên các dòng
    sampled_df = remaining_df.sample(n=num_random, random_state=random.randint(1, 100000))
    
    return specified_df, sampled_df

def main():
    parser = argparse.ArgumentParser(description="Test Random Forest Hybrid model on phishing_url.csv")
    parser.add_argument("--random-count", type=int, default=5, help="Số lượng URL ngẫu nhiên cần lấy thêm để test")
    args = parser.parse_args()

    # 1. Load model
    print("📦 Đang tải mô hình Random Forest Hybrid...")
    model, model_features, params = rfp.load_model()
    print(f"✓ Mô hình đã được tải thành công ({len(model_features)} đặc trưng).")
    
    # 2. Load URLs
    specified_df, sampled_df = load_urls_from_csv(CSV_PATH, num_random=args.random_count)
    
    print("\n" + "="*80)
    print("🎯 BẮT ĐẦU KIỂM THỬ MÔ HÌNH TRÊN CÁC URL CHỈ ĐỊNH (DÒNG 887-896)")
    print("="*80)
    
    results = []
    
    # Gom tất cả URL cần test
    # Nhãn thực tế của tất cả các URL trong phishing_url.csv đều là Phishing (1)
    to_test = []
    for idx, row in specified_df.iterrows():
        to_test.append((row['url'], "Chỉ định (Dòng " + str(idx + 2) + ")"))
    for idx, row in sampled_df.iterrows():
        to_test.append((row['url'], "Ngẫu nhiên (Dòng " + str(idx + 2) + ")"))
        
    correct_count = 0
    
    for i, (url, source) in enumerate(to_test, 1):
        print(f"\n🔍 [{i}/{len(to_test)}] URL ({source}): {url}")
        try:
            # Dự đoán
            res = rfp.predict_url(url, model, model_features)
            
            # Ground truth: Phishing (1)
            prediction = res['prediction']
            is_correct = (prediction == 1)
            if is_correct:
                correct_count += 1
                status = "🟢 CHÍNH XÁC (Phát hiện Phishing)"
            else:
                status = "🔴 SAI LỆCH (Bỏ lọt / Nhận diện nhầm là Legitimate)"
                
            print(f"   - Nhãn thực tế: PHISHING (1)")
            print(f"   - Dự đoán:     {res['label']}")
            print(f"   - Xác suất:    Phishing: {res['probability_phishing']:.2%}, Legitimate: {res['probability_legitimate']:.2%}")
            print(f"   - Trạng thái:  {status}")
            
            # Lưu kết quả để tổng hợp
            results.append({
                'url': url,
                'source': source,
                'pred': prediction,
                'correct': is_correct,
                'prob_phish': res['probability_phishing']
            })
        except Exception as e:
            print(f"   ❌ Lỗi xử lý URL này: {e}")
            
    print("\n" + "="*80)
    print("📊 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
    print("="*80)
    total_tested = len(results)
    if total_tested > 0:
        accuracy = correct_count / total_tested
        print(f"📈 Tổng số URL đã test: {total_tested}")
        print(f"✅ Số URL phát hiện đúng (True Positive): {correct_count}")
        print(f"❌ Số URL bỏ lọt (False Negative): {total_tested - correct_count}")
        print(f"🎯 Tỷ lệ phát hiện (Detection Rate / Recall): {accuracy:.2%}")
    else:
        print("Không có kết quả kiểm thử nào được hoàn thành.")
    print("="*80)

if __name__ == '__main__':
    main()
