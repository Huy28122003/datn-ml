"""
xgb_random_batch_test.py
========================
Đoạn code tự động lấy ngẫu nhiên N URL từ tập dữ liệu phishing_url.csv 
và thực hiện dự đoán bằng mô hình XGBoost đã huấn luyện.

Cách chạy:
  python lib/xgb_vs_dsfull/xgb_random_batch_test.py
"""

import os
import sys
import random
import pandas as pd

# Định nghĩa đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xgb_phishing_test_predict import load_model, predict_url, display_result

CSV_PATH = os.path.join(BASE_DIR, 'data_set', 'phishing_url.csv')


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  XGBOOST - RANDOM BATCH TESTER                           ║")
    print("║  (Lấy ngẫu nhiên URL từ tập dữ liệu để kiểm thử)          ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 1. Load mô hình đã huấn luyện
    print("\n📦 Đang đọc mô hình...")
    model, features, params = load_model()
    print(f"  ✓ Đã load thành công mô hình ({len(features)} đặc trưng thuần URL)")

    # 2. Đọc file CSV chứa danh sách phishing URLs
    if not os.path.exists(CSV_PATH):
        print(f"❌ Không tìm thấy file dữ liệu tại: {CSV_PATH}")
        sys.exit(1)

    print("\n📖 Đang đọc tập dữ liệu phishing_url.csv...")
    df = pd.read_csv(CSV_PATH)
    print(f"  ✓ Đọc thành công {len(df):,} dòng dữ liệu.")

    # 3. Lấy ngẫu nhiên N URL để kiểm thử
    num_samples = 5
    print(f"🎲 Đang lấy ngẫu nhiên {num_samples} URL...")
    
    random_seed = random.randint(1, 10000)
    random_urls = df['url'].dropna().sample(n=num_samples, random_state=random_seed).tolist()

    print("=" * 80)
    
    # 4. Dự đoán từng URL
    for i, url in enumerate(random_urls, 1):
        print(f"\n🚀 [URL {i}/{num_samples}]: {url}")
        try:
            result = predict_url(url, model, features)
            display_result(result)
            print("─" * 80)
        except Exception as e:
            print(f"❌ Lỗi khi xử lý URL: {e}")
            print("─" * 80)

    print("\n✅ Hoàn thành quy trình kiểm thử ngẫu nhiên bằng XGBoost!")


if __name__ == '__main__':
    main()
