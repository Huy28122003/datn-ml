"""
xgb_phishing_test_predict.py
=============================
Chương trình dự đoán Phishing URL sử dụng mô hình XGBoost đã huấn luyện.
Hỗ trợ 3 chế độ:
  1. Kiểm tra 1 URL đơn lẻ bằng tham số truyền vào: --url "..."
  2. CLI Interactive Mode: Nhập URL liên tiếp để dự đoán, gõ 'quit' để thoát.
  3. Tích hợp cơ chế Hybrid Whitelist danh tiếng tên miền loại bỏ hoàn toàn False Positives.
"""

import os
import sys
import pickle
import argparse
import numpy as np

# Xác định đường dẫn gốc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xgb_url_extract_test_features import extract_features_for_model

# Cấu hình đường dẫn
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'xgb_vs_dsfull')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'xgb_phishing_model.pkl')

# Whitelist tên miền uy tín hàng đầu tránh data bias về độ sâu của đường dẫn
REPUTABLE_DOMAINS = {
    'google.com', 'google.com.vn', 'youtube.com', 'facebook.com', 'instagram.com', 
    'twitter.com', 'linkedin.com', 'github.com', 'gitlab.com', 'microsoft.com', 
    'apple.com', 'amazon.com', 'netflix.com', 'wikipedia.org', 'w3schools.com', 
    'stackoverflow.com', 'stackexchange.com', 'medium.com', 'docker.com', 'docker.io', 
    'kubernetes.io', 'python.org', 'npmjs.com', 'cloudflare.com', 'mozilla.org', 
    'apache.org', 'spring.io', 'oracle.com', 'git-scm.com', 'bitbucket.org'
}


def load_model():
    """Tải mô hình XGBoost và danh sách đặc trưng đã lưu."""
    if not os.path.exists(MODEL_PATH):
        print("\n❌ LỖI: Chưa có mô hình XGBoost được huấn luyện!")
        print("  → Vui lòng chạy các bước huấn luyện trước.")
        sys.exit(1)

    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    features = model_data['features']
    params = model_data['params']

    return model, features, params


def get_registered_domain(domain):
    """Trích xuất tên miền đăng ký để so khớp whitelist."""
    if not domain:
        return ""
    domain = domain.lower()
    parts = domain.split('.')
    if len(parts) >= 2:
        if len(parts) >= 3 and parts[-2] in ('co', 'com', 'org', 'net', 'edu', 'gov'):
            return '.'.join(parts[-3:])
        return '.'.join(parts[-2:])
    return domain


def predict_url(url, model, model_features):
    """
    Dự đoán một URL là độc hại hay an toàn bằng XGBoost.
    """
    # Trích xuất đặc trưng thuần URL
    feature_vector = extract_features_for_model(url, model_features)

    # Kiểm tra Whitelist
    from xgb_url_extract_test_features import _parse_url_parts
    parts = _parse_url_parts(url)
    domain = parts.get('domain', '')
    reg_domain = get_registered_domain(domain)

    is_whitelisted = reg_domain in REPUTABLE_DOMAINS

    if is_whitelisted:
        prediction = 0
        probabilities = [1.0, 0.0]
        label = 'LEGITIMATE ✅ (Danh tiếng uy tín)'
    else:
        # Dự đoán bằng mô hình XGBoost
        X = np.array([feature_vector])
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        label = 'PHISHING 🚨' if prediction == 1 else 'LEGITIMATE ✅'

    result = {
        'url': url,
        'prediction': int(prediction),
        'label': label,
        'probability_legitimate': float(probabilities[0]),
        'probability_phishing': float(probabilities[1]),
        'confidence': float(max(probabilities)),
        'features_extracted': {
            name: val for name, val in zip(model_features, feature_vector)
        },
        'features_used_count': len(model_features),
        'is_whitelisted': is_whitelisted
    }

    return result


def display_result(result):
    """Hiển thị bảng kết quả dự đoán trực quan."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print(f"║  🎯 KẾT QUẢ DỰ ĐOÁN URL (XGBOOST){'':>34}║")
    print("╠" + "═" * 68 + "╣")
    url_trimmed = result['url'][:60] + ('...' if len(result['url']) > 60 else '')
    print(f"║  🔗 URL: {url_trimmed:<58}║")
    print("╠" + "═" * 68 + "╣")

    if result['prediction'] == 1:
        print(f"║  ⚠️  KẾT QUẢ: {result['label']:<47}║")
        print(f"║  📈 Xác suất Phishing:   {result['probability_phishing']:.2%}{'':>37}║")
    else:
        print(f"║  ✅ KẾT QUẢ: {result['label']:<47}║")
        print(f"║  📈 Xác suất An toàn:    {result['probability_legitimate']:.2%}{'':>37}║")

    print(f"║  🔥 Độ tin cậy:          {result['confidence']:.2%}{'':>37}║")
    print("╠" + "═" * 68 + "╣")
    print(f"║  📊 Số đặc trưng Pure URL sử dụng: {result['features_used_count']:<31}║")
    print("╚" + "═" * 68 + "╝")

    # Chi tiết đặc trưng
    print(f"\n  📊 CHI TIẾT CÁC ĐẶC TRƯNG ĐÃ TRÍCH XUẤT:")
    print("  " + "─" * 60)
    features = result['features_extracted']
    for i, (name, val) in enumerate(features.items(), 1):
        if isinstance(val, float):
            print(f"    {i:>2}. {name:<35} = {val:.4f}")
        else:
            print(f"    {i:>2}. {name:<35} = {val}")


def interactive_mode(model, model_features):
    """CLI liên tục tương tác."""
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  XGBOOST - PURE URL PHISHING DETECTOR                    ║")
    print("║  Nhập URL cần kiểm tra (Nhập 'quit' để thoát)           ║")
    print("╚══════════════════════════════════════════════════════════╝")

    while True:
        print("\n" + "─" * 70)
        url = input("🔗 Nhập URL: ").strip()

        if url.lower() in ('quit', 'exit', 'q'):
            print("\n👋 Tạm biệt!")
            break

        if not url:
            print("⚠ Vui lòng nhập chuỗi URL hợp lệ!")
            continue

        try:
            result = predict_url(url, model, model_features)
            display_result(result)
        except Exception as e:
            print(f"❌ Có lỗi xảy ra trong quá trình dự đoán: {e}")


def main():
    parser = argparse.ArgumentParser(description="XGBoost Phishing URL Predictor")
    parser.add_argument('--url', type=str, help='Đường dẫn URL cần kiểm tra')
    args = parser.parse_args()

    print("📦 Đang đọc mô hình...")
    model, features, params = load_model()
    print(f"  ✓ Đã load thành công mô hình XGBoost ({len(features)} đặc trưng)")

    if args.url:
        result = predict_url(args.url, model, features)
        display_result(result)
    else:
        interactive_mode(model, features)


if __name__ == '__main__':
    main()
