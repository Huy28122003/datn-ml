"""
knn_phishing_test_predict.py
==============================
Script kiểm thử và tìm kiếm bản ghi tương đồng nhất trong tập dữ liệu hybrid.csv bằng mô hình KNN.
  - Trích xuất đặc trưng Lexical từ URL đầu vào.
  - Cào HTML động thời gian thực (nếu URL online) để lấy đặc trưng webpage, hoặc dùng fallback trung vị nếu offline.
  - Chuẩn hóa vector đặc trưng của URL đầu vào bằng Scaler đã huấn luyện.
  - Tìm ra Top K bản ghi có khoảng cách Euclidean nhỏ nhất trong tập dữ liệu hybrid.csv.
  - Hiển thị kết quả so sánh chi tiết dạng side-by-side giữa URL đầu vào và bản ghi tương đồng nhất.
"""

import os
import sys
import re
import pickle
import argparse
import warnings
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# Xác định đường dẫn gốc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, 'lib'))

MODEL_PATH = os.path.join(BASE_DIR, 'output', 'knn_vs_hybrid', 'knn_model.pkl')
DATASET_PATH = os.path.join(BASE_DIR, 'data_set', 'hybrid.csv')


def load_model():
    """Tải mô hình KNN và scaler đã lưu."""
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ LỖI: Không tìm thấy tệp mô hình tại: {MODEL_PATH}")
        print("  → Vui lòng chạy huấn luyện mô hình trước bằng lệnh:")
        print("    python lib/knn_vs_hybrid/knn_model_train.py")
        sys.exit(1)

    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def get_registered_domain(domain):
    """Trích xuất tên miền đã đăng ký để hỗ trợ so khớp."""
    if not domain:
        return ""
    domain = domain.lower()
    parts = domain.split('.')
    if len(parts) >= 2:
        if len(parts) >= 3 and parts[-2] in ('co', 'com', 'org', 'net', 'edu', 'gov'):
            return '.'.join(parts[-3:])
        return '.'.join(parts[-2:])
    return domain


def get_tld(domain):
    """Trích xuất TLD từ tên miền."""
    if not domain:
        return ""
    parts = domain.split('.')
    if len(parts) >= 2:
        return parts[-1]
    return ""


def compute_token_match_score(text, title):
    """Tính toán điểm khớp từ của văn bản đối với tiêu đề trang web."""
    if not text or not title:
        return 0.0
    text_tokens = set(re.findall(r'\w+', text.lower()))
    title_tokens = set(re.findall(r'\w+', title.lower()))
    # Loại bỏ các từ quá ngắn để tránh trùng khớp ngẫu nhiên
    text_tokens = {t for t in text_tokens if len(t) > 2}
    if not text_tokens:
        return 0.0
    matches = text_tokens.intersection(title_tokens)
    return float(len(matches) / len(text_tokens) * 100.0)


def extract_hybrid_features(url):
    """
    Trích xuất đầy đủ 50 đặc trưng số cho URL tương ứng với tập dữ liệu hybrid.csv.
    Tự động scraping trực tiếp nếu URL online.
    """
    # 1. Khởi tạo giá trị trung vị mặc định từ tập dữ liệu hybrid.csv
    feats = {
        'URLLength': len(url),
        'DomainLength': 20.0,
        'IsDomainIP': 0.0,
        'URLSimilarityIndex': 100.0,
        'CharContinuationRate': 1.0,
        'TLDLegitimateProb': 0.08,
        'URLCharProb': 0.058,
        'TLDLength': 3.0,
        'NoOfSubDomain': 1.0,
        'HasObfuscation': 0.0,
        'NoOfObfuscatedChar': 0.0,
        'ObfuscationRatio': 0.0,
        'NoOfLettersInURL': 14.0,
        'LetterRatioInURL': 0.519,
        'NoOfDegitsInURL': 0.0,
        'DegitRatioInURL': 0.0,
        'NoOfEqualsInURL': 0.0,
        'NoOfQMarkInURL': 0.0,
        'NoOfAmpersandInURL': 0.0,
        'NoOfOtherSpecialCharsInURL': 1.0,
        'SpacialCharRatioInURL': 0.05,
        'IsHTTPS': 1.0 if url.startswith('https') else 0.0,
        'LineOfCode': 429.0,
        'LargestLineLength': 1090.0,
        'HasTitle': 1.0,
        'DomainTitleMatchScore': 75.0,
        'URLTitleMatchScore': 100.0,
        'HasFavicon': 0.0,
        'Robots': 0.0,
        'IsResponsive': 1.0,
        'NoOfURLRedirect': 0.0,
        'NoOfSelfRedirect': 0.0,
        'HasDescription': 0.0,
        'NoOfPopup': 0.0,
        'NoOfiFrame': 0.0,
        'HasExternalFormSubmit': 0.0,
        'HasSocialNet': 0.0,
        'HasSubmitButton': 0.0,
        'HasHiddenFields': 0.0,
        'HasPasswordField': 0.0,
        'Bank': 0.0,
        'Pay': 0.0,
        'Crypto': 0.0,
        'HasCopyrightInfo': 0.0,
        'NoOfImage': 8.0,
        'NoOfCSS': 2.0,
        'NoOfJS': 6.0,
        'NoOfSelfRef': 12.0,
        'NoOfEmptyRef': 0.0,
        'NoOfExternalRef': 10.0
    }

    # 2. Phân tích Lexical từ URL tĩnh
    domain = ""
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        if ':' in domain:
            domain = domain.split(':')[0]
            
        feats['DomainLength'] = float(len(domain))
        
        # Check IP
        ip_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
        feats['IsDomainIP'] = 1.0 if ip_pattern.match(domain) else 0.0
        
        # Subdomains
        dots = domain.count('.')
        feats['NoOfSubDomain'] = float(max(0, dots - 1))
        
        # Letters & Digits
        letters = sum(1 for c in url if c.isalpha())
        digits = sum(1 for c in url if c.isdigit())
        feats['NoOfLettersInURL'] = float(letters)
        feats['LetterRatioInURL'] = letters / len(url) if len(url) > 0 else 0.5
        feats['NoOfDegitsInURL'] = float(digits)
        feats['DegitRatioInURL'] = digits / len(url) if len(url) > 0 else 0.0
        
        # Đặc trưng ký tự đặc biệt cụ thể
        feats['NoOfEqualsInURL'] = float(url.count('='))
        feats['NoOfQMarkInURL'] = float(url.count('?'))
        feats['NoOfAmpersandInURL'] = float(url.count('&'))
        
        specials = sum(1 for c in url if not c.isalnum() and c not in ('/', '.', ':', '-'))
        feats['NoOfOtherSpecialCharsInURL'] = float(specials)
        feats['SpacialCharRatioInURL'] = specials / len(url) if len(url) > 0 else 0.0
        
        # TLD
        tld = get_tld(domain)
        feats['TLDLength'] = float(len(tld))
        
    except Exception:
        pass

    # 3. Live Web Scraping (nếu online)
    print("🌐 Đang kiểm tra trạng thái online và cào HTML lấy đặc trưng động...")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=4.0) as response:
            html = response.read().decode('utf-8', errors='ignore')
            print("  ✓ Tải trang thành công! Đang phân tích mã HTML động...")
            
            # Cập nhật thông số HTML thực tế
            lines = html.split('\n')
            feats['LineOfCode'] = float(len(lines))
            feats['LargestLineLength'] = float(max(len(line) for line in lines) if lines else 0)
            
            feats['NoOfImage'] = float(len(re.findall(r'<img\s+', html, re.I)))
            feats['NoOfJS'] = float(len(re.findall(r'<script\s+', html, re.I)))
            feats['NoOfCSS'] = float(len(re.findall(r'<link\s+[^>]*rel=["\']stylesheet["\']|<style\s+', html, re.I)))
            
            # Form Submit và các loại form đặc biệt
            feats['HasSubmitButton'] = 1.0 if re.search(r'<input\s+[^>]*type=["\']submit["\']|<button', html, re.I) else 0.0
            feats['HasHiddenFields'] = 1.0 if re.search(r'type=["\']hidden["\']', html, re.I) else 0.0
            feats['HasPasswordField'] = 1.0 if re.search(r'type=["\']password["\']', html, re.I) else 0.0
            feats['HasExternalFormSubmit'] = 1.0 if re.search(r'<form\s+[^>]*action=["\']https?://', html, re.I) else 0.0
            
            # Check keywords nhạy cảm liên quan đến ngân hàng / thanh toán / crypto
            bank_keywords = ['bank', 'ebank', 'banking', 'vietcombank', 'techcombank', 'acb', 'bidv', 'agribank', 'sacombank']
            pay_keywords = ['pay', 'payment', 'paypal', 'momo', 'zalopay', 'checkout', 'card', 'visa', 'mastercard']
            crypto_keywords = ['crypto', 'bitcoin', 'btc', 'eth', 'binance', 'wallet', 'blockchain', 'trustwallet']
            
            html_lower = html.lower()
            feats['Bank'] = 1.0 if any(k in html_lower for k in bank_keywords) else 0.0
            feats['Pay'] = 1.0 if any(k in html_lower for k in pay_keywords) else 0.0
            feats['Crypto'] = 1.0 if any(k in html_lower for k in crypto_keywords) else 0.0
            
            # Links/References
            all_links = re.findall(r'href=["\'](https?://[^"\']+|/[^"\']+|#[^"\']*)["\']', html, re.I)
            external = 0
            self_ref = 0
            empty = 0
            
            domain_escaped = re.escape(domain) if domain else ''
            for link in all_links:
                if link.startswith('#') or link == "":
                    empty += 1
                elif link.startswith('/') or (domain_escaped and re.search(domain_escaped, link)):
                    self_ref += 1
                else:
                    external += 1
                    
            feats['NoOfExternalRef'] = float(external)
            feats['NoOfSelfRef'] = float(self_ref)
            feats['NoOfEmptyRef'] = float(empty)
            
            # Metadata & Tiêu đề
            feats['HasDescription'] = 1.0 if re.search(r'<meta\s+name=["\']description["\']', html, re.I) else 0.0
            feats['HasFavicon'] = 1.0 if re.search(r'rel=["\'](icon|shortcut icon)["\']', html, re.I) else 0.0
            feats['IsResponsive'] = 1.0 if re.search(r'name=["\']viewport["\']', html, re.I) else 0.0
            feats['HasSocialNet'] = 1.0 if any(s in html for s in ['facebook.com', 'twitter.com', 'instagram.com']) else 0.0
            feats['HasCopyrightInfo'] = 1.0 if any(c in html_lower for c in ['copyright', '©', '&copy;']) else 0.0
            
            # Tiêu đề khớp điểm
            title_match = re.search(r'<title>(.*?)</title>', html, re.I | re.S)
            if title_match:
                title = title_match.group(1).strip()
                feats['HasTitle'] = 1.0
                feats['DomainTitleMatchScore'] = compute_token_match_score(domain, title)
                feats['URLTitleMatchScore'] = compute_token_match_score(url, title)
            else:
                feats['HasTitle'] = 0.0
                feats['DomainTitleMatchScore'] = 0.0
                feats['URLTitleMatchScore'] = 0.0
                
    except Exception as e:
        print(f"  ⚠️  Không thể cào HTML ({e}). Sử dụng chế độ Fallback trung vị.")

    return feats


def print_comparison_table(input_feats, match_feats):
    """Hiển thị bảng so sánh đặc trưng side-by-side phân chia theo nhóm."""
    feature_groups = {
        "1. Đặc trưng Lexical (URL tĩnh)": [
            'URLLength', 'DomainLength', 'IsDomainIP', 'TLDLength', 'NoOfSubDomain',
            'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL',
            'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL',
            'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS'
        ],
        "2. Đặc trưng cấu trúc HTML (Động)": [
            'LineOfCode', 'LargestLineLength', 'NoOfImage', 'NoOfJS', 'NoOfCSS',
            'HasSubmitButton', 'HasHiddenFields', 'HasPasswordField', 'HasExternalFormSubmit'
        ],
        "3. Đặc trưng tham chiếu (Links)": [
            'NoOfExternalRef', 'NoOfSelfRef', 'NoOfEmptyRef'
        ],
        "4. Phân tích nội dung và ngữ cảnh": [
            'HasTitle', 'DomainTitleMatchScore', 'URLTitleMatchScore', 'HasFavicon',
            'HasDescription', 'HasSocialNet', 'HasCopyrightInfo', 'IsResponsive',
            'Bank', 'Pay', 'Crypto'
        ],
        "5. Đặc trưng bổ sung (Phân tích nâng cao)": [
            'URLSimilarityIndex', 'CharContinuationRate', 'TLDLegitimateProb',
            'URLCharProb', 'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio',
            'NoOfURLRedirect', 'NoOfSelfRedirect', 'NoOfPopup', 'NoOfiFrame', 'Robots'
        ]
    }

    print("\n" + "=" * 90)
    print(f"║ {'ĐẶC TRƯNG':<32} │ {'URL ĐẦU VÀO':<23} │ {'BẢN GHI TƯƠNG ĐỒNG':<23} │ MATCH ║")
    print("=" * 90)

    for group_name, features in feature_groups.items():
        print(f" 📂 \033[1m{group_name}\033[0m")
        print(" ├" + "─" * 87)
        for name in features:
            val_input = input_feats.get(name, 0.0)
            val_match = match_feats.get(name, 0.0)
            
            # Kiểm tra khớp (bằng nhau hoặc sai số cực nhỏ đối với số thực)
            is_match = abs(val_input - val_match) < 1e-4
            match_status = "✅ YES" if is_match else "❌ NO"
            
            # Định dạng hiển thị
            str_input = f"{val_input:.4f}" if isinstance(val_input, float) and not val_input.is_integer() else f"{int(val_input)}"
            str_match = f"{val_match:.4f}" if isinstance(val_match, float) and not val_match.is_integer() else f"{int(val_match)}"
            
            print(f" │ {name:<30} │ {str_input:<21} │ {str_match:<21} │ {match_status:<5} │")
        print(" └" + "─" * 87)
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description='KNN Hybrid URL Similarity Matcher')
    parser.add_argument('--url', type=str, help='Đường dẫn URL cần kiểm tra')
    parser.add_argument('-k', type=int, default=1, help='Số lượng bản ghi tương đồng cần tìm (mặc định: 1)')
    args = parser.parse_args()

    print("📦 Đang tải mô hình KNN và scaler...")
    model_data = load_model()
    knn = model_data['knn']
    scaler = model_data['scaler']
    features_list = model_data['features']
    print(f"  ✓ Đã load mô hình thành công ({len(features_list)} đặc trưng).")

    if not os.path.exists(DATASET_PATH):
        print(f"❌ LỖI: Không tìm thấy tập dữ liệu tại: {DATASET_PATH}")
        sys.exit(1)

    print("📖 Đang tải chỉ mục dữ liệu gốc (hybrid.csv) để tra cứu...")
    # Chỉ đọc những cột metadata cần thiết để tiết kiệm bộ nhớ khi tra cứu nhanh
    meta_cols = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title', 'label']
    df_meta = pd.read_csv(DATASET_PATH, usecols=meta_cols)
    print(f"  ✓ Đã tải chỉ mục dữ liệu ({df_meta.shape[0]:,} dòng).")

    print("\n" + "=" * 80)
    print(" ╔══════════════════════════════════════════════════════════╗")
    print(" ║        HỆ THỐNG TÌM KIẾM BẢN GHI TƯƠNG ĐỒNG KNN          ║")
    print(" ║         Nhập URL cần quét (gõ 'quit' để thoát)           ║")
    print(" ╚══════════════════════════════════════════════════════════╝")
    print("=" * 80)

    # Chế độ quét theo tham số dòng lệnh hoặc lặp tương tác
    if args.url:
        urls_to_test = [args.url]
    else:
        urls_to_test = []

    interactive = len(urls_to_test) == 0

    while True:
        if interactive:
            url_input = input("\n🔗 Nhập URL để tìm láng giềng gần nhất: ").strip()
            if url_input.lower() in ('quit', 'exit', 'q'):
                print("\n👋 Tạm biệt!")
                break
            if not url_input:
                print("⚠ Vui lòng nhập URL hợp lệ!")
                continue
        else:
            if not urls_to_test:
                break
            url_input = urls_to_test.pop(0)

        if not url_input.startswith(('http://', 'https://')):
            url_input = 'https://' + url_input

        try:
            # 1. Trích xuất đặc trưng của URL đầu vào
            input_feats_all = extract_hybrid_features(url_input)
            
            # Lấy vector theo đúng danh sách đặc trưng đã lưu
            input_vector = [input_feats_all[name] for name in features_list]
            
            # 2. Chuẩn hóa vector đầu vào
            input_scaled = scaler.transform([input_vector])
            
            # 3. Tìm láng giềng gần nhất bằng KNN
            distances, indices = knn.kneighbors(input_scaled, n_neighbors=args.k)
            
            print(f"\n🎯 ĐÃ TÌM THẤY {args.k} BẢN GHI TƯƠNG ĐỒNG NHẤT:")
            print("-" * 80)
            
            # Hiển thị tóm tắt các bản ghi tương đồng tìm được
            for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
                matched_row = df_meta.iloc[idx]
                label_str = "🚨 PHISHING" if matched_row['label'] == 1 else "✅ LEGITIMATE"
                print(f" #{rank} | Khoảng cách Euclidean: {dist:.4f}")
                print(f"    - Index trong Dataset: {idx}")
                print(f"    - Tên tệp gốc: {matched_row['FILENAME']}")
                print(f"    - URL trùng khớp: {matched_row['URL']}")
                print(f"    - Nhãn bản ghi: {label_str}")
                print("-" * 80)

            # Nếu tìm bản ghi giống nhất (k=1) hoặc muốn so sánh chi tiết bản ghi đầu tiên
            if args.k >= 1:
                first_match_idx = indices[0][0]
                
                # Đọc đầy đủ các đặc trưng của bản ghi khớp từ tập dữ liệu gốc
                df_full_match = pd.read_csv(DATASET_PATH, skiprows=first_match_idx + 1, nrows=1, header=None)
                headers = pd.read_csv(DATASET_PATH, nrows=0).columns.tolist()
                df_full_match.columns = headers
                match_row_full = df_full_match.iloc[0]
                
                match_feats_dict = {name: float(match_row_full[name]) for name in features_list}
                
                print(f"\n📊 BẢNG SO SÁNH CHI TIẾT ĐẶC TRƯNG SIDE-BY-SIDE (Với Bản Ghi #{indices[0][0]} giống nhất):")
                print_comparison_table(input_feats_all, match_feats_dict)

        except Exception as e:
            print(f"❌ Có lỗi xảy ra trong quá trình xử lý: {e}")
            import traceback
            traceback.print_exc()

        if not interactive:
            break


if __name__ == '__main__':
    main()
