
import os
import sys
import time
import pickle
import urllib.parse

# huynq - Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'output', 'knn_search', 'knn_model.pkl')

# huynq - Danh sách so khớp các hậu tố ccTLD quốc gia phổ biến trên thế giới
PUBLIC_SUFFIXES = {
    'com.vn', 'co.uk', 'com.br', 'com.cn', 'com.tr', 'com.mu', 'com.ug', 'com.bi', 'com.py', 
    'com.gr', 'com.et', 'com.bn', 'net.cn', 'gov.tr', 'gov.vn', 'org.vn', 'my.id', 'ac.uk'
}

def extract_domain_parts(url_str):
    url_str = url_str.strip()
    if not url_str:
        return "", []
        
    if not url_str.startswith(('http://', 'https://')):
        url_str_for_parse = 'http://' + url_str
    else:
        url_str_for_parse = url_str
        
    try:
        parsed = urllib.parse.urlparse(url_str_for_parse)
        hostname = parsed.netloc.lower()
        if ':' in hostname:
            hostname = hostname.split(':')[0]
        if hostname.startswith('www.'):
            hostname = hostname[4:]
    except Exception:
        hostname = url_str.lower()

    labels = hostname.split('.')
    if len(labels) <= 1:
        return hostname, []

    suffix_len = 1
    if len(labels) >= 2:
        last_2 = ".".join(labels[-2:])
        # huynq - 1. So khớp danh sách ccTLD chuẩn
        if last_2 in PUBLIC_SUFFIXES:
            suffix_len = 2
        # huynq - 2. Quy luật thuật toán ccTLD quốc gia dự phòng
        elif len(labels) >= 3:
            last_label = labels[-1]
            prev_label = labels[-2]
            # huynq - Quy luật ccTLD quốc gia đuôi kép (Ví dụ: .com.vn, .co.uk)
            if len(last_label) == 2 and prev_label in {'com', 'co', 'net', 'org', 'edu', 'gov', 'ac'}:
                suffix_len = 2

    reg_labels = labels[-(suffix_len + 1):]
    registered_domain = ".".join(reg_labels)
    subdomain_labels = labels[:-(suffix_len + 1)]
    
    return registered_domain, subdomain_labels

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def levenshtein_similarity(s1, s2):
    dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return (1.0 - dist / max_len)

def main():
    print("================================================================")
    print("HỆ THỐNG TRUY VẤN KNN & PHÂN TÍCH GIẢ MẠO DOMAIN PHÂN CẤP (LAI)")
    
    if not os.path.exists(MODEL_PATH):
        print(f" Chưa tìm thấy mô hình tại {MODEL_PATH}")
        sys.exit(1)
        
    print(" Đang nạp mô hình KNN và cơ sở dữ liệu tên miền...")
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
        
    vectorizer = model_data['vectorizer']
    knn = model_data['knn']
    domains = model_data['domains']
    print(f" Nạp thành công mô hình. Tổng số tên miền: {len(domains):,}")
    
    # huynq - Rút trích danh sách thương hiệu uy tín từ toàn bộ 1 triệu tên miền sạch để làm quy chuẩn đối sánh
    clean_brands = {d.split('.')[0] for d in domains if len(d.split('.')[0]) > 3}
    print(f" danh sách {len(clean_brands):,} thương hiệu uy tín từ dữ liệu huấn luyện.")
    print("-" * 64)
    print(" Nhập 'exit' hoặc 'quit' để dừng.")
    print("-" * 64)
    
    while True:
        try:
            user_input = input("\n Nhập URL / Domain cần kiểm tra: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ('exit', 'quit'):
                break
                
            t_query_start = time.time()
            
            # huynq - 1. Phân tích Registered Domain và các nhãn Subdomain
            reg_domain, sub_labels = extract_domain_parts(user_input)
            if not reg_domain:
                continue
                
            print(f"Domain chính (Registered Domain): '{reg_domain}'")
            if sub_labels:
                print(f"Subdomains phân tích: {sub_labels}")
                
            # huynq - Đưa domain chính vào KNN
            X_input = vectorizer.transform([reg_domain])
            distances, indices = knn.kneighbors(X_input, n_neighbors=1)
            matched_domain = domains[indices[0][0]]
            
            # huynq - Tính độ tương đồng Levenshtein của Domain chính
            lev_score = levenshtein_similarity(reg_domain, matched_domain) * 100.0
            
            # huynq - 3.  phân cấp theo các quy tắc giả mạo
            print("-" * 55)
            
            if lev_score == 100.0:
                # huynq - 1
                print(f"Tên miền chính thống khớp: {matched_domain}")
                print(f"Trạng thái: đây là website của 1 đơn vị uy tín hoặc 1 thành phần thuộc đơn vị uy tín")
            else:
                # huynq - 2. Quét các nhãn subdomain
                triggered_sub_brand = None
                triggered_matched_domain = None
                
                for label in sub_labels:
                    # huynq - Loại bỏ các nhãn kỹ thuật chung theo quy luật độ dài (<= 3 ký tự)
                    if len(label) <= 3:
                        continue
                        
                    X_sub = vectorizer.transform([label])
                    sub_distances, sub_indices = knn.kneighbors(X_sub, n_neighbors=1)
                    sub_matched_domain = domains[sub_indices[0][0]]
                    sub_matched_brand = sub_matched_domain.split('.')[0]
                    
                    # huynq - Kiểm tra trùng khớp 100% với nhãn thương hiệu uy tín phổ biến
                    if label == sub_matched_brand and sub_matched_brand in clean_brands:
                        triggered_sub_brand = label
                        triggered_matched_domain = sub_matched_domain
                        break
                
                # huynq - 2.5: Nếu subdomain không khớp thương hiệu uy tín, nhưng domain chính có dấu '-'
                if not triggered_sub_brand:
                    reg_domain_brand = reg_domain.split('.')[0]
                    domain_words = reg_domain_brand.split('-')
                    if len(domain_words) >= 2:
                        for word in domain_words:
                            if len(word) <= 3:
                                continue
                            
                            X_word = vectorizer.transform([word])
                            word_distances, word_indices = knn.kneighbors(X_word, n_neighbors=1)
                            word_matched_domain = domains[word_indices[0][0]]
                            word_matched_brand = word_matched_domain.split('.')[0]
                            
                            if word == word_matched_brand and word_matched_brand in clean_brands:
                                triggered_sub_brand = word
                                triggered_matched_domain = word_matched_domain
                                break
                
                if triggered_sub_brand:
                    print(f"So khớp từ khóa thương hiệu: {triggered_sub_brand} ➔ {triggered_matched_domain}")
                    print(f"Cảnh báo: website không thuộc thương hiệu uy tín nhưng lại đang cố tình chèn thương hiệu đó vào đường dẫn")
                else:
                    # huynq - 3: Kiểm tra tỷ lệ tương đồng
                    if lev_score >= 80.0:
                        print(f" Tên miền chính thống giống nhất: {matched_domain} (Độ tương đồng: {lev_score:.2f}%)")
                        print(f" Cảnh báo: không nằm trong danh sách đơn vị uy tín đã được xác thực nhưng lại quá giống đơn vị đó")
                    else:
                        print(f"Tên miền chính thống giống nhất: {matched_domain} (Độ tương đồng: {lev_score:.2f}%)")
                        print(f" Trạng thái: An toàn. Không phát hiện hành vi mạo danh rõ rệt.")
                    
            query_time_ms = (time.time() - t_query_start) * 1000.0
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f" Đã xảy ra lỗi: {e}")

if __name__ == '__main__':
    main()
