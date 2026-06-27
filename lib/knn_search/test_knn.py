"""
test_knn.py
===========
Kiểm thử mô hình KNN tương tác kết hợp đánh giá Heuristic theo luồng phân cấp:
1. Tách Registered Domain (Domain chính) và Subdomain.
2. Kiểm tra Domain chính qua KNN. Nếu khớp 100% với tên miền chính thống -> Xử lý an toàn.
3. Nếu Domain chính không khớp 100%, tách toàn bộ các từ của subdomain bằng dấu chấm ".",
   sau đó truy vấn KNN từng từ xem có trùng khớp 100% với thương hiệu lớn nào không.
"""

import os
import sys
import time
import pickle
import urllib.parse

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'output', 'knn_search', 'knn_model.pkl')

# Danh sách các Public Suffixes đuôi kép phổ biến
PUBLIC_SUFFIXES = {
    'com.vn', 'co.uk', 'com.br', 'com.cn', 'com.tr', 'com.mu', 'com.ug', 'com.bi', 'com.py', 'com.gr', 'com.et', 'com.bn', 'net.cn', 'gov.tr', 'gov.vn', 'org.vn', 'my.id', 'ac.uk',
    'pages.dev', 'github.io', 'vercel.app', 'systeme.io', 'replit.app', 'edgeone.app', 'workers.dev', 'r2.dev', 'framer.website', 'backblazeb2.com', 'webflow.io', 'cpanel.site', 'temporary.site', 'typedream.app', 'framer.ai', 'blogspot.com'
}

# Bộ lọc các từ khóa kỹ thuật chung để tránh cảnh báo sai (False Positives)
GENERIC_SUBDOMAINS = {
    'com', 'net', 'org', 'gov', 'edu', 'pages', 'vercel', 'workers', 'api', 'app', 'cdn', 
    'dev', 'mail', 'www', 'web', 'login', 'admin', 'portal', 'secure', 'support', 'status', 
    'info', 'service', 'blog', 'shop', 'test', 'demo', 'doc', 'docs', 'files', 'account', 
    'accounts', 'user', 'users', 'client', 'clients', 'server', 'servers', 'update', 'updates', 
    'security', 'download', 'downloads', 'git', 'auth', 'media', 'static', 'images', 'image', 
    'video', 'videos', 'chat', 'help', 'jobs', 'careers', 'about', 'contact', 'privacy', 'terms', 
    'legal', 'm', 'mobile', 'sys', 'system'
}

def extract_domain_parts(url_str):
    """Phân tích URL thành (registered_domain, subdomain_labels)"""
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

    suffix_found = ""
    suffix_len = 1
    
    if len(labels) >= 2:
        last_2 = ".".join(labels[-2:])
        if last_2 in PUBLIC_SUFFIXES:
            suffix_found = last_2
            suffix_len = 2

    if not suffix_found:
        suffix_found = labels[-1]
        suffix_len = 1

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

def get_levenshtein_level(score):
    if score < 30.0:
        return "🟢 XANH (Thấp)"
    elif score <= 50.0:
        return "🟡 VÀNG (Trung bình)"
    else:
        return "🔴 ĐỎ (Cao)"

def main():
    print("================================================================")
    print("🔍 HỆ THỐNG TRUY VẤN KNN & PHÂN TÍCH HÀ HÀNG LOẠT SUBDOMAINS")
    print("================================================================")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ LỖI: Chưa tìm thấy mô hình tại {MODEL_PATH}")
        sys.exit(1)
        
    print("📖 Đang nạp mô hình KNN và cơ sở dữ liệu tên miền...")
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
        
    vectorizer = model_data['vectorizer']
    knn = model_data['knn']
    domains = model_data['domains']
    print(f"✓ Nạp thành công mô hình. Tổng số tên miền: {len(domains):,}")
    print("-" * 64)
    print("💡 Hướng dẫn: Paste một URL đầy đủ hoặc tên miền cần kiểm tra.")
    print("💡 Nhập 'exit' hoặc 'quit' để dừng.")
    print("-" * 64)
    
    while True:
        try:
            user_input = input("\n📥 Nhập URL / Domain cần kiểm tra: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ('exit', 'quit'):
                print("👋 Tạm biệt!")
                break
                
            t_query_start = time.time()
            
            # 1. Phân tích Registered Domain và các nhãn Subdomain
            reg_domain, sub_labels = extract_domain_parts(user_input)
            if not reg_domain:
                print("⚠️ Lỗi: Không thể trích xuất tên miền.")
                continue
                
            print(f"🌐 Domain chính (Registered Domain): '{reg_domain}'")
            if sub_labels:
                print(f"🏷️ Subdomains phân tích: {sub_labels}")
                
            # 2. Đưa domain chính vào KNN
            X_input = vectorizer.transform([reg_domain])
            distances, indices = knn.kneighbors(X_input, n_neighbors=1)
            matched_domain = domains[indices[0][0]]
            
            # Tính độ tương đồng Levenshtein của Domain chính
            lev_score = levenshtein_similarity(reg_domain, matched_domain) * 100.0
            
            # 3. Phán quyết phân cấp
            print("-" * 55)
            if lev_score == 100.0:
                print(f"🥇 Tên miền chính thống giống nhất: {matched_domain}")
                print("-" * 55)
                print("📊 KẾT QUẢ ĐÁNH GIÁ:")
                print(f"  🟢 Tên miền chính thống hợp lệ (100% Khớp).")
                print(f"  🟢 Trạng thái: An toàn / Bỏ qua cảnh báo.")
            else:
                lev_level = get_levenshtein_level(lev_score)
                
                # Quét từng nhãn subdomain qua KNN để tìm thương hiệu lớn khớp 100%
                triggered_sub_brand = None
                triggered_matched_domain = None
                
                for label in sub_labels:
                    if label in GENERIC_SUBDOMAINS or len(label) <= 2:
                        continue
                        
                    # Truy vấn KNN cho nhãn subdomain này
                    X_sub = vectorizer.transform([label])
                    sub_distances, sub_indices = knn.kneighbors(X_sub, n_neighbors=1)
                    sub_match_idx = sub_indices[0][0]
                    sub_matched_domain = domains[sub_match_idx]
                    
                    # Lấy nhãn thương hiệu của tên miền khớp
                    sub_matched_brand = sub_matched_domain.split('.')[0]
                    
                    # So khớp 100% với nhãn thương hiệu của tên miền chính thống phổ biến (Top 100k)
                    if label == sub_matched_brand and sub_match_idx <= 100000:
                        triggered_sub_brand = label
                        triggered_matched_domain = sub_matched_domain
                        break
                        
                # Kiểm tra Combosquatting ở domain chính
                brand_name = matched_domain.split('.')[0]
                contains_brand = brand_name in reg_domain
                brand_status = f"🔴 CÓ (Chứa thương hiệu '{brand_name}')" if contains_brand else "🟢 KHÔNG"
                
                print(f"🥇 Tên miền chính thống giống nhất: {matched_domain}")
                print("-" * 55)
                print("📊 KẾT QUẢ ĐÁNH GIÁ:")
                print(f"  ⚡ Levenshtein Similarity (Domain chính): {lev_score:.2f}% -> {lev_level}")
                print(f"  ⚡ Domain chính chứa thương hiệu gốc:      {brand_status}")
                
                if triggered_sub_brand:
                    print(f"  🚨 CẢNH BÁO: Subdomain chứa thương hiệu chính thức: {triggered_sub_brand.upper()} (Khớp 100% với {triggered_matched_domain})")
                else:
                    print("  🟢 Subdomain an toàn: Không phát hiện thương hiệu bị mạo danh.")
                    
            query_time_ms = (time.time() - t_query_start) * 1000.0
            print(f"⚡ Thời gian xử lý:                      {query_time_ms:.2f} ms")
            print("-" * 55)
            
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"⚠️ Đã xảy ra lỗi: {e}")

if __name__ == '__main__':
    main()
