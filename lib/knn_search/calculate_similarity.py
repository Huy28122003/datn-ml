# Thử nghiệm thuật toán Heuristic khác nhau (Jaro-Winkler, Levenshtein, và Sub-string).
 #Nó dùng để so sánh xem 2 text giống nhau bao nhiêu %
# file này được thử nghiệm bởi AI để tìm ra thuật toàn thích hợp
# huynq

import sys

def jaro_winkler_similarity(s1, s2):
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    max_dist = max(len1, len2) // 2 - 1
    if max_dist < 0:
        max_dist = 0

    hash_s1 = [0] * len1
    hash_s2 = [0] * len2
    match = 0

    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(len2, i + max_dist + 1)
        for j in range(start, end):
            if s1[i] == s2[j] and hash_s2[j] == 0:
                hash_s1[i] = 1
                hash_s2[j] = 1
                match += 1
                break

    if match == 0:
        return 0.0

    t = 0
    point = 0
    for i in range(len1):
        if hash_s1[i]:
            while hash_s2[point] == 0:
                point += 1
            if s1[i] != s2[point]:
                t += 1
            point += 1
    t /= 2

    jaro = (match / len1 + match / len2 + (match - t) / match) / 3.0
    
    prefix_limit = min(4, min(len1, len2))
    prefix = 0
    for i in range(prefix_limit):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
            
    return jaro + prefix * 0.1 * (1.0 - jaro)

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

def clean_domain(domain):
    domain = domain.strip().lower()
    if domain.startswith(('http://', 'https://')):
        # huynq - Tách lấy netloc sơ bộ
        from urllib.parse import urlparse
        try:
            domain = urlparse(domain).netloc
        except Exception:
            pass
    if domain.startswith('www.'):
        domain = domain[4:]
    if ':' in domain:
        domain = domain.split(':')[0]
    return domain

def main():
    # huynq - Cho phép nhập qua dòng lệnh hoặc hỏi trực tiếp
    if len(sys.argv) >= 3:
        dom_legit = sys.argv[1]
        dom_user = sys.argv[2]
    else:
        dom_legit = input("1. Nhập tên miền chính thống (ví dụ: business.com): ").strip()
        dom_user = input("2. Nhập URL/Domain cần đối chiếu: ").strip()

    if not dom_legit or not dom_user:
        print("❌ LỖI: Hãy nhập đầy đủ cả hai tên miền!")
        return

    # huynq - Làm sạch dữ liệu
    clean_legit = clean_domain(dom_legit)
    clean_user = clean_domain(dom_user)

    print("-" * 65)
    print(f"Domain chính thống (đã làm sạch): {clean_legit}")
    print(f"Domain người dùng (đã làm sạch):   {clean_user}")
    print("-" * 65)

    # huynq - 1. Tính Jaro-Winkler
    jw_score = jaro_winkler_similarity(clean_legit, clean_user) * 100.0

    # huynq - 2. Tính Levenshtein
    lev_score = levenshtein_similarity(clean_legit, clean_user) * 100.0

    # huynq - 3. Kiểm tra Combosquatting (tên miền phụ hoặc tên miền gốc chứa hoàn toàn brand name chính thống)
    # huynq - Lấy thương hiệu ví dụ "business" từ "business.com"
    brand_name = clean_legit.split('.')[0] 
    contains_brand = brand_name in clean_user
    
    print("📊 KẾT QUẢ ĐO LƯỜNG HEURISTIC:")
    print(f"🟢 Jaro-Winkler Similarity:   {jw_score:.2f}%")
    print(f"🟢 Levenshtein Similarity:    {lev_score:.2f}%")
    print(f"🟢 Chứa thương hiệu gốc ({brand_name}): {'CÓ (Combosquatting)' if contains_brand else 'KHÔNG'}")
    print("-" * 65)
    
    print("💡 NHẬN XÉT HỌC THUẬT:")
    if contains_brand and lev_score < 40.0:
        print("➔ Đây là kiểu tấn công COMBOSQUATTING. Kẻ xấu chèn nguyên chữ")
        print(f"  '{brand_name}' vào một tên miền phụ rất dài để đánh lừa thị giác.")
        print("  Gợi ý: Jaro-Winkler phù hợp phát hiện dạng này hơn Levenshtein.")
    elif lev_score > 75.0:
        print("➔ Đây là kiểu tấn công TYPOSQUATTING (gõ sai). Tên miền cực kỳ")
        print("  giống nhau về mặt ký tự và chiều dài.")
        print("  Gợi ý: Cả Jaro-Winkler và Levenshtein đều hoạt động rất tốt.")
    else:
        print("➔ Mức độ tương đồng ký tự thấp. Hai tên miền này ít có khả năng giả mạo trực tiếp.")
    print("=================================================================")

if __name__ == '__main__':
    main()
