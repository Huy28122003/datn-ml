# Chuong trinh chinh noi ket noi model va phan tich dau vao

import os
import sys
import pickle
import urllib.parse

# huynq - Cau hinh duong dan
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'output', 'knn_search', 'knn_model.pkl')

# huynq - Danh sach so khop cac hau to ccTLD quoc gia pho bien tren the gioi
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
        # huynq - So khop danh sach ccTLD chuan
        if last_2 in PUBLIC_SUFFIXES:
            suffix_len = 2
        # huynq - Quy luat thuat toan ccTLD quoc gia du phong
        # ten mien kep bat buoc deu co 2 chu cai
        elif len(labels) >= 3:
            last_label = labels[-1]
            prev_label = labels[-2]
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
    if not os.path.exists(MODEL_PATH):
        print(f" Chua tim thay mo hinh tai {MODEL_PATH}")
        sys.exit(1)

    input_url = ""
    if len(sys.argv) > 1:
        input_url = " ".join(sys.argv[1:])
    else:
        try:
            input_url = input("Hay paste URL can kiem tra vao day: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n Da huy thao tac.")
            sys.exit(0)

    if not input_url:
        print("Loi: Khong nhan duoc URL dau vao.")
        sys.exit(1)

    reg_domain, sub_labels = extract_domain_parts(input_url)
    if not reg_domain:
        print("Loi: Khong the trich xuat ten mien.")
        sys.exit(1)

    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
        
    vectorizer = model_data['vectorizer']
    knn = model_data['knn']
    domains = model_data['domains']

    # huynq - Lay thuong hieu uy tin tu toan bo 1 trieu ten mien sach
    clean_brands = {d.split('.')[0] for d in domains if len(d.split('.')[0]) > 3}

    X = vectorizer.transform([reg_domain])
    distances, indices = knn.kneighbors(X, n_neighbors=1)
    matched_domain = domains[indices[0][0]]

    # huynq - Tinh Levenshtein
    lev_score = levenshtein_similarity(reg_domain, matched_domain) * 100.0

    print("\n" + "=" * 55)
    print(f"URL dau vao:         {input_url}")
    print(f"Domain chinh:         {reg_domain}")
    if sub_labels:
        print(f"Nhan phu:             {sub_labels}")
    print("-" * 55)
    print(" KET QUA DANH GIA:")
    
    if lev_score == 100.0:
        print(f"  Ten mien chinh thong khop: {matched_domain}")
        print("  Trang thai: day la website cua 1 don vi uy tin hoac 1 thanh phan thuoc don vi uy tin")
    else:
        triggered_sub_brand = None
        triggered_matched_domain = None
        
        for label in sub_labels:
            # huynq - Quy luat do dai de bo qua nhan ky thuat ngan
            if len(label) <= 3:
                continue
                
            X_sub = vectorizer.transform([label])
            sub_distances, sub_indices = knn.kneighbors(X_sub, n_neighbors=1)
            sub_matched_domain = domains[sub_indices[0][0]]
            sub_matched_brand = sub_matched_domain.split('.')[0]
            
            if label == sub_matched_brand and sub_matched_brand in clean_brands:
                triggered_sub_brand = label
                triggered_matched_domain = sub_matched_domain
                break
                
        # huynq - Quy tac 2.5: Phan tich tu khoa gach ngang trong domain chinh
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
            print(f"  So khop tu khoa thuong hieu: {triggered_sub_brand} ➔ {triggered_matched_domain}")
            print("   Canh bao: website khong thuoc thuong hieu uy tin nhung lai dang co tinh chen thuong hieu do vao duong dan")
        else:
            if lev_score >= 80.0:
                print(f"   Ten mien chinh thong giong nhat: {matched_domain} (Do tuong dong: {lev_score:.2f}%)")
                print("   Canh bao: khong nam trong danh sach don vi uy tin da duoc xac thuc nhung lai qua giong don vi do")
            else:
                print(f"   Ten mien chinh thong giong nhat: {matched_domain} (Do tuong dong: {lev_score:.2f}%)")
                print("   Trang thai: An toan. Khong phat hien hanh vi mao danh ro ret.")
            
    print("=" * 55 + "\n")

if __name__ == '__main__':
    main()
