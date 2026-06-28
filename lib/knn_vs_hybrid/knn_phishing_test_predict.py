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
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, 'lib'))
MODEL_PATH = os.path.join(BASE_DIR, 'output', 'knn_vs_hybrid', 'knn_model.pkl')
DATASET_PATH = os.path.join(BASE_DIR, 'data_set', 'hybrid.csv')

def load_model():
    if not os.path.exists(MODEL_PATH):
        sys.exit(1)
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def get_registered_domain(domain):
    if not domain:
        return ''
    domain = domain.lower()
    parts = domain.split('.')
    if len(parts) >= 2:
        if len(parts) >= 3 and parts[-2] in ('co', 'com', 'org', 'net', 'edu', 'gov'):
            return '.'.join(parts[-3:])
        return '.'.join(parts[-2:])
    return domain

def get_tld(domain):
    if not domain:
        return ''
    parts = domain.split('.')
    if len(parts) >= 2:
        return parts[-1]
    return ''

def compute_token_match_score(text, title):
    if not text or not title:
        return 0.0
    text_tokens = set(re.findall('\\w+', text.lower()))
    title_tokens = set(re.findall('\\w+', title.lower()))
    text_tokens = {t for t in text_tokens if len(t) > 2}
    if not text_tokens:
        return 0.0
    matches = text_tokens.intersection(title_tokens)
    return float(len(matches) / len(text_tokens) * 100.0)

def extract_hybrid_features(url):
    feats = {'URLLength': len(url), 'DomainLength': 20.0, 'IsDomainIP': 0.0, 'URLSimilarityIndex': 100.0, 'CharContinuationRate': 1.0, 'TLDLegitimateProb': 0.08, 'URLCharProb': 0.058, 'TLDLength': 3.0, 'NoOfSubDomain': 1.0, 'HasObfuscation': 0.0, 'NoOfObfuscatedChar': 0.0, 'ObfuscationRatio': 0.0, 'NoOfLettersInURL': 14.0, 'LetterRatioInURL': 0.519, 'NoOfDegitsInURL': 0.0, 'DegitRatioInURL': 0.0, 'NoOfEqualsInURL': 0.0, 'NoOfQMarkInURL': 0.0, 'NoOfAmpersandInURL': 0.0, 'NoOfOtherSpecialCharsInURL': 1.0, 'SpacialCharRatioInURL': 0.05, 'IsHTTPS': 1.0 if url.startswith('https') else 0.0, 'LineOfCode': 429.0, 'LargestLineLength': 1090.0, 'HasTitle': 1.0, 'DomainTitleMatchScore': 75.0, 'URLTitleMatchScore': 100.0, 'HasFavicon': 0.0, 'Robots': 0.0, 'IsResponsive': 1.0, 'NoOfURLRedirect': 0.0, 'NoOfSelfRedirect': 0.0, 'HasDescription': 0.0, 'NoOfPopup': 0.0, 'NoOfiFrame': 0.0, 'HasExternalFormSubmit': 0.0, 'HasSocialNet': 0.0, 'HasSubmitButton': 0.0, 'HasHiddenFields': 0.0, 'HasPasswordField': 0.0, 'Bank': 0.0, 'Pay': 0.0, 'Crypto': 0.0, 'HasCopyrightInfo': 0.0, 'NoOfImage': 8.0, 'NoOfCSS': 2.0, 'NoOfJS': 6.0, 'NoOfSelfRef': 12.0, 'NoOfEmptyRef': 0.0, 'NoOfExternalRef': 10.0}
    domain = ''
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        if ':' in domain:
            domain = domain.split(':')[0]
        feats['DomainLength'] = float(len(domain))
        ip_pattern = re.compile('^\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}$')
        feats['IsDomainIP'] = 1.0 if ip_pattern.match(domain) else 0.0
        dots = domain.count('.')
        feats['NoOfSubDomain'] = float(max(0, dots - 1))
        letters = sum((1 for c in url if c.isalpha()))
        digits = sum((1 for c in url if c.isdigit()))
        feats['NoOfLettersInURL'] = float(letters)
        feats['LetterRatioInURL'] = letters / len(url) if len(url) > 0 else 0.5
        feats['NoOfDegitsInURL'] = float(digits)
        feats['DegitRatioInURL'] = digits / len(url) if len(url) > 0 else 0.0
        feats['NoOfEqualsInURL'] = float(url.count('='))
        feats['NoOfQMarkInURL'] = float(url.count('?'))
        feats['NoOfAmpersandInURL'] = float(url.count('&'))
        specials = sum((1 for c in url if not c.isalnum() and c not in ('/', '.', ':', '-')))
        feats['NoOfOtherSpecialCharsInURL'] = float(specials)
        feats['SpacialCharRatioInURL'] = specials / len(url) if len(url) > 0 else 0.0
        tld = get_tld(domain)
        feats['TLDLength'] = float(len(tld))
    except Exception:
        pass
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        with urllib.request.urlopen(req, timeout=4.0) as response:
            html = response.read().decode('utf-8', errors='ignore')
            lines = html.split('\n')
            feats['LineOfCode'] = float(len(lines))
            feats['LargestLineLength'] = float(max((len(line) for line in lines)) if lines else 0)
            feats['NoOfImage'] = float(len(re.findall('<img\\s+', html, re.I)))
            feats['NoOfJS'] = float(len(re.findall('<script\\s+', html, re.I)))
            feats['NoOfCSS'] = float(len(re.findall('<link\\s+[^>]*rel=["\\\']stylesheet["\\\']|<style\\s+', html, re.I)))
            feats['HasSubmitButton'] = 1.0 if re.search('<input\\s+[^>]*type=["\\\']submit["\\\']|<button', html, re.I) else 0.0
            feats['HasHiddenFields'] = 1.0 if re.search('type=["\\\']hidden["\\\']', html, re.I) else 0.0
            feats['HasPasswordField'] = 1.0 if re.search('type=["\\\']password["\\\']', html, re.I) else 0.0
            feats['HasExternalFormSubmit'] = 1.0 if re.search('<form\\s+[^>]*action=["\\\']https?://', html, re.I) else 0.0
            bank_keywords = ['bank', 'ebank', 'banking', 'vietcombank', 'techcombank', 'acb', 'bidv', 'agribank', 'sacombank']
            pay_keywords = ['pay', 'payment', 'paypal', 'momo', 'zalopay', 'checkout', 'card', 'visa', 'mastercard']
            crypto_keywords = ['crypto', 'bitcoin', 'btc', 'eth', 'binance', 'wallet', 'blockchain', 'trustwallet']
            html_lower = html.lower()
            feats['Bank'] = 1.0 if any((k in html_lower for k in bank_keywords)) else 0.0
            feats['Pay'] = 1.0 if any((k in html_lower for k in pay_keywords)) else 0.0
            feats['Crypto'] = 1.0 if any((k in html_lower for k in crypto_keywords)) else 0.0
            all_links = re.findall('href=["\\\'](https?://[^"\\\']+|/[^"\\\']+|#[^"\\\']*)["\\\']', html, re.I)
            external = 0
            self_ref = 0
            empty = 0
            domain_escaped = re.escape(domain) if domain else ''
            for link in all_links:
                if link.startswith('#') or link == '':
                    empty += 1
                elif link.startswith('/') or (domain_escaped and re.search(domain_escaped, link)):
                    self_ref += 1
                else:
                    external += 1
            feats['NoOfExternalRef'] = float(external)
            feats['NoOfSelfRef'] = float(self_ref)
            feats['NoOfEmptyRef'] = float(empty)
            feats['HasDescription'] = 1.0 if re.search('<meta\\s+name=["\\\']description["\\\']', html, re.I) else 0.0
            feats['HasFavicon'] = 1.0 if re.search('rel=["\\\'](icon|shortcut icon)["\\\']', html, re.I) else 0.0
            feats['IsResponsive'] = 1.0 if re.search('name=["\\\']viewport["\\\']', html, re.I) else 0.0
            feats['HasSocialNet'] = 1.0 if any((s in html for s in ['facebook.com', 'twitter.com', 'instagram.com'])) else 0.0
            feats['HasCopyrightInfo'] = 1.0 if any((c in html_lower for c in ['copyright', '©', '&copy;'])) else 0.0
            title_match = re.search('<title>(.*?)</title>', html, re.I | re.S)
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
    return feats

def print_comparison_table(input_feats, match_feats):
    feature_groups = {'1. Đặc trưng Lexical (URL tĩnh)': ['URLLength', 'DomainLength', 'IsDomainIP', 'TLDLength', 'NoOfSubDomain', 'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL', 'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL', 'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS'], '2. Đặc trưng cấu trúc HTML (Động)': ['LineOfCode', 'LargestLineLength', 'NoOfImage', 'NoOfJS', 'NoOfCSS', 'HasSubmitButton', 'HasHiddenFields', 'HasPasswordField', 'HasExternalFormSubmit'], '3. Đặc trưng tham chiếu (Links)': ['NoOfExternalRef', 'NoOfSelfRef', 'NoOfEmptyRef'], '4. Phân tích nội dung và ngữ cảnh': ['HasTitle', 'DomainTitleMatchScore', 'URLTitleMatchScore', 'HasFavicon', 'HasDescription', 'HasSocialNet', 'HasCopyrightInfo', 'IsResponsive', 'Bank', 'Pay', 'Crypto'], '5. Đặc trưng bổ sung (Phân tích nâng cao)': ['URLSimilarityIndex', 'CharContinuationRate', 'TLDLegitimateProb', 'URLCharProb', 'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio', 'NoOfURLRedirect', 'NoOfSelfRedirect', 'NoOfPopup', 'NoOfiFrame', 'Robots']}
    for (group_name, features) in feature_groups.items():
        for name in features:
            val_input = input_feats.get(name, 0.0)
            val_match = match_feats.get(name, 0.0)
            is_match = abs(val_input - val_match) < 0.0001
            match_status = '✅ YES' if is_match else '❌ NO'
            str_input = f'{val_input:.4f}' if isinstance(val_input, float) and (not val_input.is_integer()) else f'{int(val_input)}'
            str_match = f'{val_match:.4f}' if isinstance(val_match, float) and (not val_match.is_integer()) else f'{int(val_match)}'

def main():
    parser = argparse.ArgumentParser(description='KNN Hybrid URL Similarity Matcher')
    parser.add_argument('--url', type=str, help='Đường dẫn URL cần kiểm tra')
    parser.add_argument('-k', type=int, default=1, help='Số lượng bản ghi tương đồng cần tìm (mặc định: 1)')
    args = parser.parse_args()
    model_data = load_model()
    knn = model_data['knn']
    scaler = model_data['scaler']
    features_list = model_data['features']
    if not os.path.exists(DATASET_PATH):
        sys.exit(1)
    meta_cols = ['FILENAME', 'URL', 'Domain', 'TLD', 'Title', 'label']
    df_meta = pd.read_csv(DATASET_PATH, usecols=meta_cols)
    if args.url:
        urls_to_test = [args.url]
    else:
        urls_to_test = []
    interactive = len(urls_to_test) == 0
    while True:
        if interactive:
            url_input = input('\n🔗 Nhập URL để tìm láng giềng gần nhất: ').strip()
            if url_input.lower() in ('quit', 'exit', 'q'):
                break
            if not url_input:
                continue
        else:
            if not urls_to_test:
                break
            url_input = urls_to_test.pop(0)
        if not url_input.startswith(('http://', 'https://')):
            url_input = 'https://' + url_input
        try:
            input_feats_all = extract_hybrid_features(url_input)
            input_vector = [input_feats_all[name] for name in features_list]
            input_scaled = scaler.transform([input_vector])
            (distances, indices) = knn.kneighbors(input_scaled, n_neighbors=args.k)
            for (rank, (dist, idx)) in enumerate(zip(distances[0], indices[0]), 1):
                matched_row = df_meta.iloc[idx]
                label_str = '🚨 PHISHING' if matched_row['label'] == 1 else '✅ LEGITIMATE'
            if args.k >= 1:
                first_match_idx = indices[0][0]
                df_full_match = pd.read_csv(DATASET_PATH, skiprows=first_match_idx + 1, nrows=1, header=None)
                headers = pd.read_csv(DATASET_PATH, nrows=0).columns.tolist()
                df_full_match.columns = headers
                match_row_full = df_full_match.iloc[0]
                match_feats_dict = {name: float(match_row_full[name]) for name in features_list}
                print_comparison_table(input_feats_all, match_feats_dict)
        except Exception as e:
            import traceback
            traceback.print_exc()
        if not interactive:
            break
if __name__ == '__main__':
    main()
