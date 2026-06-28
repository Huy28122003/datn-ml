import os
import sys
import re
import json
import pickle
import argparse
import warnings
import urllib.request
import urllib.parse
import numpy as np

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, 'lib'))

OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'rf_vs_hybrid')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'rf_phishing_model.pkl')


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Loi: Thieu model!")
        sys.exit(1)

    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    features = model_data['features']
    params = model_data['params']

    return model, features, params


# huyny - whitelist domain
REPUTABLE_DOMAINS = {
    'google.com', 'google.com.vn', 'youtube.com', 'facebook.com', 'instagram.com', 
    'twitter.com', 'linkedin.com', 'github.com', 'gitlab.com', 'microsoft.com', 
    'apple.com', 'amazon.com', 'netflix.com', 'wikipedia.org', 'w3schools.com', 
    'stackoverflow.com', 'stackexchange.com', 'medium.com', 'docker.com', 'docker.io', 
    'kubernetes.io', 'python.org', 'npmjs.com', 'cloudflare.com', 'mozilla.org', 
    'apache.org', 'spring.io', 'oracle.com', 'git-scm.com', 'bitbucket.org'
}


def get_registered_domain(domain):
    if not domain:
        return ""
    domain = domain.lower()
    parts = domain.split('.')
    if len(parts) >= 2:
        if len(parts) >= 3 and parts[-2] in ('co', 'com', 'org', 'net', 'edu', 'gov'):
            return '.'.join(parts[-3:])
        return '.'.join(parts[-2:])
    return domain


def extract_hybrid_features(url):
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

    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        if ':' in domain:
            domain = domain.split(':')[0]
            
        feats['DomainLength'] = len(domain)
        
        ip_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
        feats['IsDomainIP'] = 1.0 if ip_pattern.match(domain) else 0.0
        
        dots = domain.count('.')
        feats['NoOfSubDomain'] = float(max(0, dots - 1))
        
        letters = sum(1 for c in url if c.isalpha())
        digits = sum(1 for c in url if c.isdigit())
        feats['NoOfLettersInURL'] = float(letters)
        feats['LetterRatioInURL'] = letters / len(url) if len(url) > 0 else 0.5
        feats['NoOfDegitsInURL'] = float(digits)
        feats['DegitRatioInURL'] = digits / len(url) if len(url) > 0 else 0.0
        
        specials = sum(1 for c in url if not c.isalnum() and c not in ('/', '.', ':', '-'))
        feats['NoOfOtherSpecialCharsInURL'] = float(specials)
        feats['SpacialCharRatioInURL'] = specials / len(url) if len(url) > 0 else 0.0
        
    except Exception:
        pass

    # huyny - cao html
    print("Check online & cao HTML...")
    is_online = False
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=3.0) as response:
            html = response.read().decode('utf-8', errors='ignore')
            print("Scrape ok, dang parse...")
            is_online = True
            
            lines = html.split('\n')
            feats['LineOfCode'] = float(len(lines))
            feats['LargestLineLength'] = float(max(len(line) for line in lines) if lines else 0)
            
            feats['NoOfImage'] = float(len(re.findall(r'<img\s+', html, re.I)))
            feats['NoOfJS'] = float(len(re.findall(r'<script\s+', html, re.I)))
            feats['NoOfCSS'] = float(len(re.findall(r'<link\s+[^>]*rel=["\']stylesheet["\']|<style\s+', html, re.I)))
            
            all_links = re.findall(r'href=["\'](https?://[^"\']+|/[^"\']+|#[^"\']*)["\']', html, re.I)
            external = 0
            self_ref = 0
            empty = 0
            
            domain_escaped = re.escape(domain) if 'domain' in locals() else ''
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
            
            feats['HasDescription'] = 1.0 if re.search(r'<meta\s+name=["\']description["\']', html, re.I) else 0.0
            feats['HasFavicon'] = 1.0 if re.search(r'rel=["\'](icon|shortcut icon)["\']', html, re.I) else 0.0
            feats['IsResponsive'] = 1.0 if re.search(r'name=["\']viewport["\']', html, re.I) else 0.0
            feats['HasSocialNet'] = 1.0 if any(s in html for s in ['facebook.com', 'twitter.com', 'instagram.com']) else 0.0
            feats['HasSubmitButton'] = 1.0 if re.search(r'<input\s+[^>]*type=["\']submit["\']|<button', html, re.I) else 0.0
            feats['HasHiddenFields'] = 1.0 if re.search(r'type=["\']hidden["\']', html, re.I) else 0.0
            feats['HasCopyrightInfo'] = 1.0 if any(c in html.lower() for c in ['copyright', '©', '&copy;']) else 0.0
            
    except Exception as e:
        print(f"Loi HTML ({e}). Fallback offline.")

    return feats, is_online


def predict_url(url, model, model_features):
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc
        if ':' in domain:
            domain = domain.split(':')[0]
    except Exception:
        domain = ""
        
    reg_domain = get_registered_domain(domain)
    is_whitelisted = reg_domain in REPUTABLE_DOMAINS

    if is_whitelisted:
        prediction = 0
        probabilities = [1.0, 0.0]
        label = 'LEGITIMATE (Whitelisted)'
        feats_all, is_online = extract_hybrid_features(url)
    else:
        feats_all, is_online = extract_hybrid_features(url)
        feature_vector = [feats_all[name] for name in model_features]
        
        X = np.array([feature_vector])
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        label = 'PHISHING' if prediction == 1 else 'LEGITIMATE'

    result = {
        'url': url,
        'prediction': int(prediction),
        'label': label,
        'probability_legitimate': float(probabilities[0]),
        'probability_phishing': float(probabilities[1]),
        'confidence': float(max(probabilities)),
        'features_extracted': {
            name: feats_all[name] for name in model_features
        },
        'features_used_count': len(model_features),
        'is_whitelisted': is_whitelisted,
        'is_online': is_online
    }

    return result


def display_result(result):
    print("\nKET QUA PHAN TICH URL:")
    print(f"URL: {result['url']}")

    if not result.get('is_online', True):
        print("TRANG THAI: OFFLINE")
        return

    print(f"KET QUA: {result['label']}")
    if result['prediction'] == 1:
        print(f"Xac suat Phishing:   {result['probability_phishing']:.2%}")
    else:
        print(f"Xac suat An toan:    {result['probability_legitimate']:.2%}")

    print(f"Do tin cay:          {result['confidence']:.2%}")
    print(f"So dac trung su dung: {result['features_used_count']}")

    print("\nCHI TIET CAC DAC TRUNG:")
    features = result['features_extracted']
    for i, (name, val) in enumerate(features.items(), 1):
        if isinstance(val, float):
            print(f"  {i:>2}. {name:<35} = {val:.4f}")
        else:
            print(f"  {i:>2}. {name:<35} = {val}")


def interactive_mode(model, model_features):
    print("\nRANDOM FOREST - HYBRID PHISHING DETECTOR")
    print("Nhap 'quit' de thoat")

    while True:
        url = input("\nNhap URL: ").strip()

        if url.lower() in ('quit', 'exit', 'q'):
            break

        if not url:
            continue

        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        try:
            result = predict_url(url, model, model_features)
            display_result(result)
        except Exception as e:
            print(f"Loi: {e}")


def main():
    parser = argparse.ArgumentParser(description='Random Forest Hybrid URL Phishing Detector')
    parser.add_argument('--url', type=str, help='Đường dẫn URL cần kiểm tra')
    args = parser.parse_args()

    print("Load model...")
    model, model_features, params = load_model()
    print(f"Loaded model ({len(model_features)} features)")

    if args.url:
        result = predict_url(args.url, model, model_features)
        display_result(result)
    else:
        interactive_mode(model, model_features)


if __name__ == '__main__':
    main()
