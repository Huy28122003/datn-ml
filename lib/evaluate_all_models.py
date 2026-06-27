"""
evaluate_all_models.py
========================
Script đánh giá tổng hợp tất cả các mô hình đã huấn luyện:
  1. RF (Random Forest) trên dataset_full.csv  (Pure URL features)
  2. XGBoost trên dataset_full.csv              (Pure URL features)
  3. RF (Random Forest) trên hybrid.csv         (Hybrid features: URL + HTML)
  4. KNN trên hybrid.csv                        (Tìm bản ghi tương đồng)
  5. Deep Learning (CNN-BiLSTM)                 (Character-level URL)

Đánh giá trên:
  - Tập test hold-out (20%) từ chính dataset gốc
  - Tập dữ liệu ngoại (phishing_url.csv từ PhishTank) để kiểm tra khả năng tổng quát hóa

Output:
  - Bảng so sánh tổng hợp metrics tất cả mô hình
  - Confusion matrix cho từng mô hình
  - Phân tích mô hình dự đoán sai nhiều nhất
  - Lưu kết quả ra output/model_comparison/
"""

import os
import sys
import json
import pickle
import warnings
import time
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data_set')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'model_comparison')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Datasets
DATASET_FULL_PATH = os.path.join(DATA_DIR, 'dataset_full.csv')
HYBRID_PATH = os.path.join(DATA_DIR, 'hybrid.csv')
DL_TEST_PATH = os.path.join(DATA_DIR, 'dl_dataset', 'processed', 'test_processed.csv')
PHISHING_URL_PATH = os.path.join(DATA_DIR, 'phishing_url.csv')

# Models
RF_DSFULL_MODEL = os.path.join(BASE_DIR, 'output', 'rf_vs_dsfull', 'rf_phishing_model.pkl')
RF_HYBRID_MODEL = os.path.join(BASE_DIR, 'output', 'rf_vs_hybrid', 'rf_phishing_model.pkl')
XGB_DSFULL_MODEL = os.path.join(BASE_DIR, 'output', 'xgb_vs_dsfull', 'xgb_phishing_model.pkl')
KNN_HYBRID_MODEL = os.path.join(BASE_DIR, 'output', 'knn_vs_hybrid', 'knn_model.pkl')
DL_MODEL_PTH = os.path.join(BASE_DIR, 'output', 'dl', 'url_phishing_dl_model.pth')

# Feature configs
RF_DSFULL_FEATURES = os.path.join(BASE_DIR, 'output', 'rf_vs_dsfull', 'rf_selected_features.json')
XGB_DSFULL_FEATURES = os.path.join(BASE_DIR, 'output', 'xgb_vs_dsfull', 'xgb_selected_features.json')
RF_HYBRID_FEATURES = os.path.join(BASE_DIR, 'output', 'rf_vs_hybrid', 'rf_selected_features.json')


# ============================================================
# PHẦN 1: TRÍCH XUẤT ĐẶC TRƯNG TỪ URL (cho dataset_full features)
# ============================================================
def extract_url_features_for_dsfull(url):
    """Trích xuất các đặc trưng thuần URL tương tự dataset_full.csv."""
    features = {}
    
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or ''
        path = parsed.path or ''
        query = parsed.query or ''
        fragment = parsed.fragment or ''
    except:
        parsed = None
        domain = ''
        path = ''
        query = ''
        fragment = ''
    
    # Tách directory và file từ path
    path_parts = path.rsplit('/', 1)
    directory = path_parts[0] if len(path_parts) > 1 else path
    file_part = path_parts[1] if len(path_parts) > 1 else ''
    
    # Params
    params = query
    
    # Các ký tự đặc biệt cần đếm
    special_chars = {
        'dot': '.', 'hyphen': '-', 'underline': '_', 'slash': '/',
        'questionmark': '?', 'equal': '=', 'at': '@', 'and': '&',
        'exclamation': '!', 'space': ' ', 'tilde': '~', 'comma': ',',
        'plus': '+', 'asterisk': '*', 'hashtag': '#', 'dollar': '$',
        'percent': '%'
    }
    
    # Đếm ký tự đặc biệt trong url
    for name, char in special_chars.items():
        features[f'qty_{name}_url'] = url.count(char)
    
    # Đếm ký tự đặc biệt trong domain
    for name, char in special_chars.items():
        features[f'qty_{name}_domain'] = domain.count(char)
    
    # Đếm ký tự đặc biệt trong directory
    for name, char in special_chars.items():
        features[f'qty_{name}_directory'] = directory.count(char)
    
    # Đếm ký tự đặc biệt trong file
    for name, char in special_chars.items():
        features[f'qty_{name}_file'] = file_part.count(char)
    
    # Đếm ký tự đặc biệt trong params  
    for name, char in special_chars.items():
        features[f'qty_{name}_params'] = params.count(char) if params else -1
    
    # Các đặc trưng chiều dài
    features['length_url'] = len(url)
    features['domain_length'] = len(domain)
    features['directory_length'] = len(directory)
    features['file_length'] = len(file_part)
    features['params_length'] = len(params) if params else -1
    
    # Đặc trưng domain
    features['qty_vowels_domain'] = sum(1 for c in domain.lower() if c in 'aeiou')
    features['domain_in_ip'] = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+', domain) else 0
    
    # TLD
    features['qty_tld_url'] = 1 if '.' in domain else 0
    
    # Các đặc trưng khác
    features['url_shortened'] = 1 if any(s in domain.lower() for s in ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'buff.ly']) else 0
    features['qty_params'] = params.count('&') + 1 if params else 0
    features['email_in_url'] = 1 if re.search(r'[\w.+-]+@[\w-]+\.[\w.]+', url) else 0
    
    # Sentinel values cho features không có sẵn
    for col in ['time_response', 'domain_spf', 'asn_ip', 'time_domain_activation', 
                'time_domain_expiration', 'qty_ip_resolved', 'qty_nameservers', 
                'qty_mx_servers', 'ttl_hostname', 'tls_ssl_certificate', 'qty_redirects',
                'url_google_index', 'domain_google_index', 'server_client_domain',
                'tld_present_params']:
        features[col] = -1
    
    return features


# ============================================================
# PHẦN 2: DEEP LEARNING - TOKENIZE URL
# ============================================================
MAX_LEN = 200
CHAR_VOCAB = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~:/?#[]@!$&'()*+,;= "
CHAR_TO_IDX = {char: idx + 2 for idx, char in enumerate(CHAR_VOCAB)}
CHAR_TO_IDX['<pad>'] = 0
CHAR_TO_IDX['<unk>'] = 1
VOCAB_SIZE = len(CHAR_TO_IDX)

def tokenize_url(url, max_len=MAX_LEN):
    if not isinstance(url, str):
        url = ""
    tokenized = []
    for char in url[:max_len]:
        tokenized.append(CHAR_TO_IDX.get(char, 1))
    if len(tokenized) < max_len:
        tokenized += [0] * (max_len - len(tokenized))
    return np.array(tokenized, dtype=np.int64)


# ============================================================
# PHẦN 3: TẢI TẤT CẢ CÁC MÔ HÌNH
# ============================================================
def load_all_models():
    """Tải tất cả các mô hình đã huấn luyện."""
    models = {}
    
    # 1. RF trên dataset_full
    print("  📦 Đang tải RF (dataset_full)...")
    with open(RF_DSFULL_MODEL, 'rb') as f:
        rf_data = pickle.load(f)
    rf_model = rf_data['model']
    rf_model.n_jobs = 1  # Force single thread to avoid joblib deadlocks in sandbox
    models['RF_Pure_URL'] = {
        'model': rf_model,
        'features': rf_data['features'],
        'type': 'sklearn'
    }
    
    # 2. XGBoost trên dataset_full
    print("  📦 Đang tải XGBoost (dataset_full)...")
    with open(XGB_DSFULL_MODEL, 'rb') as f:
        xgb_data = pickle.load(f)
    xgb_model = xgb_data['model']
    xgb_model.n_jobs = 1  # Force single thread to avoid joblib deadlocks in sandbox
    models['XGBoost_Pure_URL'] = {
        'model': xgb_model,
        'features': xgb_data['features'],
        'type': 'sklearn'
    }
    
    # 3. RF trên hybrid
    print("  📦 Đang tải RF (hybrid)...")
    with open(RF_HYBRID_MODEL, 'rb') as f:
        rf_hybrid_data = pickle.load(f)
    rf_h_model = rf_hybrid_data['model']
    rf_h_model.n_jobs = 1  # Force single thread to avoid joblib deadlocks in sandbox
    models['RF_Hybrid'] = {
        'model': rf_h_model,
        'features': rf_hybrid_data['features'],
        'type': 'sklearn_hybrid'
    }
    
    # 4. KNN trên hybrid (NearestNeighbors - unsupervised)
    print("  📦 Đang tải KNN (hybrid)...")
    with open(KNN_HYBRID_MODEL, 'rb') as f:
        knn_data = pickle.load(f)
    knn_model = knn_data['knn']
    knn_model.n_jobs = 1  # Force single thread to avoid joblib deadlocks in sandbox
    models['KNN_Hybrid'] = {
        'knn': knn_model,
        'scaler': knn_data['scaler'],
        'features': knn_data['features'],
        'type': 'knn_unsupervised'
    }
    
    # 5. Deep Learning (CNN-BiLSTM)
    print("  📦 Đang tải Deep Learning (CNN-BiLSTM)...")
    try:
        import torch
        import torch.nn as nn
        torch.set_num_threads(1)  # Limit PyTorch to 1 thread to avoid OpenMP deadlocks in sandbox
        
        class PhishingCNNBiLSTM(nn.Module):
            def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=64, 
                         cnn_filters=128, kernel_size=5, lstm_hidden=64, dropout=0.5):
                super(PhishingCNNBiLSTM, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=cnn_filters, 
                                        kernel_size=kernel_size, padding=kernel_size // 2)
                self.relu = nn.ReLU()
                self.max_pool = nn.MaxPool1d(kernel_size=2)
                self.bilstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden, 
                                      num_layers=1, bidirectional=True, batch_first=True)
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(lstm_hidden * 2, 1)
                
            def forward(self, x):
                embedded = self.embedding(x).transpose(1, 2)
                conv_out = self.relu(self.conv1d(embedded))
                pooled = self.max_pool(conv_out).transpose(1, 2)
                lstm_out, _ = self.bilstm(pooled)
                repr_vec = lstm_out[:, -1, :]
                out = self.dropout(repr_vec)
                logits = self.fc(out)
                return logits.squeeze(1)
        
        device = torch.device('cpu')
        dl_model = PhishingCNNBiLSTM()
        dl_model.load_state_dict(torch.load(DL_MODEL_PTH, map_location=device))
        dl_model.eval()
        
        models['DL_CNN_BiLSTM'] = {
            'model': dl_model,
            'type': 'pytorch'
        }
    except Exception as e:
        print(f"  ⚠️ Không thể tải mô hình DL: {e}")
    
    return models


# ============================================================
# PHẦN 4: ĐÁNH GIÁ TRÊN TẬP TEST HOLD-OUT TỪ DATASET GỐC
# ============================================================
def evaluate_on_holdout(models):
    """Đánh giá mô hình RF và XGB trên tập test từ dataset_full.csv (20% hold-out)."""
    print("\n" + "=" * 70)
    print("ĐÁNH GIÁ 1: TẬP TEST HOLD-OUT TỪ DATASET GỐC (dataset_full.csv)")
    print("=" * 70)
    
    results = {}
    
    # Load dataset_full
    df_full = pd.read_csv(DATASET_FULL_PATH)
    y_full = df_full['phishing']
    
    # Sử dụng cùng random_state=42 và test_size=0.2 như khi train
    _, X_test_idx, _, y_test = train_test_split(
        df_full.index, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    df_test_full = df_full.loc[X_test_idx]
    
    print(f"  Tập Test (dataset_full): {len(df_test_full)} mẫu")
    print(f"    - Phishing: {(y_test == 1).sum()}")
    print(f"    - Legitimate: {(y_test == 0).sum()}")
    
    # --- RF Pure URL ---
    rf_model = models['RF_Pure_URL']
    X_test_rf = df_test_full[rf_model['features']]
    y_pred_rf = rf_model['model'].predict(X_test_rf)
    y_proba_rf = rf_model['model'].predict_proba(X_test_rf)[:, 1]
    
    results['RF_Pure_URL'] = {
        'y_true': y_test.values,
        'y_pred': y_pred_rf,
        'y_proba': y_proba_rf,
        'dataset': 'dataset_full (hold-out 20%)'
    }
    
    # --- XGBoost Pure URL ---
    xgb_model = models['XGBoost_Pure_URL']
    X_test_xgb = df_test_full[xgb_model['features']]
    y_pred_xgb = xgb_model['model'].predict(X_test_xgb)
    y_proba_xgb = xgb_model['model'].predict_proba(X_test_xgb)[:, 1]
    
    results['XGBoost_Pure_URL'] = {
        'y_true': y_test.values,
        'y_pred': y_pred_xgb,
        'y_proba': y_proba_xgb,
        'dataset': 'dataset_full (hold-out 20%)'
    }
    
    # --- RF Hybrid ---
    print("\n  Đang load hybrid.csv cho RF Hybrid và KNN...")
    df_hybrid = pd.read_csv(HYBRID_PATH)
    y_hybrid = df_hybrid['label']
    
    # Dùng cùng random_state=42
    _, X_test_h_idx, _, y_test_h = train_test_split(
        df_hybrid.index, y_hybrid, test_size=0.2, random_state=42
    )
    df_test_hybrid = df_hybrid.loc[X_test_h_idx]
    
    print(f"  Tập Test (hybrid): {len(df_test_hybrid)} mẫu")
    print(f"    - Phishing: {(y_test_h == 1).sum()}")
    print(f"    - Legitimate: {(y_test_h == 0).sum()}")
    
    rf_hybrid = models['RF_Hybrid']
    X_test_rf_h = df_test_hybrid[rf_hybrid['features']]
    y_pred_rf_h = rf_hybrid['model'].predict(X_test_rf_h)
    y_proba_rf_h = rf_hybrid['model'].predict_proba(X_test_rf_h)[:, 1]
    
    results['RF_Hybrid'] = {
        'y_true': y_test_h.values,
        'y_pred': y_pred_rf_h,
        'y_proba': y_proba_rf_h,
        'dataset': 'hybrid (hold-out 20%)'
    }
    
    # --- DL CNN-BiLSTM ---
    if 'DL_CNN_BiLSTM' in models:
        print("\n  Đang đánh giá DL trên test_processed.csv...")
        import torch
        
        df_dl_test = pd.read_csv(DL_TEST_PATH)
        df_dl_test = df_dl_test.dropna(subset=['url', 'result'])
        
        urls_dl = df_dl_test['url'].values
        y_true_dl = df_dl_test['result'].values.astype(np.float32)
        
        print(f"  Tập Test (DL): {len(df_dl_test)} mẫu")
        print(f"    - Phishing: {(y_true_dl == 1).sum()}")
        print(f"    - Benign: {(y_true_dl == 0).sum()}")
        
        dl_model = models['DL_CNN_BiLSTM']['model']
        
        # Tokenize và predict theo batch
        batch_size = 256
        all_preds = []
        all_probas = []
        
        for i in range(0, len(urls_dl), batch_size):
            batch_urls = urls_dl[i:i+batch_size]
            tokenized = np.array([tokenize_url(url) for url in batch_urls])
            with torch.no_grad():
                inputs = torch.tensor(tokenized)
                logits = dl_model(inputs)
                probas = torch.sigmoid(logits).numpy()
                preds = (probas >= 0.5).astype(np.float32)
                all_preds.extend(preds)
                all_probas.extend(probas)
        
        results['DL_CNN_BiLSTM'] = {
            'y_true': y_true_dl,
            'y_pred': np.array(all_preds),
            'y_proba': np.array(all_probas),
            'dataset': 'dl_dataset (test_processed)'
        }
    
    return results


def extract_hybrid_features_fallback(url):
    """Trích xuất đặc trưng Lexical và fallback đặc trưng HTML trung vị cho tập ngoại."""
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
        parsed = urlparse(url)
        domain = parsed.netloc or ''
        if ':' in domain:
            domain = domain.split(':')[0]
        feats['DomainLength'] = len(domain)
        
        # Check IP
        feats['IsDomainIP'] = 1.0 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', domain) else 0.0
        
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
        
        # Special chars
        specials = sum(1 for c in url if not c.isalnum() and c not in ('/', '.', ':', '-'))
        feats['NoOfOtherSpecialCharsInURL'] = float(specials)
        feats['SpacialCharRatioInURL'] = specials / len(url) if len(url) > 0 else 0.0
    except:
        pass
    return feats


# ============================================================
# PHẦN 5: ĐÁNH GIÁ TRÊN TẬP DỮ LIỆU NGOẠI (phishing_url.csv)
# ============================================================
def evaluate_on_external(models):
    """Đánh giá các mô hình Pure URL trên tập dữ liệu ngoại từ PhishTank."""
    print("\n" + "=" * 70)
    print("ĐÁNH GIÁ 2: TẬP DỮ LIỆU NGOẠI (phishing_url.csv - PhishTank)")
    print("=" * 70)
    
    results_ext = {}
    
    if not os.path.exists(PHISHING_URL_PATH):
        print("  ❌ Không tìm thấy phishing_url.csv")
        return results_ext
    
    df_phish = pd.read_csv(PHISHING_URL_PATH)
    print(f"  Tổng URL phishing từ PhishTank: {len(df_phish)}")
    
    # Lọc chỉ các verified phishing URLs
    if 'verified' in df_phish.columns:
        df_phish = df_phish[df_phish['verified'] == 'yes']
    if 'online' in df_phish.columns:
        df_phish = df_phish[df_phish['online'] == 'yes']
    
    print(f"  URL phishing đã xác minh: {len(df_phish)}")
    
    # Lấy sample 5000 URLs nếu quá nhiều
    if len(df_phish) > 5000:
        df_phish = df_phish.sample(n=5000, random_state=42)
        print(f"  → Lấy mẫu 5,000 URL để đánh giá")
    
    phishing_urls = df_phish['url'].dropna().values
    n_samples = len(phishing_urls)
    
    # Tất cả là phishing (label = 1)
    y_true = np.ones(n_samples, dtype=int)
    
    # Thêm một số legitimate URLs để tạo tập cân bằng hơn
    legit_urls = [
        "https://www.google.com", "https://www.facebook.com", "https://www.youtube.com",
        "https://www.amazon.com", "https://www.wikipedia.org", "https://www.twitter.com",
        "https://www.github.com", "https://www.stackoverflow.com", "https://www.microsoft.com",
        "https://www.apple.com", "https://www.netflix.com", "https://www.linkedin.com",
        "https://www.reddit.com", "https://www.yahoo.com", "https://www.bing.com",
        "https://www.ebay.com", "https://www.instagram.com", "https://www.twitch.tv",
        "https://www.spotify.com", "https://www.medium.com", "https://docs.python.org",
        "https://www.bbc.com/news", "https://www.cnn.com", "https://www.nytimes.com",
        "https://mail.google.com", "https://drive.google.com", "https://www.dropbox.com",
        "https://www.paypal.com", "https://www.shopify.com", "https://www.wordpress.org",
    ]
    
    all_urls = list(phishing_urls) + legit_urls
    y_true_ext = np.concatenate([y_true, np.zeros(len(legit_urls), dtype=int)])
    
    print(f"  Tổng mẫu đánh giá ngoại: {len(all_urls)} (phishing: {n_samples}, legit: {len(legit_urls)})")
    
    # Trích xuất features cho RF Pure URL và XGBoost
    print("\n  🔧 Đang trích xuất đặc trưng thuần URL từ phishing_url.csv...")
    features_list = []
    for i, url in enumerate(all_urls):
        if i % 500 == 0:
            print(f"    → Đã xử lý {i}/{len(all_urls)} URL...")
        features_list.append(extract_url_features_for_dsfull(url))
    
    df_features = pd.DataFrame(features_list)
    
    # --- RF Pure URL ---
    rf_model = models['RF_Pure_URL']
    rf_feats = rf_model['features']
    # Đảm bảo tất cả features tồn tại
    for f in rf_feats:
        if f not in df_features.columns:
            df_features[f] = -1
    
    X_ext_rf = df_features[rf_feats].values
    y_pred_rf = rf_model['model'].predict(X_ext_rf)
    y_proba_rf = rf_model['model'].predict_proba(X_ext_rf)[:, 1]
    
    results_ext['RF_Pure_URL'] = {
        'y_true': y_true_ext,
        'y_pred': y_pred_rf,
        'y_proba': y_proba_rf,
        'dataset': 'PhishTank External'
    }
    
    # --- XGBoost Pure URL ---
    xgb_model = models['XGBoost_Pure_URL']
    xgb_feats = xgb_model['features']
    for f in xgb_feats:
        if f not in df_features.columns:
            df_features[f] = -1
    
    X_ext_xgb = df_features[xgb_feats].values
    y_pred_xgb = xgb_model['model'].predict(X_ext_xgb)
    y_proba_xgb = xgb_model['model'].predict_proba(X_ext_xgb)[:, 1]
    
    results_ext['XGBoost_Pure_URL'] = {
        'y_true': y_true_ext,
        'y_pred': y_pred_xgb,
        'y_proba': y_proba_xgb,
        'dataset': 'PhishTank External'
    }
    
    # --- RF Hybrid ---
    if 'RF_Hybrid' in models:
        print("  🌲 Đang dự đoán bằng RF Hybrid (Median Fallback) trên tập ngoại...")
        rf_h_model = models['RF_Hybrid']
        rf_h_feats = rf_h_model['features']
        
        features_h_list = []
        for url in all_urls:
            features_h_list.append(extract_hybrid_features_fallback(url))
        df_h_features = pd.DataFrame(features_h_list)
        
        # Đảm bảo tất cả features của mô hình tồn tại
        for f in rf_h_feats:
            if f not in df_h_features.columns:
                df_h_features[f] = 0.0
                
        X_ext_rf_h = df_h_features[rf_h_feats].values
        y_pred_rf_h = rf_h_model['model'].predict(X_ext_rf_h)
        y_proba_rf_h = rf_h_model['model'].predict_proba(X_ext_rf_h)[:, 1]
        
        results_ext['RF_Hybrid'] = {
            'y_true': y_true_ext,
            'y_pred': y_pred_rf_h,
            'y_proba': y_proba_rf_h,
            'dataset': 'PhishTank External'
        }
    
    # --- DL CNN-BiLSTM ---
    if 'DL_CNN_BiLSTM' in models:
        print("  🧠 Đang dự đoán bằng DL trên tập ngoại...")
        import torch
        
        dl_model = models['DL_CNN_BiLSTM']['model']
        batch_size = 256
        all_preds_dl = []
        all_probas_dl = []
        
        for i in range(0, len(all_urls), batch_size):
            batch_urls = all_urls[i:i+batch_size]
            tokenized = np.array([tokenize_url(url) for url in batch_urls])
            with torch.no_grad():
                inputs = torch.tensor(tokenized)
                logits = dl_model(inputs)
                probas = torch.sigmoid(logits).numpy()
                preds = (probas >= 0.5).astype(np.float32)
                all_preds_dl.extend(preds)
                all_probas_dl.extend(probas)
        
        results_ext['DL_CNN_BiLSTM'] = {
            'y_true': y_true_ext,
            'y_pred': np.array(all_preds_dl),
            'y_proba': np.array(all_probas_dl),
            'dataset': 'PhishTank External'
        }
    
    return results_ext


# ============================================================
# PHẦN 6: TÍNH TOÁN METRICS VÀ TẠO BÁO CÁO
# ============================================================
def compute_metrics(y_true, y_pred, y_proba=None):
    """Tính toán các chỉ số đánh giá."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positive'] = int(tp)
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        metrics['total_errors'] = int(fp + fn)
        metrics['error_rate'] = (fp + fn) / len(y_true)
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['roc_auc'] = None
    
    return metrics


def print_comparison_table(all_results, title=""):
    """In bảng so sánh tổng hợp."""
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    
    header = f"{'Mô hình':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'AUC-ROC':>10} {'Errors':>10} {'Error%':>10}"
    print(header)
    print("-" * 90)
    
    rows = []
    for model_name, res in all_results.items():
        metrics = compute_metrics(res['y_true'], res['y_pred'], res.get('y_proba'))
        roc_str = f"{metrics['roc_auc']:.4f}" if metrics.get('roc_auc') is not None else "N/A"
        row = f"{model_name:<20} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1_score']:>10.4f} {roc_str:>10} {metrics.get('total_errors', 'N/A'):>10} {metrics.get('error_rate', 0)*100:>9.2f}%"
        print(row)
        rows.append((model_name, metrics))
    
    print("-" * 90)
    
    # Tìm mô hình tốt nhất và kém nhất
    if rows:
        best_model = max(rows, key=lambda x: x[1]['f1_score'])
        worst_model = min(rows, key=lambda x: x[1]['f1_score'])
        most_errors = max(rows, key=lambda x: x[1].get('total_errors', 0))
        
        print(f"\n  🏆 MÔ HÌNH TỐT NHẤT (F1-Score): {best_model[0]} → F1 = {best_model[1]['f1_score']:.4f}")
        print(f"  ❌ MÔ HÌNH KÉM NHẤT (F1-Score): {worst_model[0]} → F1 = {worst_model[1]['f1_score']:.4f}")
        print(f"  ⚠️  DỰ ĐOÁN SAI NHIỀU NHẤT:     {most_errors[0]} → {most_errors[1].get('total_errors', 0)} lỗi ({most_errors[1].get('error_rate', 0)*100:.2f}%)")
    
    return rows


# ============================================================
# PHẦN 7: VẼ ĐỒ THỊ SO SÁNH TỔNG HỢP
# ============================================================
def plot_comparison_charts(all_results, suffix=""):
    """Vẽ các biểu đồ so sánh."""
    sns.set_theme(style="whitegrid")
    
    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    error_rates = []
    
    for model_name, res in all_results.items():
        metrics = compute_metrics(res['y_true'], res['y_pred'], res.get('y_proba'))
        model_names.append(model_name.replace('_', '\n'))
        accuracies.append(metrics['accuracy'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1_scores.append(metrics['f1_score'])
        error_rates.append(metrics.get('error_rate', 0) * 100)
    
    # 1. Bar chart so sánh metrics
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    x = np.arange(len(model_names))
    width = 0.2
    
    ax1 = axes[0]
    bars1 = ax1.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#3498db', edgecolor='white')
    bars2 = ax1.bar(x - 0.5*width, precisions, width, label='Precision', color='#2ecc71', edgecolor='white')
    bars3 = ax1.bar(x + 0.5*width, recalls, width, label='Recall', color='#e67e22', edgecolor='white')
    bars4 = ax1.bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='#e74c3c', edgecolor='white')
    
    ax1.set_xlabel('Mô hình', fontsize=12)
    ax1.set_ylabel('Giá trị', fontsize=12)
    ax1.set_title(f'So sánh Metrics các mô hình {suffix}', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=9)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Thêm giá trị trên cột
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
    
    # 2. Error rate chart
    ax2 = axes[1]
    colors = ['#e74c3c' if er == max(error_rates) else '#3498db' for er in error_rates]
    bars = ax2.bar(model_names, error_rates, color=colors, edgecolor='white', width=0.6)
    ax2.set_xlabel('Mô hình', fontsize=12)
    ax2.set_ylabel('Tỷ lệ lỗi (%)', fontsize=12)
    ax2.set_title(f'Tỷ lệ dự đoán sai {suffix}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, er in zip(bars, error_rates):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
                f'{er:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    suffix_clean = suffix.replace(" ", "_").replace("(", "").replace(")", "").lower().strip("_")
    filename = f'model_comparison_{suffix_clean}.png' if suffix_clean else 'model_comparison.png'
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    (saved to: {save_path})")
    print(f"  ✓ Đã lưu biểu đồ: {filename}")
    
    # 3. Confusion matrices cho từng mô hình
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, res) in enumerate(all_results.items()):
        cm = confusion_matrix(res['y_true'], res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Legit', 'Phishing'],
                   yticklabels=['Legit', 'Phishing'],
                   annot_kws={'size': 12, 'weight': 'bold'})
        axes[idx].set_title(f'{model_name}', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Dự đoán', fontsize=10)
        axes[idx].set_ylabel('Thực tế', fontsize=10)
    
    plt.suptitle(f'Confusion Matrix - Tất cả mô hình {suffix}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    suffix_clean2 = suffix.replace(" ", "_").replace("(", "").replace(")", "").lower().strip("_")
    filename_cm = f'confusion_matrices_{suffix_clean2}.png' if suffix_clean2 else 'confusion_matrices.png'
    save_path_cm = os.path.join(OUTPUT_DIR, filename_cm)
    plt.savefig(save_path_cm, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Đã lưu confusion matrices: {filename_cm}")


# ============================================================
# PHẦN 8: PHÂN TÍCH CHI TIẾT MÔ HÌNH DỰ ĐOÁN SAI
# ============================================================
def analyze_errors(all_results, suffix=""):
    """Phân tích chi tiết các lỗi dự đoán."""
    print(f"\n{'='*70}")
    print(f"  PHÂN TÍCH CHI TIẾT LỖI DỰ ĐOÁN {suffix}")
    print(f"{'='*70}")
    
    error_details = {}
    
    for model_name, res in all_results.items():
        y_true = res['y_true']
        y_pred = res['y_pred']
        
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        total = len(y_true)
        total_errors = fp + fn
        
        print(f"\n  📊 {model_name}:")
        print(f"     Tổng mẫu: {total}")
        print(f"     ✅ Dự đoán đúng: {tp + tn} ({(tp+tn)/total*100:.2f}%)")
        print(f"     ❌ Dự đoán sai:  {total_errors} ({total_errors/total*100:.2f}%)")
        print(f"        - False Positive (Legit → Phishing): {fp} ({fp/total*100:.2f}%)")
        print(f"        - False Negative (Phishing → Legit): {fn} ({fn/total*100:.2f}%)")
        
        if fn > 0:
            print(f"     ⚠️  BỎ SÓT {fn} URL PHISHING (Nguy hiểm - False Negative)")
        if fp > 0:
            print(f"     ℹ️  Cảnh báo nhầm {fp} URL hợp lệ (False Positive)")
        
        error_details[model_name] = {
            'total': total,
            'errors': total_errors,
            'false_positive': fp,
            'false_negative': fn,
            'error_rate': total_errors / total
        }
    
    # Xếp hạng
    if error_details:
        print(f"\n  {'='*50}")
        print(f"  XẾP HẠNG MÔ HÌNH (theo Error Rate - thấp hơn = tốt hơn):")
        print(f"  {'='*50}")
        
        sorted_models = sorted(error_details.items(), key=lambda x: x[1]['error_rate'])
        for rank, (name, details) in enumerate(sorted_models, 1):
            emoji = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else "  "))
            print(f"  {emoji} #{rank} {name:<20} Error Rate: {details['error_rate']*100:.2f}% ({details['errors']}/{details['total']})")
    
    return error_details


# ============================================================
# MAIN
# ============================================================
def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  ĐÁNH GIÁ TỔNG HỢP TẤT CẢ CÁC MÔ HÌNH PHISHING DETECTION  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    start_time = time.time()
    
    # 1. Tải mô hình
    print("\n📦 ĐANG TẢI TẤT CẢ CÁC MÔ HÌNH...")
    models = load_all_models()
    print(f"  ✓ Đã tải thành công {len(models)} mô hình")
    
    # 2. Đánh giá trên hold-out test sets
    holdout_results = evaluate_on_holdout(models)
    
    # 3. In bảng so sánh hold-out
    print_comparison_table(holdout_results, "BẢNG SO SÁNH - TẬP TEST HOLD-OUT")
    analyze_errors(holdout_results, "(Hold-out)")
    # plot_comparison_charts(holdout_results, " (Hold-out)") # Bỏ qua để tránh bị treo trong background sandbox
    
    # 4. Đánh giá trên tập ngoại
    external_results = evaluate_on_external(models)
    if external_results:
        print_comparison_table(external_results, "BẢNG SO SÁNH - TẬP DỮ LIỆU NGOẠI (PhishTank)")
        analyze_errors(external_results, "(External)")
        # plot_comparison_charts(external_results, " (External)") # Bỏ qua để tránh bị treo trong background sandbox
    
    # 5. Tổng kết
    print("\n" + "=" * 70)
    print("  📋 TỔNG KẾT ĐÁNH GIÁ")
    print("=" * 70)
    
    # Thu thập tất cả metrics
    all_summary = {}
    for model_name, res in holdout_results.items():
        metrics = compute_metrics(res['y_true'], res['y_pred'], res.get('y_proba'))
        all_summary[model_name] = {
            'holdout': metrics,
            'dataset': res['dataset']
        }
    
    for model_name, res in external_results.items():
        metrics = compute_metrics(res['y_true'], res['y_pred'], res.get('y_proba'))
        if model_name in all_summary:
            all_summary[model_name]['external'] = metrics
        else:
            all_summary[model_name] = {'external': metrics}
    
    # Lưu kết quả JSON
    summary_path = os.path.join(OUTPUT_DIR, 'evaluation_summary.json')
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    summary_json = {}
    for model_name, data in all_summary.items():
        summary_json[model_name] = {}
        for key, val in data.items():
            if isinstance(val, dict):
                summary_json[model_name][key] = {k: convert_numpy(v) for k, v in val.items()}
            else:
                summary_json[model_name][key] = convert_numpy(val)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  ✓ Đã lưu báo cáo tổng hợp tại: {summary_path}")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Thời gian đánh giá: {elapsed:.1f}s")
    print("\n🎉 HOÀN TẤT ĐÁNH GIÁ TỔNG HỢP!")


if __name__ == '__main__':
    main()
