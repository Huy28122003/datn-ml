import os
import csv
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOP1M_PATH = os.path.join(BASE_DIR, 'data_set', 'top-1m.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'knn_search')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'knn_model.pkl')

def clean_domain(domain_str):
    d = domain_str.strip().lower()
    if d.startswith('www.'):
        d = d[4:]
    return d

def train():
    # huynq - doc du lieu top-1m
    if not os.path.exists(TOP1M_PATH):
        print(f"Khong tim thay file du lieu tai {TOP1M_PATH}")
        return
        
    t_start = time.time()
    
    domains = []
    with open(TOP1M_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            if row[1].lower() == 'domain':
                continue
            cleaned = clean_domain(row[1])
            if cleaned:
                domains.append(cleaned)
                
    print(f"Doc thanh cong {len(domains):,} ten mien trong {time.time() - t_start:.2f} giay.")

    t_feat = time.time()
    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 3),
        max_features=15000,
        sublinear_tf=True
    )
    
    X = vectorizer.fit_transform(domains)
    print(f"Trich xuat dac trung xong trong {time.time() - t_feat:.2f} giay.")
    print(f"Kich thuoc ma tran dac trung: {X.shape[0]:,} x {X.shape[1]:,}")

    # huynq - train model knn
    t_train = time.time()
    knn = NearestNeighbors(
        n_neighbors=1,
        metric='cosine',
        algorithm='brute',
        n_jobs=-1
    )
    knn.fit(X)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    t_save = time.time()
    
    model_data = {
        'vectorizer': vectorizer,
        'knn': knn,
        'domains': domains
    }
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    file_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)

if __name__ == '__main__':
    train()
