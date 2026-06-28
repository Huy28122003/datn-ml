import os
import sys
import json
import pickle
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, 'data_set', 'hybrid.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output', 'rf_vs_hybrid')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'rf_phishing_model.pkl')


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Loi: Khong tim thay model tai {MODEL_PATH}")
        sys.exit(1)
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def main():
    model_data = load_model()
    model = model_data['model']
    features = model_data['features']
    
    if not os.path.exists(DATASET_PATH):
        print(f"Loi: Khong tim thay dataset tai {DATASET_PATH}")
        sys.exit(1)
        
    df = pd.read_csv(DATASET_PATH)
    
    print(f"Loaded model ({len(features)} dac trung).")
    
    sampled_df = df.sample(n=5, random_state=np.random.randint(1, 100000))
    
    for idx, (index, row) in enumerate(sampled_df.iterrows(), 1):
        url = row['URL']
        actual_label = int(row['label'])
        filename = row.get('FILENAME', 'N/A')
        
        feature_vector = [row[name] for name in features]
        
        X = np.array([feature_vector])
        pred = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        
        label_text = "PHISHING" if pred == 1 else "LEGITIMATE"
        actual_text = "PHISHING" if actual_label == 1 else "LEGITIMATE"
        
        print(f"\n[URL {idx}/5]: {url}")
        print(f"   - File goc:    {filename}")
        print(f"   - Nhan thuc te: {actual_text}")
        print(f"   - Du doan:     {label_text} (Xac suat: {probs[pred]:.2%})")
        print(f"   - Trang thai:  {'CHINH XAC' if pred == actual_label else 'SAI LECH'}")
        
        print("   Chi tiet 10 dac trung Hybrid quan trong nhat cua mau nay:")
        for i, name in enumerate(features[:10], 1):
            val = row[name]
            print(f"     {i:>2}. {name:<30} = {val}")


if __name__ == '__main__':
    main()
