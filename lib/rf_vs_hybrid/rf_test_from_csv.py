import os
import sys
import pandas as pd
import numpy as np
import random
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE_DIR, 'lib'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rf_phishing_test_predict as rfp

CSV_PATH = os.path.join(BASE_DIR, 'data_set', 'phishing_url.csv')

def load_urls_from_csv(csv_path, num_random=5):
    if not os.path.exists(csv_path):
        print(f"Loi: Thieu file {csv_path}")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    
    # huyny - lay indices csv
    specified_indices = list(range(885, min(895, len(df))))
    specified_df = df.iloc[specified_indices]
    
    remaining_df = df.drop(index=specified_indices)
    sampled_df = remaining_df.sample(n=num_random, random_state=random.randint(1, 100000))
    
    return specified_df, sampled_df

def main():
    parser = argparse.ArgumentParser(description="Test Random Forest Hybrid model on phishing_url.csv")
    parser.add_argument("--random-count", type=int, default=5, help="So luong URL ngau nhien can lay")
    args = parser.parse_args()

    model, model_features, params = rfp.load_model()
    print(f"Loaded model ({len(model_features)} features)")
    
    specified_df, sampled_df = load_urls_from_csv(CSV_PATH, num_random=args.random_count)
    
    results = []
    
    to_test = []
    for idx, row in specified_df.iterrows():
        to_test.append((row['url'], "Chi dinh (Dong " + str(idx + 2) + ")"))
    for idx, row in sampled_df.iterrows():
        to_test.append((row['url'], "Ngau nhien (Dong " + str(idx + 2) + ")"))
        
    correct_count = 0
    
    for i, (url, source) in enumerate(to_test, 1):
        print(f"\n[{i}/{len(to_test)}] Test: {url}")
        try:
            res = rfp.predict_url(url, model, model_features)
            
            prediction = res['prediction']
            is_correct = (prediction == 1)
            if is_correct:
                correct_count += 1
                status = "CHINH XAC"
            else:
                status = "SAI LECH"
                
            print(f"   - Actual: PHISHING")
            print(f"   - Pred:   {res['label']}")
            print(f"   - Prob:   Phishing: {res['probability_phishing']:.2%}")
            print(f"   - Status: {status}")
            
            results.append({
                'url': url,
                'source': source,
                'pred': prediction,
                'correct': is_correct,
                'prob_phish': res['probability_phishing']
            })
        except Exception as e:
            print(f"    Loi: {e}")
            
    print("\nEvaluation:")
    total_tested = len(results)
    
    # huyny - tinh recall
    if total_tested > 0:
        accuracy = correct_count / total_tested
        print(f"Total:  {total_tested}")
        print(f"TP:     {correct_count}")
        print(f"FN:     {total_tested - correct_count}")
        print(f"Recall: {accuracy:.2%}")
    else:
        print("Empty test results.")

if __name__ == '__main__':
    main()
