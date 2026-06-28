import os
import sys
import random
import pandas as pd
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xgb_phishing_test_predict import load_model, predict_url, display_result
CSV_PATH = os.path.join(BASE_DIR, 'data_set', 'phishing_url.csv')

def main():
    (model, features, params) = load_model()
    if not os.path.exists(CSV_PATH):
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    num_samples = 5
    random_seed = random.randint(1, 10000)
    random_urls = df['url'].dropna().sample(n=num_samples, random_state=random_seed).tolist()
    for (i, url) in enumerate(random_urls, 1):
        try:
            result = predict_url(url, model, features)
            display_result(result)
        except Exception as e:
if __name__ == '__main__':
    main()
