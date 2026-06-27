import os
import sys
import argparse
import pandas as pd

# Add the workspace root to sys.path to enable imports starting with 'lib'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from lib.dl.preprocessing.normalizer import deduplicate_dataset
from lib.dl.preprocessing.splitter import split_by_domain
from lib.dl.preprocessing.analyzer import analyze_distributions
from lib.dl.preprocessing.augmenter import fetch_live_malicious, generate_hard_benign_urls
from lib.dl.preprocessing.balancer import filter_malformed_urls, balance_by_length, balance_tlds

def run_pipeline(input_csv_path: str,
                 output_dir: str,
                 augment_benign_count: int = 25000,
                 fetch_live_phishing: bool = True,
                 train_ratio: float = 0.70,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_state: int = 42):
    """
    Runs the complete URL dataset optimization pipeline.
    """
    print("==================================================================")
    # Highlight the current step
    print("🚀 STARTING PHISHING DATASET OPTIMIZATION PIPELINE")
    print("==================================================================")
    
    os.makedirs(output_dir, exist_ok=True)
    analysis_before_dir = os.path.join(output_dir, "analysis_before")
    analysis_after_dir = os.path.join(output_dir, "analysis_after")
    
    # --- Step 1: Load Dataset ---
    print(f"\n[Step 1] Loading raw dataset from: {input_csv_path}")
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Dataset path {input_csv_path} does not exist!")
        
    df = pd.read_csv(input_csv_path)
    print(f"Loaded {len(df)} records.")
    
    # Clean column names
    df = df.rename(columns={
        'url': 'url',
        'label': 'label',
        'result': 'result'
    })
    
    # --- Step 2 & 3: Normalize and Deduplicate ---
    print("\n[Step 2 & 3] Normalizing URLs & removing duplicates...")
    df = deduplicate_dataset(df, url_col='url', label_col='label')
    
    # --- Step 4 & 5: Split by Domain (prevent leakage) ---
    print("\n[Step 4 & 5] Splitting dataset by registered domain...")
    train_df, val_df, test_df = split_by_domain(
        df, url_col='url', label_col='label',
        train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        random_state=random_state
    )
    
    # --- Step 6: Analyze Initial Distributions ---
    print("\n[Step 6] Analyzing initial distributions (before augmentation & balancing)...")
    print("\n>>> Training Set Initial Distribution:")
    analyze_distributions(train_df, url_col='url', label_col='label', output_dir=analysis_before_dir)
    
    # --- Step 7: Augment Benign and Malicious ---
    print("\n[Step 7] Augmenting dataset...")
    # Fetch live phishing feeds
    live_phishing_df = None
    if fetch_live_phishing:
        try:
            live_phishing_df = fetch_live_malicious()
            if len(live_phishing_df) > 0:
                print(f"Adding {len(live_phishing_df)} live phishing URLs to training set.")
                train_df = pd.concat([train_df, live_phishing_df], ignore_index=True)
        except Exception as e:
            print(f"Failed to fetch live phishing URLs: {e}. Skipping live phishing augmentation.")
            
    # Synthesize hard benign URLs (long, query parameters, OAuth, CDN)
    if augment_benign_count > 0:
        hard_benign_df = generate_hard_benign_urls(count=augment_benign_count)
        print(f"Adding {len(hard_benign_df)} synthetically generated hard benign URLs to training set.")
        train_df = pd.concat([train_df, hard_benign_df], ignore_index=True)
        
    # Re-deduplicate train_df in case of duplicate additions
    train_df = train_df.drop_duplicates(subset=['url'], keep='first')
    
    # --- Step 8: Balance Distributions in Train Set ---
    print("\n[Step 8] Balancing training set distributions...")
    
    # 8.1 Filter malformed URLs
    train_df = filter_malformed_urls(train_df, url_col='url')
    val_df = filter_malformed_urls(val_df, url_col='url')
    test_df = filter_malformed_urls(test_df, url_col='url')
    
    # 8.2 Balance length distribution to remove shortcut bias
    train_df = balance_by_length(train_df, url_col='url', label_col='label', bin_size=15, random_state=random_state)
    
    # 8.3 Balance TLDs
    train_df = balance_tlds(train_df, url_col='url', label_col='label', max_tld_ratio=0.25, random_state=random_state)
    
    # Make sure binary labels 'result' align with 'label'
    train_df['result'] = train_df['label'].apply(lambda l: 1 if l == 'malicious' else 0)
    val_df['result'] = val_df['label'].apply(lambda l: 1 if l == 'malicious' else 0)
    test_df['result'] = test_df['label'].apply(lambda l: 1 if l == 'malicious' else 0)
    
    # Ensure final clean formats (only select required columns)
    final_cols = ['url', 'label', 'result']
    train_df = train_df[final_cols].copy()
    val_df = val_df[final_cols].copy()
    test_df = test_df[final_cols].copy()
    
    # Shuffle training set
    train_df = train_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    # --- Step 6 (Part 2): Analyze Post-processed Training Set ---
    print("\n[Step 6 - Verify] Analyzing distributions after optimization...")
    print("\n>>> Training Set Final Balanced Distribution:")
    analyze_distributions(train_df, url_col='url', label_col='label', output_dir=analysis_after_dir)
    
    # Save processed splits
    print(f"\nSaving processed datasets to output directory: {output_dir}")
    train_df.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_processed.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_processed.csv"), index=False)
    
    print("\n==================================================================")
    print("🎉 PIPELINE SUCCESSFULLY COMPLETED!")
    print(f"  Processed dataset outputs stored at: {output_dir}")
    print(f"  Training set size:   {len(train_df)} samples")
    print(f"  Validation set size: {len(val_df)} samples")
    print(f"  Test set size:       {len(test_df)} samples")
    print("==================================================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Phishing URL Dataset Optimization Pipeline")
    parser.add_argument("--input", type=str, default="data_set/balanced_urls.csv", help="Path to input raw CSV")
    parser.add_argument("--output_dir", type=str, default="data_set/processed", help="Directory to save output splits")
    parser.add_argument("--augment_benign", type=int, default=25000, help="Number of hard benign URLs to synthesize")
    parser.add_argument("--skip_live", action="store_true", help="Skip fetching live phishing URLs")
    parser.add_argument("--train_ratio", type=float, default=0.70, help="Train ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    
    args = parser.parse_args()
    
    run_pipeline(
        input_csv_path=args.input,
        output_dir=args.output_dir,
        augment_benign_count=args.augment_benign,
        fetch_live_phishing=not args.skip_live,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed
    )
