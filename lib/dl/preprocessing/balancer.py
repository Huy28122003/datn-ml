import re
import numpy as np
import pandas as pd
from urllib.parse import urlparse
import tldextract

def is_valid_url(url: str) -> bool:
    """Basic validation to filter out malformed or dead/nonsense URL strings."""
    if not isinstance(url, str) or len(url) < 8:
        return False
    
    # Check for basic scheme and structure
    if not (url.startswith('http://') or url.startswith('https://')):
        return False
        
    try:
        parsed = urlparse(url)
        # Must have a valid network location (host)
        if not parsed.netloc or '.' not in parsed.netloc:
            return False
        return True
    except Exception:
        return False

def filter_malformed_urls(df: pd.DataFrame, url_col: str = 'url') -> pd.DataFrame:
    """Removes malformed, empty, or invalid URLs from the dataset."""
    print("Filtering malformed URLs...")
    before = len(df)
    df = df[df[url_col].apply(is_valid_url)].copy()
    after = len(df)
    print(f"Removed {before - after} malformed or invalid URLs.")
    return df

def balance_by_length(df: pd.DataFrame, 
                      url_col: str = 'url', 
                      label_col: str = 'label', 
                      bin_size: int = 15,
                      min_bin_count: int = 10,
                      random_state: int = 42) -> pd.DataFrame:
    """
    Groups URLs into length bins (e.g. 0-15, 15-30, etc.) and balances
    the count of benign and malicious URLs in each bin by downsampling 
    the majority class.
    
    This ensures the model cannot use URL length as a predictive shortcut.
    """
    print(f"Balancing length distribution (bin size = {bin_size})...")
    
    # Calculate lengths
    df = df.copy()
    df['url_len'] = df[url_col].apply(len)
    
    # Create bins
    max_len = df['url_len'].max()
    bins = list(range(0, int(max_len) + bin_size, bin_size))
    df['len_bin'] = pd.cut(df['url_len'], bins=bins, labels=bins[:-1])
    
    balanced_chunks = []
    
    # Labels (usually 'benign' and 'malicious')
    labels = df[label_col].unique()
    if len(labels) != 2:
        print("Warning: Dataset is not binary. Length balancing is optimized for binary classification.")
        
    for bin_val, group in df.groupby('len_bin', observed=True):
        label_counts = group[label_col].value_counts()
        
        # If bin is too small or doesn't have both classes, skip or handle
        if len(label_counts) < 2 or label_counts.min() < min_bin_count:
            # We can skip extremely small bins (e.g. URLs of length 1000+) to remove outliers
            continue
            
        min_class_size = label_counts.min()
        
        # Resample each class in the bin to match the minority class size
        bin_balanced = []
        for label in labels:
            label_group = group[group[label_col] == label]
            sampled_label_group = label_group.sample(n=min_class_size, random_state=random_state)
            bin_balanced.append(sampled_label_group)
            
        balanced_chunks.append(pd.concat(bin_balanced))
        
    if not balanced_chunks:
        print("Error: No bins could be balanced. Check URL length overlap between classes.")
        return df
        
    balanced_df = pd.concat(balanced_chunks).reset_index(drop=True)
    balanced_df = balanced_df.drop(columns=['url_len', 'len_bin'])
    
    print(f"Dataset size after length balancing: {len(balanced_df)} samples")
    return balanced_df

def balance_tlds(df: pd.DataFrame,
                 url_col: str = 'url',
                 label_col: str = 'label',
                 max_tld_ratio: float = 0.25,
                 random_state: int = 42) -> pd.DataFrame:
    """
    Caps the representation of any single TLD in both classes to prevent TLD bias.
    If a TLD (like .com) makes up more than `max_tld_ratio` of a class, 
    we downsample it.
    """
    print(f"Balancing TLD distribution (max ratio per TLD = {max_tld_ratio})...")
    
    ext = tldextract.TLDExtract()
    df = df.copy()
    df['tld'] = df[url_col].apply(lambda u: ext(u).suffix.lower())
    
    balanced_chunks = []
    
    for label, group in df.groupby(label_col):
        tld_counts = group['tld'].value_counts()
        total_samples = len(group)
        max_allowed = int(total_samples * max_tld_ratio)
        
        label_tld_chunks = []
        for tld, count in tld_counts.items():
            tld_group = group[group['tld'] == tld]
            if count > max_allowed:
                # Downsample
                tld_group = tld_group.sample(n=max_allowed, random_state=random_state)
            label_tld_chunks.append(tld_group)
            
        balanced_chunks.append(pd.concat(label_tld_chunks))
        
    balanced_df = pd.concat(balanced_chunks).reset_index(drop=True)
    balanced_df = balanced_df.drop(columns=['tld'])
    
    print(f"Dataset size after TLD balancing: {len(balanced_df)} samples")
    return balanced_df
