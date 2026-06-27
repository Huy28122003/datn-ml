import tldextract
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def extract_registered_domain(url: str, ext) -> str:
    """
    Extracts registered domain (domain + suffix) using tldextract.
    For example: 'https://sub.paypal-security.com/path' -> 'paypal-security.com'
    """
    try:
        extracted = ext(url)
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}"
        elif extracted.domain:
            return extracted.domain
        return ""
    except Exception:
        return ""

def split_by_domain(df: pd.DataFrame, 
                    url_col: str = 'url', 
                    label_col: str = 'label', 
                    train_ratio: float = 0.7, 
                    val_ratio: float = 0.15, 
                    test_ratio: float = 0.15,
                    random_state: int = 42) -> tuple:
    """
    Splits the dataset into Train, Validation, and Test sets based on registered domain.
    This guarantees that any registered domain appears in ONLY ONE of the splits.
    
    Returns:
        train_df, val_df, test_df
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Splits must sum to 1.0"
    
    print("Extracting registered domains for splitting...")
    ext = tldextract.TLDExtract()
    df['registered_domain'] = df[url_col].apply(lambda u: extract_registered_domain(u, ext))
    
    # Filter out empty domains (e.g., malformed URLs)
    df = df[df['registered_domain'] != ""].copy()
    
    # We will use GroupShuffleSplit to split domains
    # Step 1: Split train vs temp (validation + test)
    temp_ratio = val_ratio + test_ratio
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_state)
    
    train_idx, temp_idx = next(gss1.split(df, groups=df['registered_domain']))
    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()
    
    # Step 2: Split temp into validation and test
    # The ratio of validation in temp is val_ratio / temp_ratio
    val_in_temp_ratio = val_ratio / temp_ratio
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_in_temp_ratio, random_state=random_state)
    
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df['registered_domain']))
    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()
    
    # Clean up temporary columns
    train_df = train_df.drop(columns=['registered_domain'])
    val_df = val_df.drop(columns=['registered_domain'])
    test_df = test_df.drop(columns=['registered_domain'])
    
    print("\n--- Split Summary (Domain-Disjoint) ---")
    print(f"Total dataset: {len(df)} samples")
    print(f"Train set:      {len(train_df)} samples ({len(train_df)/len(df):.2%})")
    print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df):.2%})")
    print(f"Test set:       {len(test_df)} samples ({len(test_df)/len(df):.2%})")
    
    # Verify no overlap
    ext2 = tldextract.TLDExtract()
    train_domains = set(train_df[url_col].apply(lambda u: extract_registered_domain(u, ext2)))
    val_domains = set(val_df[url_col].apply(lambda u: extract_registered_domain(u, ext2)))
    test_domains = set(test_df[url_col].apply(lambda u: extract_registered_domain(u, ext2)))
    
    overlap_train_val = train_domains.intersection(val_domains)
    overlap_train_test = train_domains.intersection(test_domains)
    overlap_val_test = val_domains.intersection(test_domains)
    
    print(f"Overlap (Train & Val):  {len(overlap_train_val)} domains")
    print(f"Overlap (Train & Test): {len(overlap_train_test)} domains")
    print(f"Overlap (Val & Test):   {len(overlap_val_test)} domains")
    
    assert len(overlap_train_val) == 0, "Leakage detected between Train and Val!"
    assert len(overlap_train_test) == 0, "Leakage detected between Train and Test!"
    assert len(overlap_val_test) == 0, "Leakage detected between Val and Test!"
    print("✓ Verification successful: 0 domain leakage!")
    
    return train_df, val_df, test_df
