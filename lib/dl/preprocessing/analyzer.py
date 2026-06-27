import os
import math
import numpy as np
import pandas as pd
from collections import Counter
from urllib.parse import urlparse
import tldextract
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_entropy(text: str) -> float:
    """Calculates the Shannon entropy of a string."""
    if not text:
        return 0.0
    entropy = 0.0
    length = len(text)
    counts = Counter(text)
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy

def get_subdomain_depth(url: str, ext) -> int:
    """Calculates subdomain depth (number of segments in subdomain)."""
    try:
        extracted = ext(url)
        subdomain = extracted.subdomain
        if not subdomain:
            return 0
        return len(subdomain.split('.'))
    except Exception:
        return 0

def get_tld(url: str, ext) -> str:
    """Gets the top-level domain (suffix)."""
    try:
        return ext(url).suffix.lower()
    except Exception:
        return ""

def count_special_chars(url: str) -> dict:
    """Counts special characters commonly biased in phishing URLs."""
    chars = ['?', '=', '&', '-', '.', '_', '@', '//']
    counts = {}
    for char in chars:
        counts[f'count_{char}'] = url.count(char)
    return counts

def analyze_distributions(df: pd.DataFrame, 
                         url_col: str = 'url', 
                         label_col: str = 'label', 
                         output_dir: str = None) -> pd.DataFrame:
    """
    Computes distribution statistics for length, entropy, subdomain depth, 
    special characters, and TLDs for benign vs malicious URLs.
    Generates comparison plots if output_dir is provided.
    """
    print("Running distribution analysis...")
    # Drop rows with NaN in critical columns
    df = df.dropna(subset=[url_col, label_col]).copy()
    
    ext = tldextract.TLDExtract()
    
    # Feature extraction for analysis
    analysis_df = pd.DataFrame()
    analysis_df['label'] = df[label_col].astype(str)
    analysis_df['length'] = df[url_col].apply(len)
    analysis_df['entropy'] = df[url_col].apply(calculate_entropy)
    analysis_df['subdomain_depth'] = df[url_col].apply(lambda u: get_subdomain_depth(u, ext))
    analysis_df['tld'] = df[url_col].apply(lambda u: get_tld(u, ext))
    
    # Extract special character counts
    special_char_counts = list(df[url_col].apply(count_special_chars))
    char_df = pd.DataFrame(special_char_counts)
    analysis_df = pd.concat([analysis_df, char_df], axis=1)
    
    # Calculate summary statistics grouped by label
    stats = []
    features_to_summary = ['length', 'entropy', 'subdomain_depth'] + list(char_df.columns)
    
    for feature in features_to_summary:
        group_stats = analysis_df.groupby('label')[feature].agg(['mean', 'std', 'median', 'min', 'max'])
        group_stats['feature'] = feature
        stats.append(group_stats)
        
    summary_stats = pd.concat(stats).reset_index()
    print("\n=== DISTRIBUTION SUMMARY ===")
    print(summary_stats.to_string(index=False))
    
    # Top TLDs by label
    print("\n=== TOP TLD DISTRIBUTION BY LABEL ===")
    for label in analysis_df['label'].unique():
        tlds = analysis_df[analysis_df['label'] == label]['tld'].value_counts().head(5)
        print(f"\nLabel: {label}")
        for tld, count in tlds.items():
            pct = count / len(analysis_df[analysis_df['label'] == label])
            print(f"  .{tld}: {count} ({pct:.2%})")
            
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        
        # Plot 1: URL Length Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(data=analysis_df, x='length', hue='label', bins=100, kde=True, multiple="dodge")
        plt.title('URL Length Distribution (capped at 200 chars for visibility)')
        plt.xlim(0, 200)
        plt.xlabel('Length (characters)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'url_length_dist.png'), dpi=150)
        plt.close()
        
        # Plot 2: Entropy Distribution
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=analysis_df, x='entropy', hue='label', fill=True, common_norm=False, alpha=0.5)
        plt.title('Shannon Entropy Distribution')
        plt.xlabel('Entropy')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'url_entropy_dist.png'), dpi=150)
        plt.close()
        
        # Plot 3: Subdomain Depth
        plt.figure(figsize=(10, 6))
        sns.countplot(data=analysis_df, x='subdomain_depth', hue='label')
        plt.title('Subdomain Depth Distribution')
        plt.xlabel('Subdomain Depth')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'subdomain_depth_dist.png'), dpi=150)
        plt.close()
        
        # Plot 4: Special Characters Boxplots
        plt.figure(figsize=(12, 8))
        char_melted = pd.melt(analysis_df, id_vars=['label'], value_vars=list(char_df.columns), 
                              var_name='char', value_name='count')
        sns.boxplot(data=char_melted, x='char', y='count', hue='label')
        plt.title('Special Character Counts')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'special_chars_dist.png'), dpi=150)
        plt.close()
        
        print(f"\n✓ Distribution plots saved to {output_dir}")
        
    return summary_stats
