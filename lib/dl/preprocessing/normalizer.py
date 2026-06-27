import re
from urllib.parse import urlparse, unquote
import pandas as pd
import tldextract

def normalize_url(url: str) -> str:
    """
    Normalizes a URL:
    - Lowercase scheme and host.
    - Decodes percent-encoded characters.
    - Removes fragments.
    - Strips leading/trailing spaces and trailing slashes.
    """
    if not isinstance(url, str):
        return ""
    
    url = url.strip()
    
    # URL Decode percent encoding (e.g., %20 -> space, %3D -> =)
    try:
        url = unquote(url)
    except Exception:
        pass
    
    # Parse URL
    try:
        # Standardize scheme if missing
        if not re.match(r'^[a-zA-Z]+://', url):
            url_to_parse = 'http://' + url
        else:
            url_to_parse = url
            
        parsed = urlparse(url_to_parse)
        
        # Lowercase scheme and host
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        
        # Reconstruct path and query, remove fragment
        path = parsed.path
        if path.endswith('/') and len(path) > 1:
            path = path[:-1]
            
        query = parsed.query
        
        # Reconstruct normalized URL
        normalized = f"{scheme}://{netloc}{path}"
        if query:
            normalized += f"?{query}"
            
        return normalized
    except Exception:
        # Fallback to simple string cleaning if urlparse fails
        url = url.lower()
        url = url.split('#')[0]
        if url.endswith('/'):
            url = url[:-1]
        return url

def get_structural_signature(url: str, ext) -> tuple:
    """
    Computes a structural signature for a URL to detect near-duplicates.
    Signature consists of:
    1. Registered domain (domain + suffix)
    2. Path depth (number of / in path)
    3. Query parameter keys (sorted)
    4. Path length bin (to allow slight length variations)
    """
    try:
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        path_depth = len(path.split('/')) if path else 0
        path_len_bin = len(path) // 10  # Bin path length by chunks of 10 chars
        
        # Extract query keys
        query_keys = []
        if parsed.query:
            # Query keys separated by & or ;
            pairs = re.split(r'[&;]', parsed.query)
            for pair in pairs:
                parts = pair.split('=')
                if parts:
                    query_keys.append(parts[0])
        query_keys_sig = tuple(sorted(query_keys))
        
        # Extract registered domain
        extracted = ext(url)
        reg_domain = f"{extracted.domain}.{extracted.suffix}" if extracted.suffix else extracted.domain
        
        return (reg_domain, path_depth, path_len_bin, query_keys_sig)
    except Exception:
        return ("", 0, 0, ())

def deduplicate_dataset(df: pd.DataFrame, url_col: str = 'url', label_col: str = 'label') -> pd.DataFrame:
    """
    Performs URL normalization, exact deduplication, and structural near-deduplication.
    """
    print(f"Initial dataset shape: {df.shape}")
    
    # 1. Normalization
    print("Normalizing URLs...")
    df['normalized_url'] = df[url_col].apply(normalize_url)
    
    # Remove rows where normalization failed or became empty
    df = df[df['normalized_url'] != ""].copy()
    
    # 2. Exact Deduplication
    print("Removing exact duplicates...")
    before_exact = len(df)
    # If duplicates exist with conflicting labels, we drop them or keep the malicious one for safety
    # We sort by label/result to prioritize malicious (usually 1) if duplicate check is run
    if 'result' in df.columns:
        df = df.sort_values(by='result', ascending=False)
    
    df = df.drop_duplicates(subset=['normalized_url'], keep='first')
    after_exact = len(df)
    print(f"Removed {before_exact - after_exact} exact duplicates.")
    
    # 3. Structural Near-Deduplication
    # Pairwise Levenshtein is too slow (O(N^2)) for 600k URLs.
    # Instead, we construct a structural signature: registered domain + path structure + query structure.
    # Within the same domain, if URLs have the same path depth, path length bin, and query parameters,
    # they are very likely near-duplicates (e.g. paypal-login.com/index.php?id=1 and paypal-login.com/index.php?id=2).
    print("Extracting domain-level structures for near-deduplication...")
    ext = tldextract.TLDExtract()
    
    # Calculate signatures
    df['signature'] = df['normalized_url'].apply(lambda u: get_structural_signature(u, ext))
    
    # Remove near duplicates
    before_near = len(df)
    df = df.drop_duplicates(subset=['signature'], keep='first')
    after_near = len(df)
    print(f"Removed {before_near - after_near} near-duplicates based on structural similarity.")
    
    # Clean up temporary columns
    df = df.drop(columns=['signature'])
    
    print(f"Final deduplicated dataset shape: {df.shape}")
    return df
