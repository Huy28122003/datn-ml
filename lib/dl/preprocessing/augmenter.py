import re
import random
import requests
import pandas as pd
import tldextract

# List of typical benign domains to use for synthetic complex benign generation if needed
BENIGN_DOMAINS = [
    "google.com", "github.com", "shopee.vn", "amazon.com", "youtube.com", "facebook.com",
    "microsoft.com", "apple.com", "wikipedia.org", "yahoo.com", "netflix.com", "linkedin.com",
    "chase.com", "bankofamerica.com", "paypal.com", "dropbox.com", "drive.google.com",
    "docs.google.com", "trello.com", "slack.com", "zoom.us", "spotify.com", "medium.com"
]

def fetch_phishtank() -> list:
    """Fetches live phishing URLs from PhishTank (JSON format)."""
    print("Fetching live phishing URLs from PhishTank...")
    url = "http://data.phishtank.com/data/online-valid.json"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        # Request with a timeout
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            urls = [item['url'] for item in data]
            print(f"  ✓ Fetched {len(urls)} URLs from PhishTank")
            return urls
        else:
            print(f"  ✗ PhishTank returned status code {response.status_code}")
    except Exception as e:
        print(f"  ✗ PhishTank download failed: {str(e)}")
    return []

def fetch_openphish() -> list:
    """Fetches live phishing URLs from OpenPhish (text format)."""
    print("Fetching live phishing URLs from OpenPhish...")
    url = "https://openphish.com/feed.txt"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            urls = [line.strip() for line in response.text.split('\n') if line.strip()]
            print(f"  ✓ Fetched {len(urls)} URLs from OpenPhish")
            return urls
        else:
            print(f"  ✗ OpenPhish returned status code {response.status_code}")
    except Exception as e:
        print(f"  ✗ OpenPhish download failed: {str(e)}")
    return []

def fetch_urlhaus() -> list:
    """Fetches live phishing URLs from URLHaus (recent URLs text format)."""
    print("Fetching live URLs from URLHaus...")
    url = "https://urlhaus.abuse.ch/downloads/text_recent/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            urls = []
            for line in response.text.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    urls.append(line)
            print(f"  ✓ Fetched {len(urls)} URLs from URLHaus")
            return urls
        else:
            print(f"  ✗ URLHaus returned status code {response.status_code}")
    except Exception as e:
        print(f"  ✗ URLHaus download failed: {str(e)}")
    return []

def fetch_live_malicious() -> pd.DataFrame:
    """Aggregates all live malicious URLs and returns them as a DataFrame."""
    phishtank_urls = fetch_phishtank()
    openphish_urls = fetch_openphish()
    urlhaus_urls = fetch_urlhaus()
    
    all_urls = list(set(phishtank_urls + openphish_urls + urlhaus_urls))
    print(f"Total aggregated live malicious URLs: {len(all_urls)}")
    
    return pd.DataFrame({
        'url': all_urls,
        'label': 'malicious',
        'result': 1
    })

def generate_hard_benign_urls(count: int) -> pd.DataFrame:
    """
    Synthesizes complex/hard benign URLs containing queries, depth,
    auth parameters, tracking parameters, CDNs, and API formats.
    """
    print(f"Generating {count} hard benign URLs...")
    generated = []
    
    # Path structures
    api_paths = [
        "/api/v1/users/profile", "/api/v2/auth/login", "/v3/repos/{repo}/issues",
        "/api/v1/products/{id}/reviews", "/api/v2/checkout/cart", "/oauth/authorize",
        "/oauth/token", "/v1/objects/{uuid}", "/cdn/static/js/main.{hash}.js",
        "/assets/images/banner_{id}.png", "/storage/v1/bucket/file.pdf"
    ]
    
    doc_paths = [
        "/document/d/{uuid}/edit", "/spreadsheets/d/{uuid}/edit", 
        "/presentation/d/{uuid}/view", "/file/d/{uuid}/download"
    ]
    
    shop_paths = [
        "/product/{id}", "/shop/category/{cat}", "/item/details/{id}.html",
        "/search", "/cart/add", "/order/status/{id}"
    ]
    
    # Query parameters templates
    query_templates = [
        "page={page}&limit=50&sort=desc",
        "utm_source={src}&utm_medium={med}&utm_campaign={camp}&click_id={uuid}",
        "redirect_uri=https%3A%2F%2F{dom}%2Fauth%2Fcallback&client_id={cid}&response_type=code&state={state}&scope=read_user",
        "q={query}&category={cat}&ref=nav_search",
        "token={token}&expires={exp}&user_id={uid}",
        "access_token={token}&type=bearer",
        "search={query}&page={page}",
        "id={id}&format=json&version=1.4"
    ]
    
    search_queries = ["phishing+detection", "deep+learning", "machine+learning+tutorial", "best+laptop+2026", "how+to+cook+pasta", "weather+today", "datn+phishing"]
    sources = ["google", "facebook", "newsletter", "affiliate", "partner", "twitter", "adwords"]
    mediums = ["cpc", "email", "social", "banner", "cpm", "referral"]
    campaigns = ["spring_sale", "brand_awareness", "rebranding_2026", "product_launch", "user_acquisition"]
    repos = ["flutter/flutter", "tensorflow/tensorflow", "scikit-learn/scikit-learn", "pandas-dev/pandas", "python/cpython"]
    categories = ["electronics", "clothing", "books", "home-decor", "sports", "automotive"]
    
    for _ in range(count):
        dom = random.choice(BENIGN_DOMAINS)
        scheme = random.choice(["https", "http"])
        
        # Decide category of URL
        url_type = random.choice(["api", "doc", "shop", "query", "oauth"])
        
        # Generate components
        r_id = random.randint(1000, 999999)
        r_uuid = f"{random.randint(100000, 999999)}-{random.randint(10,99)}a-{random.randint(10,99)}b-{random.randint(1000000, 9999999)}"
        r_hash = "".join(random.choices("abcdef0123456789", k=8))
        r_token = "".join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=32))
        r_cid = "".join(random.choices("0123456789", k=12))
        r_state = "".join(random.choices("abcdefgh012345", k=10))
        r_query = random.choice(search_queries)
        r_cat = random.choice(categories)
        r_page = random.randint(1, 10)
        
        path = ""
        query = ""
        
        if url_type == "api":
            path_tmpl = random.choice(api_paths)
            path = path_tmpl.replace("{repo}", random.choice(repos)).replace("{id}", str(r_id)).replace("{uuid}", r_uuid).replace("{hash}", r_hash)
            # API query params
            query = f"format=json&version=2.0&sig={r_state}"
        elif url_type == "doc":
            path_tmpl = random.choice(doc_paths)
            path = path_tmpl.replace("{uuid}", r_uuid)
            query = f"usp=sharing&authuser={random.randint(0,2)}"
        elif url_type == "shop":
            path_tmpl = random.choice(shop_paths)
            path = path_tmpl.replace("{id}", str(r_id)).replace("{cat}", r_cat)
            query = f"ref=item_show_recs&spm={random.randint(1000,9999)}.{random.randint(100,999)}"
        elif url_type == "oauth":
            path = "/oauth/authorize"
            query_tmpl = query_templates[2] # oauth
            query = query_tmpl.replace("{dom}", dom).replace("{cid}", r_cid).replace("{state}", r_state)
        else: # general query / tracking
            path = "/search"
            query_tmpl = random.choice([query_templates[0], query_templates[1], query_templates[3], query_templates[4]])
            query = query_tmpl.format(
                page=r_page, src=random.choice(sources), med=random.choice(mediums),
                camp=random.choice(campaigns), uuid=r_uuid, dom=dom, cid=r_cid, state=r_state,
                query=r_query, cat=r_cat, token=r_token, exp=random.randint(1600000000, 1800000000),
                uid=r_id, id=r_id
            )
            
        # Standardize subdomains
        sub = ""
        if dom in ["google.com", "github.com", "microsoft.com"]:
            if url_type == "api":
                sub = "api."
            elif url_type == "doc" and dom == "google.com":
                sub = "docs."
            elif url_type == "oauth":
                sub = "accounts."
            else:
                sub = "www."
        else:
            sub = "www." if random.random() > 0.3 else ""
            
        url_str = f"{scheme}://{sub}{dom}{path}"
        if query:
            url_str += f"?{query}"
            
        generated.append(url_str)
        
    return pd.DataFrame({
        'url': generated,
        'label': 'benign',
        'result': 0
    })
