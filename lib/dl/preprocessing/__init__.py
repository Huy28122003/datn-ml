# Phishing URL Dataset Optimization Package
from .normalizer import normalize_url, deduplicate_dataset
from .splitter import split_by_domain
from .analyzer import analyze_distributions
from .augmenter import fetch_live_malicious, generate_hard_benign_urls
from .balancer import filter_malformed_urls, balance_by_length, balance_tlds
from .pipeline import run_pipeline
