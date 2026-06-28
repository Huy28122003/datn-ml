import re
import ipaddress
from urllib.parse import urlparse, parse_qs
COMMON_TLDS = {'com', 'org', 'net', 'edu', 'gov', 'mil', 'int', 'co', 'io', 'me', 'info', 'biz', 'name', 'mobi', 'pro', 'aero', 'asia', 'cat', 'coop', 'jobs', 'museum', 'tel', 'travel', 'uk', 'us', 'ca', 'de', 'fr', 'jp', 'cn', 'ru', 'br', 'au', 'in', 'it', 'nl', 'se', 'no', 'es', 'pl', 'ch', 'at', 'be', 'dk', 'fi', 'ie', 'pt', 'cz', 'hu', 'ro', 'bg', 'hr', 'sk', 'si', 'lt', 'lv', 'ee', 'mt', 'cy', 'lu', 'is', 'li', 'mc', 'vn', 'th', 'ph', 'my', 'sg', 'id', 'kr', 'tw', 'hk', 'za', 'ng', 'ke', 'eg', 'gh', 'tz', 'ma', 'dz', 'tn', 'ar', 'cl', 'co', 'mx', 'pe', 've', 'ec', 'uy', 'py', 'nz', 'pk', 'bd', 'lk', 'np', 'mm', 'kh', 'la', 'xyz', 'top', 'wang', 'club', 'vip', 'shop', 'site', 'online', 'store', 'tech', 'space', 'fun', 'website', 'press', 'host', 'buzz', 'blog', 'art', 'dev', 'app', 'page', 'cloud'}
URL_SHORTENERS = {'bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'ow.ly', 'is.gd', 'buff.ly', 'adf.ly', 'bit.do', 'mcaf.ee', 'su.pr', 'rebrand.ly', 'cutt.ly', 'short.io', 'rb.gy', 'tiny.cc', 'shorturl.at', 'v.gd', 'clck.ru', 'qr.ae'}

def _count_char(text, char):
    if text is None:
        return -1
    return text.count(char)

def _count_vowels(text):
    if text is None:
        return -1
    return sum((1 for c in text.lower() if c in 'aeiou'))

def _is_ip_address(domain):
    try:
        ipaddress.ip_address(domain)
        return 1
    except ValueError:
        return 0

def _check_server_client(domain):
    if domain is None:
        return -1
    domain_lower = domain.lower()
    if 'server' in domain_lower or 'client' in domain_lower:
        return 1
    return 0

def _count_tld_in_url(url):
    url_lower = url.lower()
    count = 0
    for tld in COMMON_TLDS:
        if f'.{tld}' in url_lower:
            count += 1
    return max(count, 1)

def _extract_special_chars(text):
    if text is None:
        return {'dot': -1, 'hyphen': -1, 'underline': -1, 'slash': -1, 'questionmark': -1, 'equal': -1, 'at': -1, 'and': -1, 'exclamation': -1, 'space': -1, 'tilde': -1, 'comma': -1, 'plus': -1, 'asterisk': -1, 'hashtag': -1, 'dollar': -1, 'percent': -1}
    return {'dot': text.count('.'), 'hyphen': text.count('-'), 'underline': text.count('_'), 'slash': text.count('/'), 'questionmark': text.count('?'), 'equal': text.count('='), 'at': text.count('@'), 'and': text.count('&'), 'exclamation': text.count('!'), 'space': text.count(' ') + text.count('%20'), 'tilde': text.count('~'), 'comma': text.count(','), 'plus': text.count('+'), 'asterisk': text.count('*'), 'hashtag': text.count('#'), 'dollar': text.count('$'), 'percent': text.count('%')}

def _parse_url_parts(url):
    if not url.startswith(('http://', 'https://', 'ftp://')):
        url = 'http://' + url
    parsed = urlparse(url)
    domain = parsed.netloc
    if ':' in domain and (not _is_ip_address(domain)):
        domain = domain.split(':')[0]
    path = parsed.path
    query = parsed.query
    if not path or path == '':
        directory = None
        filename = None
    else:
        last_slash_idx = path.rfind('/')
        if last_slash_idx == -1:
            directory = None
            filename = path
        else:
            directory = path[:last_slash_idx + 1]
            filename = path[last_slash_idx + 1:]
    params = query if query else None
    return {'full_url': url, 'domain': domain if domain else None, 'directory': directory, 'file': filename, 'params': params}

def extract_features_from_url(url):
    parts = _parse_url_parts(url)
    features = {}
    url_chars = _extract_special_chars(url)
    for (char_name, count) in url_chars.items():
        features[f'qty_{char_name}_url'] = count
    features['qty_tld_url'] = _count_tld_in_url(url)
    features['length_url'] = len(url)
    domain = parts['domain']
    domain_chars = _extract_special_chars(domain)
    for (char_name, count) in domain_chars.items():
        features[f'qty_{char_name}_domain'] = count
    features['qty_vowels_domain'] = _count_vowels(domain)
    features['domain_length'] = len(domain) if domain else -1
    features['domain_in_ip'] = _is_ip_address(domain) if domain else -1
    features['server_client_domain'] = _check_server_client(domain)
    directory = parts['directory']
    dir_chars = _extract_special_chars(directory)
    for (char_name, count) in dir_chars.items():
        features[f'qty_{char_name}_directory'] = count
    features['directory_length'] = len(directory) if directory is not None else -1
    filename = parts['file']
    file_chars = _extract_special_chars(filename)
    for (char_name, count) in file_chars.items():
        features[f'qty_{char_name}_file'] = count
    features['file_length'] = len(filename) if filename is not None else -1
    params = parts['params']
    params_chars = _extract_special_chars(params)
    for (char_name, count) in params_chars.items():
        features[f'qty_{char_name}_params'] = count
    features['params_length'] = len(params) if params is not None else -1
    if params is not None:
        tld_in_params = 0
        for tld in COMMON_TLDS:
            if tld in params.lower():
                tld_in_params = 1
                break
        features['tld_present_params'] = tld_in_params
        features['qty_params'] = len(parse_qs(params))
    else:
        features['tld_present_params'] = -1
        features['qty_params'] = -1
    email_pattern = '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}'
    features['email_in_url'] = 1 if re.search(email_pattern, url) else 0
    domain_lower = domain.lower() if domain else ''
    features['url_shortened'] = 1 if domain_lower in URL_SHORTENERS else 0
    return features

def extract_features_for_model(url, model_features):
    all_features = extract_features_from_url(url)
    feature_vector = []
    for feat_name in model_features:
        feature_vector.append(all_features.get(feat_name, -1))
    return feature_vector
