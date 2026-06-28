import os
import sys
import pickle
import time
import urllib.parse
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'output', 'knn_search', 'knn_model.pkl')
REPORT_PATH = os.path.join(BASE_DIR, 'output', 'knn_search', 'batch_test_report.txt')

URL_LIST = [
    "https://advana.lat/",
    "https://httpp-roblox.co/users/1704538081/profile",
    "http://omerguzelim.github.io/Facebook",
    "https://www.paguesempararbr.site/",
    "http://spectrumwebmail22.systeme.io/",
    "https://freeflow.rodoviasabr.site/",
    "https://rhinomadeusainc.shop/index.php/campaigns/xg327p5nl2403/web-version/nz391gb6wtbc8",
    "https://rodovlas-freeflow-2vla.online/",
    "https://freeflow.br-rodovias.shop/",
    "http://ckuygfdr.icu/",
    "https://www.toki-gov-tr-fatih.vercel.app/",
    "https://www.bancolaumentatucupo-net.vercel.app/",
    "http://yjasur039-cyber.github.io/instagram-main/",
    "http://sillomoo.site/wp/",
    "https://ipfs.io/ipfs/bafkreiazyj2gljuogsm4afghneayisrjwpi66ocpefoy44jkaf6go45n6a",
    "http://rbxmulti.net/",
    "https://www.borderclick.com/BC/media/Borderclick/Files/free-robux-2026.html",
    "https://rhinomadeusainc.shop/index.php/campaigns/xg327p5nl2403/web-version/wk938dx056654",
    "https://allegro-lokalnie.8h723810.xyz/lokalnie/?id=3Y8u9D0i7a8a4C4N8n7A4t3e7Z5X3l2t",
    "https://imztoken.com/security",
    "https://api.bp-web-whatsapp.com.cn/",
    "https://sumitsrivastava01.github.io/Amazon-Clone/",
    "https://sp10ct6-glurnex-biz-slatik-platem.pages.dev/",
    "https://canburnlikeacigarette.github.io/vibexarr/",
    "https://sabiboys11211-cyber.github.io/verify-savingsinvestments/",
    "http://www.xfinityrefunds.com/TokenLogin",
    "https://rhinomadeusainc.shop/index.php/campaigns/xg327p5nl2403/web-version/pq394o70yyea8",
    "https://roblox.iconce.com/rebeccantss",
    "http://ilpendolo.it/Alphabnakgr/Login.html",
    "http://tribu-tax.cl/components/com_media/upgrade",
    "https://www.robiox.com.py/games/1537690962/Bee-Swarm-Simulator?privateServerLinkCode=856133174663442586082258522371",
    "https://www.web-git-06-09-fixwebtdpallnetworksoptionupdatest-619c96-uniswap.vercel.app/",
    "https://www.web-git-arc-chain-support-uniswap.vercel.app/",
    "http://nerdstarcode.github.io/Clone-Instagram",
    "https://web-git-06-09-fixwebtdpallnetworksoptionupdatest-619c96-uniswap.vercel.app/explore",
    "https://dd1.qzz.io/dana/t0f/",
    "https://ig.do/robloxusers322367312458profile",
    "https://www.galaxypedals.com/",
    "http://www.dgyuehuiwan.com/",
    "https://barclays-grads.twineapp.com/password/invalid-tenant",
    "https://exodus-authorized.com/authentication",
    "https://www.roblox.com.et/users/532856562524/profile",
    "https://24761a05n2-netizen.github.io/netflix_webpage/",
    "http://m.1adt42o0cbo5.com/account/reg?code=20611",
    "https://swapnil-2004.github.io/Netflix_Clone-2024/",
    "https://paypal.authia-rule.xyz/",
    "http://docusign.documentfileappsuitesverification8877584990303094847578494.dlnllc.com/",
    "http://https-www-roblox.cc/users/21713489/profile",
    "http://pokerklas.yeni-lisans.icu/",
    "https://rhinomadeusainc.shop/index.php/campaigns/xg327p5nl2403/web-version/dl265g13gr0a6",
    "https://rhinomadeusainc.shop/index.php/campaigns/xg327p5nl2403/web-version/ky227ebtxeff7",
    "http://bm03.ru/zxn110/pay/",
    "https://pub-f8fde659e2dc4783b121a5445d3d74c0.r2.dev/wmx.html?Faurecia=galix7@af2a53fdf681ce9589746327d14fb4234cb5.com/",
    "https://roblox.com.ug/games/133553556652607/visit7s-Place?privateServerLinkCode=25580346291535101991913016496821",
    "https://www.roblox.com.bi/games/136066387156306/Be-Flash-For-Brainrots?privateServerLinkCode=29383763406896541593443970804931&game_id=136066387156306&game_name=Be-Flash-For-Brainrots",
    "https://rhinomadeusainc.shop/index.php/campaigns/xg327p5nl2403/web-version/zv933cpmkpcfb",
    "http://fin103.grayku.com/",
    "https://toki-gov-tr-fatih.vercel.app/giris.php/",
    "https://www.shoptkpro.com/",
    "http://www.is.gd/zqJ6ve/",
    "https://www.xptspecielty.com/",
    "http://ssusshi.com/",
    "http://web-testing-dev-entry-gateway-uniswap.vercel.app/positions",
    "http://web-testing-dev-entry-gateway-uniswap.vercel.app/portfolio",
    "http://web-testing-dev-entry-gateway-uniswap.vercel.app/swap",
    "http://web-testing-dev-entry-gateway-uniswap.vercel.app/explore",
    "http://www.pruebabancoltarjeta1.vercel.app/",
    "http://tugcekaner.github.io/movie_clone",
    "http://teachersamanthasilva.github.io/CloneHomeInstagram",
    "https://estebangt1029.github.io/Clon-Instagram",
    "http://fabioldp.github.io/Rercriando-a-Pagina-inicial-do-Instagram",
    "https://m.1188xinpujing.com/",
    "https://salathebox.com/wp-content/id.handel.mobile.de-login-service/",
    "https://www.roblox.com.mu/users/2814540882/profile",
    "https://pl.funtrip360.com/wp-content/uploads/2025/?ref=a8847c04c3ed737e6146c9bd98888cee-xcjrGmf1ZhrXqJAV99ehe4VUjlRLitHGeuJvGuWuS5I3",
    "https://www.martumaswallt.darlic.com/",
    "https://martumaswallt.darlic.com/",
    "http://hulyacrk.github.io/edevlet",
    "https://enlineagalicia.github.io/personas/",
    "https://webtrafficguard2.com/",
    "https://bdvenlineaw.github.io/galicia.com",
    "https://www.roblox.com.mu/users/6137387333/profile",
    "https://linkedin.securityreview.casa/",
    "https://www.roblox.com.bi/users/546359157504/profile",
    "http://www.crosss.top/crs/verifycertifjcate.php",
    "https://aktifkaanpaylatterss.qes.my.id/",
    "https://secure-myxfinity-update.framer.website/",
    "https://www.amzonin.com/",
    "https://aktiifkanpaylater.vwk.my.id/",
    "https://extra-cyan-zfxl6ma9.edgeone.app/",
    "https://www.roblox.com.et/games/115054138215106/RP-Tuning?privateServerLinkCode=19696371439770339491581196338406&game_id=115054138215106&game_name=RP-Tuning",
    "http://www.inoutlconfirmar.webcindario.com/",
    "http://fnworth.com/",
    "http://business-blue-tick-authorized-000012026v34.pages.dev/",
    "https://7-0070hu.vercel.app/",
    "http://pricefn.top/",
    "https://sites.google.com/l0gin-microsoftwebonlne.app/nuj78hu77/home/",
    "https://rhinomadeusainc.shop/index.php/campaigns/xg327p5nl2403/web-version/gs693lmw6k4ca",
    "https://rhinomadeusainc.shop/index.php/campaigns/xg327p5nl2403/track-url/gs693lmw6k4ca/07f2af22139b1b85fe91b7f69c6c6bffe720de3b",
    "https://hjc23.net/",
    "https://rhinomadeusainc.shop/index.php/campaigns/xg327p5nl2403/web-version/tt206zfzvsfcd",
    "https://rhinomadeusainc.shop/index.php/campaigns/xg327p5nl2403/track-url/tt206zfzvsfcd/07f2af22139b1b85fe91b7f69c6c6bffe720de3b",
    "https://rhinomadeusainc.shop/index.php/campaigns/xg327p5nl2403/track-url/qo296m6m8y90c/07f2af22139b1b85fe91b7f69c6c6bffe720de3b",
    "https://rhinomadeusainc.shop/index.php/campaigns/xg327p5nl2403/web-version/qo296m6m8y90c",
    "https://qorvanel-biz-lomvireta-k7t4pz19.pages.dev/guyywl-fdfscx-lkljhjh?welcome=223625891861766&class246=223625891861766&name_class136=Teton%20County%20Health%20Department",
    "http://fortinatexpress.shop/",
    "https://www.r.oblox.com.et/games/109983668079237/Steal-a-Brainrot?privateServerLinkCode=67017710470720403395908687292184",
    "http://gokhanbyk.github.io/Netflix-loginPage-clone",
    "http://rbxliy.com/",
    "https://rhinomadeusainc.shop/index.php/campaigns/xg327p5nl2403/web-version/pb469fh797d04",
    "https://spotify.servernotification.events/premium/campaign/rejoin6m/login/ybLO0xAsYb5CuAW1XCPZSaVKzVb0-Bdr0UUU=0AQ==2QUJdRltUS21eXVVbXA==/f_0_udCHKeYuAV5eLolBcApFxggXCyTb",
    "https://authorized-website-link.replit.app/?email=elyn8@df1b3332b039ed6674bdf9cd8419de0ed772.com",
    "https://bdvenlineaw.github.io/galicia.com/",
    "https://citibankonlline.com/?ch=1&js=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJKb2tlbiIsImV4cCI6MTc4MjUyNTQ0MywiaWF0IjoxNzgyNTE4MjQzLCJpc3MiOiJKb2tlbiIsImpzIjoxLCJqdGkiOiIzMnVjZHBnYmJjYmRxc3RxMGsxYjJrOG4iLCJuYmYiOjE3ODI1MTgyNDMsInRzIjoxNzgyNTE4MjQzODY3MDc1fQ._ZK5ZsoAjugXIYUK-0Vwh8-sQLSkVOn_Fv1Ofj0ExgI&sid=c6d1481a-71ba-11f1-ad0b-fa3939b2a76a",
    "https://toki-gov-tr-konut1.vercel.app/giris.php",
    "https://web-git-06-10-chorewebremovemultichaintokenuxfla-ee0b7b-uniswap.vercel.app/explore",
    "https://web-git-06-10-chorewebremovemultichaintokenuxfla-ee0b7b-uniswap.vercel.app/positions",
    "https://web-git-06-10-chorewebremovemultichaintokenuxfla-ee0b7b-uniswap.vercel.app/portfolio",
    "https://web-git-06-10-chorewebremovemultichaintokenuxfla-ee0b7b-uniswap.vercel.app/swap",
    "https://www.roblox.com.bn/games/75888315541325/Push-Rock-for-Brainrots?privateServerLinkCode=32521013251177191579040166837938",
    "https://www.svrbancolombianeg.vercel.app/",
    "http://auth--apps-kucooin-sso---cdnn.webflow.io/",
    "https://recargabundles.com/offer/?gad_source&#61",
    "https://creditoenlineaecuador-2026-w-23--ecuadorcredire.replit.app/",
    "https://nelienaei-7837--lineasnejj2.replit.app/",
    "https://cliente-i-limpfeirao.shop/",
    "http://svrbancolombianeg.vercel.app/",
    "http://dejadirsbm-intereccanad.replit.app/inter~cas/dejsr/app/index.html",
    "https://aifreebie.top/",
    "http://www.instagramcom-instagramcom.blogspot.com/",
    "https://www.roblox.com.bi/games/84575720768520/Lucky-Block-Rush?privateServerLinkCode=29383763406896541593443970804931&game_id=84575720768520&game_name=Lucky-Block-Rush",
    "https://web-git-05-20-featwebshowread-onlytokeninfoforex-4642a4-uniswap.vercel.app/positions",
    "https://web-git-05-20-featwebshowread-onlytokeninfoforex-4642a4-uniswap.vercel.app/?intro=true",
    "https://web-git-05-20-featwebshowread-onlytokeninfoforex-4642a4-uniswap.vercel.app/swap",
    "https://www.eliaruizwork.com/portfolio",
    "http://03365pay.com/",
    "https://areaexclusiva.s3.us-east-005.backblazeb2.com/aumento.html",
    "https://www.robiox.com.gr/users/6955289390/profile/",
    "http://jwgl.my.m.luxurylifebrand.com/",
    "https://www.borderclick.com/BC/media/Borderclick/Files/free-7up-robux.html",
    "https://www.roblox.com.ml/users/189325325212/profile",
    "https://inoutlconfirmar.webcindario.com/",
    "https://enlineaecuadorcredito-2026-enlinea--serviciosubii.replit.app/",
    "https://payme-paypal.com/login",
    "http://ig.do/usersroblox8898154689profile",
    "https://icloud.com.gr/njH/l/cnD",
    "http://icloud.com.gr/njH/lcnD",
    "https://nwp6qjrhd.kapsalontulp.nl/",
    "https://zeqaw.xyz/KKFzS/z6hwy2nfkrAeTezUP5w2F2ifVQP1vHpSNh7TFXqjmiWIeJwGVS6RQJTD0z9id54P8990hjm7A2DYnH0HjQ-Q/1782486964766/",
    "http://www.kronosquantity.vip/",
    "http://payme-paypal.com/",
    "http://metamasklogino.webflow.io/",
    "https://expensestatus.com/2513501.doc/18a80a/fad0f483-81b2-45c6-ad47-7272058d9cb6/",
    "http://bancolaumentatucupo-net.vercel.app/",
    "https://www.sonomama-net.jp/Aeroplan/shaw/index.html?id=033"
]
PUBLIC_SUFFIXES = {
    'com.vn', 'co.uk', 'com.br', 'com.cn', 'com.tr', 'com.mu', 'com.ug', 'com.bi', 'com.py', 
    'com.gr', 'com.et', 'com.bn', 'net.cn', 'gov.tr', 'gov.vn', 'org.vn', 'my.id', 'ac.uk'
}

def extract_domain_parts(url_str):
    url_str = url_str.strip()
    if not url_str:
        return "", []
    if not url_str.startswith(('http://', 'https://')):
        url_str_for_parse = 'http://' + url_str
    else:
        url_str_for_parse = url_str
    try:
        parsed = urllib.parse.urlparse(url_str_for_parse)
        hostname = parsed.netloc.lower()
        if ':' in hostname:
            hostname = hostname.split(':')[0]
        if hostname.startswith('www.'):
            hostname = hostname[4:]
    except Exception:
        hostname = url_str.lower()

    labels = hostname.split('.')
    if len(labels) <= 1:
        return hostname, []

    suffix_len = 1
    if len(labels) >= 2:
        last_2 = ".".join(labels[-2:])
        if last_2 in PUBLIC_SUFFIXES:
            suffix_len = 2
        elif len(labels) >= 3:
            last_label = labels[-1]
            prev_label = labels[-2]
            if len(last_label) == 2 and prev_label in {'com', 'co', 'net', 'org', 'edu', 'gov', 'ac'}:
                suffix_len = 2

    reg_labels = labels[-(suffix_len + 1):]
    registered_domain = ".".join(reg_labels)
    subdomain_labels = labels[:-(suffix_len + 1)]
    return registered_domain, subdomain_labels

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def levenshtein_similarity(s1, s2):
    dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return (1.0 - dist / max_len)

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Chua tim thay mo hinh tai {MODEL_PATH}")
        sys.exit(1)
        
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
        
    vectorizer = model_data['vectorizer']
    knn = model_data['knn']
    domains = model_data['domains']

    clean_brands = {d.split('.')[0] for d in domains if len(d.split('.')[0]) > 3}

    results = []
    counter_types = Counter()
    
    t_start = time.time()
    
    for idx, url in enumerate(URL_LIST, 1):
        reg_domain, sub_labels = extract_domain_parts(url)
        if not reg_domain:
            continue
            
        X = vectorizer.transform([reg_domain])
        distances, indices = knn.kneighbors(X, n_neighbors=1)
        matched_domain = domains[indices[0][0]]
        
        lev_score = levenshtein_similarity(reg_domain, matched_domain) * 100.0
        
        if lev_score == 100.0:
            status = " day la website cua 1 don vi uy tin hoac 1 thanh phan thuoc don vi uy tin"
            counter_types['legit'] += 1
        else:
            triggered_sub_brand = None
            triggered_matched_domain = None
            
            for label in sub_labels:
                if len(label) <= 3:
                    continue
                    
                X_sub = vectorizer.transform([label])
                sub_distances, sub_indices = knn.kneighbors(X_sub, n_neighbors=1)
                sub_matched_domain = domains[sub_indices[0][0]]
                sub_matched_brand = sub_matched_domain.split('.')[0]
                
                if label == sub_matched_brand and sub_matched_brand in clean_brands:
                    triggered_sub_brand = label
                    triggered_matched_domain = sub_matched_domain
                    break
            
            if not triggered_sub_brand:
                reg_domain_brand = reg_domain.split('.')[0]
                domain_words = reg_domain_brand.split('-')
                if len(domain_words) >= 2:
                    for word in domain_words:
                        if len(word) <= 3:
                            continue
                        
                        X_word = vectorizer.transform([word])
                        word_distances, word_indices = knn.kneighbors(X_word, n_neighbors=1)
                        word_matched_domain = domains[word_indices[0][0]]
                        word_matched_brand = word_matched_domain.split('.')[0]
                        
                        if word == word_matched_brand and word_matched_brand in clean_brands:
                            triggered_sub_brand = word
                            triggered_matched_domain = word_matched_domain
                            break
            
            if triggered_sub_brand:
                status = f" Canh bao: website khong thuoc thuong hieu uy tin nhung lai dang co tinh chen thuong hieu do vao duong dan (mao danh: {triggered_matched_domain})"
                counter_types['sub_spoof'] += 1
            else:
                if lev_score >= 80.0:
                    status = f"Canh bao: khong nam trong danh sach don vi uy tin da duoc xac thuc nhung lai qua giong don vi do (giong {matched_domain} tai {lev_score:.2f}%)"
                    counter_types['typo_danger'] += 1
                else:
                    status = f" An toan (Do tuong dong thap: {lev_score:.2f}%)"
                    counter_types['safe'] += 1
                
        results.append({
            'index': idx,
            'url': url,
            'reg_domain': reg_domain,
            'matched': matched_domain,
            'lev': lev_score,
            'status': status
        })
            
    elapsed = time.time() - t_start
    
    report = []
    report.append("=========================================================================================================")
    report.append("          BAO CAO PHAN TICH GIA MAO DOMAIN HANG LOAT (THEO QUY TAC PHAN CAP)")
    report.append
    report.append(f"Tong so URL thu nghiem:              {len(URL_LIST)}")
    report.append(f"Thoi gian thuc thi:                  {elapsed:.2f} giay")
    report.append("-" * 105)
    report.append(" THONG KE PHAN LOAI:")
    report.append(f"   Don vi uy tin chinh thuc (Khop 100%):            {counter_types['legit']} URL")
    report.append(f"   Mao danh thuong hieu o subdomain:                {counter_types['sub_spoof']} URL")
    report.append(f"  Qua giong don vi uy tin (Do tuong dong >= 80%):  {counter_types['typo_danger']} URL")
    report.append(f"   Khac biet hoan toan / An toan:                   {counter_types['safe']} URL")
    report.append("=========================================================================================================")
    report.append("")
    report.append(f"{'STT':<4} | {'Domain chinh':<45} | {'Ten mien sach giong nhat':<25} | {'Do Lev':<8} | {'Trang thai & Canh bao phan cap'}")
    report.append("-" * 160)
    
    for r in results:
        report.append(
            f"{r['index']:<4} | "
            f"{r['reg_domain'][:45]:<45} | "
            f"{r['matched'][:25]:<25} | "
            f"{r['lev']:>6.1f}% | "
            f"{r['status']}"
        )
    report.append("=" * 160)
    
    report_text = "\n".join(report)
    
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report_text)
        
    print(f"  ➔ Tep bao cao: {REPORT_PATH}")

if __name__ == '__main__':
    main()
