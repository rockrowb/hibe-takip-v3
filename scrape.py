#!/usr/bin/env python3
"""
KATMAN 1/3 — HAM VERİ TOPLAMA

Türkiye hibe/destek programı duyurularını sources.json'daki kaynaklardan tarar
ve SADECE data/raw.json'a yazar. Hiçbir sınıflandırma/AI işlemi burada
YAPILMAZ — bu bilinçli bir tasarım tercihi:

  data/raw.json         <- bu script yazar (ham: başlık/tarih/link)
  data/classified.json  <- classify.py yazar (+ ücretsiz olasi_tur etiketi)
  data/duyurular.json   <- enrich.py yazar (+ AI ile çıkarılan detaylar)

Neden ayrı katmanlar: sınıflandırma kuralları ya da AI mantığı değiştiğinde
(ör. yeni anahtar kelime eklendi, yeni AI alanı eklendi) SADECE ilgili script
yeniden çalıştırılır — siteleri tekrar taramaya (yavaş, kotaya bağlı) gerek
kalmaz. data/raw.json bir kere toplanan hiçbir kaydı SİLMEZ, sadece üzerine
ekler/günceller; bu yüzden bir kaynağın path'i bozulsa/değişse bile önceden
toplanmış kayıtlar kalıcı olarak saklanır.

Her kaynak için:
  1. sources.json'da tanımlı olası "path"ler sırayla denenir (ör. /duyurular,
     /destekler, /haberler...). İçinde yeterli sayıda link bulunan ilk sayfa
     kullanılır (basit bir "otomatik keşif" mekanizması).
  2. Sayfadaki linkler arasından "duyuru gibi görünenler" seçilir.
  3. Link'in bulunduğu blokta bir tarih aranır.

Not: Bazı siteler (özellikle JavaScript ile içerik yükleyen tek-sayfa
uygulamaları, ya da ortak bir bot-koruma/CAPTCHA meydan okuma sayfası
gösteren sağlayıcılar) bu basit HTTP+HTML yaklaşımıyla taranamaz — sayfa
her seferinde neredeyse aynı boyutta, boş bir "meydan okuma" HTML'i döner.
Bu durumda script otomatik olarak PLAYWRIGHT (gerçek, JavaScript çalıştıran
bir tarayıcı motoru) ile YENİDEN dener — bu, "mevka gibi" siteler dahil,
normal isteklerle açılamayan HERHANGİ bir kaynak için otomatik bir yedek
mekanizmadır, elle işaretlemeye gerek yoktur.

Kullanım:
    python scrape.py                # tüm kaynakları tarar
    python scrape.py --only kosgeb  # sadece belirli bir kaynağı tarar (test için)
    python scrape.py --no-browser   # Playwright yedeğini kapat (daha hızlı, test için)
"""
import argparse
import re
import ssl
import sys
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
from urllib3.exceptions import InsecureRequestWarning
from bs4 import BeautifulSoup

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    # ÖNEMLİ: Accept-Encoding BİLİNÇLİ OLARAK burada YOK. requests/urllib3 bunu
    # otomatik ve ortamda gerçekten çözülebilen sıkıştırma yöntemlerine göre
    # (gzip/deflate, brotli kütüphanesi kuruluysa br) kendisi ayarlıyor. Bunu
    # burada sabit "br" olarak zorlarsak ve ortamda brotli decoder kurulu
    # değilse, sunucu Brotli ile sıkıştırılmış yanıt döner ama biz onu doğru
    # açamayız — sonuç: "HTTP 200" ama içerik bozuk/boş görünür, hiç link
    # bulunamaz. Bu tam olarak yaşanan sorunun kök nedeniydi.
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}
ROOT = Path(__file__).parent
RAW_FILE = ROOT / "data" / "raw.json"
SOURCES_FILE = ROOT / "sources.json"
TIMEOUT = 20
MIN_TITLE_LEN = 12
MIN_LINKS_TO_ACCEPT_PAGE = 4
RETRYABLE_STATUS = {500, 502, 503, 504}


class LegacySSLAdapter(HTTPAdapter):
    """Bazı eski/özel yapılandırılmış sunucular (ör. bazı .gov.tr siteleri)
    modern OpenSSL varsayılanlarıyla 'handshake failure' hatası veriyor.
    Bu adaptör güvenlik seviyesini bilinçli olarak biraz düşürüp (SECLEVEL=1)
    bu tip eski sunucularla da bağlantı kurabilmeyi sağlar. Sadece normal
    bağlantı SSLError ile başarısız olduğunda, ikinci deneme olarak
    kullanılır."""
    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.set_ciphers("DEFAULT@SECLEVEL=1")
        ctx.options |= getattr(ssl, "OP_LEGACY_SERVER_CONNECT", 0)
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


_legacy_session = requests.Session()
_legacy_session.mount("https://", LegacySSLAdapter())

NOISE_WORDS = {
    "anasayfa", "iletişim", "hakkımızda", "kurumsal", "giriş", "kayıt ol",
    "gizlilik", "çerez", "sıkça sorulan", "site haritası", "erişilebilirlik",
    "facebook", "twitter", "instagram", "linkedin", "youtube", "e-bülten",
    "devamını oku", "read more", "tıklayınız", "detaylı bilgi", "paylaş",
}

TR_MONTHS = {
    "oca": "01", "şub": "02", "sub": "02", "mar": "03", "nis": "04",
    "may": "05", "haz": "06", "tem": "07", "ağu": "08", "agu": "08",
    "eyl": "09", "eki": "10", "kas": "11", "ara": "12",
}
EN_MONTHS = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05",
    "jun": "06", "jul": "07", "aug": "08", "sep": "09", "oct": "10",
    "nov": "11", "dec": "12",
}


def parse_date(text):
    m = re.search(r"(\d{2})\.(\d{2})\.(\d{4})", text)
    if m:
        d, mo, y = m.groups()
        return f"{y}-{mo}-{d}"
    m = re.search(r"(\d{1,2})\s+([A-Za-zŞşĞğÜüÖöÇçİı]{3,})\s+(\d{4})", text)
    if m:
        d, mon, y = m.groups()
        mon_key = mon.lower()[:3].replace("i̇", "i")
        mnum = TR_MONTHS.get(mon_key) or EN_MONTHS.get(mon_key)
        if mnum:
            return f"{y}-{mnum}-{int(d):02d}"
    return None


def parse_all_dates(text):
    """Bloktaki TÜM tarihleri (tekrarsız, sırayla) döndürür — ör. bir kaynak
    listelemesinde 'Başlangıç: ... Bitiş: ...' gibi iki tarih birden
    geçiyorsa ikisini de yakalar. Bu, ka.gov.tr gibi sitelerde AI'ya hiç
    gitmeden doğrudan tarih bilgisi elde etmek için kullanılır."""
    found = []
    for d, mo, y in re.findall(r"(\d{2})\.(\d{2})\.(\d{4})", text):
        iso = f"{y}-{mo}-{d}"
        if iso not in found:
            found.append(iso)
    return found


def fetch(url):
    """Normal istek dener; SSL hatası alırsa önce esnetilmiş bir SSL bağlamıyla
    (eski/zayıf şifreleme kullanan sunucular için), o da başarısız olursa
    (özellikle 'sertifika doğrulanamadı' hatalarında — bazı .gov.tr siteleri
    eksik sertifika zinciri gönderiyor) sertifika doğrulamasını atlayarak son
    bir kez dener. 500/502/503/504 gibi geçici sunucu hatalarında kısa bir
    bekleme sonrası bir kez daha dener."""
    last_exc = None
    for attempt in range(2):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code in RETRYABLE_STATUS and attempt == 0:
                time.sleep(2.0)
                continue
            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser"), r.status_code, len(r.text)
        except requests.exceptions.SSLError as e:
            last_exc = e
            try:
                r = _legacy_session.get(url, headers=HEADERS, timeout=TIMEOUT)
                r.raise_for_status()
                return BeautifulSoup(r.text, "html.parser"), r.status_code, len(r.text)
            except requests.exceptions.SSLError as e2:
                # Son çare: sertifika zinciri eksik/bozuk sunucular için
                # doğrulamayı bilinçli olarak atla. Bu bir güvenlik ödünü
                # ama burada sadece herkese açık bir duyuru sayfası
                # okunuyor, hassas veri gönderilmiyor/alınmıyor.
                try:
                    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, verify=False)
                    r.raise_for_status()
                    return BeautifulSoup(r.text, "html.parser"), r.status_code, len(r.text)
                except Exception as e3:
                    last_exc = e3
                    break
            except Exception as e2:
                last_exc = e2
                break
        except requests.exceptions.HTTPError as e:
            last_exc = e
            if attempt == 0 and e.response is not None and e.response.status_code in RETRYABLE_STATUS:
                time.sleep(2.0)
                continue
            raise
    raise last_exc


_PLAYWRIGHT_AVAILABLE = None  # None = henüz kontrol edilmedi, True/False = kontrol sonucu


def playwright_available():
    """Playwright kurulu mu diye bir kez kontrol eder, sonucu önbelleğe alır.
    Kurulu değilse (ör. yerel test ortamında) sistem hiç çökmez, sadece
    tarayıcı yedeğini atlar."""
    global _PLAYWRIGHT_AVAILABLE
    if _PLAYWRIGHT_AVAILABLE is None:
        try:
            import playwright.sync_api  # noqa: F401
            _PLAYWRIGHT_AVAILABLE = True
        except ImportError:
            _PLAYWRIGHT_AVAILABLE = False
    return _PLAYWRIGHT_AVAILABLE


def fetch_with_playwright(url, wait_ms=4000):
    """Gerçek, JavaScript çalıştıran bir tarayıcı (headless Chromium) ile
    sayfayı açar. Basit requests isteğinin aynı boyutta boş bir 'meydan okuma'
    sayfasıyla karşılaştığı siteler için son çare yedek yöntemdir — bu tip
    korumalar genelde tarayıcı gibi davranan (JS çalıştıran, birkaç saniye
    bekleyen) isteklere gerçek sayfayı gösterir."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(user_agent=HEADERS["User-Agent"], locale="tr-TR")
            page.goto(url, timeout=30000, wait_until="networkidle")
            page.wait_for_timeout(wait_ms)  # JS meydan okuması/render için ekstra bekleme
            html = page.content()
        finally:
            browser.close()
    soup = BeautifulSoup(html, "html.parser")
    return soup, 200, len(html)


def looks_like_announcement(a, link_contains):
    title = a.get_text(strip=True)
    href = a.get("href", "")
    if not title or len(title) < MIN_TITLE_LEN:
        return False
    if title.lower() in NOISE_WORDS:
        return False
    if any(w in title.lower() for w in NOISE_WORDS) and len(title) < 20:
        return False
    if not href or href.startswith(("javascript:", "#", "mailto:", "tel:")):
        return False
    if link_contains:
        if not any(token in href for token in link_contains):
            return False
    return True


def scrape_page(url, link_contains, use_browser=False):
    if use_browser:
        soup, status, html_len = fetch_with_playwright(url)
    else:
        soup, status, html_len = fetch(url)
    candidates = soup.find_all("a", href=True)
    entries = []
    for a in candidates:
        if not looks_like_announcement(a, link_contains):
            continue
        title = a.get_text(strip=True)
        href = urljoin(url, a["href"])
        block = a.find_parent(["div", "li", "article"]) or a
        block_text = block.get_text(" ", strip=True)
        date_iso = parse_date(block_text)
        # Ham, AI'sız tarih yakalama: bloktaki TÜM tarihler (ör. ka.gov.tr'de
        # "Teklif Teslimi Başlangıç/Bitiş Tarihi" gibi iki tarih birden
        # geçebiliyor). En sonuncusu genelde son başvuru/bitiş tarihidir —
        # panel bunu AI çalışmasa bile ön-tahmin olarak kullanabilir.
        all_dates = parse_all_dates(block_text)
        entries.append({"title": title, "url": href, "date": date_iso, "on_tarihler": all_dates})
    tarayici_notu = " [Playwright ile]" if use_browser else ""
    diag = f"HTTP {status}, {html_len} byte HTML, {len(candidates)} link, {len(entries)} olası duyuru{tarayici_notu}"
    return entries, diag


def scrape_source(source, use_browser_fallback=True):
    base = source["homepage"].rstrip("/")
    link_contains = source.get("link_contains") or []
    best = []
    tried = []
    all_diags = []  # denenen HER path'in sonucu — sadece sonuncusu değil, teşhis için hepsi saklanır
    for path in source["paths"]:
        url = base + path if path.startswith("/") else base + "/" + path
        tried.append(url)
        try:
            entries, diag = scrape_page(url, link_contains)
            all_diags.append(f"{path} -> {diag}")
        except requests.exceptions.Timeout:
            all_diags.append(f"{path} -> ZAMAN AŞIMI ({TIMEOUT}sn içinde yanıt gelmedi)")
            print(f"  [{source['id']}] {all_diags[-1]}")
            continue
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            all_diags.append(f"{path} -> HTTP {code}")
            print(f"  [{source['id']}] {all_diags[-1]}")
            continue
        except Exception as e:
            all_diags.append(f"{path} -> hata: {e}")
            print(f"  [{source['id']}] {all_diags[-1]}")
            continue
        entries = [e for e in entries if urlparse(e["url"]).netloc == urlparse(base).netloc]
        seen = set()
        uniq = []
        for e in entries:
            if e["url"] not in seen:
                seen.add(e["url"])
                uniq.append(e)
        if len(uniq) >= MIN_LINKS_TO_ACCEPT_PAGE:
            print(f"  [{source['id']}] OK: {url} -> {len(uniq)} olası duyuru")
            best = uniq
            break
        elif len(uniq) > len(best):
            best = uniq
        time.sleep(0.3)

    # --- PLAYWRIGHT YEDEĞİ ---
    # Normal (JS'siz) istekler hiçbir path'te yeterli içerik bulamadıysa —
    # tipik olarak bot-koruma/CAPTCHA meydan okuma sayfası ya da JS ile
    # render edilen bir site anlamına gelir — gerçek bir tarayıcı motoruyla
    # İLK path'i bir kez daha deneriz. Bu, elle "bu site JS gerektiriyor"
    # diye işaretlemeye gerek kalmadan otomatik çalışır.
    if not best and use_browser_fallback and playwright_available() and source["paths"]:
        first_path = source["paths"][0]
        url = base + first_path if first_path.startswith("/") else base + "/" + first_path
        try:
            print(f"  [{source['id']}] normal istek 0 sonuç verdi, Playwright ile deneniyor: {url}")
            entries, diag = scrape_page(url, link_contains, use_browser=True)
            all_diags.append(f"{first_path} (Playwright) -> {diag}")
            entries = [e for e in entries if urlparse(e["url"]).netloc == urlparse(base).netloc]
            seen, uniq = set(), []
            for e in entries:
                if e["url"] not in seen:
                    seen.add(e["url"]); uniq.append(e)
            if len(uniq) > len(best):
                best = uniq
                print(f"  [{source['id']}] Playwright OK: {len(uniq)} olası duyuru")
        except Exception as e:
            all_diags.append(f"{first_path} (Playwright) -> hata: {e}")
            print(f"  [{source['id']}] Playwright de başarısız: {e}")

    if not best:
        print(f"  [{source['id']}] UYARI: denenen hiçbir sayfada yeterli içerik bulunamadı: {tried}")
    for e in best:
        e["source"] = source["name"]
        e["source_id"] = source["id"]
    return best, " | ".join(all_diags)


def load_existing():
    if RAW_FILE.exists():
        return json.loads(RAW_FILE.read_text(encoding="utf-8"))
    return {"last_updated": None, "items": {}, "source_status": {}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Sadece bu source id'sini tara (test için)")
    parser.add_argument("--no-browser", action="store_true",
                         help="Playwright yedeğini kapat (daha hızlı, sadece requests ile test için)")
    args = parser.parse_args()

    sources = json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
    if args.only:
        sources = [s for s in sources if s["id"] == args.only]
        if not sources:
            print(f"'{args.only}' id'li kaynak sources.json içinde bulunamadı.")
            sys.exit(1)

    store = load_existing()
    items = store.get("items", {})
    source_status = store.get("source_status", {})
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    added_total = 0
    for source in sources:
        print(f"Taranıyor: {source['name']}")
        try:
            entries, diag = scrape_source(source, use_browser_fallback=not args.no_browser)
            source_status[source["id"]] = {
                "ok": len(entries) > 0, "checked": today, "found": len(entries),
                "diag": diag,  # HTTP durum kodu + kaç link/duyuru bulundu — "0 sonuç"un GERÇEK sebebini gösterir
            }
        except Exception as e:
            print(f"  [{source['id']}] KAYNAK ERİŞİLEMEDİ: {e}")
            source_status[source["id"]] = {"ok": False, "checked": today, "error": str(e)}
            entries = []

        added = 0
        for entry in entries:
            key = entry["url"]
            if key not in items:
                entry["first_seen"] = today
                items[key] = entry
                added += 1
            else:
                # ÖNEMLİ: kayıt SİLİNMEZ/değiştirilmez, sadece başlık/tarih tazelenir.
                items[key]["title"] = entry["title"] or items[key]["title"]
                if entry.get("date"):
                    items[key]["date"] = entry["date"]
                if entry.get("on_tarihler"):
                    items[key]["on_tarihler"] = entry["on_tarihler"]
        added_total += added
        print(f"  -> yeni: {added}")
        time.sleep(1.0)  # kaynaklar arasında kısa bekleme — art arda çok hızlı istek atıp
                          # bot-koruması tetiklemekten kaçınmak için

    store["items"] = items
    store["source_status"] = source_status
    store["last_updated"] = today
    RAW_FILE.parent.mkdir(exist_ok=True)
    RAW_FILE.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[raw.json] Toplam kayıt: {len(items)} | Bu çalıştırmada yeni: {added_total}")


if __name__ == "__main__":
    main()
