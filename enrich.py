#!/usr/bin/env python3
"""
KATMAN 3/3 — OPSİYONEL AI ZENGİNLEŞTİRME

data/classified.json'ı OKUR (data/raw.json değil — sınıflandırma katmanının
üzerine inşa edilir), data/duyurular.json'a YAZAR. Panel (index.html) sadece
data/duyurular.json'ı okur.

Her yeni (henüz "details" alanı olmayan) kayıt için TEK bir AI çağrısında:
  1. Sınıflandırma: bu GERÇEKTEN AÇIK/GÜNCEL bir hibe-destek çağrısı mı, yoksa
     haber/sonuç ilanı/genel bilgilendirme mi? ("tur" alanı)
  2. Eğer açık bir hibe çağrısıysa: kimler başvurabilir, hibe miktarı, son
     başvuru tarihleri (birden fazla dönem/aşama varsa hepsi), desteklenen
     temel hizmetler/aktiviteler, ve ilgili tema etiketleri (tekstil, yazılım,
     tarım vb.) metinden çıkarılır.

VERİ KATMANLARI VE NEDEN AYRI:
  data/raw.json         <- scrape.py (ham, asla silinmez/üzerine yazılmaz)
  data/classified.json  <- classify.py (ücretsiz olasi_tur etiketi, her
                            çalıştırmada raw.json'dan yeniden üretilebilir)
  data/duyurular.json   <- BU SCRIPT (+ AI detayları). Önceden AI ile
                            işlenmiş "details" alanları HER ZAMAN korunur —
                            classify.py/scrape.py'da bir değişiklik olsa bile
                            daha önce parası ödenmiş AI sonuçları kaybolmaz.

GÜVENCE 1 — Yeni kayıt yoksa AI'a KESİNLİKLE gidilmez:
  Script en başta "details" alanı olmayan kayıt var mı diye bakar. Hiç yoksa
  API sağlayıcısını sorgulamadan, hiçbir ağ isteği atmadan sys.exit(0) ile
  çıkar.

GÜVENCE 2 — Kota/limit dolarsa o ana kadarki ilerleme kaybolmaz:
  - Her kayıt işlendikten hemen sonra data/duyurular.json diske yazılır.
  - API "kota/limit doldu" tipi bir hata döndürürse (HTTP 429, ya da
    "insufficient_quota" / "RESOURCE_EXHAUSTED" / "rate_limit" içeren
    mesajlar) döngü kalan kayıtlara dokunmadan durur, kalanlar bir sonraki
    çalıştırmada otomatik kuyruğa girer.

ÜÇ SAĞLAYICI DESTEKLENİR — hangisinin anahtarı tanımlıysa öncelik sırasıyla o kullanılır:
  - ANTHROPIC_API_KEY tanımlıysa    -> Claude (claude-sonnet-4-6) (1. öncelik)
  - yoksa GEMINI_API_KEY tanımlıysa -> Google Gemini (gemini-2.5-flash) (2. öncelik)
  - yoksa GROQ_API_KEY tanımlıysa   -> Groq (llama-3.3-70b-versatile) (3. öncelik) —
    tamamen ücretsiz, kredi kartı istemeyen bir kademesi var (console.groq.com),
    Gemini'nin ücretsiz kotası yetersiz kalırsa/çalışmazsa iyi bir alternatif.
  - hiçbiri tanımlı değilse         -> script sessizce çıkar

Token tasarrufu katmanları:
  A) classify.py her kaydı ücretsiz anahtar kelime taramasından geçirip
     olasi_tur etiketler (hibe_olabilir / sonuc_olabilir / haber_olabilir / belirsiz).
  B) "sonuc_olabilir" VE "haber_olabilir" etiketli kayıtlar AI'ya HİÇ
     GÖNDERİLMEZ, ücretsiz olarak işaretlenir.
  C) "details" alanı zaten olan kayıtlar tekrar gönderilmez.

Kullanım:
    export ANTHROPIC_API_KEY=sk-ant-...    # ya da
    export GEMINI_API_KEY=AIza...          # ya da
    export GROQ_API_KEY=gsk_...
    python enrich.py                # bekleyen tüm kayıtları işler
    python enrich.py --limit 20     # tek çalıştırmada işlenecek üst sınır
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent
CLASSIFIED_FILE = ROOT / "data" / "classified.json"
DATA_FILE = ROOT / "data" / "duyurular.json"
HEADERS_WEB = {"User-Agent": "Mozilla/5.0 (compatible; HibeTakipBot/1.0)"}

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-4-6"

# Not: Google, gemini-2.5-flash modelini Ekim 2026'da kullanımdan kaldıracağını
# duyurdu. O tarihten sonra model adını güncel bir Gemini Flash modeliyle
# değiştirmek gerekebilir (ai.google.dev/api/generate-content'ten kontrol edin).
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_MODEL = "gemini-2.5-flash"

# Groq: tamamen ücretsiz, kredi kartı istemeyen bir kademesi var (2026 itibarıyla
# dakikada 30 / günde 14.400 istek limiti — bu sistemin ihtiyacının çok üzerinde).
# API'si OpenAI ile uyumlu format kullanıyor. console.groq.com'dan anahtar alınır.
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"
# Not: llama-3.3-70b-versatile'ın ücretsiz kademesi ~6.000 token/dakika (TPM)
# ile sınırlı — bizim her çağrımız (~2.850 token) bu sınırı 2-3 çağrıda
# doldurup "kota doldu" hatası veriyordu. llama-3.1-8b-instant çok daha
# yüksek TPM/RPD sunduğu için (bu sınıflandırma/çıkarım işi için kalitesi
# de yeterli) varsayılan bu yapıldı.

QUOTA_ERROR_SIGNS = [
    "429", "insufficient_quota", "resource_exhausted", "rate_limit",
    "quota", "too many requests", "billing",
]

# Panelde filtre olarak sunulacak sabit tema listesi. AI bu listeden 0-4 tane
# uygun olanı seçer; hiçbiri uymuyorsa boş bırakır (uydurma yeni etiket üretmez).
# DEĞİŞTİRİLEBİLİR: bu listeyi düzenleyip enrich.py'ı tekrar çalıştırman yeterli.
TEMA_LISTESI = [
    "tarım", "hayvancılık", "gıda", "tekstil", "turizm", "enerji", "çevre",
    "yazılım", "donanım", "inovasyon", "ar-ge", "girişimcilik", "kadın girişimciliği",
    "gençlik", "istihdam", "ihracat", "dijital dönüşüm", "yapay zeka", "sağlık",
    "eğitim", "kültür-sanat", "sosyal girişimcilik", "kentsel dönüşüm",
    "afet yönetimi", "ulaştırma", "savunma sanayii", "biyoteknoloji", "oyun",
    "e-ticaret", "sivil toplum",
]

# Sürekli/periyodik olarak açılan, tanınmış "program serisi" adları. AI bunlardan
# 0-2 tane eşleşeni seçer — serbest metin üretmediği için standart temalara göre
# DAHA AZ token harcar ve arama/filtrelemede tutarlı sonuç verir.
# DEĞİŞTİRİLEBİLİR: yeni bir sürekli program fark edersen buraya ekle.
PROGRAM_SERISI_LISTESI = [
    "Erasmus+", "Ufuk Avrupa (Horizon Europe)", "TÜBİTAK 1001", "TÜBİTAK 1501",
    "TÜBİTAK 1507", "TÜBİTAK 1512 (BiGG)", "TÜBİTAK 1812 (BiGG Yatırım)",
    "TEKNOFEST", "KOSGEB KOBİGEL", "KOSGEB Ar-Ge ve İnovasyon",
    "KOSGEB Girişimci Destek Programı", "IPARD (TKDK)", "SOGEP",
    "Yerel Kalkınma Hamlesi", "Sivil Düşün", "CSSP",
]

# "Kimler başvurabilir" için standart, kapsamlı etiket listesi. AI serbest metin
# YAZMAK yerine bu listeden 0-5 tane işaretler — bu hem daha az token harcar
# (üretim yerine seçim yapıyor) hem de filtrelemeyi tutarlı hale getirir.
# DEĞİŞTİRİLEBİLİR: ihtiyaca göre genişlet/daralt.
KIMLER_ETIKET_LISTESI = [
    "KOBİ", "büyük işletme", "girişimci/startup", "STK/dernek/vakıf",
    "üniversite/akademisyen", "araştırmacı", "kamu kurumu", "yerel yönetim",
    "kooperatif", "kadın girişimci", "genç (29 yaş altı)", "öğrenci",
    "sanatçı", "çiftçi/üretici", "esnaf/serbest meslek", "bireysel başvuru",
    "engelli birey", "eski hükümlü", "ihracatçı", "yazılım/teknoloji şirketi",
]

SYSTEM_PROMPT = f"""Sana bir Türkiye kamu/STK duyurusunun web sayfası metni verilecek.
Aşağıdaki şemada SADECE JSON döndür, başka hiçbir şey yazma (açıklama, markdown işareti vb. ekleme):

{{
  "tur": "hibe_duyurusu" | "yarisma_duyurusu" | "haber" | "sonuc_ilani" | "diger",
  "basvuruya_acik": true | false | null,
  "kimler_etiketleri": ["..."] (aşağıdaki KİMLER listesinden 0-5 tane uygun olanı seç),
  "kimler_basvurabilir": "kısa ek açıklama (listede karşılığı olmayan özel bir koşul varsa) veya null",
  "hibe_miktari_min_tl": sayı (ÖNCELİKLİ ALAN — sıralama/filtreleme için kullanılır. Alt/tek tutar TL cinsinden, sadece rakam, virgülsüz, ondalık yok — ör. 500000) veya null,
  "hibe_miktari_max_tl": sayı (aralık varsa üst sınır TL cinsinden) veya null (tek tutarsa min ile birebir aynı değeri yaz),
  "hibe_miktari": "İKİNCİL ALAN — yukarıdaki sayısal değerin insan tarafından okunacak, detaylı açıklaması (ör. '%75 hibe oranı, proje başına üst limit 500.000 TL, KDV hariç') veya null",
  "toplam_butce": "PROGRAMIN/ÇAĞRININ TOPLAMINDA ayrılan bütçe — tek bir başvurunun alacağı miktar DEĞİL, tüm çağrı için ayrılan toplam kaynak (ör. 'Alan Toplam Bütçesi: 13.1 Milyar Avro') — insan tarafından okunacak metin veya null",
  "son_basvuru_tarihleri": ["YYYY-MM-DD", "..."] veya null (birden fazla aşama/dönem varsa hepsini listele, tek tarihse tek elemanlı liste),
  "desteklenen_aktiviteler": "hangi temel hizmetler/faaliyetler/harcamalar destekleniyor, kısa liste veya null",
  "faaliyet_suresi": "desteklenen proje/faaliyetin süresi (ör. 'en fazla 18 ay', '6-24 ay arası') veya null",
  "temalar": ["..."] (aşağıdaki TEMA listesinden 0-4 tane uygun olanı seç),
  "program_serisi": ["..."] (aşağıdaki PROGRAM SERİSİ listesinden 0-2 tane eşleşen varsa seç, yoksa boş liste),
  "ozet": "1-2 cümlelik tarafsız özet"
}}

TEMA listesi (sadece bunlardan seç): {", ".join(TEMA_LISTESI)}

PROGRAM SERİSİ listesi (sadece bunlardan seç, metin bu programlardan birine AÇIKÇA aitse): {", ".join(PROGRAM_SERISI_LISTESI)}

KİMLER listesi (sadece bunlardan seç): {", ".join(KIMLER_ETIKET_LISTESI)}

"tur" alanını SIKI şekilde belirle:
- "hibe_duyurusu": metin AÇIKÇA yeni başvurulara açık, güncel bir hibe/destek/fon
  çağrısı olmalı (başvuru koşulları, son tarih veya başvuru şekli gibi somut
  bilgiler içermeli).
- "yarisma_duyurusu": hibe/fon değil ama ödüllü bir YARIŞMAYA başvuru çağrısıysa
  (ör. TEKNOFEST yarışmaları, inovasyon yarışmaları, tasarım yarışmaları) bunu
  kullan — hibe_duyurusu ile karıştırma, bunlar ayrı bir kategori.
- Sadece bir kurumdan/programdan genel bahseden, geçmişte açılmış bir çağrıyı
  hatırlatan ama şu an başvuru almayan, ziyaret/imza töreni/toplantı/video gibi
  genel haberler, ya da net bir çağrı içermeyen metinleri "hibe_duyurusu" SAYMA
  — bunlar "haber" ya da "diger" olsun.
- Kurumsal/idari içerikler (anket merkezi, bütçe uygulama sonuçları, faaliyet
  raporu, stratejik plan, insan kaynakları/personel ilanı, ihale ilanı,
  yönetim kurulu/genel kurul kararları, KVKK/gizlilik metinleri, denetim
  raporu gibi) KESİNLİKLE "diger" olsun, bunları asla "hibe_duyurusu" sayma.
- Başvuru sonuçları/kazananlar/asıl-yedek liste/yarışma sonucu açıklanıyorsa "sonuc_ilani".
- "basvuruya_acik": metinde başvuru tarihinin geçmiş/gelecek olduğu netse true/false yap,
  emin değilsen null bırak.
- HİBE MİKTARI ÇIKARIMINDA ÖZENLİ OL — önce sayıyı, sonra açıklamayı çıkar:
  1. Metinde TL cinsinden net bir rakam (ör. "500.000 TL", "1 milyon TL",
     "üst limit 2.000.000 TL") geçiyorsa, bunu hibe_miktari_min_tl/max_tl'ye
     SAYI olarak yaz (binlik ayraç/nokta/virgül OLMADAN, ör. 500000).
     "bin TL"/"milyon TL" gibi ifadeleri gerçek sayıya çevir (1 milyon TL -> 1000000).
  2. Aralık varsa (ör. "50.000 - 500.000 TL arası") min'e alt sınırı, max'e üst
     sınırı yaz. Tek bir tutar varsa iki alana da AYNI sayıyı yaz.
  3. Metinde döviz (EUR/USD/GBP) geçiyorsa TL'ye çevirmeye ÇALIŞMA, sayısal
     alanları null bırak (yanlış kur riskinden kaçın) — ama hibe_miktari
     (metin) alanına döviz tutarını olduğu gibi yazabilirsin.
  4. Sadece yüzde (%) oranı verilmiş, tutar belirtilmemişse sayısal alanları
     null bırak, oranı hibe_miktari metin alanına yaz.
  5. hibe_miktari (metin) alanını HER ZAMAN doldurmaya çalış — sayısal alan
     null olsa bile (ör. döviz cinsinden ya da sadece oran varsa), en azından
     insan-okunur açıklamayı ver.
- toplam_butce: hibe_miktari'nden FARKLI bir alan — hibe_miktari tek bir
  başvurucunun alabileceği miktar, toplam_butce ise TÜM ÇAĞRI/PROGRAM için
  ayrılan toplam kaynaktır. İkisi de metinde geçebilir, karıştırma.
- tur "hibe_duyurusu" ve "yarisma_duyurusu" DIŞINDAYSA kimler_etiketleri,
  kimler_basvurabilir, hibe_miktari, hibe_miktari_min_tl, hibe_miktari_max_tl,
  toplam_butce, son_basvuru_tarihleri, desteklenen_aktiviteler, faaliyet_suresi,
  temalar, program_serisi alanlarını null/boş bırak, sadece ozet'i doldur.
- Emin olmadığın alanları null bırak, metinde olmayan bilgiyi ASLA uydurma."""


def fetch_text(url, max_chars=6000):
    try:
        r = requests.get(url, headers=HEADERS_WEB, timeout=15)
        r.raise_for_status()
    except Exception as e:
        return None, str(e)
    ctype = r.headers.get("Content-Type", "")
    if "pdf" in ctype.lower() or url.lower().endswith(".pdf"):
        return None, "PDF - şimdilik atlanıyor"
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    return text[:max_chars], None


class QuotaExceeded(Exception):
    pass


def _check_quota_error(exc, response_text=""):
    blob = f"{exc} {response_text}".lower()
    if any(sign in blob for sign in QUOTA_ERROR_SIGNS):
        raise QuotaExceeded(str(exc))


def call_claude(api_key, page_text, system_prompt=None, max_tokens=600):
    resp = requests.post(
        ANTHROPIC_API_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": max_tokens,
            "system": system_prompt or SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": page_text}],
        },
        timeout=30,
    )
    if resp.status_code == 429:
        raise QuotaExceeded(f"HTTP 429: {resp.text[:200]}")
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        _check_quota_error(e, resp.text)
        raise
    data = resp.json()
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    text = re.sub(r"^```json|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


def call_gemini(api_key, page_text, system_prompt=None, max_tokens=600):
    url = GEMINI_API_URL.format(model=GEMINI_MODEL)
    resp = requests.post(
        url,
        params={"key": api_key},
        headers={"content-type": "application/json"},
        json={
            "system_instruction": {"parts": [{"text": system_prompt or SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": page_text}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
        },
        timeout=30,
    )
    if resp.status_code == 429:
        raise QuotaExceeded(f"HTTP 429: {resp.text[:200]}")
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        _check_quota_error(e, resp.text)
        raise
    data = resp.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    text = re.sub(r"^```json|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


def call_groq(api_key, page_text, system_prompt=None, max_tokens=600):
    resp = requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
                {"role": "user", "content": page_text},
            ],
        },
        timeout=30,
    )
    if resp.status_code == 429:
        raise QuotaExceeded(f"HTTP 429: {resp.text[:200]}")
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        _check_quota_error(e, resp.text)
        raise
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    text = re.sub(r"^```json|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


def get_provider():
    """Hangi anahtar(lar) tanımlıysa öncelik sırasıyla kullanılır:
    Anthropic > Gemini > Groq. Groq son sırada çünkü genelde en son eklenen
    yedek/deneme seçeneği oluyor; öncelik sırasını değiştirmek istersen bu
    fonksiyondaki sırayı değiştirmen yeterli."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", os.environ["ANTHROPIC_API_KEY"]
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini", os.environ["GEMINI_API_KEY"]
    if os.environ.get("GROQ_API_KEY"):
        return "groq", os.environ["GROQ_API_KEY"]
    return None, None


def call_ai(provider, api_key, page_text, system_prompt=None, max_tokens=600):
    if provider == "anthropic":
        return call_claude(api_key, page_text, system_prompt, max_tokens)
    elif provider == "gemini":
        return call_gemini(api_key, page_text, system_prompt, max_tokens)
    elif provider == "groq":
        return call_groq(api_key, page_text, system_prompt, max_tokens)
    raise ValueError(f"Bilinmeyen sağlayıcı: {provider}")


# Groq gibi sağlayıcılarda ücretsiz kademe limiti "günlük kota" değil,
# "dakikada token" (TPM) şeklinde olabiliyor — bu durumda 429 hatası kalıcı
# bir kota bitişi değil, sadece "bir dakika bekle" anlamına gelir. Bu yüzden
# 429 alındığında hemen pes etmek yerine BİR KEZ ~65 saniye bekleyip tekrar
# deniyoruz (TPM penceresi sıfırlanmış olur); o da başarısız olursa gerçekten
# kota/limit bitmiş kabul edip QuotaExceeded olarak yukarı fırlatıyoruz.
RATE_LIMIT_RETRY_WAIT_SECONDS = 65


def call_ai_with_retry(provider, api_key, page_text, system_prompt=None, max_tokens=600):
    try:
        return call_ai(provider, api_key, page_text, system_prompt, max_tokens)
    except QuotaExceeded as e:
        print(f"    (dakikalık limite takıldı, {RATE_LIMIT_RETRY_WAIT_SECONDS}sn bekleyip 1 kez daha deneniyor: {e})")
        time.sleep(RATE_LIMIT_RETRY_WAIT_SECONDS)
        return call_ai(provider, api_key, page_text, system_prompt, max_tokens)

def priority(item):
    order = {"hibe_olabilir": 0, "belirsiz": 1, "sonuc_olabilir": 2, "haber_olabilir": 2}
    return order.get(item.get("olasi_tur"), 1)


FREE_SKIP_DETAILS = {
    # SADECE çok net "sonuç ilanı" kalıpları ücretsiz elenir (ör. "kazananlar
    # açıklandı"). "haber_olabilir" (kurumsal/basın bülteni gibi görünenler)
    # ARTIK AI'a gönderiliyor — AI'ın sınıflandırmadaki rolünü güçlendirmek
    # için bilinçli tercih: anahtar kelime kalıpları yanılabilir (ör. "Anket
    # Merkezi" gerçekten alakasızdır ama farklı bir başlık yanlış pozitif
    # olabilir), AI son sözü söylesin.
    "sonuc_olabilir": "sonuc_ilani_tahmini",
}


def empty_details(tur_tahmini):
    return {
        "tur": tur_tahmini,
        "basvuruya_acik": False,
        "kimler_etiketleri": [],
        "kimler_basvurabilir": None,
        "hibe_miktari": None,
        "hibe_miktari_min_tl": None,
        "hibe_miktari_max_tl": None,
        "toplam_butce": None,
        "son_basvuru_tarihleri": None,
        "desteklenen_aktiviteler": None,
        "faaliyet_suresi": None,
        "temalar": [],
        "program_serisi": [],
        "ozet": None,
        "ai_ile_dogrulandi": False,
    }


def save(store):
    DATA_FILE.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


# --- HEDEFLİ KISMİ YENİDEN TARAMA (BACKFILL) ---
# Şemaya yeni bir alan eklendiğinde (ör. bu turda toplam_butce, faaliyet_suresi,
# program_serisi, kimler_etiketleri eklendi), önceden AI ile TAM işlenmiş
# binlerce kaydı sıfırdan yeniden işletmek hem yavaş hem pahalıdır. Bunun
# yerine --backfill ile SADECE eksik olan yeni alan(lar) için, çok daha kısa
# bir prompt kullanarak, SADECE o alanlar eksik olan kayıtlara gidilir.
BACKFILL_SCHEMAS = {
    "toplam_butce": '"toplam_butce": "PROGRAMIN/ÇAĞRININ TOPLAMINDA ayrılan bütçe (tek başvurunun alacağı miktar DEĞİL) — insan tarafından okunacak metin veya null"',
    "faaliyet_suresi": '"faaliyet_suresi": "desteklenen proje/faaliyetin süresi (ör. \'en fazla 18 ay\') veya null"',
    "program_serisi": '"program_serisi": ["listeden 0-2 tane eşleşen, yoksa boş liste"]',
    "kimler_etiketleri": '"kimler_etiketleri": ["listeden 0-5 tane uygun olan, yoksa boş liste"]',
}
BACKFILL_LIST_HINTS = {
    "program_serisi": PROGRAM_SERISI_LISTESI,
    "kimler_etiketleri": KIMLER_ETIKET_LISTESI,
}


def build_backfill_prompt(field_names):
    valid = [f for f in field_names if f in BACKFILL_SCHEMAS]
    if not valid:
        raise ValueError(f"Bilinmeyen alan(lar): {field_names}. Geçerli seçenekler: {list(BACKFILL_SCHEMAS)}")
    schema_lines = ",\n  ".join(BACKFILL_SCHEMAS[f] for f in valid)
    hints = "\n".join(
        f"{f} listesi: {', '.join(BACKFILL_LIST_HINTS[f])}" for f in valid if f in BACKFILL_LIST_HINTS
    )
    return valid, f"""Sana bir Türkiye kamu/STK hibe/destek duyurusunun web sayfası metni verilecek.
Bu duyuru daha önce sınıflandırıldı, sadece şu EK alanları metinden çıkarman gerekiyor.
SADECE aşağıdaki şemada JSON döndür, başka hiçbir şey yazma:

{{
  {schema_lines}
}}

{hints}

Emin olmadığın alanları null/boş liste bırak, metinde olmayan bilgiyi ASLA uydurma."""


def run_backfill(field_names, provider, api_key, items, limit):
    """items: duyurular.json'daki tüm kayıtlar (url -> item). Sadece daha önce
    AI ile TAM işlenmiş (ai_ile_dogrulandi=True) VE istenen alan(lar)ı henüz
    hiç içermeyen kayıtlar işlenir."""
    valid_fields, backfill_prompt = build_backfill_prompt(field_names)
    print(f"Backfill modu — hedef alan(lar): {valid_fields}")

    candidates = []
    for url, item in items.items():
        details = item.get("details")
        if not details or not details.get("ai_ile_dogrulandi"):
            continue
        if any(f not in details for f in valid_fields):
            candidates.append(url)

    if not candidates:
        print("Backfill gereken kayıt yok — tüm AI-onaylı kayıtlar bu alan(lar)a zaten sahip.")
        return 0

    if limit:
        candidates = candidates[:limit]
    print(f"Backfill edilecek kayıt: {len(candidates)}")

    processed = 0
    for url in candidates:
        item = items[url]
        text, err = fetch_text(item["url"])
        if err:
            print(f"  atlandı ({err}): {item['title'][:60]}")
            continue
        try:
            partial = call_ai_with_retry(provider, api_key, text, system_prompt=backfill_prompt, max_tokens=300)
            for f in valid_fields:
                if f in partial:
                    item["details"][f] = partial[f]
            processed += 1
            print(f"  OK: {item['title'][:60]}")
        except QuotaExceeded as e:
            print(f"\nAPI kotası doldu ({e}). {processed} kayıt backfill edildi, kalanlar bir sonraki çalıştırmada devam edecek.")
            break
        except Exception as e:
            print(f"  hata (atlandı): {e} -> {item['title'][:60]}")
        time.sleep(0.4)
    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Tek çalıştırmada işlenecek üst sınır")
    parser.add_argument("--backfill", default=None,
                         help="Virgülle ayrılmış alan adları — SADECE bu alanları eksik olan, "
                              "daha önce AI ile TAM işlenmiş kayıtlar için hedefli/ucuz yeniden "
                              "tarama yapar. ör: --backfill toplam_butce,faaliyet_suresi")
    args = parser.parse_args()

    if not CLASSIFIED_FILE.exists():
        print("data/classified.json bulunamadı, önce scrape.py ve classify.py çalıştırılmalı.")
        sys.exit(1)

    classified = json.loads(CLASSIFIED_FILE.read_text(encoding="utf-8"))
    classified_items = classified.get("items", {})
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Önceki AI sonuçlarını (data/duyurular.json) yükle — "details" alanları KORUNUR.
    if DATA_FILE.exists():
        existing = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        existing_items = existing.get("items", {})
    else:
        existing_items = {}

    # classified.json'daki her kayıt duyurular.json'a taşınır; daha önce
    # "details" işlenmişse o korunur, işlenmemişse pending kalır.
    items = {}
    for url, c_item in classified_items.items():
        merged = dict(c_item)
        if url in existing_items and "details" in existing_items[url]:
            merged["details"] = existing_items[url]["details"]
        items[url] = merged

    def build_store(ai_status):
        return {
            "last_updated": classified.get("last_updated"),
            "source_status": classified.get("source_status", {}),
            "tema_listesi": TEMA_LISTESI,  # panel filtre kutucuklarını hep bu sabit listeyle doldurur
            "program_serisi_listesi": PROGRAM_SERISI_LISTESI,
            "kimler_etiket_listesi": KIMLER_ETIKET_LISTESI,
            "ai_status": ai_status,
            "items": items,
        }

    # --- BACKFILL MODU: normal akıştan tamamen ayrı, erken çıkış ---
    if args.backfill:
        provider, api_key = get_provider()
        if not provider:
            print("Backfill için ANTHROPIC_API_KEY/GEMINI_API_KEY/GROQ_API_KEY tanımlı değil.")
            sys.exit(0)
        field_names = [f.strip() for f in args.backfill.split(",") if f.strip()]
        processed = run_backfill(field_names, provider, api_key, items, args.limit)
        save(build_store({"provider": provider, "quota_exceeded": False, "checked_at": now_iso,
                           "pending_count": 0, "last_backfill": {"fields": field_names, "processed": processed, "at": now_iso}}))
        print(f"\nBackfill tamamlandı: {processed} kayıt güncellendi.")
        return

    # --- GÜVENCE 1: yeni kayıt yoksa AI'a hiç gidilmez ---
    # Not: "duplicate_of" işaretli kayıtlar (classify.py'ın tekilleştirmesi)
    # AI kuyruğuna hiç girmez — zaten başka bir kayıtla aynı, gereksiz
    # maliyet/kota harcamamak için burada elenir.
    pending = [k for k, v in items.items() if "details" not in v and not v.get("duplicate_of")]
    if not pending:
        print("Yeni/bekleyen kayıt yok — AI'a hiç gidilmedi, hiçbir çağrı yapılmadı.")
        save(build_store({"provider": None, "quota_exceeded": False, "checked_at": now_iso, "pending_count": 0}))
        sys.exit(0)

    provider, api_key = get_provider()
    if not provider:
        print(f"{len(pending)} bekleyen kayıt var ama ANTHROPIC_API_KEY/GEMINI_API_KEY tanımlı değil — AI atlanıyor.")
        save(build_store({"provider": None, "quota_exceeded": False, "checked_at": now_iso, "pending_count": len(pending)}))
        sys.exit(0)
    print(f"Kullanılan AI sağlayıcı: {provider} | Bekleyen kayıt: {len(pending)}")

    pending.sort(key=lambda k: priority(items[k]))

    skipped_free = 0
    to_call_ai = []
    for key in pending:
        item = items[key]
        tur_tahmini = FREE_SKIP_DETAILS.get(item.get("olasi_tur"))
        if tur_tahmini:
            item["details"] = empty_details(tur_tahmini)
            skipped_free += 1
        else:
            to_call_ai.append(key)

    if args.limit:
        to_call_ai = to_call_ai[: args.limit]

    print(f"Ücretsiz filtre ile atlanan (muhtemel sonuç ilanı / haber): {skipped_free}")
    print(f"AI'ya gönderilecek kayıt: {len(to_call_ai)}")

    quota_hit = False
    quota_error_msg = None
    ai_status = {"provider": provider, "quota_exceeded": False, "checked_at": now_iso, "pending_count": len(to_call_ai)}
    save(build_store(ai_status))

    processed = 0
    try:
        for key in to_call_ai:
            item = items[key]
            text, err = fetch_text(item["url"])
            if err:
                print(f"  atlandı ({err}): {item['title'][:60]}")
                continue
            try:
                details = call_ai_with_retry(provider, api_key, text)
                details["ai_ile_dogrulandi"] = True
                details["ai_saglayici"] = provider
                item["details"] = details
                processed += 1
                print(f"  OK [{details.get('tur')}]: {item['title'][:60]}")
            except QuotaExceeded as e:
                print(f"\nAPI kotası/token limiti doldu ({e}).")
                print(f"Bu çalıştırmada {processed} kayıt işlendi, kalanlar bir sonraki çalıştırmada devam edecek.")
                quota_hit = True
                quota_error_msg = str(e)
                break
            except Exception as e:
                print(f"  AI hata (bu kayıt atlandı, devam ediliyor): {e} -> {item['title'][:60]}")

            ai_status["pending_count"] = sum(1 for k in to_call_ai if "details" not in items[k])
            save(build_store(ai_status))
            time.sleep(0.4)
    finally:
        remaining_now = sum(1 for k in to_call_ai if "details" not in items[k])
        ai_status = {
            "provider": provider,
            "quota_exceeded": quota_hit,
            "quota_error": quota_error_msg,
            "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pending_count": remaining_now,
        }
        save(build_store(ai_status))

    remaining = sum(1 for k in to_call_ai if "details" not in items[k])
    print(f"\nTamamlandı. Ücretsiz: {skipped_free} | AI ile işlenen: {processed}"
          + (f" | Bir sonraki çalıştırmaya kalan: {remaining}" if remaining else ""))


if __name__ == "__main__":
    main()
