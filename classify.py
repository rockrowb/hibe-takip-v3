#!/usr/bin/env python3
"""
KATMAN 2/3 — ÜCRETSİZ SINIFLANDIRMA

data/raw.json'ı okur, hiçbir ağ isteği ATMADAN (tamamen çevrimdışı, ücretsiz,
hızlı) her kaydın başlığına bakarak bir ön tahmin ("olasi_tur") üretir ve
data/classified.json'a yazar:

    hibe_olabilir   -> başlıkta hibe/destek/çağrı/başvuru gibi kelimeler var
    sonuc_olabilir  -> başlıkta "sonuçlandı", "kazananlar açıklandı" gibi kalıplar var
    haber_olabilir  -> başlıkta genel haber/basın bülteni kalıpları var (video,
                        ziyaret, imza töreni, toplantı vb.) ama hibe kelimesi yok
    belirsiz        -> hiçbiri net değil

Bu etiket HİÇBİR KAYDI SİLMEZ ya da GİZLEMEZ — sadece (a) panelin varsayılan
sekme yerleşimine ve (b) enrich.py'ın hangi kayıtları AI'ya göndermeden
ücretsiz eleyeceğine karar vermek için kullanılır.

Kurallar/anahtar kelimeler değiştiğinde SADECE bu script yeniden çalıştırılır
— siteleri tekrar taramaya gerek yoktur, çünkü girdi (raw.json) zaten diskte
duruyor. Bu yüzden data/raw.json'u SİLMEK YERİNE, sınıflandırma mantığını
düzeltip bu scripti tekrar çalıştırmak yeterlidir.

Kullanım:
    python classify.py
"""
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).parent
RAW_FILE = ROOT / "data" / "raw.json"
CLASSIFIED_FILE = ROOT / "data" / "classified.json"

# Bu kaynaklar "merkezi/agregatör" niteliğinde — aynı destek başka bir ajans
# kaynağında da bulunursa, tekilleştirmede AJANS kaynağı esas alınır, merkezi
# kaynaktaki kopya "duplicate_of" ile işaretlenir (silinmez, sadece panelde
# gizlenir).
AGGREGATOR_SOURCE_IDS = {"ka_gov", "yatirimadestek"}
DUPLICATE_SIMILARITY_THRESHOLD = 0.88

HIBE_KEYWORDS = [
    "hibe", "destek", "çağrı", "cagri", "başvuru", "basvuru", "fon",
    "program", "teklif çağrısı", "grant", "proje çağrısı", "burs",
    "yarışma başvur", "competition apply",
]

SONUC_KEYWORDS = [
    "sonuçlandı", "sonuclandi", "kazananlar", "ilan edildi",
    "değerlendirme tamamlandı", "degerlendirme tamamlandi",
    "başarılı bulunan", "basarili bulunan", "asil liste", "asıl liste",
    "yedek liste", "kabul edilen projeler",
    "sonuçları açıklandı", "sonucu açıklandı", "sonuçlarını açıkladı",
    "sonuçları belli oldu", "kazananları belli oldu", "dereceye giren",
    "ödül töreni", "ödül aldı", "birinci oldu",
]

# Genel haber/basın bülteni VE kurumsal/rapor tipi kalıplar — bir hibe çağrısı
# DEĞİL, ama "sonuç ilanı" da değil (ör. ziyaret, imza töreni, anket, bütçe
# raporu, stratejik plan gibi kurumsal/idari içerikler).
HABER_KEYWORDS = [
    "ziyaret etti", "ziyaret gerçekleştir", "imza töreni", "imzalandı",
    "açılışını gerçekleştir", "açılışı yapıldı", "toplantısı gerçekleştirildi",
    "toplantı düzenlendi", "video", "simülasyon raporu", "sunum sonuçları",
    "eğitim verildi", "eğitimi düzenlendi", "ile buluştu", "kutlandı",
    "tebrik", "davet edildi", "katıldı", "katılım sağladı", "ağırladı",
    "işbirliği protokolü", "protokol imzalandı", "bilgilendirme toplantısı",
    "farkındalık", "anma etkinliği", "fuarına katıldı",
    # kurumsal/idari/rapor içerikleri
    "anket merkezi", "memnuniyet anketi", "bütçe uygulama sonuçları",
    "faaliyet raporu", "stratejik plan", "insan kaynakları", "personel alımı",
    "personel ilanı", "ihale ilanı", "mali tablo", "yönetim kurulu",
    "genel kurul", "kvkk", "gizlilik politikası", "çerez politikası",
    "iç kontrol", "denetim raporu", "sayıştay", "kamu zararı",
    "basın bülteni", "basında biz",
]

# Yarışma/ödül programları — hibe değil ama "hibe_olabilir" ile karışmaması
# için ayrı bir kova.
YARISMA_KEYWORDS = [
    "yarışma", "yarışması", "competition", "challenge", "hackathon",
]


def guess_type(title):
    t = title.lower()
    has_sonuc = any(k in t for k in SONUC_KEYWORDS)
    has_hibe = any(k in t for k in HIBE_KEYWORDS)
    has_haber = any(k in t for k in HABER_KEYWORDS)
    has_yarisma = any(k in t for k in YARISMA_KEYWORDS)

    if has_sonuc:
        return "sonuc_olabilir"
    if has_hibe:
        return "hibe_olabilir"
    if has_yarisma:
        return "yarisma_olabilir"
    if has_haber:
        return "haber_olabilir"
    return "belirsiz"


def normalize_title(title):
    t = title.lower()
    t = re.sub(r"[^\wşğüöçı ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _mark_duplicate(classified, dup_url, canonical_url):
    """Kopya kaydı işaretler VE kanonik kayda 'ayrıca şu kaynakta da bulundu'
    bilgisini ekler — kullanıcı tekilleştirmeyi panelden açtığında hangi
    kaynaklardan geldiğini görebilsin diye."""
    dup_item = classified[dup_url]
    canonical_item = classified[canonical_url]
    dup_item["duplicate_of"] = canonical_url
    dup_item["duplicate_of_source"] = canonical_item.get("source")
    also_found = canonical_item.setdefault("also_found_in", [])
    also_found.append({
        "source": dup_item.get("source"),
        "source_id": dup_item.get("source_id"),
        "url": dup_url,
    })


def find_duplicates(classified):
    """Farklı kaynaklardan gelen, başlığı çok benzeyen kayıtları eşleştirir.
    Kanonik (asıl gösterilecek) kayıt tercihen AJANS kaynağıdır; agregatör
    (ka_gov, yatirimadestek) kaynağındaki kopya 'duplicate_of' ile işaretlenir.
    Hiçbir kayıt SİLİNMEZ, sadece panelde varsayılan olarak gizlenir."""
    urls = list(classified.keys())
    norm = {u: normalize_title(classified[u].get("title", "")) for u in urls}

    # Aynı normalize başlığa sahip olanları grupla (hızlı ön-eşleştirme)
    groups = {}
    for u in urls:
        groups.setdefault(norm[u], []).append(u)

    dup_count = 0
    for key, group_urls in groups.items():
        if len(group_urls) < 2 or not key:
            continue
        # Kanonik: agregatör OLMAYAN bir kaynak varsa onu seç, yoksa ilkini seç.
        non_aggregator = [u for u in group_urls
                           if classified[u].get("source_id") not in AGGREGATOR_SOURCE_IDS]
        canonical = non_aggregator[0] if non_aggregator else group_urls[0]
        for u in group_urls:
            if u != canonical:
                _mark_duplicate(classified, u, canonical)
                dup_count += 1

    # Ekstra: tam eşleşmeyen ama çok benzeyen başlıkları da yakala (farklı
    # kaynaklarda ufak yazım farkıyla geçen aynı program adı gibi). Sadece
    # agregatör <-> ajans çiftlerine bakarak maliyeti düşük tutuyoruz.
    aggregator_urls = [u for u in urls if classified[u].get("source_id") in AGGREGATOR_SOURCE_IDS
                        and "duplicate_of" not in classified[u]]
    agency_urls = [u for u in urls if classified[u].get("source_id") not in AGGREGATOR_SOURCE_IDS]
    for au in aggregator_urls:
        best_match, best_score = None, 0
        for gu in agency_urls:
            score = SequenceMatcher(None, norm[au], norm[gu]).ratio()
            if score > best_score:
                best_score, best_match = score, gu
        if best_match and best_score >= DUPLICATE_SIMILARITY_THRESHOLD:
            _mark_duplicate(classified, au, best_match)
            dup_count += 1

    return dup_count


def main():
    if not RAW_FILE.exists():
        print("data/raw.json bulunamadı, önce scrape.py çalıştırılmalı.")
        return

    raw = json.loads(RAW_FILE.read_text(encoding="utf-8"))
    raw_items = raw.get("items", {})

    classified = {}
    counts = {"hibe_olabilir": 0, "sonuc_olabilir": 0, "yarisma_olabilir": 0, "haber_olabilir": 0, "belirsiz": 0}
    for url, item in raw_items.items():
        tur = guess_type(item.get("title", ""))
        counts[tur] += 1
        entry = {**item, "olasi_tur": tur}
        entry.pop("duplicate_of", None)  # her çalıştırmada temiz baştan hesapla
        entry.pop("duplicate_of_source", None)
        entry.pop("also_found_in", None)
        classified[url] = entry

    dup_count = find_duplicates(classified)

    out = {
        "last_updated": raw.get("last_updated"),
        "source_status": raw.get("source_status", {}),
        "items": classified,
    }
    CLASSIFIED_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[classified.json] Toplam: {len(classified)} | " +
          " | ".join(f"{k}: {v}" for k, v in counts.items()) +
          f" | Tekilleştirilen (duplicate): {dup_count}")


if __name__ == "__main__":
    main()
