# AGENTS.md — Istanbul City Pulse

Bu dosya, projede çalışan tüm **Yapay Zeka Ajanları ve Kod Asistanları** için bağlamı, mimari prensipleri ve uyulması gereken zorunlu kuralları tanımlar. Çelişki halinde bu dosya, ajanın kendi varsayılan alışkanlıklarının önüne geçer.

---

##  1. Proje Özeti ve Amacı
* **Proje Adı:** Istanbul City Pulse
* **Tanım:** İstanbul Büyükşehir Belediyesi (İBB) Açık Veri Portalı ve şehir API'lerinden dinamik veri toplayan, depolayan, dbt ile dönüştüren ve makine öğrenimi modelleriyle (LightGBM/MLflow) şehir dinamiklerini analiz ve tahmin eden uçtan uca modern bir Veri Mühendisliği ve MLOps platformudur.
* **Teknoloji Yığını:** Python 3.12+, `uv`, Polars, DuckDB, PostgreSQL (SQLAlchemy + Alembic), dbt, Apache Airflow, Docker Compose, LightGBM, MLflow, Ruff, Pytest.
* **Kalite Hedefi:** Proje, kıdemli (senior) bir veri mühendisinin portföyünde gururla gösterebileceği düzeyde olmalıdır. Bu hedef, **istenen işi eksiksiz, doğru, test edilmiş ve okunaklı yapmak** demektir; istenmeyen özellik eklemek demek **değildir** (bkz. Bölüm 2).

---

##  2. Çalışma Biçimi ve Kapsam Disiplini (EN ÖNEMLİ BÖLÜM)
1. **Sadece istenen işi yap.** Kullanıcının talep etmediği özellik, dosya, bağımlılık, klasör, ayar, otomasyon, belge veya "iyileştirme" ekleme. İstenmeyen ekleme, hata sayılır.
2. **Kapsam dışı fikirler için öner, uygulama.** Faydalı bir şey görürsen işin sonunda kısa bir **öneri** olarak yaz ve kullanıcının onayını bekle. Onay olmadan uygulama.
3. **Belirsizlik varsa tahmin etme, sor.** Birden fazla makul yorum varsa en fazla bir kısa soruyla netleştir. Soruyu yanıtlamadan önce varsayımla büyük iş yapma.
4. **Çalışan koda gereksiz dokunma.** Görevle ilgisiz dosyaları yeniden biçimlendirme, yeniden adlandırma veya yeniden düzenleme. Değişiklikleri küçük ve gözden geçirilebilir tut.
5. **Yeni bağımlılık eklemeden önce sor.** `uv add` yalnızca kullanıcı istediğinde veya görev açıkça gerektirdiğinde ve önce gerekçesi belirtilerek yapılır.
6. **Yapılandırmayı kendiliğinden değiştirme.** `pyproject.toml` (ruff/mypy/pytest ayarları dahil), `docker-compose.yml`, CI ve `.gitignore` değişiklikleri kullanıcı istemedikçe yapılmaz; gerekliyse öner.
7. **Dürüst raporla.** Çalıştırmadığın kodu "çalışıyor" diye sunma. Neyi çalıştırdığını (test, lint, tip denetimi), neyi **çalıştıramadığını** (örn. gerçek API, ağ erişimi) açıkça yaz. Başarısız bir kontrolü gizleme.
8. **Uydurma.** Var olduğunu doğrulamadığın API uç noktası, kolon adı, kütüphane fonksiyonu veya veri seti alanı yazma. Emin değilsen bunu belirt ve doğrulama yolunu öner.

---

##  3. Paket Yönetimi ve Çalıştırma Kuralları
1. **Asla `pip` Kullanma:** Paket yönetimi için sadece `uv` kullanılmalıdır.
   * Paket ekleme: `uv add <paket>`
   * Gruba paket ekleme: `uv add --group <dev|ml|dbt|quality> <paket>`
   * Ortam eşitleme: `uv sync` veya `uv sync --all-groups`
2. **Komutları `uv run` ile Koştur:**
   * Testler: `uv run pytest`
   * Linter & Format kontrolü: `uv run ruff check .` / `uv run ruff format .`
   * Tip denetimi: `uv run mypy src`
   * Python scriptleri ve modüller: `uv run python -m <paket.modül>` veya `uv run python <dosya_yolu>`
3. **Kilit Dosyası Bütünlüğü:** `uv.lock` dosyasını asla elle düzenleme; `pyproject.toml` üzerinden `uv` aracılığıyla güncelle.
4. **Airflow ve dbt ayrı tutulur:** Çok sayıda bağımlılığı sabitledikleri için ana ortama karıştırılmaz; kendi grup/imajlarında çalışır.

---

##  4. Dizin Yapısı ve Sorumluluklar
* **`src/city_pulse/`:** Tüm çekirdek Python kaynak kodları buraya yazılır (`src` layout). Kök dizine doğrudan `.py` kodu bırakma. *(Paket adı `pyproject.toml` ve importlarla birebir aynı olmalıdır.)*
* **`dags/`:** Sadece Apache Airflow DAG (iş akışı) dosyaları barındırılır. DAG'lar **ince** olur, iş mantığı `src/` içinde yaşar.
* **`dbt/`:** Veri ambarı dönüşüm (ELT) SQL modelleri, `seeds/` ve `schema.yml` dosyaları burada yer alır.
* **`docker/`:** Servislere ait özel `Dockerfile`'lar tutulur.
* **`data/`:** Yerel ham ve işlenmiş veri dosyaları (CSV, Parquet) tutulur. **Proje kökünde** durur, `src/` içinde değil. Asla Git'e gönderilmez.
* **`tests/`:** Pytest birim ve entegrasyon testleri (`tests/unit`, `tests/integration`) buraya yazılır.
* **`ZZ_NOTLAR/`:** Geliştiricinin kişisel notlarıdır. Kod bağımlılığı içermez, otomatik olarak değiştirilmemelidir.

### Katmanlar ve bağımlılık yönü
```
domain  ←  ingest / transform / storage  ←  pipelines  ←  dags / CLI
```
* `domain`: saf veri modelleri ve sabitler; hiçbir şeye bağımlı değil.
* `ingest`: dış dünya adaptörleri (HTTP istemcileri) **ve** saf `*_to_frame` dönüştürücüleri ayrı modüllerde.
* `storage`: disk/veritabanı yazımı, `Protocol` arayüzü arkasında.
* `pipelines`: orkestrasyon (çek → sakla → dönüştür → sakla); bağımlılıkları **parametre olarak** alır.
* Bağımlılık **içe** doğru akar: alt katmanlar üst katmanları import etmez. Döngüsel import yasaktır.

---

##  5. Kodlama ve Mimari Standartları

### 5.1 Kalite çıtası: SOLID ve Pythonic (ZORUNLU)
Yazılan **her** kod SOLID ilkelerine ve Pythonic yazıma uygun olmalıdır:
* **S, Tek Sorumluluk:** Her modülün, sınıfın ve fonksiyonun değişmesi için tek bir nedeni olur. **I/O ile saf mantık ayrılır** (ağ, disk, saat, rastgelelik bir yanda; dönüştürme diğer yanda). Fonksiyonlar kısa ve tek işli olur.
* **O, Açık/Kapalı:** Yeni bir kaynak/biçim eklemek, var olan akışı değiştirmeden **yeni bir adaptör yazarak** mümkün olmalı. Kod kopyalayıp yapıştırarak çoğaltma yok.
* **L, Liskov:** Bir `Protocol`'ü karşılayan her gerçekleme aynı sözleşmeyi (şema, dönüş tipi, hata davranışı) korur.
* **I, Arayüz Ayrımı:** Küçük, amaca özel `Protocol`'ler. Tüketici yalnızca ihtiyaç duyduğu arayüze bağlıdır.
* **D, Bağımlılığın Tersine Çevrilmesi:** Üst katmanlar somut `requests`, dosya sistemi, `datetime.now()` gibi şeylere değil, **enjekte edilen soyutlamalara** (`Protocol`, `Callable`) bağlıdır. Modül düzeyinde gizli global durum veya singleton yok.

**Pythonic yazım:**
* Kompozisyon, kalıtımdan önce gelir. `Protocol`, `@dataclass(frozen=True, slots=True)`, `NamedTuple`, `Enum` tercih edilir. `ABC`, metaclass ve fabrika kalıpları yalnızca gerçekten gerektiğinde kullanılır.
* Aşırı mühendislikten kaçın: **tek gerçeklemesi olan ve I/O sınırı olmayan yere soyutlama ekleme.** SOLID bir araçtır, amaç değil; ama I/O sınırlarında bağımlılık enjeksiyonu zorunludur.
* PEP 8 isimlendirme, `pathlib`, f-string, context manager, generator (büyük akışlarda), okunaklı comprehension. `from x import *`, değiştirilebilir varsayılan argüman, çıplak `except:`, ölü kod ve yorum satırına alınmış kod yasaktır.
* Sihirli sayı/metin yok: adlandırılmış sabit kullan.
* Herkese açık fonksiyon ve sınıflara kısa **docstring** yaz. Yorumlar "ne"yi değil **"neden"i** açıklar.
* `# type: ignore` ve `# noqa` yalnızca nedeni yanında yazılıysa kullanılır.

### 5.2 Dil kuralı
* **Tanımlayıcılar (değişken, fonksiyon, sınıf, modül, kolon adı) İngilizce**, docstring ve yorumlar **Türkçe**, kullanıcıya açıklamalar Türkçe ve sade dilde yazılır.

### 5.3 Veri, tip ve yapılandırma
* **Veri İşleme Önceliği:** Büyük veri manipülasyonu ve ETL işlemlerinde `pandas` yerine **`polars`** veya **`duckdb`** tercih edilmelidir. Pandas sadece ML modellerinin doğrudan girdi beklediği noktalarda kullanılmalıdır.
* **Tip Güvenliği (Type Hints):** Tüm fonksiyon, metod ve sınıf değişkenlerinde Python tip belirteçleri (`typing` / built-in types) eksiksiz yazılmalıdır. Yeni kod `mypy` temiz olmalıdır.
* **Veri Doğrulama:** Dış kaynaklardan gelen tüm JSON ve API yanıtları **Pydantic (v2)** modelleri ile şemaya bağlanmalı ve doğrulanmalıdır. Tablo şemaları (kolon adları ve **tipleri**) açıkça tanımlanır; tip çıkarımına (inference) güvenilmez.
* **Konfigürasyon Yönetimi:** Gizli anahtarlar, veritabanı bağlantı URI'ları ve ayarlar **`pydantic-settings`** kullanılarak `.env` üzerinden okunmalıdır. Koda hardcode anahtar yazmak **kesinlikle yasaktır**.
* **Loglama:** `print()` yerine standart Python `logging` kullanılmalıdır. Loglara gizli bilgi (anahtar, parola, kişisel veri) yazılmaz.

### 5.4 Hata yönetimi
* Spesifik istisnalar yakala; hatayı sessizce yutma. Gerekirse projeye özgü istisna sınıfları tanımla.
* Her HTTP çağrısında **`timeout` zorunludur.** Yeniden deneme (retry) yalnızca **idempotent** isteklerde (GET) ve geri çekilme (backoff) ile yapılır.
* Hata durumunda kısmi/yarım çıktı bırakılmaz (bkz. atomik yazma).

---

##  6. Veri Mühendisliği Kuralları
1. **Bronze / Silver / Gold:** Bronze = API'nin ham yanıtı, **değiştirilmez ve silinmez**. Silver = temizlenmiş, tipli, ortak şemalı veri. Gold/mart = model ve rapor için türetilmiş tablolar (silver'dan yeniden üretilebilir).
2. **Zaman:** Tüm zaman damgaları **UTC** saklanır; yerel saate çevirme yalnızca gösterim/gün-bazlı toplulaştırma anında yapılır. Saat eşleme kuralı (saat başı mı, sonu mu) tek yerde tanımlanır.
3. **İdempotency:** Aynı girdiyle pipeline'ı tekrar çalıştırmak aynı sonucu verir, çift kayıt üretmez. Kod içinde `datetime.now()` gizlenmez; saat bir `clock` parametresiyle enjekte edilir.
4. **Atomik yazma:** Dosyalar önce geçici dosyaya yazılır, sonra yerine taşınır. Yarım dosya bırakılmaz.
5. **Eksik veriyi sessizce doldurma.** Eksik/boş saatler işaretlenir; nasıl ele alınacağı açıkça belirlenir.
6. **Kaynağı etiketle:** Her satır hangi kaynaktan (`source`) ve ne zaman (`fetched_at_utc`) geldiğini taşır. Farklı kaynakları (geçmiş, canlı, tahmin) tek kolon gibi karıştırma.
7. **Şema tutarlılığı:** Aynı kolon her dosyada aynı tipte olmalıdır (örn. sayısal ölçümler `Float64`).
8. **Konum:** Sokak/cadde adlarına bağımlı kalma; koordinat, ilçe veya ızgara (grid/geohash) anahtarı kullan. Konum eşleme kuralı tek yerde yaşar.
9. **Veri kaynağı kullanımı:** Kaynağın lisansına, kotasına ve kullanım şartlarına uy. `robots.txt` ve şartları ihlal eden otomatik erişim yazma. Belgelenmemiş/özel (mobil uygulama vb.) uç noktaları tersine mühendislikle kullanma. Gerekli atıfları (örn. Open-Meteo) README'de koru.

---

##  7. Makine Öğrenimi Kuralları
1. **Veri sızıntısı (leakage) yasak:** Doğrulama/test ayrımı **zamana göre** yapılır, rastgele değil. Gelecek bilgisi eğitime girmez.
2. **Gerçekleşen ve tahmin edilen hava ayrı tutulur.** Eğitimde ve canlıda kullanılan özelliklerin kaynağı tutarlı olmalı (training-serving skew); fark varsa ölçülür ve raporlanır.
3. **Baz model (baseline) zorunlu:** Her model, basit bir baz modelle (örn. haftanın günü × saat ortalaması) karşılaştırılır. Baz modeli geçemeyen sonuç dürüstçe raporlanır.
4. **Tekrarlanabilirlik:** Rastgelelik tohumları (seed) sabitlenir; eğitim verisi **değişmez snapshot** olarak saklanır; model ve parametreler MLflow'a kaydedilir.
5. **Yeniden eğitimde champion/challenger:** Yeni model mevcut modelden iyi değilse yayınlanmaz.

---

##  8. SQL, dbt, Airflow, Docker
* **SQL:** Yalnızca parametreli sorgular (string birleştirme yasak). Tanımlayıcılar güvenli biçimde kaçırılır. Join'den önce anahtar tekilliği düşünülür (satır çoğalması/fan-out).
* **dbt:** Katmanlar `staging → intermediate → marts`. Her modelin **grain'i** belgelenir; anahtarlara `unique` ve `not_null`, ilişkilere `relationships` testi yazılır. Mart'larda `source()` değil `ref()` kullanılır.
* **Airflow:** DAG dosyasının üst düzeyinde ağır iş yapılmaz. Zaman için `data_interval`/`logical_date` kullanılır, `datetime.now()` kullanılmaz. Task'lar idempotent ve tarih parametrelidir. XCom'a yalnızca küçük veri (yol, id) konur.
* **Docker:** Sabit (pinlenmiş) temel imaj sürümü, `uv sync --locked`, root olmayan kullanıcı, imaja gizli bilgi gömülmez.

---

##  9. Test ve "Bitti" Tanımı
* Her yeni davranış için **pytest** testi yazılır. Birim testler **ağsız ve diskten bağımsız** çalışır (sahte `Protocol` gerçeklemeleri, `tmp_path`). Testler deterministik, birbirinden bağımsız ve herhangi bir sırada çalışabilir olmalıdır.
* Hata düzeltmeleri önce hatayı yakalayan bir testle başlar.
* **Bir iş ancak şu kontroller geçince "bitti" sayılır:**
  ```
  uv run ruff check .
  uv run ruff format --check .
  uv run mypy src
  uv run pytest
  ```
  Çalıştırılamayan kontrol varsa nedeni raporda açıkça yazılır.

---

##  10. Güvenlik ve Git Kuralları
* `.env`, `*.duckdb`, `data/`, `.venv/` ve geçici önbellek dizinleri asla Git'e eklenmemelidir.
* Sırlar (API anahtarı, parola, token) kodda, testte, logda, imajda ve commit geçmişinde bulunmaz. `.env.example` yalnızca örnek (sahte) değerler içerir.
* Commit mesajları **Conventional Commits** standartlarına uygun olmalıdır (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`). Bir commit tek bir mantıksal değişiklik içerir.
* Ajan, kullanıcı açıkça istemedikçe `git commit`, `git push`, dal silme veya geçmiş yeniden yazma işlemi yapmaz.

---

##  11. Bilinen Veri Gerçekleri (doğrulanmış; yeni bilgi çıkınca güncelle)
* **İBB "İstanbul Trafik Endeksi" CSV'si günlük ve şehir geneldedir:** her gün için tek satır; `minimum/maximum/average_traffic_index` kolonları. **Saat ve konum yoktur.** `trafficindexdate` içindeki saat (03:00/02:00) gerçek bir saat değildir, yalnızca tarih kullanılır. İndirilen dosya **2024-10-18'de** bitiyor; son satır eksik gün gibi görünüyor (atılmalı). Yaklaşık 30 gün eksiktir.
* **Saat ve konum hedefi için** İBB'nin "saatlik trafik yoğunluğu veri seti" (aylık dosyalar) gerekir; kolonları indirilip doğrulanmadan şema varsayılmaz.
* **Belgelenmiş, resmi bir anlık (canlı) İBB trafik API'si doğrulanmamıştır.** Uç nokta uydurma; keşif için portalın CKAN API'sini (`package_search`) kullan ve sonucu kullanıcıyla paylaş.
* **Open-Meteo:** ERA5 uç noktası geçmiş (yeniden analiz) veri, `forecast` uç noktası anlık + tahmin verir. ERA5 bir istasyon ölçümü değil, yeniden analiz modelidir. Veri CC BY 4.0, atıf gerekir.
* **Meteosource ücretsiz planı** günlük 400 çağrı ile sınırlıdır ve geçmiş veri içermez.

---

##  12. İletişim
* Yanıtlar Türkçe, kısa ve net olur. Önce **ne yapıldığı**, sonra **neyin doğrulandığı / doğrulanamadığı**, en son (varsa) **öneriler** yazılır.
* Kullanıcı veri mühendisliğini öğrenmektedir: yeni bir kavram veya kalıp kullanıldığında bir iki cümleyle **neden** o seçimin yapıldığı açıklanır; ama kapsam genişletilmez.