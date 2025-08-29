# project/prep.py
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
import os, glob, shutil, json

# ─────────────────────────────────────────────────────────
# Utility base (stile cleaning.py)
# ─────────────────────────────────────────────────────────
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_single_jsonl(df: DataFrame, final_out_file: str):
    """
    Scrive df in UN SOLO file .jsonl:
    - coalesce(1) per una sola partizione
    - scrive in cartella temporanea
    - rinomina 'part-*.json' nel path finale
    - se il file finale esiste, lo SOVRASCRIVE
    """
    ensure_dir(os.path.dirname(final_out_file))
    temp_dir = final_out_file + ".tmpdir"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    # Scrittura (non compresso): genererà part-*.json (+ _SUCCESS, .crc interni)
    df.coalesce(1).write.mode("overwrite").json(temp_dir)

    # Trova il part file
    part_files = glob.glob(os.path.join(temp_dir, "part-*.json"))
    if not part_files:
        part_files = glob.glob(os.path.join(temp_dir, "part-*.json.gz"))
    if not part_files:
        print(f"[ERRORE] Nessun part file in {temp_dir}. Contenuto directory:")
        try:
            for name in os.listdir(temp_dir):
                print("   -", name)
        except Exception as e:
            print("   [EXC] Impossibile listare la dir temporanea:", e)
        raise RuntimeError(f"Nessun part file trovato in {temp_dir}")

    part_file = part_files[0]

    # Normalizza estensione a .jsonl
    if final_out_file.endswith(".json"):
        final_out_file = final_out_file[:-5] + ".jsonl"
    elif not final_out_file.endswith(".jsonl"):
        final_out_file = final_out_file + ".jsonl"

    # Sovrascrivi se esiste
    if os.path.exists(final_out_file):
        os.remove(final_out_file)

    # Sposta il singolo file e rimuovi la tmp (niente .crc persistenti)
    shutil.move(part_file, final_out_file)
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"[OK] File unico scritto: {final_out_file}")

def list_cleaned_files(folder: str):
    """
    Restituisce i file 'cleaned' in new_ds (primo livello):
    - *_cleaned.jsonl preferiti
    - fallback: .jsonl / .json
    """
    if not os.path.isdir(folder):
        print(f"[ATTENZIONE] Cartella cleaned non valida: {folder}")
        return []
    # Preferisci i *_cleaned.jsonl se presenti
    cleaned = sorted(glob.glob(os.path.join(folder, "*_cleaned.jsonl")))
    if cleaned:
        print(f"[INFO] Trovati {len(cleaned)} file *_cleaned.jsonl")
        return cleaned

    # Altrimenti prendi jsonl/json singoli
    patterns = ["*.jsonl", "*.json"]
    out = []
    for pat in patterns:
        out.extend(glob.glob(os.path.join(folder, pat)))
    out = sorted([p for p in out if os.path.isfile(p)])
    print(f"[INFO] Trovati {len(out)} file (fallback json/jsonl).")
    return out

# ─────────────────────────────────────────────────────────
# Normalizzazione 'place' (no UDF) — la tua versione
# ─────────────────────────────────────────────────────────
def _canon_place(col):
    c = F.lower(F.trim(col))
    c = F.regexp_replace(c, r"\([^)]*\)", "")                   # rimuovi parentesi
    c = F.regexp_replace(c, r"[.']", "")                        # rimuovi . e '
    c = F.regexp_replace(c, r"\s+", " ")                        # normalizza spazi
    c = F.regexp_replace(c, r"\s*,\s*", ", ")                   # normalizza virgole

    # alias città/aree frequenti
    c = F.regexp_replace(c, r"\bnew york city\b", "new york")
    c = F.regexp_replace(c, r"\bnyc\b", "new york")
    c = F.when(c == F.lit("ny"), F.lit("new york")).otherwise(c)
    c = F.regexp_replace(c, r"\bsf\b", "san francisco")
    c = F.when(c == F.lit("la"), F.lit("los angeles")).otherwise(c)
    c = F.regexp_replace(c, r"\bwashington,? d ?c\b", "washington")

    us_state_codes = "(?:al|ak|az|ar|ca|co|ct|dc|de|fl|ga|hi|ia|id|il|in|ks|ky|la|ma|md|me|mi|mn|mo|ms|mt|nc|nd|ne|nh|nj|nm|nv|ny|oh|ok|or|pa|ri|sc|sd|tn|tx|ut|va|vt|wa|wi|wv)"
    us_state_names = "(?:alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|mississippi|missouri|montana|nebraska|nevada|new hampshire|new jersey|new mexico|new york|north carolina|north dakota|ohio|oklahoma|oregon|pennsylvania|rhode island|south carolina|south dakota|tennessee|texas|utah|vermont|virginia|washington|west virginia|wisconsin|wyoming|district of columbia|dc)"
    ca_prov_codes = "(?:ab|bc|mb|nb|nl|ns|nt|nu|on|pe|qc|sk|yt)"
    ca_prov_names = "(?:ontario|quebec|british columbia|alberta|manitoba|saskatchewan|nova scotia|new brunswick|newfoundland and labrador|prince edward island|northwest territories|nunavut|yukon)"
    countries = "(?:united states|usa|america|united kingdom|uk|england|scotland|wales|northern ireland|canada|india|nigeria|australia)"

    tail_with_comma = f"(?:,\\s*{countries}|,\\s*{us_state_codes}|,\\s*{us_state_names}|,\\s*{ca_prov_codes}|,\\s*{ca_prov_names})"
    tail_no_comma   = f"(?:\\s+{countries}|\\s+{us_state_codes}|\\s+{us_state_names}|\\s+{ca_prov_codes}|\\s+{ca_prov_names})"
    c = F.regexp_replace(c, f"(?:{tail_with_comma})+$", "")
    c = F.regexp_replace(c, f"(?:{tail_no_comma})+$", "")

    c = F.regexp_replace(c, r"(,?\s*\b(city|county)\b)+$", "")  # rimuovi city/county finali
    c = F.regexp_replace(c, r"\s*,\s*$", "")                    # pulizia finale
    c = F.regexp_replace(c, r"\s+", " ")
    return F.trim(c)

# ─────────────────────────────────────────────────────────
# Prep (come nel tuo query.py) + place_norm
# ─────────────────────────────────────────────────────────
def prep(df: DataFrame) -> DataFrame:
    # parsing robusto del created_at di Twitter
    arr = F.split(F.col("created_at"), r"\s+")
    mon_abbr = arr.getItem(1)
    day      = F.lpad(arr.getItem(2), 2, "0")
    time_    = arr.getItem(3)
    zone     = arr.getItem(4)
    year     = arr.getItem(5)

    month_map = F.create_map(
        F.lit("Jan"), F.lit("01"), F.lit("Feb"), F.lit("02"), F.lit("Mar"), F.lit("03"),
        F.lit("Apr"), F.lit("04"), F.lit("May"), F.lit("05"), F.lit("Jun"), F.lit("06"),
        F.lit("Jul"), F.lit("07"), F.lit("Aug"), F.lit("08"), F.lit("Sep"), F.lit("09"),
        F.lit("Oct"), F.lit("10"), F.lit("Nov"), F.lit("11"), F.lit("Dec"), F.lit("12"),
    )
    mon_num = F.element_at(month_map, mon_abbr)

    iso_like = F.concat_ws(" ", F.concat_ws("-", year, mon_num, day), time_, zone)

    ts_robust = F.when(
        (F.size(arr) >= 6) & mon_num.isNotNull(),
        F.to_timestamp(iso_like, "yyyy-MM-dd HH:mm:ss Z")
    )
    ts = F.coalesce(ts_robust, F.to_timestamp(F.col("created_at")))

    fav_src = F.coalesce(F.col("retweeted_status.favorite_count").cast("long"),
                         F.col("favorite_count").cast("long"))
    rt_src  = F.coalesce(F.col("retweeted_status.retweet_count").cast("long"),
                         F.col("retweet_count").cast("long"))

    base = (
        df.withColumn("ts", ts)
          .withColumn("hour", F.date_trunc("hour", F.col("ts")))
          .withColumn("text_full", F.coalesce(F.col("extended_tweet.full_text"), F.col("text")))
          .withColumn("verified", F.coalesce(F.col("user.verified"), F.lit(False)))
          .withColumn("is_rt", F.col("retweeted_status").isNotNull() | F.col("text").rlike(r"^RT @"))
          .withColumn("engagement", fav_src + rt_src)
          .withColumn(
              "device_norm",
              F.trim(F.regexp_replace(F.regexp_replace(F.col("source"), r"<[^>]+>", ""), r"\s+", " "))
          )
          .withColumn(
              "hashtags_arr",
              F.when(F.col("entities.hashtags").isNotNull(),
                     F.expr("transform(entities.hashtags, x -> lower(x.text))"))
               .otherwise(F.array())
          )
          .withColumn(
              "mentions_count",
              F.when(F.col("entities.user_mentions").isNotNull(),
                     F.size(F.col("entities.user_mentions"))).otherwise(F.lit(0))
          )
          .withColumn("has_mentions", F.col("mentions_count") > 0)
          .withColumn(
              "place",
              F.coalesce(F.col("place.full_name"), F.col("place.name"), F.col("user.location"))
          )
    )

    return base.withColumn("place_norm", _canon_place(F.coalesce(F.col("place"), F.lit(""))))

# ─────────────────────────────────────────────────────────
# Pipeline per singolo file
# ─────────────────────────────────────────────────────────
def process_single_file(spark, in_file: str, out_dir: str):
    """
    Legge un file 'cleaned', applica prep() e scrive UN SOLO .jsonl in out_dir.
    Mantiene il naming coerente: <stem>_prepped.jsonl
    (se era *_cleaned.jsonl, rimpiazza _cleaned -> _prepped).
    """
    base = os.path.basename(in_file)
    stem, ext = os.path.splitext(base)        # ext include ".jsonl" o ".json"
    if stem.endswith("_cleaned"):
        stem_out = stem[:-8] + "_prepped"     # rimuove "_cleaned" e mette "_prepped"
    else:
        stem_out = stem + "_prepped"

    print(f"\n[INFO] Prep su: {base}")

    df = spark.read.option("multiLine", "false").json(in_file)
    df_prep = prep(df)

    final_file = os.path.join(out_dir, f"{stem_out}.jsonl")
    write_single_jsonl(df_prep, final_file)

# ─────────────────────────────────────────────────────────
# Pipeline per cartella
# ─────────────────────────────────────────────────────────
def run_prep(spark, cleaned_dir: str, prepared_dir: str):
    """
    Applica la preparazione a tutti i file in 'cleaned_dir' e salva
    UN file .jsonl per ciascun input in 'prepared_dir'.
    """
    ensure_dir(prepared_dir)
    files = list_cleaned_files(cleaned_dir)
    if not files:
        print(f"[ATTENZIONE] Nessun file in {cleaned_dir}")
        return
    for f in files:
        process_single_file(spark, in_file=f, out_dir=prepared_dir)
    print(f"\n[OK] Preparazione completata. Output: {prepared_dir}")
