# project/cleaning.py
from pyspark.sql import functions as F
from pyspark.sql.types import StructType
import json, os, glob, shutil

# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def column_exists(df, dotted: str) -> bool:
    """
    Verifica l'esistenza di una colonna, supportando path con dot notation (es. 'user.id').
    """
    parts = dotted.split(".")
    schema = df.schema
    for p in parts:
        if not isinstance(schema, StructType):
            return False
        names = [f.name for f in schema.fields]
        if p not in names:
            return False
        field = next(f for f in schema.fields if f.name == p)
        schema = field.dataType
    return True

def drop_list(df, cols):
    """
    Droppa solo colonne TOP-LEVEL (Spark non supporta df.drop su nested).
    Per i nested usare funzioni dedicate (entities/user/retweeted_status).
    """
    top = [c for c in cols if "." not in c and column_exists(df, c)]
    if top:
        df = df.drop(*top)
    return df, top

# ──────────────────────────────────────────────────────────────────────────────
# ENTITIES: rimuovi urls, pulisci
# ──────────────────────────────────────────────────────────────────────────────
def strip_entities_clean(df):
    """
    - Elimina COMPLETAMENTE entities.urls (ricostruendo la struct)
    - user_mentions -> conserva: id, name, screen_name
    - hashtags -> conserva: text
    - symbols -> lascia invariato
    - Se 'entities' resta vuoto, droppalo
    """
    if not column_exists(df, "entities"):
        return df

    parts = []

    if column_exists(df, "entities.user_mentions"):
        parts.append(
            F.transform(
                F.col("entities.user_mentions"),
                lambda m: F.struct(
                    m["id"].alias("id"),
                    m["name"].alias("name"),
                    m["screen_name"].alias("screen_name")
                )
            ).alias("user_mentions")
        )

    if column_exists(df, "entities.hashtags"):
        parts.append(
            F.transform(
                F.col("entities.hashtags"),
                lambda h: F.struct(h["text"].alias("text"))
            ).alias("hashtags")
        )

    if column_exists(df, "entities.symbols"):
        parts.append(F.col("entities.symbols").alias("symbols"))

    # Ricostruisci entities SENZA urls (e senza altri campi non whitelisted)
    if parts:
        df = df.withColumn("entities", F.struct(*parts))
    else:
        df = df.drop("entities")

    return df

# ──────────────────────────────────────────────────────────────────────────────
# USER: whitelist semplice
# ──────────────────────────────────────────────────────────────────────────────
def strip_user_simple(df, keep_fields=None, drop_if_empty=True):
    """
    Ricostruisce 'user' mantenendo solo i campi della whitelist (keep_fields).
    Se non resta nulla e drop_if_empty=True, droppa 'user'.
    """
    if not column_exists(df, "user"):
        return df

    if keep_fields is None:
        keep_fields = [
            "id",
            "name",
            "screen_name",
            "location",
            "description",
            "verified",
            "created_at",
            # "followers_count", "friends_count", "favourites_count",
            # "listed_count", "statuses_count",
        ]

    existing = [k for k in keep_fields if column_exists(df, f"user.{k}")]
    if not existing:
        return df.drop("user") if drop_if_empty else df

    cols = [F.col(f"user.{k}").alias(k) for k in existing]
    return df.withColumn("user", F.struct(*cols))

# ──────────────────────────────────────────────────────────────────────────────
# RETWEETED_STATUS: whitelist richiesta
# ──────────────────────────────────────────────────────────────────────────────
def strip_retweeted_status_keep(df, keep_fields=None, drop_if_empty=True):
    """
    Ricostruisce 'retweeted_status' mantenendo solo i campi specificati.
    Se non resta nulla e drop_if_empty=True, droppa 'retweeted_status'.

    Campi richiesti (default):
      - created_at
      - favorite_count
      - retweet_count
      - id
      - geo
      - filter_level
      - full_text
      - text
      - truncated
      - source
    """
    if not column_exists(df, "retweeted_status"):
        return df

    if keep_fields is None:
        keep_fields = [
            "created_at",
            "favorite_count",
            "retweet_count",
            "id",
            "geo",
            "filter_level",
            "full_text",
            "text",
            "truncated",
            "source",
        ]

    existing = [k for k in keep_fields if column_exists(df, f"retweeted_status.{k}")]
    if not existing:
        return df.drop("retweeted_status") if drop_if_empty else df

    cols = [F.col(f"retweeted_status.{k}").alias(k) for k in existing]
    return df.withColumn("retweeted_status", F.struct(*cols))

# ──────────────────────────────────────────────────────────────────────────────
# Output helper
# ──────────────────────────────────────────────────────────────────────────────

def save_first_row_pretty(df, out_file: str):
    rows = df.head(1)
    if not rows:
        print(f"[ATTENZIONE] DF pulito vuoto; nulla da salvare per {out_file}.")
        return
    record = rows[0].asDict(recursive=True)
    ensure_dir(os.path.dirname(out_file))
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[OK] Prima riga pulita INDENTATA: {out_file}")

def write_single_jsonl(df, final_out_file: str):
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

    # crea se non esiste, sovrascrive se esiste
    df.coalesce(1).write.mode("overwrite").json(temp_dir)

    # prova a trovare il part file
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

    # normalizza estensione a .jsonl
    if final_out_file.endswith(".json"):
        final_out_file = final_out_file[:-5] + ".jsonl"
    elif not final_out_file.endswith(".jsonl"):
        final_out_file = final_out_file + ".jsonl"

    # sovrascrivi se esiste
    if os.path.exists(final_out_file):
        os.remove(final_out_file)

    shutil.move(part_file, final_out_file)
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"[OK] File unico scritto: {final_out_file}")

def list_input_files(folder: str):
    """
    Restituisce i file di primo livello in 'folder' con estensioni tipiche JSON.
    Se i tuoi file sono senza estensione, rimuovi il filtro oppure aggiungile qui.
    """
    exts = {".jsonl", ".json", ".jsonl.gz", ".json.gz"}
    if not os.path.isdir(folder):
        print(f"[ATTENZIONE] INPUT_PATH non è una cartella valida: {folder}")
        return []
    files = []
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if os.path.isfile(p):
            _, ext = os.path.splitext(p)
            if ext.lower() in exts:
                files.append(p)
    print(f"[INFO] Trovati {len(files)} file in input.")
    if not files:
        print("[SUGGERIMENTO] Controlla le estensioni: accetto .json, .jsonl, .json.gz, .jsonl.gz")
    return files

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline per singolo file
# ──────────────────────────────────────────────────────────────────────────────
def process_single_file(spark, in_file: str, out_dir: str, first_dir: str):
    """
    Esegue l'intera pipeline di pulizia per un singolo file e scrive:
      - file pulito come JSONL in out_dir
      - prima riga pulita (pretty JSON) in first_dir
    """
    # Colonne TOP-LEVEL da droppare (come nella tua versione)
    DROP_COLS = [
        "contributors", "favorited", "retweeted", "display_text_range",
        "id_str", "quoted_status_id_str", "quoted_status",
        "extended_entities",
        "in_reply_to_status_id", "in_reply_to_status_id_str", "in_reply_to_user_id_str",
    ]

    base = os.path.basename(in_file)
    stem, _ = os.path.splitext(base)
    print(f"\n[INFO] Processo: {base}")

    # 0) Leggi e cache
    df = spark.read.option("multiLine", "false").json(in_file).cache()
    _ = df.count()  # materializza la cache

    # FILTRA record corrotti
    if column_exists(df, "_corrupt_record"):
        df = df.filter(F.col("_corrupt_record").isNull()).drop("_corrupt_record")

    # 1) Drop top-level
    df, actually_dropped = drop_list(df, DROP_COLS)
    if actually_dropped:
        print(f"[INFO] Droppate {len(actually_dropped)} colonne TOP-LEVEL in {base}: {actually_dropped}")

    # 2) Ripulisci entities (rimuove urls e altro superfluo non whitelisted)
    df = strip_entities_clean(df)

    # 3) Ripulisci user (solo whitelist)
    df = strip_user_simple(df)

    # 4) Ripulisci retweeted_status (solo i campi richiesti)
    df = strip_retweeted_status_keep(df)

    # 5) Scrivi UN SOLO file JSONL per questo input
    final_file = os.path.join(out_dir, f"{stem}_cleaned.jsonl")
    write_single_jsonl(df, final_file)

    # 6) Salva la prima riga pulita
    ensure_dir(first_dir)
    first_row_file = os.path.join(first_dir, f"{stem}_first_row.json")
    save_first_row_pretty(df, first_row_file)

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline per cartella
# ──────────────────────────────────────────────────────────────────────────────
def run_cleaning(spark, input_dir: str, output_dir: str, first_preview_dir: str):
    """
    Pulisce tutti i file supportati in input_dir usando la pipeline sopra.
    """
    ensure_dir(output_dir)
    ensure_dir(first_preview_dir)

    files = list_input_files(input_dir)
    for f in files:
        process_single_file(
            spark,
            in_file=f,
            out_dir=output_dir,
            first_dir=first_preview_dir,
        )
    print(f"\n[OK] Pulizia completata. Output: {output_dir} | First-rows: {first_preview_dir}")
