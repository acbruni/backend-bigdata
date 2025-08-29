# main.py (radice repo)
import os
import uvicorn
from pyspark.sql import SparkSession

from project import cleaning as cl
from project import prep
from project.api_query import create_app   # deve supportare files_limit opzionale, come da tua versione
from project.model import add_model_routes

def main():
    # 1) Percorsi
    INPUT_PATH        = "/Users/acb_23/Documents/materialeBigData/progettoBigData/backend-bigdata-1/project/dataset"
    CLEANED_DIR       = "/Users/acb_23/Documents/materialeBigData/progettoBigData/backend-bigdata-1/project/new_ds"
    FIRST_PREVIEW_DIR = "/Users/acb_23/Documents/materialeBigData/progettoBigData/backend-bigdata-1/project/first"
    PREP_DIR          = "/Users/acb_23/Documents/materialeBigData/progettoBigData/backend-bigdata-1/project/prep_ds"
    MODEL_DIR         = "/Users/acb_23/Documents/materialeBigData/progettoBigData/backend-bigdata-1/project/models"

    # Limiti opzionali presi da env (0 o vuoto = None)
    QUERY_FILES_LIMIT = int(os.getenv("QUERY_FILES_LIMIT", "15")) or None
    MODEL_FILES_LIMIT = int(os.getenv("MODEL_FILES_LIMIT", "15")) or None

    # 2) SparkSession
    spark = (
        SparkSession.builder
        .appName("DisasterTweets Analysis")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.driver.memory", "7g")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.shuffle.partitions", "256")
        .config("spark.sql.debug.maxToStringFields", "2000")
        .getOrCreate()
    )
    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
    spark.conf.set("spark.sql.session.timeZone", "UTC")

    # 3) Cleaning (salta se già fatto)
    if os.path.isdir(CLEANED_DIR) and os.listdir(CLEANED_DIR) \
       and os.path.isdir(FIRST_PREVIEW_DIR) and os.listdir(FIRST_PREVIEW_DIR):
        print("[INFO] 'new_ds' e 'first' presenti: salto cleaning.")
    else:
        print("[INFO] Eseguo cleaning...")
        cl.run_cleaning(
            spark,
            input_dir=INPUT_PATH,
            output_dir=CLEANED_DIR,
            first_preview_dir=FIRST_PREVIEW_DIR,
        )

    # 4) Prep (salta se già fatto)
    if os.path.isdir(PREP_DIR) and os.listdir(PREP_DIR):
        print("[INFO] 'prep_ds' presente: salto prep.")
    else:
        print("[INFO] Eseguo prep...")
        prep.run_prep(spark, cleaned_dir=CLEANED_DIR, prepared_dir=PREP_DIR)

    # 5) App query (lavora su prep_ds invece che su new_ds)
    app = create_app(spark, PREP_DIR, files_limit=QUERY_FILES_LIMIT)

    # 6) Route modello (lavora su prep_ds invece che su new_ds)
    add_model_routes(app, spark, prepared_dir=PREP_DIR, model_root=MODEL_DIR, default_files_limit=MODEL_FILES_LIMIT)

    # 7) Avvio server
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
