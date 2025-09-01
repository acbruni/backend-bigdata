# project/api_query.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark import StorageLevel
import os, glob, json

# ─────────────────────────────
# Helpers / compat
# ─────────────────────────────
try:
    # FastAPI re-export (preferito)
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    # Fallback per versioni vecchie
    from starlette.middleware.cors import CORSMiddleware
    
def _MEMORY_ONLY_SER():
    try:
        return StorageLevel.MEMORY_ONLY_SER
    except AttributeError:
        return StorageLevel(False, True, False, False, 1)

def _MEMORY_AND_DISK_SER():
    try:
        return StorageLevel.MEMORY_AND_DISK_SER
    except AttributeError:
        return StorageLevel(True, True, False, False, 1)

def _list_prep_files(prepared_dir: str) -> List[str]:
    """
    Cerca file JSON/JSONL nella cartella dei dati preparati (prep_ds).
    """
    if not os.path.isdir(prepared_dir):
        return []
    patterns = [
        os.path.join(prepared_dir, "*.jsonl"),
        os.path.join(prepared_dir, "*/*.jsonl"),
        os.path.join(prepared_dir, "*.json"),
        os.path.join(prepared_dir, "*/*.json"),
    ]
    out: List[str] = []
    for pat in patterns:
        out.extend(glob.glob(pat))
    return sorted([f for f in out if os.path.isfile(f) and os.path.getsize(f) > 0])

def _df_to_list(df: DataFrame, limit: int = 100):
    rows = df.limit(max(1, limit)).toJSON().collect()
    return [json.loads(r) for r in rows]


# ─────────────────────────────
# Costruttore dell'app: passa Spark e cartella PREPARATA dal main
# ─────────────────────────────
def create_app(spark: SparkSession, prepared_dir: str, files_limit: Optional[int] = None) -> FastAPI:
    """
    prepared_dir: cartella 'prep_ds' già prodotta da prep.py
    """
    app = FastAPI(title="Disaster Tweets API (queries su dati preparati)")

    # CORS: permette localhost e 127.0.0.1 su qualunque porta
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127.0.0.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=[""],
        allow_headers=[""],
    )

    # stato interno condiviso dagli endpoint
    df_prepared: DataFrame | None = None  # chiusura
    used_files: List[str] = []            # per health/debug

    def _load():
        nonlocal df_prepared, used_files
        all_files = _list_prep_files(prepared_dir)
        if not all_files:
            raise RuntimeError(f"Nessun file preparato in {prepared_dir}")

        if files_limit is not None and files_limit > 0:
            used_files = all_files[:files_limit]   # deterministico
        else:
            used_files = all_files

        # Carica direttamente i dati preparati (niente prep() qui)
        df = spark.read.json(used_files)

        # cache prudente
        df_prepared = df.persist(_MEMORY_AND_DISK_SER())
        _ = df_prepared.take(1)  # warm-up leggero

    @app.on_event("startup")
    def _on_startup():
        _load()

    @app.on_event("shutdown")
    def _on_shutdown():
        try:
            spark.stop()
        except Exception:
            pass

    # ─────────────────────────────
    # Endpoint semplici
    # ─────────────────────────────

    @app.get("/health")
    def health():
        try:
            total = len(_list_prep_files(prepared_dir))
            used = used_files if isinstance(used_files, list) else []
            df_cached = bool(globals().get('df_prepared') is not None)

            return JSONResponse({
                "status": "ok",
                "files_total_in_dir": total,
                "files_used_for_queries": len(used),
                "sample_used_files": used[:5],
                "df_cached": df_cached,
            })
        except Exception as e:
            # In caso di errore, torna 500 con un dettaglio utile
            raise HTTPException(status_code=500, detail=f"health failed: {e}")

    @app.post("/reload")
    def reload():
        _load()
        return JSONResponse({"reloaded": True, "files_used_for_queries": len(used_files)})

    @app.get("/tot_rows")
    def tot_rows():
        result = df_prepared.count()
        return JSONResponse({"tot_rows": int(result)})

    @app.get("/top-viral-post")
    def top_viral_post(limit: int = Query(1, ge=1, le=50)):
        res = (df_prepared
               .orderBy(F.col("engagement").desc_nulls_last())
               .limit(limit))
        return JSONResponse(_df_to_list(res, limit))

    @app.get("/tweets-per-hour")
    def tweets_per_hour(limit: int = Query(100, ge=1, le=2000)):
        res = (df_prepared.groupBy("hour")
               .agg(F.count("*").alias("tweets"))
               .orderBy("hour"))
        return JSONResponse(_df_to_list(res, limit))

    @app.get("/top-hashtags")
    def top_hashtags(min_len: int = Query(1, ge=1, le=100),
                     limit: int = Query(100, ge=1, le=2000)):
        res = (
            df_prepared.withColumn("hashtag", F.explode_outer("hashtags_arr"))
                       .filter((F.col("hashtag").isNotNull()) & (F.length("hashtag") >= min_len))
                       .groupBy("hashtag").agg(F.count("*").alias("uses"))
                       .orderBy(F.col("uses").desc())
        )
        return JSONResponse(_df_to_list(res, limit))

    @app.get("/top-languages")
    def top_languages(limit: int = Query(100, ge=1, le=2000)):
        res = df_prepared.groupBy("lang").agg(F.count(F.lit(1)).alias("tweets")).orderBy(F.col("tweets").desc())
        return JSONResponse(_df_to_list(res, limit))

    @app.get("/top-places")
    def top_places(limit: int = Query(100, ge=1, le=2000)):
        # usa la versione canonica per accorpare alias (già calcolata in prep)
        base = df_prepared.filter(F.col("place_norm") != "")
        res = (base.groupBy("place_norm")
               .agg(F.count("*").alias("tweets"))
               .orderBy(F.col("tweets").desc()))
        res = res.select(F.col("place_norm").alias("place"), F.col("tweets"))
        return JSONResponse(_df_to_list(res, limit))

    @app.get("/sensitive-stats")
    def sensitive_stats(limit: int = Query(100, ge=1, le=2000)):
        res = (
            df_prepared.groupBy(F.coalesce(F.col("possibly_sensitive"), F.lit(False)).alias("possibly_sensitive"))
                       .agg(F.count(F.lit(1)).alias("tweets"))
                       .orderBy(F.col("tweets").desc())
        )
        return JSONResponse(_df_to_list(res, limit))

    @app.get("/peaks-lang-verified")
    def peaks_lang_verified(limit: int = Query(100, ge=1, le=2000)):
        res = (df_prepared.groupBy("hour", "lang", "verified")
               .agg(F.count("*").alias("volume"),
                    F.avg("engagement").alias("avg_engagement"))
               .orderBy(F.col("avg_engagement").desc_nulls_last()))
        return JSONResponse(_df_to_list(res, limit))

    @app.get("/efficient-hashtags")
    def efficient_hashtags(min_uses: int = Query(20, ge=1, le=100000),
                           limit: int = Query(100, ge=1, le=2000)):
        stats = (df_prepared.withColumn("hashtag", F.explode_outer("hashtags_arr"))
                 .filter((F.col("hashtag").isNotNull()) & (F.col("hashtag") != ""))
                 .groupBy("hashtag")
                 .agg(F.count("*").alias("uses"),
                      F.avg("engagement").alias("avg_engagement"),
                      F.sum("engagement").alias("tot_engagement")))
        res = stats.filter(F.col("uses") >= min_uses).orderBy(F.col("avg_engagement").desc_nulls_last())
        return JSONResponse(_df_to_list(res, limit))

    @app.get("/device-verified-effect")
    def device_verified_effect(limit: int = Query(100, ge=1, le=2000)):
        base = df_prepared.withColumn(
            "device_norm",
            F.when(F.trim(F.col("device_norm")) == "", F.lit("Unknown")).otherwise(F.col("device_norm"))
        )
        res = (base.groupBy("device_norm", "verified")
               .agg(F.count("*").alias("volume"),
                    F.avg("engagement").alias("avg_engagement"))
               .orderBy(F.col("volume").desc()))
        return JSONResponse(_df_to_list(res, limit))

    @app.get("/geo-temporal-hotspots")
    def geo_temporal_hotspots(limit: int = Query(100, ge=1, le=2000)):
        base = df_prepared.filter(F.col("place_norm") != "")
        res = (base.groupBy("place_norm", "hour")
               .agg(F.count("*").alias("volume"),
                    F.avg("engagement").alias("avg_engagement"))
               .orderBy(F.col("volume").desc(), F.col("avg_engagement").desc_nulls_last()))
        res = res.select(F.col("place_norm").alias("place"), "hour", "volume", "avg_engagement")
        return JSONResponse(_df_to_list(res, limit))

    @app.get("/early-vs-late")
    def early_vs_late(hours: int = Query(2, ge=1, le=48),
                      limit: int = Query(100, ge=1, le=2000)):
        min_ts = df_prepared.select(F.min("ts").alias("m")).first()["m"]
        phased = df_prepared.withColumn(
            "phase",
            F.when(F.col("ts") < (F.lit(min_ts) + F.expr(f"INTERVAL {int(hours)} HOURS")), F.lit("early"))
             .otherwise(F.lit("late"))
        )
        res = (phased.groupBy("phase")
               .agg(F.count(F.lit(1)).alias("tweets"),
                    F.avg("engagement").alias("avg_engagement"),
                    F.sum("engagement").alias("tot_engagement"))
               .orderBy("phase"))
        return JSONResponse(_df_to_list(res, limit))

    @app.get("/mentions-impact-proxy")
    def mentions_impact_proxy(limit: int = Query(100, ge=1, le=2000)):
        res = (df_prepared.groupBy("verified", "has_mentions")
               .agg(F.count(F.lit(1)).alias("volume"),
                    F.avg("engagement").alias("avg_engagement"),
                    F.sum("engagement").alias("tot_engagement"))
               .orderBy("verified", F.col("has_mentions").desc()))
        return JSONResponse(_df_to_list(res, limit))

    @app.get("/top-hours-by-engagement")
    def top_hours_by_engagement(min_volume: int = Query(20, ge=1, le=100000),
                                limit: int = Query(100, ge=1, le=2000)):
        agg = (df_prepared.groupBy("hour")
               .agg(F.count(F.lit(1)).alias("volume"),
                    F.avg("engagement").alias("avg_engagement"),
                    F.sum("engagement").alias("tot_engagement")))
        res = agg.filter(F.col("volume") >= min_volume).orderBy(F.col("avg_engagement").desc_nulls_last())
        return JSONResponse(_df_to_list(res, limit))

    return app
