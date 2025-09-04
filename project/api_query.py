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
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
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
# Costruttore dell'app
# ─────────────────────────────
def create_app(spark: SparkSession, prepared_dir: str, files_limit: Optional[int] = None) -> FastAPI:
    app = FastAPI(title="Disaster Tweets API (queries su dati preparati)")

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127.0.0.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=[""],
        allow_headers=[""],
    )

    df_prepared: DataFrame | None = None
    used_files: List[str] = []

    def _load():
        nonlocal df_prepared, used_files
        all_files = _list_prep_files(prepared_dir)
        if not all_files:
            raise RuntimeError(f"Nessun file preparato in {prepared_dir}")
        used_files = all_files[:files_limit] if (files_limit and files_limit > 0) else all_files
        df = spark.read.json(used_files)
        df_prepared = df.persist(_MEMORY_AND_DISK_SER())
        _ = df_prepared.take(1)

    @app.on_event("startup")
    def _on_startup():
        _load()

    @app.on_event("shutdown")
    def _on_shutdown():
        try:
            spark.stop()
        except Exception:
            pass

    # ---------- helper: epoch ms robusto da hour_ts (timestamp O stringa ISO) ----------
    def _hour_ms():
        # se è già timestamp, usalo; altrimenti parse ISO 8601 con timezone (X gestisce 'Z')
        ts_parsed = F.coalesce(
            F.col("hour_ts").cast("timestamp"),
            F.to_timestamp(F.col("hour_ts"), "yyyy-MM-dd'T'HH:mm:ss.SSSX")
        )
        return (ts_parsed.cast("long") * F.lit(1000))

    # ─────────────────────────────
    # Endpoint
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
        # Root id: originale se presente, altrimenti l'id del tweet retwittato
        base = df_prepared.withColumn(
            "root_id",
            F.coalesce(F.col("retweeted_status.id").cast("long"),
                    F.col("id").cast("long"))
        )

        # Top root per engagement massimo
        top_roots = (
            base.groupBy("root_id")
                .agg(F.max("engagement").alias("engagement"))
                .orderBy(F.col("engagement").desc_nulls_last())
                .limit(int(limit))
        )

        # Prendi una sola riga per root (preferisci originali)
        picked = (
            top_roots.join(base, on=["root_id", "engagement"], how="inner")
                    .orderBy(F.col("is_rt").asc(), F.col("engagement").desc_nulls_last())
                    .dropDuplicates(["root_id"])
        )

        # Testo da esporre: per RT usa il testo dell'originale se presente
        text_display = F.when(
            F.col("is_rt"),
            F.coalesce(F.col("retweeted_status.text"),
                    F.col("text_full"),
                    F.col("text"))
        ).otherwise(F.coalesce(F.col("text_full"), F.col("text"))).alias("text_full")

        res = (
            picked.select(
                text_display,
                F.col("engagement"),
                F.col("lang"),
                F.col("device_norm"),
                F.col("created_at"),
                F.col("user.name").alias("user_name"),
                F.col("user.screen_name").alias("user_handle"),
            )
            .orderBy(F.col("engagement").desc_nulls_last())
        )

        return JSONResponse(_df_to_list(res, limit))

    # ====== FIX: timeline oraria (hour_ts) + hour_ms robusto ======
    @app.get("/tweets-per-hour")
    def tweets_per_hour(limit: int = Query(100, ge=1, le=2000)):
        # Ora del giorno 0..23 dalla nuova prep
        res = (
            df_prepared.groupBy("hour_of_day")
                    .agg(F.count("*").alias("tweets"))
                    .orderBy("hour_of_day")
                    .select(
                        F.col("hour_of_day").cast("int").alias("hour"),
                        F.col("tweets").cast("long").alias("tweets")
                    )
        )
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

    # ====== FIX: peaks per lang/verified su hour_ts + hour_ms robusto ======
    @app.get("/peaks-lang-verified")
    def peaks_lang_verified(limit: int = Query(100, ge=1, le=2000)):
        res = (
            df_prepared.groupBy("hour_ts", "lang", "verified")
                       .agg(
                           F.count("*").alias("volume"),
                           F.avg("engagement").alias("avg_engagement")
                       )
                       .orderBy(F.col("avg_engagement").desc_nulls_last())
                       .select(
                           F.date_format(F.col("hour_ts"), "yyyy-MM-dd'T'HH:mm:ss'Z'").alias("hour"),
                           _hour_ms().alias("hour_ms"),
                           "lang", "verified", "volume", "avg_engagement"
                       )
        )
        return JSONResponse(_df_to_list(res, limit))

    @app.get("/efficient-hashtags")
    def efficient_hashtags(min_uses: int = Query(100, ge=1, le=100000),
                           limit: int = Query(10, ge=1, le=2000)):
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

    # ====== FIX: geo-temporal con hour_ts + hour_ms robusto ======
    @app.get("/geo-temporal-hotspots")
    def geo_temporal_hotspots(limit: int = Query(100, ge=1, le=2000)):
        base = df_prepared.filter(F.col("place_norm") != "")
        res = (
            base.groupBy("place_norm", "hour_of_day")
                .agg(
                    F.count("*").alias("volume"),
                    F.avg("engagement").alias("avg_engagement"),
                )
                .orderBy(F.col("volume").desc(), F.col("avg_engagement").desc_nulls_last())
                .select(
                    F.col("place_norm").alias("place"),
                    F.col("hour_of_day").alias("hour"),     # <-- 0..23 già pronto
                    F.col("volume"),
                    F.col("avg_engagement"),
                )
        )
        return JSONResponse(_df_to_list(res, limit))


    @app.get("/early-vs-late")
    def early_vs_late(hours: int = Query(2, ge=1, le=48),
                    limit: int = Query(100, ge=1, le=2000)):
        # 1) Ore valide e distinte
        valid_hours = (
            df_prepared
            .select("hour_ts")
            .where(F.col("hour_ts").isNotNull() & (F.col("hour_ts") != ""))
            .distinct()
        )

        # 2) Prendi le prime hours ore ordinate e ricava la soglia (max di quelle)
        #    NB: ordine lessicografico su ISO-Z coincide con ordine temporale.
        topN = valid_hours.orderBy(F.col("hour_ts").asc()).limit(int(hours))
        row = topN.agg(F.max("hour_ts").alias("threshold")).first()
        threshold = row["threshold"]

        # 3) Se non esiste alcuna ora valida -> tutto 'late'
        if threshold is None:
            phased = df_prepared.withColumn("phase", F.lit("late"))
        else:
            phased = df_prepared.withColumn(
                "phase",
                F.when(
                    F.col("hour_ts").isNotNull() &
                    (F.col("hour_ts") != "") &
                    (F.col("hour_ts") <= F.lit(threshold)),  # confronto tra stringhe ISO
                    F.lit("early")
                ).otherwise(F.lit("late"))
            )

        # 4) Aggregazione finale
        res = (
            phased.groupBy("phase")
                .agg(F.count(F.lit(1)).alias("tweets"),
                    F.avg("engagement").alias("avg_engagement"),
                    F.sum("engagement").alias("tot_engagement"))
                .orderBy("phase")
        )
        return JSONResponse(_df_to_list(res, limit))

    @app.get("/mentions-impact-proxy")
    def mentions_impact_proxy(limit: int = Query(100, ge=1, le=2000)):
        res = (
            df_prepared
            .groupBy("verified", "has_mentions")
            .agg(
                F.count("*").alias("volume"),
                F.avg("engagement").alias("avg_engagement"),
            )
            .orderBy(F.col("verified").asc(), F.col("has_mentions").desc())
            .select("verified", "has_mentions", "volume", "avg_engagement")
        )
        return JSONResponse(_df_to_list(res, limit))

    # ====== invariato concettualmente: ora del giorno (0–23) ======
    @app.get("/top-hours-by-engagement")
    def top_hours_by_engagement(min_volume: int = Query(20, ge=1, le=100000),
                                limit: int = Query(100, ge=1, le=2000)):
        agg = (
            df_prepared.groupBy("hour_of_day")
                       .agg(
                           F.count(F.lit(1)).alias("volume"),
                           F.avg("engagement").alias("avg_engagement"),
                           F.sum("engagement").alias("tot_engagement")
                       )
                       .select(
                           F.col("hour_of_day").cast("int").alias("hour"),
                           "volume", "avg_engagement", "tot_engagement"
                       )
        )
        res = agg.filter(F.col("volume") >= min_volume)\
                 .orderBy(F.col("avg_engagement").desc_nulls_last())
        return JSONResponse(_df_to_list(res, limit))

    return app
