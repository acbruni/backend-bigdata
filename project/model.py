# project/model.py
from __future__ import annotations

import os, glob, json, shutil
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import JSONResponse

from pyspark.sql import SparkSession, DataFrame, functions as F, types as T
from pyspark import StorageLevel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
    VectorAssembler,
)
from pyspark.ml.classification import LogisticRegression


# ─────────────────────────────────────────────────────────────
# Parametri e percorsi modello
# ─────────────────────────────────────────────────────────────
MODEL_SUBDIR  = "info_mlr_model"         # il train sovrascrive qui
META_FILENAME = "info_mlr_meta.json"

# classi: indice → nome
CLASS_INDEX = {
    0: "Request/Need",
    1: "Offer/Donation",
    2: "Damage/Impact",
    3: "Other"
}
CLASS_NAME_TO_INDEX = {v: k for k, v in CLASS_INDEX.items()}

# parole chiave per etichettatura "silver" (minuscole)
KEYWORDS = {
    "Request/Need": {
        "text": ["need", "help", "urgent", "please send", "request", "rescue", "shelter needed", "trapped", "asap"],
        "tags": ["help", "rescue", "needs", "shelter"]
    },
    "Offer/Donation": {
        "text": ["donate", "donations", "volunteer", "offering", "provide", "supplies available", "fundraiser"],
        "tags": ["donate", "volunteer", "relief", "fundraiser"]
    },
    "Damage/Impact": {
        "text": ["damage", "flooded", "power out", "outage", "sewage", "water", "roads closed", "bridge", "hospital"],
        "tags": ["flood", "damage", "powerout", "water", "sewage"]
    }
}


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────
def _list_json_files(prepared_dir: str, limit: Optional[int]) -> List[str]:
    """Lista deterministica di file .json/.jsonl dentro prepared_dir (e sottocartelle). Applica lo slice al limite."""
    if not os.path.isdir(prepared_dir):
        return []
    patterns = [
        os.path.join(prepared_dir, "*.json"),
        os.path.join(prepared_dir, "*.jsonl"),
        os.path.join(prepared_dir, "*/*.json"),
        os.path.join(prepared_dir, "*/*.jsonl"),
    ]
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(p))
    files = sorted(set(files))
    if limit is not None and limit > 0:
        return files[:limit]
    return files


def _require_cols(df: DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Mancano colonne richieste: {missing}")


# ─────────────────────────────────────────────────────────────
# Etichette silver (4 classi)
# ─────────────────────────────────────────────────────────────
def _label_silver(df: DataFrame) -> DataFrame:
    empty_arr = F.lit([]).cast("array<string>")
    txt  = F.lower(F.coalesce(F.col("text_full").cast("string"), F.lit("")))
    tags = F.coalesce(F.col("hashtags_arr").cast("array<string>"), empty_arr)

    def contains_any_text(col, words: List[str]):
        cond = F.lit(False)
        for w in words:
            cond = cond | (F.instr(col, F.lit(w)) > F.lit(0))
        return cond

    def contains_any_tag(col, words: List[str]):
        cond = F.lit(False)
        for w in words:
            cond = cond | F.array_contains(col, F.lit(w))
        return cond

    req = contains_any_text(txt, KEYWORDS["Request/Need"]["text"]) | contains_any_tag(tags, KEYWORDS["Request/Need"]["tags"])
    off = contains_any_text(txt, KEYWORDS["Offer/Donation"]["text"]) | contains_any_tag(tags, KEYWORDS["Offer/Donation"]["tags"])
    dmg = contains_any_text(txt, KEYWORDS["Damage/Impact"]["text"]) | contains_any_tag(tags, KEYWORDS["Damage/Impact"]["tags"])

    return df.withColumn(
        "label",
        F.when(req, F.lit(CLASS_NAME_TO_INDEX["Request/Need"]))
         .when(off, F.lit(CLASS_NAME_TO_INDEX["Offer/Donation"]))
         .when(dmg, F.lit(CLASS_NAME_TO_INDEX["Damage/Impact"]))
         .otherwise(F.lit(CLASS_NAME_TO_INDEX["Other"]))
         .cast("int")
    )


# ─────────────────────────────────────────────────────────────
# Pesi di classe smorzati: sqrt per non “sovra-spingere” minoritarie
# ─────────────────────────────────────────────────────────────
def _add_class_weights(df: DataFrame, label_col: str = "label") -> DataFrame:
    counts = df.groupBy(label_col).count()
    total = df.count()
    num_classes = len(CLASS_INDEX)
    weights = counts.withColumn(
        "weight_raw",
        (F.lit(float(total)) / F.lit(float(num_classes))) / F.col("count").cast("double")
    ).withColumn(
        "weight", F.sqrt(F.col("weight_raw"))
    ).select(label_col, "weight")
    return df.join(weights, on=label_col, how="left")


# ─────────────────────────────────────────────────────────────
# Pipeline leggera: StopWords → CV → IDF → Assembler → LR
# ─────────────────────────────────────────────────────────────
def _build_pipeline(min_df_tokens: int = 10, vocab_size: int = 5000, binary_cv: bool = True) -> Pipeline:
    remover = StopWordsRemover(
        inputCol="all_tokens",
        outputCol="clean_tokens",
        stopWords=StopWordsRemover.loadDefaultStopWords("english") + [
            "rt", "amp", "https", "http", "co", "www", "com"
        ],
    )

    cv = CountVectorizer(
        inputCol="clean_tokens",
        outputCol="tf_text",
        minDF=int(min_df_tokens),
        vocabSize=int(vocab_size),
        binary=binary_cv
    )

    idf = IDF(inputCol="tf_text", outputCol="vec_text")

    assembler = VectorAssembler(
        inputCols=["vec_text", "mentions_count", "is_rt", "verified", "hour_of_day"],
        outputCol="features"
    )

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        weightCol="weight",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        maxIter=40,
        regParam=0.2,
        elasticNetParam=0.0,   # L2
        family="multinomial"
    )

    return Pipeline(stages=[remover, cv, idf, assembler, lr])


# ─────────────────────────────────────────────────────────────
# Metriche
# ─────────────────────────────────────────────────────────────
def _compute_metrics(pred_df: DataFrame) -> Dict[str, Any]:
    pairs = pred_df.select(
        F.col("label").cast("int").alias("label"),
        F.col("prediction").cast("int").alias("pred")
    )
    counts = pairs.groupBy("label", "pred").count()
    n = len(CLASS_INDEX)
    mat = [[0 for _ in range(n)] for _ in range(n)]
    for r in counts.collect():
        mat[int(r["label"])][int(r["pred"])] = int(r["count"])

    per_class: Dict[str, Dict[str, float]] = {}
    total_correct, total = 0, sum(sum(row) for row in mat)
    for i in range(n):
        tp = mat[i][i]
        fp = sum(mat[r][i] for r in range(n) if r != i)
        fn = sum(mat[i][c] for c in range(n) if c != i)
        tn = total - tp - fp - fn
        prec = tp/(tp+fp) if (tp+fp) else 0.0
        rec  = tp/(tp+fn) if (tp+fn) else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) else 0.0
        per_class[CLASS_INDEX[i]] = {"precision": prec, "recall": rec, "f1": f1, "TP": tp, "FP": fp, "TN": tn, "FN": fn}
        total_correct += tp

    macro_f1 = sum(per_class[c]["f1"] for c in per_class)/float(n) if n else 0.0
    accuracy = total_correct/float(total) if total else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": {"labels": [CLASS_INDEX[i] for i in range(n)], "as_table": mat}
    }


# ─────────────────────────────────────────────────────────────
# Route FastAPI: /model/*
# ─────────────────────────────────────────────────────────────
def add_model_routes(app: FastAPI,
                     spark: SparkSession,
                     prepared_dir: str,
                     model_root: str,
                     default_files_limit: Optional[int] = None) -> None:
    """
    Monta le route del modello su un'app FastAPI già esistente.
    """
    MODEL_PATH = os.path.join(model_root, MODEL_SUBDIR)
    META_PATH  = os.path.join(model_root, META_FILENAME)
    os.makedirs(model_root, exist_ok=True)

    def _load_model_if_exists() -> Optional[PipelineModel]:
        try:
            if os.path.isdir(MODEL_PATH):
                return PipelineModel.load(MODEL_PATH)
        except Exception:
            pass
        return None

    def _load_meta() -> Dict[str, Any]:
        if os.path.isfile(META_PATH):
            try:
                with open(META_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    @app.get("/model/status")
    def model_status():
        present = os.path.isdir(MODEL_PATH)
        meta = _load_meta()
        return JSONResponse({
            "model_present": present,
            "model_path": MODEL_PATH,
            "classes": CLASS_INDEX,
            "keywords": KEYWORDS,
            "meta": meta,
            "default_files_limit_from_main": default_files_limit
        })

    @app.get("/model/labels")
    def get_labels():
        return JSONResponse({"classes": CLASS_INDEX, "keywords": KEYWORDS})

    @app.get("/model/rules")  # alias retro-compatibile
    def get_rules_alias():
        return get_labels()

    @app.post("/model/train")
    def train_model(
        files_limit: Optional[int] = Query(None, description="Override del limite file (se assente, usa quello del main)"),
        min_df_tokens: int = Query(10, description="minDF per CountVectorizer (filtra token rari)"),
        vocab_size: int = Query(5000, description="limite massimo del vocabolario per CountVectorizer"),
        dynamic_shuffle: bool = Query(True, description="adatta spark.sql.shuffle.partitions al numero di file"),
        binary_cv: bool = Query(True, description="CountVectorizer in modalità binaria (meno burstiness)")
    ):
        # 1) Limite file effettivo
        resolved_limit = files_limit if (files_limit is not None) else (default_files_limit if default_files_limit is not None else None)

        # 2) Shuffle partitions conservative
        if dynamic_shuffle:
            if resolved_limit is not None:
                target_parts = max(2, min(resolved_limit * 2, 64))
            else:
                target_parts = 32
            spark.conf.set("spark.sql.shuffle.partitions", str(target_parts))
        else:
            target_parts = int(spark.conf.get("spark.sql.shuffle.partitions", "256"))

        # 3) Input files
        files = _list_json_files(prepared_dir, resolved_limit)
        if not files:
            raise HTTPException(404, detail="Nessun file trovato in prepared_dir.")
        files_sample = files[:min(3, len(files))]

        # 4) Lettura dataset minimo
        df = spark.read.option("mode", "PERMISSIVE").json(files)
        _require_cols(df, ["text_full", "hashtags_arr", "mentions_count", "is_rt", "verified", "hour_of_day"])

        empty_arr = F.lit([]).cast("array<string>")
        base = df.select(
            F.coalesce(F.col("text_full").cast("string"), F.lit("")).alias("text_full"),
            F.coalesce(F.col("hashtags_arr").cast("array<string>"), empty_arr).alias("hashtags_arr"),
            F.col("mentions_count").cast("double").alias("mentions_count"),
            F.col("is_rt").cast("double").alias("is_rt"),
            F.col("verified").cast("double").alias("verified"),
            F.col("hour_of_day").cast("double").alias("hour_of_day"),
        )

        # 5) Etichette silver e pesi
        labeled = _label_silver(base)
        labeled = _add_class_weights(labeled)

        # 6) Split
        train_df, test_df = labeled.randomSplit([0.8, 0.2], seed=42)

        # 7) Tokenizzazione + concat con hashtag
        tokenizer = RegexTokenizer(
            inputCol="text_full", outputCol="tokens",
            pattern="[^\\p{L}\\p{N}]+", minTokenLength=2, toLowercase=True
        )
        train_tok = tokenizer.transform(train_df).withColumn(
            "all_tokens",
            F.concat(
                F.coalesce(F.col("tokens"), empty_arr),
                F.coalesce(F.col("hashtags_arr"), empty_arr)
            )
        )
        test_tok = tokenizer.transform(test_df).withColumn(
            "all_tokens",
            F.concat(
                F.coalesce(F.col("tokens"), empty_arr),
                F.coalesce(F.col("hashtags_arr"), empty_arr)
            )
        )

        # 8) Persist con spill su disco
        train_tok = train_tok.persist(StorageLevel.MEMORY_AND_DISK)
        test_tok  = test_tok.persist(StorageLevel.MEMORY_AND_DISK)
        _ = train_tok.count(); _ = test_tok.count()

        # 9) Pipeline e fit
        pipe = _build_pipeline(min_df_tokens=min_df_tokens, vocab_size=vocab_size, binary_cv=binary_cv)
        model = pipe.fit(train_tok)

        # 10) Valutazione
        pred_test = model.transform(test_tok)
        metrics = _compute_metrics(pred_test)

        # 11) Salva modello (overwrite)
        if os.path.isdir(MODEL_PATH):
            shutil.rmtree(MODEL_PATH, ignore_errors=True)
        model.write().overwrite().save(MODEL_PATH)

        # 12) Metadati
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "classes": CLASS_INDEX,
                "keywords": KEYWORDS,
                "min_df_tokens": min_df_tokens,
                "vocab_size": vocab_size,
                "binary_cv": binary_cv,
                "default_files_limit_from_main": default_files_limit
            }, f, ensure_ascii=False, indent=2)

        # 13) Cleanup persist
        try:
            train_tok.unpersist()
            test_tok.unpersist()
        except Exception:
            pass

        return JSONResponse({
            "resolved_files_limit": resolved_limit,
            "used_files": len(files),
            "files_sample": files_sample,
            "shuffle_partitions_used": target_parts,
            "min_df_tokens_used": min_df_tokens,
            "vocab_size_used": vocab_size,
            "binary_cv_used": binary_cv,
            "model_path": MODEL_PATH,
            "metrics": metrics
        })

    @app.post("/model/predict")
    def predict(payload: Dict[str, Any] = Body(...)):
        """
        Predizione con body JSON. Esempio body:
        {
          "text_full": "help me please",
          "hashtags_arr": ["help"],
          "mentions_count": 1,
          "is_rt": 1,
          "verified": 1,
          "hour_of_day": 12
        }
        """
        # 1) Modello
        model = _load_model_if_exists()
        if model is None:
            raise HTTPException(404, detail="Modello non addestrato. Esegui /model/train prima.")

        # 2) Input minimo
        text_full = str(payload.get("text_full", "") or "")
        hashtags  = payload.get("hashtags_arr", []) or []
        mentions  = float(payload.get("mentions_count", 0.0))
        is_rt     = float(payload.get("is_rt", 0.0))
        verified  = float(payload.get("verified", 0.0))
        hour      = float(payload.get("hour_of_day", 0.0))

        schema = T.StructType([
            T.StructField("text_full", T.StringType(), True),
            T.StructField("hashtags_arr", T.ArrayType(T.StringType()), True),
            T.StructField("mentions_count", T.DoubleType(), True),
            T.StructField("is_rt", T.DoubleType(), True),
            T.StructField("verified", T.DoubleType(), True),
            T.StructField("hour_of_day", T.DoubleType(), True),
        ])
        df_one = spark.createDataFrame([(text_full, hashtags, mentions, is_rt, verified, hour)], schema=schema)

        # 3) Token + concat coerente col training
        empty_arr = F.lit([]).cast("array<string>")
        tokenizer = RegexTokenizer(
            inputCol="text_full", outputCol="tokens",
            pattern="[^\\p{L}\\p{N}]+", minTokenLength=2, toLowercase=True
        )
        df_one_tok = tokenizer.transform(df_one).withColumn(
            "all_tokens",
            F.concat(
                F.coalesce(F.col("tokens"), empty_arr),
                F.coalesce(F.col("hashtags_arr"), empty_arr)
            )
        )

        # 4) Predizione
        out = model.transform(df_one_tok).select("prediction", "probability").first()
        pred_idx = int(out["prediction"])
        prob_vec = out["probability"]
        try:
            probs = [float(x) for x in prob_vec.toArray().tolist()]
        except AttributeError:
            probs = [float(x) for x in prob_vec]

        return JSONResponse({
            "predicted_class_index": pred_idx,
            "predicted_class_name": CLASS_INDEX.get(pred_idx, "Unknown"),
            "probabilities": probs
        })