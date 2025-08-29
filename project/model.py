# project/model.py
from __future__ import annotations

import os, json, glob, math
from datetime import datetime
from typing import Dict, Optional, List, Tuple

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ─────────────────────────────────────────────────────────
# Utilità base
# ─────────────────────────────────────────────────────────
_TS_FMT = "%Y%m%d-%H%M%S"

def _now_tag() -> str:
    return datetime.now().strftime(_TS_FMT)

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _write_small_json(obj: dict, out_path: str):
    _ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _list_prep_files(prepared_dir: str, limit: Optional[int] = None) -> List[str]:
    patterns = [
        os.path.join(prepared_dir, "*_prepped.jsonl"),
        os.path.join(prepared_dir, "*.jsonl"),
        os.path.join(prepared_dir, "*/*.jsonl"),
        os.path.join(prepared_dir, "*.json"),
        os.path.join(prepared_dir, "*/*.json"),
    ]
    out: List[str] = []
    for p in patterns:
        out.extend(glob.glob(p))
    out = sorted([f for f in out if os.path.isfile(f) and os.path.getsize(f) > 0])
    if limit is not None:
        out = out[: int(limit)]
    return out

def _find_latest_run(model_root: str) -> Optional[str]:
    txt = os.path.join(model_root, "latest_path.txt")
    if os.path.isfile(txt):
        with open(txt, "r", encoding="utf-8") as f:
            return f.read().strip()
    cand = sorted([p for p in glob.glob(os.path.join(model_root, "rf_*")) if os.path.isdir(p)])
    return cand[-1] if cand else None

# ─────────────────────────────────────────────────────────
# Target binario: aggiunge SOLO la colonna label
# ─────────────────────────────────────────────────────────
def add_label(df: DataFrame, viral_quantile: float = 0.70, viral_threshold: Optional[int] = None) -> DataFrame:
    """
    Crea label binaria: 1 se 'engagement' >= soglia (se fornita) altrimenti > quantile.
    Non crea nuove feature: si assume che quelle necessarie siano già nel dataset preparato.
    """
    if viral_threshold is not None:
        return df.withColumn("label", (F.col("engagement") >= F.lit(int(viral_threshold))).cast("int"))
    q = df.approxQuantile("engagement", [viral_quantile], 0.01)[0]
    return df.withColumn("label", (F.col("engagement") > F.lit(q)).cast("int"))

# Candidati: useremo SOLO quelli realmente presenti nel dataset preparato
CANDIDATE_FEATURES = [
    "len_text","word_count","hashtags_count","mentions_count",
    "is_rt","verified","hour_sin","hour_cos",
    "log_followers","log_friends","log_statuses",
    # se nel prep hai altre numeriche, lasciale qui: verranno usate solo se presenti
]

# ─────────────────────────────────────────────────────────
# Metriche: built-in evaluator + confusion matrix
# ─────────────────────────────────────────────────────────
def _compute_metrics(pred_df: DataFrame) -> Tuple[float, float, float, List[Tuple[int,int,int]]]:
    ev = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    precision = float(ev.setMetricName("weightedPrecision").evaluate(pred_df))
    recall    = float(ev.setMetricName("weightedRecall").evaluate(pred_df))
    f1        = float(ev.setMetricName("f1").evaluate(pred_df))
    cm_rows = (pred_df.groupBy("label","prediction").count().orderBy("label","prediction").collect())
    cm = [(int(r["label"]), int(r["prediction"]), int(r["count"])) for r in cm_rows]
    return precision, recall, f1, cm

# ─────────────────────────────────────────────────────────
# Train / Load
# ─────────────────────────────────────────────────────────
def train_classifier(
    spark: SparkSession,
    prepared_dir: str,
    model_root: str,
    files_limit: Optional[int] = None,
    viral_quantile: float = 0.70,
    viral_threshold: Optional[int] = None,
    n_trees: int = 100,
    # compatibilità con main/route: accettiamo ma NON usiamo questi parametri
    max_bins: int = 128,
    top_k_lang: int = 30,
    top_k_device: int = 30,
    debug_logs: bool = True,
) -> Dict:
    files = _list_prep_files(prepared_dir, limit=files_limit)
    if not files:
        raise RuntimeError(f"Nessun file preparato trovato in {prepared_dir} (files_limit={files_limit})")

    df = spark.read.json(files)
    df = add_label(df, viral_quantile=viral_quantile, viral_threshold=viral_threshold)

    # Usa SOLO le colonne che esistono davvero
    present = set(df.schema.names)
    used_feats = [c for c in CANDIDATE_FEATURES if c in present]
    if not used_feats:
        raise RuntimeError(
            "Nessuna feature numerica trovata nel dataset preparato. "
            f"Attese (almeno una): {CANDIDATE_FEATURES}"
        )
    if debug_logs:
        missing = [c for c in CANDIDATE_FEATURES if c not in present]
        print("[DEBUG] available columns:", df.schema.names)
        print("[DEBUG] used_feats:", used_feats)
        if missing:
            print("[DEBUG] missing_feats (ignorate):", missing)

    # ⚠️ fill SOLO in subset (evita UNRESOLVED_COLUMN)
    df = df.na.fill(0, subset=used_feats)

    # split
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    # class imbalance → pesi
    dist_tr = train.groupBy("label").count().collect()
    cnt_tr = {int(r["label"]): int(r["count"]) for r in dist_tr}
    pos_tr, neg_tr = cnt_tr.get(1, 1), cnt_tr.get(0, 1)
    tot_tr = float(pos_tr + neg_tr)
    w_pos = float(neg_tr) / max(1.0, tot_tr)  # più peso alla minoritaria
    w_neg = float(pos_tr) / max(1.0, tot_tr)
    class_weights = {"pos": w_pos, "neg": w_neg}

    train = train.withColumn(
        "cls_weight",
        F.when(F.col("label") == 1, F.lit(w_pos)).otherwise(F.lit(w_neg))
    )

    assembler = VectorAssembler(inputCols=used_feats, outputCol="features")
    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        weightCol="cls_weight",
        numTrees=int(n_trees),
        seed=42,
    )

    pipe = Pipeline(stages=[assembler, rf])
    model = pipe.fit(train)

    # valutazione
    pred_test = model.transform(test)
    precision, recall, f1, cm = _compute_metrics(pred_test)

    if debug_logs:
        print(f"[DEBUG] class_weights: {class_weights}")
        print(f"[DEBUG] Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")
        print("[DEBUG] CM:", cm)

    # salvataggi
    ts = _now_tag()
    run_dir = f"{model_root.rstrip('/')}/rf_{ts}"
    model.write().overwrite().save(os.path.join(run_dir, "model"))

    metrics = {
        "viral_quantile": float(viral_quantile),
        "row_used": int(df.count()),
        "file_used": len(files),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "n_trees": int(n_trees),
        "class_weights": class_weights,
    }
    _ensure_dir(model_root)
    _write_small_json(metrics, os.path.join(run_dir, "metrics.json"))
    with open(os.path.join(model_root, "latest_path.txt"), "w", encoding="utf-8") as f:
        f.write(run_dir + "\n")

    return metrics

def load_latest_model(model_root: str) -> Optional[PipelineModel]:
    run_dir = _find_latest_run(model_root)
    if not run_dir:
        return None
    try:
        return PipelineModel.load(os.path.join(run_dir, "model"))
    except Exception:
        return None

def _load_run_metrics(model_root: str, run_dir: Optional[str]) -> Optional[dict]:
    if not run_dir:
        return None
    p = os.path.join(run_dir, "metrics.json")
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

# ─────────────────────────────────────────────────────────
# API (identica: nessuna modifica al main)
# ─────────────────────────────────────────────────────────
def add_model_routes(
    app: FastAPI,
    spark: SparkSession,
    prepared_dir: str,
    model_root: str,
    default_files_limit: Optional[int] = None,
):
    state: Dict[str, Optional[str]] = {"last_run_dir": _find_latest_run(model_root)}

    def _effective_limit(param_limit: Optional[int]) -> Optional[int]:
        return param_limit if param_limit is not None else default_files_limit

    # ---- TRAIN
    @app.post("/model/train")
    def model_train(
        files_limit: Optional[int] = Query(None, ge=1, le=100000),
        viral_quantile: float = Query(0.70, ge=0.5, le=0.999),
        viral_threshold: Optional[int] = Query(None, ge=1),
        n_trees: int = Query(100, ge=10, le=500),
        max_bins: int = Query(128, ge=32, le=2048),     # compat (ignorati)
        top_k_lang: int = Query(30, ge=5, le=200),      # compat (ignorati)
        top_k_device: int = Query(30, ge=5, le=200),    # compat (ignorati)
        debug_logs: bool = Query(True),
    ):
        eff_limit = _effective_limit(files_limit)
        metrics = train_classifier(
            spark=spark,
            prepared_dir=prepared_dir,
            model_root=model_root,
            files_limit=eff_limit,
            viral_quantile=viral_quantile,
            viral_threshold=viral_threshold,
            n_trees=n_trees,
            max_bins=max_bins,
            top_k_lang=top_k_lang,
            top_k_device=top_k_device,
            debug_logs=bool(debug_logs),
        )
        state["last_run_dir"] = _find_latest_run(model_root)
        return JSONResponse({"status": "ok", "metrics": metrics})

    # ---- LATEST
    @app.get("/model/latest")
    def model_latest():
        run_dir = state["last_run_dir"] or _find_latest_run(model_root)
        if not run_dir:
            return JSONResponse({"status": "empty", "message": "Nessun modello trovato"}, status_code=404)

        out = {"run_dir": run_dir}
        metrics_path = os.path.join(run_dir, "metrics.json")
        if os.path.isfile(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    out["metrics"] = json.load(f)
            except Exception:
                pass
        return JSONResponse(out)

    # ---- PREDICT
    @app.get("/model/predict")
    def model_predict(
        text: str = Query(..., min_length=1),
        hashtags: int = Query(0, ge=0),
        mentions: int = Query(0, ge=0),
        is_rt: int = Query(0, ge=0, le=1),
        verified: int = Query(0, ge=0, le=1),
        hour: int = Query(12, ge=0, le=23),
        followers: int = Query(0, ge=0),
        friends: int = Query(0, ge=0),
        statuses: int = Query(0, ge=0),
        # compat: accettiamo ma NON usiamo direttamente
        lang: str = Query("unknown"),
        device: str = Query("Unknown"),
    ):
        run_dir = state["last_run_dir"] or _find_latest_run(model_root)
        if not run_dir:
            return JSONResponse({"status": "empty", "message": "Nessun modello trovato"}, status_code=404)

        pipe = load_latest_model(model_root)
        if pipe is None:
            return JSONResponse({"status": "error", "message": "Impossibile caricare il modello"}, status_code=500)

        # prova a recuperare l'assembler per sapere quali feature sono state usate
        try:
            stages = pipe.stages
            assembler = next(s for s in stages if isinstance(s, VectorAssembler))
            used_feats = list(assembler.getInputCols())
        except Exception:
            # fallback (non dovrebbe servire): usa i candidati
            used_feats = [c for c in CANDIDATE_FEATURES]

        # valori derivabili dagli input della route
        hour_sin = math.sin(2.0 * math.pi * (hour / 24.0))
        hour_cos = math.cos(2.0 * math.pi * (hour / 24.0))
        values = {
            "len_text": len(text),
            "word_count": len(text.split()),
            "hashtags_count": int(hashtags),
            "mentions_count": int(mentions),
            "is_rt": int(is_rt),
            "verified": int(verified),
            "hour_sin": float(hour_sin),
            "hour_cos": float(hour_cos),
            "log_followers": math.log1p(max(0, followers)),
            "log_friends": math.log1p(max(0, friends)),
            "log_statuses": math.log1p(max(0, statuses)),
        }
        # costruisci SOLO le colonne usate in training (manca? → 0)
        row = [tuple(values.get(c, 0) for c in used_feats)]
        df_in = spark.createDataFrame(row, used_feats)

        res = pipe.transform(df_in).select("probability","prediction").first()
        return JSONResponse({
            "prob_viral": float(res["probability"][1]),
            "prediction": int(res["prediction"])
        })
