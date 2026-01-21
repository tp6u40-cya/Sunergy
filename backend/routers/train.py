from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from database import get_db
from models import SiteData
from schemas import TrainRequest

# Optional imports for models (detect per-library)
HAS_SKLEARN = False
HAS_XGBOOST = False

try:
    from sklearn.model_selection import train_test_split, ParameterGrid
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.svm import SVR as SKSVR
    from sklearn.ensemble import RandomForestRegressor
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

router = APIRouter(prefix="/train", tags=["Train"])


def _uploads_base_dir() -> Path:
    base = Path(__file__).resolve().parent.parent / "uploads"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _cleaned_dir() -> Path:
    p = _uploads_base_dir() / "cleaned"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_cleaned_or_json(entry: SiteData) -> pd.DataFrame:
    # prefer cleaned csv if exists
    rel = (entry.processed_meta or {}).get("cleaned_path") if entry.processed_meta else None
    if rel:
        p = _uploads_base_dir() / rel
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    # fallback to processed_json, else json_data
    data = entry.processed_json or entry.json_data
    if not data:
        raise HTTPException(status_code=400, detail="no data available for training")
    return pd.DataFrame(data)


def _to_native(obj):
    try:
        import numpy as _np  # local import to avoid hard dep in type hints
    except Exception:  # pragma: no cover
        _np = None
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    # numpy scalar -> built-in
    if _np is not None and isinstance(obj, (_np.integer,)):
        return int(obj)
    if _np is not None and isinstance(obj, (_np.floating,)):
        return float(obj)
    return obj


@router.get("/info")
def train_info(data_id: int, db: Session = Depends(get_db)):
    entry = db.query(SiteData).filter(SiteData.data_id == data_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="data_id not found")
    cleaned_path = (entry.processed_meta or {}).get("cleaned_path") if entry.processed_meta else None
    return {
        "data_id": data_id,
        "cleaned_file": cleaned_path,
        "has_processed_json": bool(entry.processed_json),
        "rows": len(entry.processed_json or entry.json_data or []),
    }


def _train_single_model(model_id: str, X_train, y_train, X_test, y_test, strategy: str, param_spec: Dict[str, Any]):
    # Dependency checks per model
    if model_id in ("SVR", "RandomForest", "LSTM") and not HAS_SKLEARN:
        raise HTTPException(status_code=500, detail="scikit-learn not available on server")
    if model_id == "XGBoost" and not HAS_XGBOOST:
        raise HTTPException(status_code=500, detail="xgboost not available on server")

    best = None
    tried = []

    def evaluate(m):
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        # Weighted MAPE (avoid div by zero): weight by actual
        eps = 1e-6
        wmape = float(np.sum(np.abs(y_test - y_pred)) / (np.sum(np.abs(y_test) + eps)))
        return {"r2": r2, "rmse": rmse, "mae": mae, "wmape": wmape}

    # build candidate param grid
    def build_grid(spec: Dict[str, Any]):
        grid = {}
        for k, v in (spec or {}).items():
            if isinstance(v, dict) and {"start", "end", "step"}.issubset(v.keys()):
                start, end, step = v["start"], v["end"], max(v.get("step", 1), 1)
                grid[k] = list(np.arange(start, end + (step or 1), step).astype(int))
            elif isinstance(v, dict) and {"values"}.issubset(v.keys()):
                grid[k] = list(v["values"])  # explicit list
            else:
                grid[k] = [v]
        return list(ParameterGrid(grid)) if grid else [{}]

    candidates = [{}]
    if strategy == "manual":
        candidates = [param_spec or {}]
    elif strategy == "grid":
        candidates = build_grid(param_spec or {})
    elif strategy == "bayes":
        # Simple surrogate for now: same as grid (placeholder)
        candidates = build_grid(param_spec or {})

    for params in candidates:
        if model_id == "SVR":
            C = float(params.get("C", 1.0))
            kernel = params.get("kernel", "rbf")
            gamma = params.get("gamma", "scale")
            model = SKSVR(C=C, kernel=kernel, gamma=gamma)
        elif model_id == "RandomForest":
            n_estimators = int(params.get("n_estimators", 200))
            max_depth = params.get("max_depth", None)
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        elif model_id == "XGBoost":
            n_estimators = int(params.get("n_estimators", 300))
            learning_rate = float(params.get("learning_rate", 0.1))
            max_depth = int(params.get("max_depth", 6))
            model = xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, tree_method="hist")
        elif model_id == "LSTM":
            # Placeholder: map to RandomForest for now; real LSTM would require PyTorch/TF
            n_estimators = int(params.get("n_estimators", 200))
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        else:
            continue

        metrics = evaluate(model)
        tried.append({"params": params, **metrics})
        if (best is None) or (metrics["wmape"] < best["wmape"]):
            best = {"params": params, **metrics}

    return {"best": best, "trials": tried}


@router.post("/run")
def run_training(payload: TrainRequest, db: Session = Depends(get_db)):
    entry = db.query(SiteData).filter(SiteData.data_id == payload.data_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="data_id not found")

    df = _load_cleaned_or_json(entry)
    if payload.target not in df.columns:
        raise HTTPException(status_code=400, detail=f"target column '{payload.target}' not found")

    # basic feature selection
    feature_cols = [c for c in df.columns if c != payload.target]
    num_df = df[feature_cols].select_dtypes(include=[np.number]).fillna(0.0)
    if num_df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="no numeric features found for training; ensure cleaned data has numeric columns besides target")
    X = num_df.values
    y = df[payload.target].astype(float).values

    if len(y) < 10:
        raise HTTPException(status_code=400, detail="not enough rows to train")

    test_size = max(0.05, min(0.5, 1.0 - float(payload.split_ratio)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    results: Dict[str, Any] = {}
    for m in payload.models:
        spec = (payload.params or {}).get(m, {})
        try:
            res = _train_single_model(m, X_train, y_train, X_test, y_test, payload.strategy, spec)
            best = res.get("best") or {}
            results[m] = {
                "id": m,
                "status": "ok" if best else "failed",
                "r2": round(best.get("r2", 0.0), 3),
                "rmse": round(best.get("rmse", 0.0), 3),
                "mae": round(best.get("mae", 0.0), 3),
                "wmape": round(best.get("wmape", 0.0), 4),
                "best_params": best.get("params", {}),
            }
        except HTTPException as e:
            results[m] = {
                "id": m,
                "status": "error",
                "error": e.detail if hasattr(e, 'detail') else str(e),
            }
        except Exception as e:
            results[m] = {"id": m, "status": "error", "error": str(e)}

    # attach metadata for which file used
    cleaned_path = (entry.processed_meta or {}).get("cleaned_path") if entry.processed_meta else None
    return _to_native({
        "data_id": payload.data_id,
        "cleaned_file": cleaned_path,
        "split": {
            "train": float(payload.split_ratio),
            "test": 1.0 - float(payload.split_ratio),
        },
        "results": results,
    })
