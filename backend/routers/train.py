from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from database import get_db
from models import SiteData, TrainedModel
from schemas import TrainRequest, PredictRequest

# Optional imports for models (detect per-library)
HAS_SKLEARN = False
HAS_XGBOOST = False
HAS_OPTUNA = False
HAS_TORCH = False

try:
    from sklearn.model_selection import train_test_split, ParameterGrid
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    from sklearn.svm import SVR as SKSVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    import optuna  # type: ignore
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

try:  # Prefer PyTorch for LSTM
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

router = APIRouter(prefix="/train", tags=["Train"])


def _uploads_base_dir() -> Path:
    base = Path(__file__).resolve().parent.parent / "uploads"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _cleaned_dir() -> Path:
    p = _uploads_base_dir() / "cleaned"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _models_dir(data_id: int) -> Path:
    p = _uploads_base_dir() / "models" / str(data_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _list_artifacts(data_id: int):
    d = _models_dir(data_id)
    items = []
    for p in sorted(d.glob("*")):
        if p.suffix in {".joblib", ".json"}:
            meta = p.with_suffix(".meta.json")
            items.append({
                "artifact": p.name,
                "meta": meta.name if meta.exists() else None,
                "size": p.stat().st_size,
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
            })
    return items


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


@router.get("/models")
def list_models(data_id: int):
    return {"data_id": data_id, "artifacts": _list_artifacts(data_id)}


def _train_single_model(model_id: str, X_train, y_train, X_test, y_test, strategy: str, param_spec: Dict[str, Any]):
    # Dependency checks per model
    if model_id in ("SVR", "RandomForest") and not HAS_SKLEARN:
        raise HTTPException(status_code=500, detail="scikit-learn not available on server")
    if model_id == "XGBoost" and not HAS_XGBOOST:
        raise HTTPException(status_code=500, detail="xgboost not available on server")
    if model_id == "LSTM" and not HAS_TORCH:
        raise HTTPException(status_code=500, detail="PyTorch not available on server for LSTM")

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
        def _make_range(start, end, step):
            # choose int or float grid based on inputs
            def _is_intlike(x):
                try:
                    return float(x).is_integer()
                except Exception:
                    return isinstance(x, int)
            is_int_grid = _is_intlike(start) and _is_intlike(end) and _is_intlike(step)
            arr = np.arange(float(start), float(end) + (float(step) or 1.0), float(step))
            if is_int_grid:
                return [int(round(x)) for x in arr]
            return [float(np.round(x, 6)) for x in arr]

        grid = {}
        for k, v in (spec or {}).items():
            # ignore private control keys (e.g., _max_combinations)
            if isinstance(k, str) and k.startswith('_'):
                continue
            if isinstance(v, dict) and {"start", "end", "step"}.issubset(v.keys()):
                start, end, step = v.get("start"), v.get("end"), v.get("step", 1)
                grid[k] = _make_range(start, end, step)
            elif isinstance(v, dict) and {"values"}.issubset(v.keys()):
                grid[k] = list(v["values"])  # explicit list
            else:
                grid[k] = [v]
        return list(ParameterGrid(grid)) if grid else [{}]

    # Bayesian Optimization (real, Optuna-based) for non-LSTM models
    if strategy == "bayes":
        if not HAS_OPTUNA:
            raise HTTPException(status_code=500, detail="bayesian optimization requires optuna on server")

        def suggest_from_spec(trial, spec: Dict[str, Any]):
            params = {}
            for k, v in (spec or {}).items():
                if isinstance(v, dict) and {"values"}.issubset(v.keys()):
                    params[k] = trial.suggest_categorical(k, list(v["values"]))
                elif isinstance(v, dict) and {"start", "end"}.issubset(v.keys()):
                    start = float(v["start"])
                    end = float(v["end"])
                    step = float(v.get("step", 0))
                    is_int = float(start).is_integer() and float(end).is_integer() and float(step or 0).is_integer()
                    if is_int:
                        if step and step > 0:
                            params[k] = trial.suggest_int(k, int(start), int(end), step=int(step))
                        else:
                            params[k] = trial.suggest_int(k, int(start), int(end))
                    else:
                        if step and step > 0:
                            params[k] = trial.suggest_float(k, start, end, step=step)
                        else:
                            params[k] = trial.suggest_float(k, start, end)
                else:
                    params[k] = v
            return params

        def build_model(mid: str, p: Dict[str, Any]):
            if mid == "SVR":
                C = float(p.get("C", 1.0))
                kernel = p.get("kernel", "rbf")
                gamma = p.get("gamma", "scale")
                return Pipeline([
                    ("scaler", StandardScaler()),
                    ("svr", SKSVR(C=C, kernel=kernel, gamma=gamma)),
                ])
            if mid == "RandomForest":
                n_estimators = int(p.get("n_estimators", 200))
                max_depth = p.get("max_depth", None)
                return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
            if mid == "XGBoost":
                xgb_kwargs = {
                    'n_estimators': int(p.get('n_estimators', 300)),
                    'learning_rate': float(p.get('learning_rate', 0.1)),
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'n_jobs': -1,
                    'tree_method': 'hist',
                }
                md = p.get('max_depth', 6)
                if md is not None and md != '' and md != 'None':
                    xgb_kwargs['max_depth'] = int(md)
                return xgb.XGBRegressor(**xgb_kwargs)
            raise ValueError("unsupported model for bayes")

        def objective(trial):
            p = suggest_from_spec(trial, param_spec or {})
            model = build_model(model_id, p)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # minimize WMAPE
            eps = 1e-6
            wmape = float(np.sum(np.abs(y_test - y_pred)) / (np.sum(np.abs(y_test)) + eps))
            return wmape

        trials = int((param_spec or {}).get('_trials', 30))
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=trials)
        best_p = study.best_params

        # evaluate metrics for best params
        model = build_model(model_id, best_p)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = float(r2_score(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        eps = 1e-6
        wmape = float(np.sum(np.abs(y_test - y_pred)) / (np.sum(np.abs(y_test)) + eps))
        return {"best": {"params": best_p, "r2": r2, "rmse": rmse, "mae": mae, "wmape": wmape}, "trials": []}

    candidates = [{}]
    if strategy == "manual":
        candidates = [param_spec or {}]
    elif strategy == "grid":
        candidates = build_grid(param_spec or {})
        # XGBoost-only: cap number of combinations if requested
        if model_id == "XGBoost":
            try:
                max_n = int((param_spec or {}).get('_max_combinations', 0) or 0)
            except Exception:
                max_n = 0
            if max_n > 0 and len(candidates) > max_n:
                # pick evenly spaced indices to cover the space
                idx = np.linspace(0, len(candidates) - 1, num=max_n)
                idx = [int(round(i)) for i in idx]
                # ensure unique and sorted
                seen = set()
                selected = []
                for i in idx:
                    if i not in seen:
                        selected.append(i)
                        seen.add(i)
                candidates = [candidates[i] for i in selected]

    for params in candidates:
        if model_id == "SVR":
            C = float(params.get("C", 1.0))
            kernel = params.get("kernel", "rbf")
            gamma = params.get("gamma", "scale")
            # use scaling during training to be consistent with persistence
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("svr", SKSVR(C=C, kernel=kernel, gamma=gamma)),
            ])
        elif model_id == "RandomForest":
            n_estimators = int(params.get("n_estimators", 200))
            max_depth = params.get("max_depth", None)
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        elif model_id == "XGBoost":
            xgb_kwargs = {
                'n_estimators': int(params.get('n_estimators', 300)),
                'learning_rate': float(params.get('learning_rate', 0.1)),
                'subsample': float(params.get('subsample', 0.8)),
                'colsample_bytree': float(params.get('colsample_bytree', 0.8)),
                'random_state': 42,
                'n_jobs': -1,
                'tree_method': 'hist',
            }
            md = params.get('max_depth', 6)
            if md is not None and md != '' and md != 'None':
                xgb_kwargs['max_depth'] = int(md)
            # optional regularization / constraints
            if 'min_child_weight' in params:
                xgb_kwargs['min_child_weight'] = float(params.get('min_child_weight'))
            if 'reg_lambda' in params:
                xgb_kwargs['reg_lambda'] = float(params.get('reg_lambda'))
            if 'reg_alpha' in params:
                xgb_kwargs['reg_alpha'] = float(params.get('reg_alpha'))
            model = xgb.XGBRegressor(**xgb_kwargs)
        elif model_id == "LSTM":
            # True LSTM implementation using PyTorch
            lookback = int(params.get("lookback", 24))
            hidden_size = int(params.get("hidden_size", 64))
            num_layers = int(params.get("num_layers", 1))
            dropout = float(params.get("dropout", 0.0))
            lr = float(params.get("lr", 1e-3))
            epochs = int(params.get("epochs", 20))
            batch_size = int(params.get("batch_size", 64))

            # build sequences from time-ordered X_train/X_test
            def make_seq(X, y, win):
                if len(X) <= win:
                    raise HTTPException(status_code=400, detail=f"LSTM lookback {win} too large for dataset")
                Xs, ys = [], []
                for i in range(win, len(X)):
                    Xs.append(X[i-win:i])
                    ys.append(y[i])
                return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

            # Standardize by train stats
            m = X_train.mean(axis=0)
            s = X_train.std(axis=0)
            s[s == 0] = 1.0
            Xtr = ((X_train - m) / s).astype(np.float32)
            Xte = ((X_test - m) / s).astype(np.float32)
            Xtr_seq, ytr_seq = make_seq(Xtr, y_train, lookback)
            Xte_seq, yte_seq = make_seq(Xte, y_test, lookback)

            class LSTMRegressor(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, dropout):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
                    self.fc = nn.Linear(hidden_size, 1)
                def forward(self, x):
                    out, _ = self.lstm(x)
                    out = out[:, -1, :]
                    out = self.fc(out)
                    return out.squeeze(-1)

            device = torch.device("cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu")
            model = LSTMRegressor(input_size=Xtr_seq.shape[-1], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            ds = TensorDataset(torch.from_numpy(Xtr_seq), torch.from_numpy(ytr_seq))
            dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
            model.train()
            for _ in range(epochs):
                for xb, yb in dl:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    opt.zero_grad()
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    opt.step()

            # Evaluate
            model.eval()
            with torch.no_grad():
                y_pred = model(torch.from_numpy(Xte_seq).to(device)).cpu().numpy()
            r2 = float(r2_score(yte_seq, y_pred))
            rmse = float(np.sqrt(mean_squared_error(yte_seq, y_pred)))
            mae = float(mean_absolute_error(yte_seq, y_pred))
            eps = 1e-6
            wmape = float(np.sum(np.abs(yte_seq - y_pred)) / (np.sum(np.abs(yte_seq)) + eps))
            metrics = {"r2": r2, "rmse": rmse, "mae": mae, "wmape": wmape}
            tried.append({"params": params, **metrics})
            if (best is None) or (metrics["wmape"] < best["wmape"]):
                best = {"params": params, **metrics}
            # continue to next candidate (skip generic sklearn evaluate)
            continue
        else:
            continue

        metrics = evaluate(model)
        tried.append({"params": params, **metrics})
        if (best is None) or (metrics["wmape"] < best["wmape"]):
            best = {"params": params, **metrics}

    return {"best": best, "trials": tried}


def _ensure_time_features(df: pd.DataFrame, time_col: Optional[str]) -> pd.DataFrame:
    # derive basic time features if a time column is available
    if time_col and time_col in df.columns:
        s = pd.to_datetime(df[time_col], errors="coerce")
        df = df.copy()
        df['hour'] = s.dt.hour
        df['dayofweek'] = s.dt.dayofweek
        df['month'] = s.dt.month
        # cyclical hour
        df['hour_sin'] = np.sin(2 * np.pi * (df['hour'].fillna(0) / 24))
        df['hour_cos'] = np.cos(2 * np.pi * (df['hour'].fillna(0) / 24))
    return df


def _validate_clean_data(df: pd.DataFrame, gi_col: str, tm_col: str, target_col: str):
    missing_cols = [c for c in [gi_col, tm_col, target_col] if c not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"missing required column(s): {','.join(missing_cols)}")

    # Ensure convertible to float and no NaNs (since data is claimed cleaned)
    for c in [gi_col, tm_col, target_col]:
        try:
            df[c].astype(float)
        except Exception:
            raise HTTPException(status_code=400, detail=f"column '{c}' cannot be parsed as float")
        if pd.isna(df[c]).any():
            raise HTTPException(status_code=400, detail=f"column '{c}' has NaN but data should be cleaned")


@router.post("/run")
def run_training(payload: TrainRequest, db: Session = Depends(get_db)):
    entry = db.query(SiteData).filter(SiteData.data_id == payload.data_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="data_id not found")

    df = _load_cleaned_or_json(entry)
    target_col = payload.target or 'EAC'
    if target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"target column '{payload.target}' not found")

    # enforce default features GI + TM
    features: List[str] = payload.features or ['GI','TM']
    missing = [c for c in ['GI','TM'] if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"required feature(s) missing: {','.join(missing)}")

    # Lightweight validation only (no interpolation/cleaning here)
    _validate_clean_data(df, gi_col='GI', tm_col='TM', target_col=target_col)

    # time features & strict time sorting on df level
    time_col = payload.time_col
    if time_col is None:
        # try to auto detect a time column used earlier in visualize
        for cand in ['timestamp','datetime','time','recordtime','thedate','Date','Time']:
            if cand in df.columns:
                time_col = cand
                break
    if time_col and time_col in df.columns:
        # parse, drop NaT, sort, then derive time features
        parsed = pd.to_datetime(df[time_col], errors='coerce')
        df = df.loc[parsed.notna()].copy()
        df[time_col] = parsed[parsed.notna()]
        df = df.sort_values(time_col).reset_index(drop=True)
        df = _ensure_time_features(df, time_col)

    # build design matrix
    extra_time_feats = [c for c in ['hour','dayofweek','month','hour_sin','hour_cos'] if c in df.columns]
    feature_cols = features + [c for c in extra_time_feats if c not in features]
    num_df = df[feature_cols].select_dtypes(include=[np.number])
    if num_df.isna().any().any():
        raise HTTPException(status_code=400, detail="numeric features contain NaN; cleaned data should not have missing values in features")
    if num_df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="no numeric features after preprocessing; ensure GI/TM exist and are numeric")
    X = num_df.values
    y = df[target_col].astype(float).values

    if len(y) < 10:
        raise HTTPException(status_code=400, detail="not enough rows to train")

    # time-based split or random split (df already sorted if time_col given)

    if payload.split_method == 'time' and len(y) > 10:
        test_size = max(0.05, min(0.5, 1.0 - float(payload.split_ratio)))
        n_test = max(1, int(len(y) * test_size))
        X_train, X_test = X[:-n_test], X[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]
    else:
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

    # Optional artifact saving (do not require sklearn globally, as XGBoost can save without it)
    if payload.save_model:
        import json as _json
        from datetime import datetime as _dt
        import joblib as _joblib
        saved = []
        save_errors = []
        timestamp = _dt.utcnow().strftime('%Y%m%d_%H%M%S%f')
        models_dir = _models_dir(payload.data_id)
        for mid, res in results.items():
            if res.get('status') != 'ok':
                continue
            try:
                if mid == 'XGBoost' and HAS_XGBOOST:
                    # retrain best param briefly to persist (handle optional/None params robustly)
                    best_params = res.get('best_params', {}) or {}
                    xgb_kwargs = {
                        'n_estimators': int(best_params.get('n_estimators', 300)),
                        'learning_rate': float(best_params.get('learning_rate', 0.1)),
                        'subsample': float(best_params.get('subsample', 0.8)),
                        'colsample_bytree': float(best_params.get('colsample_bytree', 0.8)),
                        'random_state': 42,
                        'n_jobs': -1,
                        'tree_method': 'hist',
                    }
                    md = best_params.get('max_depth', 6)
                    if md is not None and md != '' and md != 'None':
                        xgb_kwargs['max_depth'] = int(md)
                    if 'min_child_weight' in best_params:
                        xgb_kwargs['min_child_weight'] = float(best_params.get('min_child_weight'))
                    if 'reg_lambda' in best_params:
                        xgb_kwargs['reg_lambda'] = float(best_params.get('reg_lambda'))
                    if 'reg_alpha' in best_params:
                        xgb_kwargs['reg_alpha'] = float(best_params.get('reg_alpha'))
                    model = xgb.XGBRegressor(**xgb_kwargs)
                    model.fit(X_train, y_train)
                    # Prefer native JSON; fall back to joblib if JSON save fails
                    try:
                        path = models_dir / f"{timestamp}_{mid}.json"
                        model.save_model(str(path))
                    except Exception:
                        # Fallback to pickle-based artifact
                        path = models_dir / f"{timestamp}_{mid}.joblib"
                        _joblib.dump(model, path)
                else:
                    # retrain a simple model using best params (RF/SVR)
                    if mid == 'RandomForest':
                        n_estimators = int(res.get('best_params', {}).get('n_estimators', 200))
                        max_depth = res.get('best_params', {}).get('max_depth', None)
                        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth if max_depth is not None else None, random_state=42, n_jobs=-1)
                    elif mid == 'SVR':
                        C = float(res.get('best_params', {}).get('C', 1.0))
                        model = Pipeline([('scaler', StandardScaler()), ('svr', SKSVR(C=C, kernel='rbf', gamma='scale'))])
                    else:
                        continue
                    model.fit(X_train, y_train)
                    path = models_dir / f"{timestamp}_{mid}.joblib"
                    _joblib.dump(model, path)
                # write sidecar metadata to lock feature order & config
                meta_path = models_dir / f"{timestamp}_{mid}.meta.json"
                sidecar = {
                    'model_id': mid,
                    'artifact': str(path.name),
                    'feature_cols_used': feature_cols,
                    'target': target_col,
                    'time_col': time_col,
                    'split_method': payload.split_method,
                    'trained_at': timestamp,
                }
                with open(meta_path, 'w', encoding='utf-8') as fh:
                    _json.dump(sidecar, fh, ensure_ascii=False, indent=2)
                saved.append({'model_id': mid, 'artifact': str(path.name), 'meta': meta_path.name})
            except Exception as e:
                # collect error to surface in response
                try:
                    save_errors.append(f"save_failed:{mid}:{str(e)}")
                except Exception:
                    save_errors.append(f"save_failed:{mid}")
                continue
        # persist simple registry into processed_meta
        meta = entry.processed_meta or {}
        history = meta.get('trained_models', [])
        history.append({
            'trained_at': timestamp,
            'data_id': payload.data_id,
            'target': target_col,
            'features': feature_cols,
            'split_method': payload.split_method,
            'artifacts': saved,
            'cleaned_file': cleaned_path,
        })
        meta['trained_models'] = history
        entry.processed_meta = meta
        db.add(entry)
        # also insert trained models into a dedicated table for querying
        for art in saved:
            try:
                tm = TrainedModel(
                    site_id=entry.site_id,
                    data_id=payload.data_id,
                    model_type=art.get('model_id'),
                    parameters=results.get(art.get('model_id'), {}).get('best_params', {}),
                    file_path=str((_models_dir(payload.data_id) / art.get('artifact')).relative_to(_uploads_base_dir())),
                    trained_at=datetime.utcnow(),
                )
                db.add(tm)
            except Exception:
                continue
        db.commit()

    warnings = []
    if payload.strategy == 'bayes':
        warnings.append("strategy=bayes behaves like grid (placeholder)")
    if any(m == 'LSTM' for m in payload.models):
        warnings.append("LSTM is not implemented; consider removing or implementing sequence model")
    # surface any artifact save errors
    if 'save_errors' in locals() and save_errors:
        warnings.extend(save_errors)

    return _to_native({
        "data_id": payload.data_id,
        "cleaned_file": cleaned_path,
        "split": {
            "train": float(payload.split_ratio),
            "test": 1.0 - float(payload.split_ratio),
        },
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "feature_cols_used": feature_cols,
        "results": results,
        "warnings": warnings,
    })


@router.post("/predict")
def predict(payload: PredictRequest, db: Session = Depends(get_db)):
    entry = db.query(SiteData).filter(SiteData.data_id == payload.data_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="data_id not found")

    # locate artifact
    models_dir = _models_dir(payload.data_id)
    artifact_path: Optional[Path] = None
    meta_path: Optional[Path] = None
    if payload.artifact:
        artifact_path = models_dir / payload.artifact
        if not artifact_path.exists():
            raise HTTPException(status_code=404, detail="artifact not found")
        if artifact_path.suffix in {".joblib", ".json"}:
            meta_path = artifact_path.with_suffix(".meta.json")
    else:
        # find by model_id + trained_at
        if not (payload.model_id and payload.trained_at):
            raise HTTPException(status_code=400, detail="provide artifact filename or (model_id + trained_at)")
        candidates = list(models_dir.glob(f"{payload.trained_at}_{payload.model_id}.*"))
        if not candidates:
            raise HTTPException(status_code=404, detail="artifact by model_id+trained_at not found")
        artifact_path = candidates[0]
        meta_path = artifact_path.with_suffix(".meta.json")

    # load metadata
    if not meta_path or not meta_path.exists():
        raise HTTPException(status_code=400, detail="missing meta sidecar for artifact")
    import json as _json
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = _json.load(fh)
    feature_cols = meta.get("feature_cols_used") or []
    target_col = meta.get("target") or "EAC"

    # build dataframe
    if payload.rows:
        df = pd.DataFrame(payload.rows)
    else:
        df = _load_cleaned_or_json(entry)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"missing feature columns: {','.join(missing)}")
    X = df[feature_cols].select_dtypes(include=[np.number]).values

    # dispatch by artifact type
    if artifact_path.suffix == ".json":
        if not HAS_XGBOOST:
            raise HTTPException(status_code=500, detail="xgboost not available to load json model")
        booster = xgb.XGBRegressor()
        booster.load_model(str(artifact_path))
        y_pred = booster.predict(X)
    else:
        # joblib models (RF/SVR or XGB fallback)
        if not HAS_SKLEARN and artifact_path.suffix == ".joblib":
            # Might still be XGB via joblib, try loading anyway
            pass
        import joblib as _joblib
        model = _joblib.load(artifact_path)
        # If it's an SVR pipeline, it handles scaling internally
        y_pred = model.predict(X)

    return _to_native({
        "artifact": artifact_path.name,
        "n": int(len(y_pred)),
        "target": target_col,
        "pred": y_pred.tolist(),
    })
