# backend/routers/visualize.py
from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from processors.dataprocessor import DataProcessor
from database import get_db
from models import SiteData

router = APIRouter(prefix="/visualize", tags=["Visualize"]) 


def safe_json(obj):
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json(v) for v in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.integer, np.floating)):
        v = obj.item()
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return v
    return obj


# Local object storage helpers (backend/uploads)

def _uploads_base_dir() -> Path:
    base = Path(__file__).resolve().parent.parent / "uploads"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _cleaned_dir() -> Path:
    p = _uploads_base_dir() / "cleaned"
    p.mkdir(parents=True, exist_ok=True)
    return p


@router.get("/site-data")
def visualize_site_data(
    data_id: int = Query(...),
    remove_outliers: bool = Query(False),
    outlier_method: str = Query("iqr", description="none, iqr, zscore, isolation_forest, custom"),
    iqr_factor: float = Query(1.5, description="IQR factor"),
    zscore_threshold: float = Query(3.0, description="Z-Score threshold"),
    iso_contamination: float = Query(0.05, description="Isolation Forest contamination", ge=0.01, le=0.5),
    db: Session = Depends(get_db),
):
    # 1) Load
    entry = db.query(SiteData).filter(SiteData.data_id == data_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="data_id not found")
    if not entry.json_data:
        raise HTTPException(status_code=400, detail="no json data found for this entry")

    # 2) DataFrame
    df = pd.DataFrame(entry.json_data)

    # 3) Detect time column and derive features
    time_col = None
    possible_keywords = ["date", "time", "timestamp", "datetime", "recordtime", "thedate"]
    for c in df.columns:
        if any(keyword in str(c).lower() for keyword in possible_keywords):
            time_col = c
            df[c] = pd.to_datetime(df[c], errors="coerce")
            break
    if time_col is None:
        for c in df.columns:
            if df[c].dtype == "object":
                sample = df[c].dropna().head(5)
                if pd.to_datetime(sample, errors="coerce").notna().all():
                    time_col = c
                    df[c] = pd.to_datetime(df[c], errors="coerce")
                    break
    if time_col:
        df["month"] = df[time_col].dt.month
        df["day_of_year"] = df[time_col].dt.dayofyear
        df["hour"] = df[time_col].dt.hour

    # 4) Outlier mask via DataProcessor (or custom)
    dp = DataProcessor()
    outlier_mask = pd.Series(False, index=df.index)
    method_to_use = outlier_method if remove_outliers else "iqr"
    columns = ["EAC", "GI", "TM"]

    if method_to_use != "none":
        if method_to_use == "iqr":
            outlier_mask = dp.detect_outliers_iqr_mask(df, columns, iqr_factor=iqr_factor)
        elif method_to_use == "zscore":
            outlier_mask = dp.detect_outliers_zscore_mask(df, columns, threshold=zscore_threshold)
        elif method_to_use == "isolation_forest":
            outlier_mask = dp.detect_outliers_isoforest_mask(df, columns, contamination=iso_contamination)
        elif method_to_use == "custom":
            if "GI" in df.columns and "EAC" in df.columns:
                outlier_mask |= (df["GI"] < 100) & (df["EAC"] > 3.0)
                outlier_mask |= (df["GI"] < 30) & (df["EAC"] > 1.5)
            if "TM" in df.columns and "EAC" in df.columns:
                outlier_mask |= (df["TM"] > 40) & (df["EAC"] < 3.0)

    # 5) Apply removal and persist cleaned result when toggle is ON
    if remove_outliers:
        cleaned_df = df.loc[~outlier_mask].reset_index(drop=True)

        # sanitize datetimes for JSON
        json_ready = cleaned_df.map(lambda x: x.isoformat() if hasattr(x, "isoformat") else x)
        json_records = json_ready.replace({pd.NA: None}).to_dict(orient="records")

        # save to disk
        cleaned_folder = _cleaned_dir()
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S%f")
        file_stub = f"cleaned_{data_id}_{method_to_use}_{ts}.csv"
        cleaned_path = cleaned_folder / file_stub
        try:
            cleaned_df.to_csv(cleaned_path, index=False)
            cleaned_size = cleaned_path.stat().st_size
        except Exception as e:
            cleaned_size = None
            print(f"[warn] failed to save cleaned file: {e}")

        # update DB entry (processed fields)
        extra_meta = {
            "cleaned_path": str(cleaned_path.relative_to(_uploads_base_dir())),
            "cleaned_rows": int(len(cleaned_df)),
            "cleaned_method": method_to_use,
            "cleaned_params": {
                "iqr_factor": iqr_factor,
                "zscore_threshold": zscore_threshold,
                "iso_contamination": iso_contamination,
            },
            "cleaned_size": cleaned_size,
        }
        entry.processed_json = json_records
        entry.processed_at = datetime.utcnow()
        entry.processed_by = "visualize_site_data"
        entry.processed_meta = {**(entry.processed_meta or {}), **extra_meta}
        db.add(entry)
        db.commit()

        # continue visualization with cleaned df
        df = cleaned_df
        outlier_mask = outlier_mask.loc[df.index]

    # 6) Stats & sample
    stats = df.describe(include="all").to_dict()
    sample = df.head(20).to_dict(orient="records")

    # 7) Scatter matrix
    scatter_cols = ["EAC", "GI", "TM"]
    pairs = {}
    for x in scatter_cols:
        for y in scatter_cols:
            if x == y:
                continue
            if x in df.columns and y in df.columns:
                sub = df[[x, y]].dropna()
                pairs[f"{x}__{y}"] = {
                    "x": sub[x].tolist(),
                    "y": sub[y].tolist(),
                }
    scatter_matrix = {"columns": scatter_cols, "pairs": pairs}

    # 8) Boxplot values (by month/day/hour)
    def compute_box_values(dataframe: pd.DataFrame, group_col: str):
        result = {}
        if group_col not in dataframe.columns or "EAC" not in dataframe.columns:
            return result
        for g, sub in dataframe.groupby(group_col):
            values = sub["EAC"].dropna().tolist()
            if len(values) >= 1:
                result[str(g)] = {"values": values}
        return result

    boxplot_by_month = compute_box_values(df, "month")
    boxplot_by_day = compute_box_values(df, "day_of_year")
    boxplot_by_hour = compute_box_values(df, "hour")

    # 9) Response
    return safe_json({
        "columns": df.columns.tolist(),
        "stats": stats,
        "sample": sample,
        "scatter_matrix": scatter_matrix,
        "boxplot_by_month": boxplot_by_month,
        "boxplot_by_day": boxplot_by_day,
        "boxplot_by_hour": boxplot_by_hour,
        "outlier_mask": outlier_mask.tolist(),
    })
