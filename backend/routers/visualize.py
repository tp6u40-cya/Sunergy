# backend/routers/visualize.py
from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

from database import get_db
from models import SiteData

router = APIRouter(prefix="/visualize", tags=["Visualize"])


# =========================================================
# 把 NaN / inf 轉成 JSON 合法的 None
# =========================================================
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


# =========================================================
# API
# =========================================================
@router.get("/site-data")
def visualize_site_data(
    data_id: int = Query(...),
    remove_outliers: bool = Query(False),
    outlier_method: str = Query("iqr", description="離群值偵測方法: none, iqr, zscore, isolation_forest, custom"),
    iqr_factor: float = Query(1.5, description="IQR 方法的倍數"),
    zscore_threshold: float = Query(3.0, description="Z-Score 方法的閾值"),
    iso_contamination: float = Query(0.05, description="Isolation Forest 的污染率", ge=0.01, le=0.5),
    db: Session = Depends(get_db)
):
    # -----------------------------------------------------
    # 1️⃣ 撈資料
    # -----------------------------------------------------
    entry = db.query(SiteData).filter(SiteData.data_id == data_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="data_id 不存在")

    if not entry.json_data:
        raise HTTPException(status_code=400, detail="此資料沒有 json data")

    # -----------------------------------------------------
    # 2️⃣ DataFrame
    # -----------------------------------------------------
    df = pd.DataFrame(entry.json_data)

    # -----------------------------------------------------
    # 3️⃣ 處理時間欄位
    # -----------------------------------------------------
    time_col = None
    possible_keywords = ["date", "time", "timestamp", "datetime", "recordtime", "thedate"]
    for c in df.columns:
        if any(keyword in c.lower() for keyword in possible_keywords):
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
    else:
        print("警告：完全找不到時間欄位，箱型圖將無法顯示")

    # -----------------------------------------------------
    # 4️⃣ 建立離群點 mask（支援多種方法與係數調整）
    # -----------------------------------------------------
    outlier_mask = pd.Series(False, index=df.index)

    # 決定使用哪個方法（初始顯示用 iqr，移除時用選擇的方法）
    method_to_use = outlier_method if remove_outliers else "iqr"

    if method_to_use != "none":
        columns = ["EAC", "GI", "TM"]

        if method_to_use == "iqr":
            for col in columns:
                if col not in df.columns:
                    continue
                s = df[col].dropna()
                if len(s) < 5:
                    continue
                q1 = s.quantile(0.25)
                q3 = s.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower = q1 - iqr_factor * iqr
                upper = q3 + iqr_factor * iqr
                mask = (df[col] < lower) | (df[col] > upper)
                outlier_mask |= mask.fillna(False)

        elif method_to_use == "zscore":
            for col in columns:
                if col not in df.columns:
                    continue
                s = df[col].dropna()
                if len(s) < 5:
                    continue
                mean = s.mean()
                std = s.std()
                if std == 0:
                    continue
                mask = (df[col] - mean).abs() > zscore_threshold * std
                outlier_mask |= mask.fillna(False)

        elif method_to_use == "isolation_forest":
            numeric_df = df[columns].select_dtypes(include=[np.number]).dropna()
            if numeric_df.shape[0] > 10 and numeric_df.shape[1] > 0:
                iso = IsolationForest(
                    contamination=max(0.01, min(0.5, iso_contamination)),
                    random_state=42
                )
                preds = iso.fit_predict(numeric_df)
                iso_mask = pd.Series(preds == -1, index=numeric_df.index)
                outlier_mask.loc[iso_mask.index] = iso_mask

        elif method_to_use == "custom":
            # 精準物理規則（避免誤標 TM vs GI 正常點）
            if "GI" in df.columns and "EAC" in df.columns:
                # GI 低但 EAC 高 → 異常
                outlier_mask |= (df["GI"] < 100) & (df["EAC"] > 3.0)
                outlier_mask |= (df["GI"] < 30) & (df["EAC"] > 1.5)

            if "TM" in df.columns and "EAC" in df.columns:
                # 高溫低換氣 → 異常
                outlier_mask |= (df["TM"] > 40) & (df["EAC"] < 3.0)

            # 冬天高輻射 → 可能異常，但不強制
            # if "GI" in df.columns and "TM" in df.columns:
            #     outlier_mask |= (df["GI"] > 700) & (df["TM"] < 22)

    # -----------------------------------------------------
    # 5️⃣ 移除離群值（僅在 remove_outliers=True 時）
    # -----------------------------------------------------
    if remove_outliers:
        df = df.loc[~outlier_mask].reset_index(drop=True)
        outlier_mask = outlier_mask.loc[df.index]

    # -----------------------------------------------------
    # 6️⃣ 統計量 & Sample
    # -----------------------------------------------------
    stats = df.describe(include="all").to_dict()
    sample = df.head(20).to_dict(orient="records")

    # -----------------------------------------------------
    # 7️⃣ Scatter Matrix
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # 8️⃣ Boxplot（只回傳 values）
    # -----------------------------------------------------
    def compute_box_values(df, group_col):
        result = {}
        if group_col not in df.columns or "EAC" not in df.columns:
            return result

        for g, sub in df.groupby(group_col):
            values = sub["EAC"].dropna().tolist()
            if len(values) >= 1:
                result[str(g)] = {"values": values}
        return result

    boxplot_by_month = compute_box_values(df, "month")
    boxplot_by_day = compute_box_values(df, "day_of_year")
    boxplot_by_hour = compute_box_values(df, "hour")

    # -----------------------------------------------------
    # 9️⃣ 回傳
    # -----------------------------------------------------
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