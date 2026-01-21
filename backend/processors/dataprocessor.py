# processors/dataprocessor.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

class DataProcessor:
    def __init__(self):
        pass

    def detect_outliers_iqr_mask(self, df, columns, iqr_factor=1.5):
        mask = pd.Series(False, index=df.index)
        for col in columns:
            if col not in df.columns: continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lb = Q1 - iqr_factor * IQR
            ub = Q3 + iqr_factor * IQR
            mask = mask | (df[col] < lb) | (df[col] > ub)
        return mask

    def detect_outliers_zscore_mask(self, df, columns, threshold=3.0):
        mask = pd.Series(False, index=df.index)
        for col in columns:
            if col not in df.columns: continue
            std = df[col].std()
            if std == 0 or pd.isna(std):
                continue
            z = (df[col] - df[col].mean()) / std
            mask = mask | (z.abs() > threshold)
        return mask

    def detect_outliers_isoforest_mask(self, df, columns, contamination=0.05):
        numeric_data = df[columns].select_dtypes(include=[np.number]).dropna()
        if numeric_data.shape[0] == 0:
            return pd.Series(False, index=df.index)
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = pd.Series(iso.fit_predict(numeric_data), index=numeric_data.index)
        mask = pd.Series(False, index=df.index)
        mask.loc[preds[preds == -1].index] = True
        return mask

    def remove_outliers(self, df, method="iqr", params=None):
        params = params or {}
        cols = [c for c in ["EAC","GI","TM"] if c in df.columns]
        if len(cols) == 0:
            return df.copy(), pd.DataFrame(columns=df.columns)

        if method == "iqr":
            mask = self.detect_outliers_iqr_mask(df, cols, iqr_factor=params.get("iqr_factor", 1.5))
        elif method == "zscore":
            mask = self.detect_outliers_zscore_mask(df, cols, threshold=params.get("zscore_threshold", 3.0))
        elif method == "isolation_forest":
            mask = self.detect_outliers_isoforest_mask(df, cols, contamination=params.get("contamination", 0.05))
        elif method == "default":
            # default: drop NA GI, iso forest then iqr
            df2 = df.copy()
            if "GI" in df2.columns:
                df2 = df2.dropna(subset=["GI"])
            mask_iso = self.detect_outliers_isoforest_mask(df2, cols, contamination=params.get("contamination", 0.05))
            mask = pd.Series(False, index=df.index)
            mask.loc[mask_iso[mask_iso].index] = True
            mask_iqr = self.detect_outliers_iqr_mask(df2, cols, iqr_factor=params.get("iqr_factor", 1.5))
            mask.loc[mask_iqr[mask_iqr].index] = True
        else:
            raise ValueError("unknown method")

        outliers = df.loc[mask].reset_index(drop=False)  # keep original index if needed
        cleaned = df.loc[~mask].reset_index(drop=True)
        return cleaned, outliers

    def compute_hist(self, arr, bins=10):
        if len(arr) == 0:
            return {"bins": [], "counts": []}
        counts, bin_edges = np.histogram(np.array(arr, dtype=float), bins=bins)
        return {"bins": bin_edges.tolist(), "counts": counts.tolist()}

    def compute_scatter_pairs(self, df, numeric_cols):
        pairs = {}
        hist = {}
        for i, xi in enumerate(numeric_cols):
            for j, yj in enumerate(numeric_cols):
                if xi == yj: continue
                x = df[xi].replace({np.nan: None}).tolist()
                y = df[yj].replace({np.nan: None}).tolist()
                L = min(len(x), len(y))
                pairs[f"{xi}__{yj}"] = {"x": [self._sanitize(v) for v in x[:L]], "y": [self._sanitize(v) for v in y[:L]]}
        for c in numeric_cols:
            arr = df[c].dropna().tolist()
            hist[c] = self.compute_hist(arr, bins=10)
        return pairs, hist

    def compute_box_by_group(self, df, group_col, value_col):
        if not group_col or group_col not in df.columns or value_col not in df.columns:
            return {}
        groups = {}
        for key, g in df.groupby(df[group_col].fillna("NA")):
            vals = g[value_col].dropna().astype(float)
            if len(vals) == 0:
                groups[str(key)] = {"min": None, "q1": None, "median": None, "q3": None, "max": None}
            else:
                groups[str(key)] = {
                    "min": float(vals.min()),
                    "q1": float(vals.quantile(0.25)),
                    "median": float(vals.median()),
                    "q3": float(vals.quantile(0.75)),
                    "max": float(vals.max()),
                }
        return groups

    def _sanitize(self, v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        try:
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            return v
        except:
            return str(v)
