# schemas.py
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Literal

class RegisterUser(BaseModel):
    user_name: str
    user_account: str
    user_pw: str

class LoginUser(BaseModel):
    user_account: str
    user_pw: str

class CreateSite(BaseModel):
    site_code: str
    site_name: str
    location: str
    user_id: int

class ProcessRequest(BaseModel):
    data_id: int
    method: str                 # 'iqr' | 'zscore' | 'isolation_forest' | 'default'
    params: Optional[Dict[str, Any]] = None


class TrainRequest(BaseModel):
    data_id: int
    split_ratio: float  # e.g., 0.8 means 80% train
    models: List[str]   # ['LSTM','XGBoost','RandomForest','SVR'] any subset
    strategy: Literal['manual','grid','bayes'] = 'grid'
    params: Optional[Dict[str, Dict[str, Any]]] = None  # per-model param specs (fixed or ranges)
    target: Optional[str] = 'EAC'
    # new, optional but with sensible defaults applied in code
    features: Optional[List[str]] = None           # default to ['GI','TM'] if None
    time_col: Optional[str] = None                 # attempt auto-detect if None
    split_method: Literal['time','random'] = 'time'
    save_model: bool = True
    model_name: Optional[str] = None
