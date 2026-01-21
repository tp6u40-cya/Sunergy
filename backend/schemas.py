# schemas.py
from pydantic import BaseModel
from typing import Optional, Dict, Any

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
