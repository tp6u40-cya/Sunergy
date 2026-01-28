# models.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, LargeBinary, JSON, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class User(Base):
    __tablename__ = "user"
    user_id = Column(Integer, primary_key=True, index=True)
    user_name = Column(String, nullable=False)
    user_account = Column(String, unique=True, nullable=False)
    user_pw = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    sites = relationship("Site", back_populates="owner")

class Site(Base):
    __tablename__ = "site"
    site_id = Column(Integer, primary_key=True, index=True)
    site_code = Column(String, nullable=False)
    site_name = Column(String, nullable=False)
    location = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("user.user_id"), nullable=False)
    owner = relationship("User", back_populates="sites")
    site_data = relationship("SiteData", back_populates="site")

class SiteData(Base):
    __tablename__ = "site_data"
    data_id = Column(Integer, primary_key=True, index=True)
    site_id = Column(Integer, ForeignKey("site.site_id"), nullable=False)
    data_name = Column(String, nullable=True)
    original_rows = Column(Integer, nullable=True)
    # file_bytes column removed: we persist raw files on disk under backend/uploads/raw
    json_data = Column(JSON, nullable=True)            # 原始資料 (records)
    processed_json = Column(JSON, nullable=True)       # 處理後資料 (records)
    processed_at = Column(DateTime, nullable=True)
    processed_by = Column(String, nullable=True)
    processed_meta = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    site = relationship("Site", back_populates="site_data")


class TrainedModel(Base):
    __tablename__ = "trained_model"

    # primary key
    model_id = Column(Integer, primary_key=True, index=True)

    # links
    site_id = Column(Integer, ForeignKey("site.site_id"), nullable=False, index=True)
    data_id = Column(Integer, ForeignKey("site_data.data_id"), nullable=True, index=True)

    # metadata
    model_type = Column(String, nullable=False)            # e.g., 'XGBoost', 'SVR', 'RandomForest', 'LSTM'
    parameters = Column(JSONB, nullable=True)              # best params used to train this artifact
    file_path = Column(String, nullable=False)             # relative path under backend/uploads
    trained_at = Column(DateTime, default=datetime.utcnow, nullable=False)
