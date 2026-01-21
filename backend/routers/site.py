# routers/site.py
from fastapi import APIRouter, UploadFile, File, Query, HTTPException, Depends
from sqlalchemy.orm import Session
from io import BytesIO
import pandas as pd
from datetime import datetime

from database import get_db
from models import Site, SiteData, User
from schemas import CreateSite

router = APIRouter(prefix="/site", tags=["Site"])

@router.get("/list")
def list_sites(user_id: int, db: Session = Depends(get_db)):
    sites = db.query(Site).filter(Site.user_id == user_id).order_by(Site.created_at.desc()).all()
    return [
        {"site_id": s.site_id, "site_code": s.site_code, "site_name": s.site_name, "location": s.location, "created_at": s.created_at.isoformat(), "user_id": s.user_id}
        for s in sites
    ]

@router.post("/create")
def create_site(payload: CreateSite, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.user_id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=400, detail="指定的 user_id 不存在")
    new_site = Site(site_code=payload.site_code, site_name=payload.site_name, location=payload.location, user_id=payload.user_id)
    db.add(new_site)
    db.commit()
    db.refresh(new_site)
    return {"message":"案場建立成功", "site_id": new_site.site_id}

@router.post("/upload-data")
async def upload_site_data(
    site_id: int = Query(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    site = db.query(Site).filter(Site.site_id == site_id).first()
    if not site:
        raise HTTPException(status_code=400, detail="site_id 不存在")

    content = await file.read()
    bio = BytesIO(content)

    try:
        fname = file.filename.lower()
        if fname.endswith(".csv"):
            df = pd.read_csv(bio)
        else:
            df = pd.read_excel(bio)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"無法解析檔案: {str(e)}")

    original_rows = int(len(df))
    features = list(df.columns)

    # ⭐⭐ 轉換 Timestamp → 字串（避免 SQLAlchemy JSON 失敗）
    df = df.map(lambda x: x.isoformat() if hasattr(x, "isoformat") else x)

    # 轉成 JSON 可接受的格式
    json_data = df.replace({pd.NA: None}).to_dict(orient="records")

    new_entry = SiteData(
        site_id=site_id,
        data_name=file.filename,
        original_rows=original_rows,
        file_bytes=content,
        json_data=json_data
    )
    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)

    return {
        "message": "上傳成功",
        "data_id": new_entry.data_id,
        "file_name": new_entry.data_name,
        "rows": original_rows,
        "features": features
    }

@router.get("/download")
def download_data(data_id: int, db: Session = Depends(get_db)):
    entry = db.query(SiteData).filter(SiteData.data_id == data_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="data_id not found")
    if not entry.file_bytes:
        raise HTTPException(status_code=404, detail="no file stored")
    # 這裡回傳 metadata; 若要下載 file bytes 作流式傳輸可以改成 StreamingResponse
    return {"file_name": entry.data_name, "size": len(entry.file_bytes)}
