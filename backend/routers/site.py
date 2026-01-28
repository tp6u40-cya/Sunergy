# routers/site.py
from fastapi import APIRouter, UploadFile, File, Query, HTTPException, Depends
from sqlalchemy.orm import Session
from io import BytesIO
import pandas as pd
from datetime import datetime
from pathlib import Path
import re

from database import get_db
from models import Site, SiteData, User
from schemas import CreateSite

router = APIRouter(prefix="/site", tags=["Site"])


def _uploads_base_dir() -> Path:
    # backend/routers/site.py -> backend/uploads
    base = Path(__file__).resolve().parent.parent / "uploads"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _raw_dir() -> Path:
    p = _uploads_base_dir() / "raw"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cleaned_dir() -> Path:
    p = _uploads_base_dir() / "cleaned"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_filename(name: str) -> str:
    name = name.replace("\\", "/").split("/")[-1]
    # keep letters, numbers, dash, underscore, dot
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


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
        raise HTTPException(status_code=400, detail="user_id not found")
    new_site = Site(site_code=payload.site_code, site_name=payload.site_name, location=payload.location, user_id=payload.user_id)
    db.add(new_site)
    db.commit()
    db.refresh(new_site)
    return {"message": "site created", "site_id": new_site.site_id}


@router.post("/upload-data")
async def upload_site_data(
    site_id: int = Query(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    site = db.query(Site).filter(Site.site_id == site_id).first()
    if not site:
        raise HTTPException(status_code=400, detail="site_id not found")

    content = await file.read()
    bio = BytesIO(content)

    try:
        fname = (file.filename or "").lower()
        if fname.endswith(".csv"):
            df = pd.read_csv(bio)
        else:
            df = pd.read_excel(bio)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to parse file: {str(e)}")

    original_rows = int(len(df))
    features = list(df.columns)

    # convert timestamps to strings if needed for JSON
    df = df.map(lambda x: x.isoformat() if hasattr(x, "isoformat") else x)
    json_data = df.replace({pd.NA: None}).to_dict(orient="records")

    # Save raw file to local object storage (uploads/raw)
    raw_folder = _raw_dir()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S%f")
    safe_name = _safe_filename(file.filename or f"upload_{ts}.bin")
    stored_name = f"{ts}_site{site_id}_{safe_name}"
    raw_path = raw_folder / stored_name
    try:
        with open(raw_path, "wb") as f:
            f.write(content)
        raw_size = raw_path.stat().st_size
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store file: {str(e)}")

    new_entry = SiteData(
        site_id=site_id,
        data_name=file.filename,
        original_rows=original_rows,
        json_data=json_data,
        processed_meta={
            "raw_path": str(raw_path.relative_to(_uploads_base_dir())),
            "raw_size": raw_size,
        }
    )
    db.add(new_entry)
    db.commit()
    db.refresh(new_entry)

    return {
        "message": "uploaded",
        "data_id": new_entry.data_id,
        "file_name": new_entry.data_name,
        "rows": original_rows,
        "features": features,
        "storage": {
            "raw_path": new_entry.processed_meta.get("raw_path") if new_entry.processed_meta else None,
            "raw_size": new_entry.processed_meta.get("raw_size") if new_entry.processed_meta else None,
        },
    }


@router.get("/download")
def download_data(data_id: int, db: Session = Depends(get_db)):
    entry = db.query(SiteData).filter(SiteData.data_id == data_id).first()
    if not entry:
        raise HTTPException(status_code=404, detail="data_id not found")

    # Prefer local object storage if available
    raw_rel = (entry.processed_meta or {}).get("raw_path") if entry.processed_meta else None
    if raw_rel:
        raw_abs = _uploads_base_dir() / raw_rel
        if raw_abs.exists():
            try:
                size = raw_abs.stat().st_size
            except Exception:
                size = None
            return {"file_name": entry.data_name, "size": size, "path": str(raw_rel)}

    # No DB-bytes fallback anymore: files are stored on disk only

    raise HTTPException(status_code=404, detail="no file stored")
