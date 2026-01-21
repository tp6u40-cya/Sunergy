# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import Base, engine
from pathlib import Path
import os

# import models so SQLAlchemy knows table classes
import models
from routers.auth import router as auth_router
from routers.site import router as site_router
from routers.visualize import router as visualize_router

# create tables if not exist
Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)      # routes prefixed by /auth (router defines prefix)
app.include_router(site_router)      # routes prefixed by /site
app.include_router(visualize_router) # routes prefixed by /visualize

@app.get("/")
def root():
    return {"message": "Backend running!"}


# Ensure local object storage directories exist
def _ensure_upload_dirs():
    base_dir = Path(__file__).resolve().parent
    # backend/ directory
    uploads = base_dir / "uploads"
    raw_dir = uploads / "raw"
    cleaned_dir = uploads / "cleaned"
    for d in (uploads, raw_dir, cleaned_dir):
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass


_ensure_upload_dirs()
