# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import Base, engine
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
