# routers/auth.py
from fastapi import APIRouter, HTTPException, Depends, Header, status
from sqlalchemy.orm import Session
from passlib.context import CryptContext

from database import get_db
from models import User
from schemas import RegisterUser, LoginUser
from auth_utils import create_access_token, decode_access_token

router = APIRouter(prefix="/auth", tags=["Auth"])
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")


@router.post("/register")
def register(user: RegisterUser, db: Session = Depends(get_db)):
    exists = db.query(User).filter(User.user_account == user.user_account).first()
    if exists:
        raise HTTPException(status_code=400, detail="Account already exists")
    hashed = pwd_context.hash(user.user_pw)
    new_user = User(user_name=user.user_name, user_account=user.user_account, user_pw=hashed)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token({
        "sub": str(new_user.user_id),
        "user_id": new_user.user_id,
        "user_name": new_user.user_name,
        "user_account": new_user.user_account,
    })
    return {
        "message": "Registered",
        "user_id": new_user.user_id,
        "user_name": new_user.user_name,
        "user_account": new_user.user_account,
        "access_token": token,
        "token_type": "bearer",
    }


@router.post("/login")
def login(user: LoginUser, db: Session = Depends(get_db)):
    u = db.query(User).filter(User.user_account == user.user_account).first()
    if not u or not pwd_context.verify(user.user_pw, u.user_pw):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    token = create_access_token({
        "sub": str(u.user_id),
        "user_id": u.user_id,
        "user_name": u.user_name,
        "user_account": u.user_account,
    })
    return {
        "message": "Login success",
        "user_id": u.user_id,
        "user_name": u.user_name,
        "user_account": u.user_account,
        "access_token": token,
        "token_type": "bearer",
    }


def _extract_bearer(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header")
    return parts[1]


@router.get("/me")
def me(authorization: str | None = Header(None), db: Session = Depends(get_db)):
    token = _extract_bearer(authorization)
    try:
        payload = decode_access_token(token)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    u = db.query(User).filter(User.user_id == user_id).first()
    if not u:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return {
        "user_id": u.user_id,
        "user_name": u.user_name,
        "user_account": u.user_account,
    }

