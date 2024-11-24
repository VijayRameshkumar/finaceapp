import os
from ..schemas import user_schemas
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
from ..models import user_models
from ..database import get_db
from dotenv import load_dotenv

router = APIRouter()

# Load environment variables from a .env file
load_dotenv()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Secret key and algorithm for JWT
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 3600  # Token expiration time in minutes

# Function to create a new access token
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Login route
@router.post("/login")
async def login(form_data: user_schemas.TokenRequest, db: Session = Depends(get_db)):
    db_user = db.query(user_models.User).filter(user_models.User.email == form_data.email).first()
    if not db_user or not pwd_context.verify(form_data.password, db_user.password):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"id": db_user.id,"email": db_user.email,"is_admin":db_user.is_admin}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}
