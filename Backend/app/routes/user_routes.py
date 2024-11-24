from ..utils.email import send_email_via_graph_api
from ..models import user_models
from ..schemas import user_schemas
from ..dependencies import get_current_user
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from .. import models, schemas
from typing import Optional
from ..database import get_db

router = APIRouter()

@router.post("/create_user/")
def create_user(user: user_schemas.UserCreate, db: Session = Depends(get_db), current_user: user_schemas.User = Depends(get_current_user)):
    db_user = db.query(user_models.User).filter(user_models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="User already registered")
    
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    default_password=user.password
    hashed_password = pwd_context.hash(default_password)

    new_user = user_models.User(email=user.email, name=user.name, password=hashed_password, is_admin=user.is_admin)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    # send email to user
    send_email_via_graph_api(new_user.email,user.password)

    # Return a response model without the password
    return user_schemas.UserResponse(
        id=new_user.id,
        email=new_user.email,
        name=new_user.name,
        is_admin=new_user.is_admin,
        can_access_budget_analysis=True,   
        can_access_accounts=False,         
        can_access_analysis_report=False,
        can_download=False
        
    )

@router.delete("/delete_user/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db), current_user: user_models.User = Depends(get_current_user)):
    db_user = db.query(user_models.User).filter(user_models.User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    db.delete(db_user)
    db.commit()
    return {"message": "User deleted successfully"}

@router.patch("/reset_password/{user_id}")
def reset_password(user_id: int, request: user_schemas.ResetPasswordRequest, db: Session = Depends(get_db), current_user: user_models.User = Depends(get_current_user)):
    db_user = db.query(user_models.User).filter(user_models.User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    default_password=request.new_password 
    hashed_password = pwd_context.hash(default_password)

    db_user.password = hashed_password  # Assuming proper hashing will be applied
    db.commit()
    return {"message": "Password reset successful"}

@router.patch("/update_permissions/{user_id}")
def update_user_permissions(user_id: int, permissions: user_schemas.UserUpdatePermissions, db: Session = Depends(get_db), current_user: user_models.User = Depends(get_current_user)):
    db_user = db.query(user_models.User).filter(user_models.User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    db_user.can_access_budget_analysis = permissions.can_access_budget_analysis
    db_user.can_access_accounts = permissions.can_access_accounts
    db_user.can_access_analysis_report = permissions.can_access_analysis_report
    db_user.can_download = permissions.can_download
    

    db.commit()
    db.refresh(db_user)
    return {"message": "User permissions updated successfully", "user": db_user}

@router.get("/user_permissions/{user_id}")
def get_user_permissions(user_id: int, db: Session = Depends(get_db), current_user: user_models.User = Depends(get_current_user)):
    db_user = db.query(user_models.User).filter(user_models.User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "email": db_user.email,
        "name": db_user.name,
        "is_admin": db_user.is_admin,
        "can_access_budget_analysis": db_user.can_access_budget_analysis,
        "can_access_accounts": db_user.can_access_accounts,
        "can_access_analysis_report": db_user.can_access_analysis_report,
        "can_download": db_user.can_download
    }


@router.get("/users/")
def get_all_users(
    db: Session = Depends(get_db),
    current_user: user_models.User = Depends(get_current_user),
    email: Optional[str] = Query(None, description="Filter users by email")
):
    # If email is provided, filter by email, else return all users
    if email:
        users = db.query(user_models.User).filter(user_models.User.email.ilike(f"%{email}%")).all()
    else:
        users = db.query(user_models.User).all()
        
    return [
        {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "is_admin": user.is_admin,
            "can_access_budget_analysis": user.can_access_budget_analysis,
            "can_access_accounts": user.can_access_accounts,
            "can_access_analysis_report": user.can_access_analysis_report,
            "can_download": user.can_download
        }
        for user in users
    ]


