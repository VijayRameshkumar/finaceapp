from sqlalchemy import Column, Integer, String, Boolean
from ..database import Base

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False)
    can_access_budget_analysis = Column(Boolean, default=False)
    can_access_accounts = Column(Boolean, default=False)
    can_access_analysis_report = Column(Boolean, default=False)
    can_download= Column(Boolean, default=False)
