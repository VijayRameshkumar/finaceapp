from pydantic import BaseModel

class UserBase(BaseModel):
    email: str
    name: str

class UserCreate(UserBase):
    is_admin: bool = False
    password: str

class UserUpdatePermissions(BaseModel):
    can_access_budget_analysis: bool
    can_access_accounts: bool
    can_access_analysis_report: bool
    can_download: bool

class User(UserBase):
    id: int
    is_admin: bool
    can_access_budget_analysis: bool
    can_access_accounts: bool
    can_access_analysis_report: bool
    can_download:bool

    class Config:
        from_attributes = True


class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    is_admin: bool
    can_access_budget_analysis: bool
    can_access_accounts: bool
    can_access_analysis_report: bool
    can_download:bool

    class Config:
        # Use from_attributes instead of orm_mode
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    id: int
    email: str
    name: str
    is_admin: bool
    can_access_budget_analysis: bool
    can_access_accounts: bool
    can_access_analysis_report: bool
    can_download: bool

class TokenRequest(BaseModel):
    email: str
    password: str

class ResetPasswordRequest(BaseModel):
    new_password: str

