"""
Pydantic models for authentication API requests and responses.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, EmailStr, Field, validator


class RegisterRequest(BaseModel):
    """User registration request."""
    username: str = Field(..., min_length=3, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = Field(None, max_length=255)
    
    @validator('username')
    def username_alphanumeric(cls, v):
        """Validate username is alphanumeric with underscores."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must be alphanumeric (underscores and hyphens allowed)')
        return v


class LoginRequest(BaseModel):
    """User login request."""
    username: str = Field(..., description="Username or email")
    password: str


class TokenResponse(BaseModel):
    """Token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    """Change password request."""
    old_password: str
    new_password: str = Field(..., min_length=8)


class UserResponse(BaseModel):
    """User response."""
    id: str
    username: str
    email: str
    full_name: Optional[str]
    role: str
    status: str
    email_verified: bool
    last_login_at: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    
    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    """Generic message response."""
    message: str
    success: bool = True


# ===== User Preferences Models =====

class ChatPreferences(BaseModel):
    """Chat-specific preferences."""
    selectedWorkspaceIds: list[str] = Field(default_factory=list, description="List of selected workspace UUIDs for RAG search filtering")
    selectedDocumentIds: list[str] = Field(default_factory=list, description="List of selected document UUIDs for RAG search filtering")
    lastUpdated: Optional[str] = None


class AppPreferences(BaseModel):
    """Application-wide preferences."""
    theme: str = Field(default="saga-blue", description="Selected PrimeReact theme")
    lastUpdated: Optional[str] = None


class UserPreferences(BaseModel):
    """Full user preferences object."""
    chat: Optional[ChatPreferences] = None
    app: Optional[AppPreferences] = None

    class Config:
        extra = "allow"  # Allow additional preference keys


class UpdatePreferencesRequest(BaseModel):
    """Request to update user preferences."""
    preferences: Dict[str, Any]


class UpdateChatPreferencesRequest(BaseModel):
    """Request to update chat preferences."""
    selectedWorkspaceIds: list[str] = Field(default_factory=list)
    selectedDocumentIds: list[str] = Field(default_factory=list)


class UpdateThemePreferenceRequest(BaseModel):
    """Request to update theme preference."""
    theme: str = Field(..., description="PrimeReact theme name")


class PreferencesResponse(BaseModel):
    """User preferences response."""
    preferences: Dict[str, Any]
    success: bool = True


# ===== User Profile Models =====

class UpdateProfileRequest(BaseModel):
    """Request to update user profile information."""
    email: EmailStr = Field(..., description="User's email address (required)")
    full_name: Optional[str] = Field(None, max_length=255, description="User's full name")
    phone_number: Optional[str] = Field(None, max_length=20, description="User's phone number")
    address: Optional[str] = Field(None, max_length=500, description="User's address")
    city: Optional[str] = Field(None, max_length=100, description="User's city")
    state: Optional[str] = Field(None, max_length=100, description="User's state/province")
    country: Optional[str] = Field(None, max_length=100, description="User's country")
    postal_code: Optional[str] = Field(None, max_length=20, description="User's postal code")
    bio: Optional[str] = Field(None, max_length=1000, description="User's bio/description")


class ProfileResponse(BaseModel):
    """User profile response."""
    id: str
    username: str
    email: str
    full_name: Optional[str]
    phone_number: Optional[str]
    address: Optional[str]
    city: Optional[str]
    state: Optional[str]
    country: Optional[str]
    postal_code: Optional[str]
    bio: Optional[str]
    role: str
    status: str
    email_verified: bool
    last_login_at: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    success: bool = True

    class Config:
        from_attributes = True

