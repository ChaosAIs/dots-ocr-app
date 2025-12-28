"""
Authentication API endpoints.
"""
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from db.database import get_db
from db.models import User, UserRole
from db.user_repository import UserRepository
from auth.models import (
    RegisterRequest, LoginRequest, TokenResponse, RefreshTokenRequest,
    ChangePasswordRequest, UserResponse, MessageResponse,
    UpdatePreferencesRequest, UpdateChatPreferencesRequest, PreferencesResponse
)
from auth.dependencies import get_current_active_user, require_admin
from auth.password_utils import PasswordUtils
from auth.jwt_utils import JWTUtils

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(
    request: RegisterRequest,
    http_request: Request,
    db: Session = Depends(get_db)
):
    """Register a new user."""
    user_repo = UserRepository(db)
    
    # Validate password strength
    is_valid, error_msg = PasswordUtils.validate_password_strength(request.password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )
    
    # Check if username or email already exists
    if user_repo.get_user_by_username(request.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    if user_repo.get_user_by_email(request.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    try:
        # Create user
        user = user_repo.create_user(
            username=request.username,
            email=request.email,
            password=request.password,
            full_name=request.full_name,
            role=UserRole.USER
        )
        
        # Generate tokens
        access_token = JWTUtils.create_access_token(
            user_id=user.id,
            username=user.username,
            role=user.role.value
        )
        
        refresh_token, expires_at = JWTUtils.create_refresh_token(
            user_id=user.id,
            username=user.username
        )
        
        # Store refresh token
        user_repo.create_refresh_token(
            user_id=user.id,
            token=refresh_token,
            expires_at=expires_at,
            user_agent=http_request.headers.get("user-agent"),
            ip_address=http_request.client.host if http_request.client else None
        )
        
        logger.info(f"User registered: {user.username}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user=user.to_dict()
        )
    
    except IntegrityError as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Registration failed. Username or email may already exist."
        )


@router.post("/login", response_model=TokenResponse)
def login(
    request: LoginRequest,
    http_request: Request,
    db: Session = Depends(get_db)
):
    """Login user."""
    user_repo = UserRepository(db)
    
    # Authenticate user
    user = user_repo.authenticate_user(
        identifier=request.username,
        password=request.password,
        ip_address=http_request.client.host if http_request.client else None
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate tokens
    access_token = JWTUtils.create_access_token(
        user_id=user.id,
        username=user.username,
        role=user.role.value
    )
    
    refresh_token, expires_at = JWTUtils.create_refresh_token(
        user_id=user.id,
        username=user.username
    )
    
    # Store refresh token
    user_repo.create_refresh_token(
        user_id=user.id,
        token=refresh_token,
        expires_at=expires_at,
        user_agent=http_request.headers.get("user-agent"),
        ip_address=http_request.client.host if http_request.client else None
    )
    
    logger.info(f"User logged in: {user.username}")

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=user.to_dict()
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh_token(
    request: RefreshTokenRequest,
    http_request: Request,
    db: Session = Depends(get_db)
):
    """Refresh access token using refresh token."""
    user_repo = UserRepository(db)

    # Verify refresh token
    payload = JWTUtils.verify_refresh_token(request.refresh_token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if token exists in database and is not revoked
    db_token = user_repo.get_refresh_token(request.refresh_token)
    if not db_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token not found or revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user
    user_id = payload.get("sub")
    user = user_repo.get_user_by_id(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Generate new tokens
    access_token = JWTUtils.create_access_token(
        user_id=user.id,
        username=user.username,
        role=user.role.value
    )

    new_refresh_token, expires_at = JWTUtils.create_refresh_token(
        user_id=user.id,
        username=user.username
    )

    # Revoke old refresh token and create new one
    user_repo.revoke_refresh_token(request.refresh_token)
    user_repo.create_refresh_token(
        user_id=user.id,
        token=new_refresh_token,
        expires_at=expires_at,
        user_agent=http_request.headers.get("user-agent"),
        ip_address=http_request.client.host if http_request.client else None
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        user=user.to_dict()
    )


@router.post("/logout", response_model=MessageResponse)
def logout(
    request: RefreshTokenRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Logout user by revoking refresh token."""
    user_repo = UserRepository(db)

    # Revoke refresh token
    user_repo.revoke_refresh_token(request.refresh_token)

    logger.info(f"User logged out: {current_user.username}")

    return MessageResponse(message="Logged out successfully")


@router.get("/me", response_model=UserResponse)
def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user information."""
    return UserResponse(**current_user.to_dict())


@router.post("/change-password", response_model=MessageResponse)
def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Change user password."""
    user_repo = UserRepository(db)

    # Validate new password strength
    is_valid, error_msg = PasswordUtils.validate_password_strength(request.new_password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg
        )

    # Change password
    success = user_repo.change_password(
        user_id=current_user.id,
        old_password=request.old_password,
        new_password=request.new_password
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect old password"
        )

    logger.info(f"Password changed for user: {current_user.username}")

    return MessageResponse(message="Password changed successfully")


@router.get("/users", response_model=List[UserResponse])
def list_users(
    _admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """List all users (admin only)."""
    user_repo = UserRepository(db)
    users = user_repo.get_all_users()

    return [UserResponse(**user.to_dict()) for user in users]


@router.delete("/users/{user_id}", response_model=MessageResponse)
def delete_user(
    user_id: str,
    admin_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """Delete user (admin only)."""
    user_repo = UserRepository(db)

    # Prevent admin from deleting themselves
    if str(admin_user.id) == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )

    success = user_repo.delete_user(user_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return MessageResponse(message="User deleted successfully")


# ===== User Preferences Endpoints =====

@router.get("/preferences", response_model=PreferencesResponse)
def get_preferences(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get current user's preferences."""
    user_repo = UserRepository(db)
    preferences = user_repo.get_user_preferences(current_user.id)

    return PreferencesResponse(preferences=preferences)


@router.put("/preferences", response_model=PreferencesResponse)
def update_preferences(
    request: UpdatePreferencesRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update entire user preferences."""
    user_repo = UserRepository(db)
    preferences = user_repo.update_user_preferences(current_user.id, request.preferences)

    if preferences is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    logger.info(f"Updated preferences for user: {current_user.username}")
    return PreferencesResponse(preferences=preferences)


@router.patch("/preferences/chat", response_model=PreferencesResponse)
def update_chat_preferences(
    request: UpdateChatPreferencesRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update chat-specific preferences (workspace and document selections)."""
    user_repo = UserRepository(db)

    chat_prefs = {
        "selectedWorkspaceIds": request.selectedWorkspaceIds,
        "selectedDocumentIds": request.selectedDocumentIds
    }

    preferences = user_repo.update_chat_preferences(current_user.id, chat_prefs)

    if preferences is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    logger.info(f"Updated chat preferences for user: {current_user.username}")
    return PreferencesResponse(preferences=preferences)


@router.get("/preferences/chat")
def get_chat_preferences(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get chat-specific preferences."""
    user_repo = UserRepository(db)
    chat_prefs = user_repo.get_chat_preferences(current_user.id)

    return {
        "chat": chat_prefs,
        "success": True
    }

