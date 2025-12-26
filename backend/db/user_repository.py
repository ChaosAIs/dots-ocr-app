"""
User repository for database operations.
"""
import logging
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy import or_

from db.models import User, UserRole, UserStatus, RefreshToken
from auth.password_utils import PasswordUtils
from auth.jwt_utils import JWTUtils

logger = logging.getLogger(__name__)


class UserRepository:
    """Repository for user database operations."""

    # Account locking configuration
    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 30

    def __init__(self, db: Session):
        """Initialize repository with database session."""
        self.db = db

    @staticmethod
    def normalize_name(name: str) -> str:
        """
        Normalize a name for filesystem-safe usage.

        - Converts to lowercase
        - Replaces spaces with underscores
        - Removes special characters (keeps only alphanumeric and underscores)
        - Limits to 100 characters
        - Returns 'user' if result is empty

        Args:
            name: The name to normalize

        Returns:
            Normalized, filesystem-safe name
        """
        if not name:
            return 'user'

        # Convert to lowercase and replace spaces
        normalized = name.lower().replace(' ', '_')
        # Remove special characters (keep alphanumeric and underscores)
        normalized = re.sub(r'[^a-z0-9_]', '', normalized)
        # Limit length
        normalized = normalized[:100]
        # Ensure not empty
        if not normalized:
            normalized = 'user'
        return normalized

    def _ensure_unique_normalized_username(self, normalized: str, exclude_user_id: Optional[UUID] = None) -> str:
        """
        Ensure normalized username is unique by appending a counter if needed.

        Args:
            normalized: The base normalized username
            exclude_user_id: User ID to exclude from uniqueness check (for updates)

        Returns:
            Unique normalized username
        """
        original = normalized
        counter = 1

        while True:
            query = self.db.query(User).filter(User.normalized_username == normalized)
            if exclude_user_id:
                query = query.filter(User.id != exclude_user_id)

            if not query.first():
                return normalized

            normalized = f"{original}_{counter}"
            counter += 1

            # Safety limit
            if counter > 1000:
                raise ValueError(f"Unable to generate unique normalized username for: {original}")

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: Optional[str] = None,
        role: UserRole = UserRole.USER,
        created_by: Optional[UUID] = None
    ) -> User:
        """Create a new user."""
        password_hash = PasswordUtils.hash_password(password)

        # Generate normalized username
        normalized_username = self.normalize_name(username)
        normalized_username = self._ensure_unique_normalized_username(normalized_username)

        user = User(
            username=username,
            normalized_username=normalized_username,
            email=email,
            password_hash=password_hash,
            full_name=full_name,
            role=role,
            status=UserStatus.ACTIVE,
            created_by=created_by
        )

        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)

        logger.info(f"Created user: {username} (normalized: {normalized_username}, ID: {user.id})")
        return user
    
    def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID."""
        return self.db.query(User).filter(
            User.id == user_id,
            User.deleted_at.is_(None)
        ).first()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.db.query(User).filter(
            User.username == username,
            User.deleted_at.is_(None)
        ).first()
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.db.query(User).filter(
            User.email == email,
            User.deleted_at.is_(None)
        ).first()
    
    def get_user_by_username_or_email(self, identifier: str) -> Optional[User]:
        """Get user by username or email."""
        return self.db.query(User).filter(
            or_(User.username == identifier, User.email == identifier),
            User.deleted_at.is_(None)
        ).first()
    
    def get_all_users(self, include_deleted: bool = False) -> List[User]:
        """Get all users."""
        query = self.db.query(User)
        if not include_deleted:
            query = query.filter(User.deleted_at.is_(None))
        return query.all()
    
    def update_user(self, user: User) -> User:
        """Update user."""
        user.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(user)
        return user
    
    def delete_user(self, user_id: UUID) -> bool:
        """Soft delete user."""
        user = self.get_user_by_id(user_id)
        if user:
            user.deleted_at = datetime.utcnow()
            user.status = UserStatus.INACTIVE
            self.db.commit()
            logger.info(f"Deleted user: {user.username} (ID: {user_id})")
            return True
        return False
    
    def authenticate_user(
        self,
        identifier: str,
        password: str,
        ip_address: Optional[str] = None
    ) -> Optional[User]:
        """
        Authenticate user with username/email and password.
        Implements account locking after failed attempts.
        """
        user = self.get_user_by_username_or_email(identifier)
        
        if not user:
            return None
        
        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.utcnow():
            logger.warning(f"Account locked: {user.username}")
            return None
        
        # Verify password
        if not PasswordUtils.verify_password(password, user.password_hash):
            # Increment failed attempts
            user.failed_login_attempts += 1
            
            # Lock account if max attempts reached
            if user.failed_login_attempts >= self.MAX_FAILED_ATTEMPTS:
                user.locked_until = datetime.utcnow() + timedelta(minutes=self.LOCKOUT_DURATION_MINUTES)
                logger.warning(f"Account locked due to failed attempts: {user.username}")
            
            self.db.commit()
            return None
        
        # Check if user is active
        if user.status != UserStatus.ACTIVE:
            logger.warning(f"Inactive user login attempt: {user.username}")
            return None
        
        # Successful login - reset failed attempts
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login_at = datetime.utcnow()
        user.last_login_ip = ip_address
        self.db.commit()
        
        logger.info(f"User authenticated: {user.username}")
        return user

    def create_refresh_token(
        self,
        user_id: UUID,
        token: str,
        expires_at: datetime,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> RefreshToken:
        """Create a refresh token."""
        refresh_token = RefreshToken(
            user_id=user_id,
            token=token,
            expires_at=expires_at,
            user_agent=user_agent,
            ip_address=ip_address
        )

        self.db.add(refresh_token)
        self.db.commit()
        self.db.refresh(refresh_token)

        return refresh_token

    def get_refresh_token(self, token: str) -> Optional[RefreshToken]:
        """Get refresh token by token string."""
        return self.db.query(RefreshToken).filter(
            RefreshToken.token == token,
            RefreshToken.revoked_at.is_(None),
            RefreshToken.expires_at > datetime.utcnow()
        ).first()

    def revoke_refresh_token(self, token: str) -> bool:
        """Revoke a refresh token."""
        refresh_token = self.db.query(RefreshToken).filter(
            RefreshToken.token == token
        ).first()

        if refresh_token:
            refresh_token.revoked_at = datetime.utcnow()
            self.db.commit()
            return True
        return False

    def revoke_all_user_tokens(self, user_id: UUID) -> int:
        """Revoke all refresh tokens for a user."""
        count = self.db.query(RefreshToken).filter(
            RefreshToken.user_id == user_id,
            RefreshToken.revoked_at.is_(None)
        ).update({"revoked_at": datetime.utcnow()})

        self.db.commit()
        return count

    def change_password(
        self,
        user_id: UUID,
        old_password: str,
        new_password: str
    ) -> bool:
        """Change user password."""
        user = self.get_user_by_id(user_id)

        if not user:
            return False

        # Verify old password
        if not PasswordUtils.verify_password(old_password, user.password_hash):
            return False

        # Update password
        user.password_hash = PasswordUtils.hash_password(new_password)
        user.password_changed_at = datetime.utcnow()
        self.db.commit()

        # Revoke all refresh tokens
        self.revoke_all_user_tokens(user_id)

        logger.info(f"Password changed for user: {user.username}")
        return True

    # ===== User Preferences Methods =====

    def get_user_preferences(self, user_id: UUID) -> Dict[str, Any]:
        """
        Get user preferences.

        Args:
            user_id: The user's UUID

        Returns:
            User preferences dictionary, or empty dict if not set
        """
        user = self.get_user_by_id(user_id)
        if not user:
            return {}
        return user.preferences or {}

    def update_user_preferences(self, user_id: UUID, preferences: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Replace entire user preferences.

        Args:
            user_id: The user's UUID
            preferences: New preferences dictionary

        Returns:
            Updated preferences, or None if user not found
        """
        user = self.get_user_by_id(user_id)
        if not user:
            return None

        user.preferences = preferences
        user.updated_at = datetime.utcnow()
        flag_modified(user, "preferences")
        self.db.commit()
        self.db.refresh(user)

        logger.info(f"Updated preferences for user: {user.username}")
        return user.preferences

    def update_preference_key(self, user_id: UUID, key: str, value: Any) -> Optional[Dict[str, Any]]:
        """
        Update a specific preference key while preserving other preferences.

        Args:
            user_id: The user's UUID
            key: The preference key to update (e.g., 'chat', 'ui')
            value: The value to set for this key

        Returns:
            Updated preferences, or None if user not found
        """
        user = self.get_user_by_id(user_id)
        if not user:
            return None

        # Get current preferences or initialize empty dict
        current_preferences = user.preferences or {}

        # Update the specific key
        current_preferences[key] = value

        # Save back to user - use copy to ensure SQLAlchemy detects the change
        user.preferences = dict(current_preferences)
        user.updated_at = datetime.utcnow()

        # Flag the JSONB column as modified to ensure SQLAlchemy tracks the change
        flag_modified(user, "preferences")

        self.db.commit()
        self.db.refresh(user)

        logger.info(f"Updated preference key '{key}' for user: {user.username}")
        return user.preferences

    def get_chat_preferences(self, user_id: UUID) -> Dict[str, Any]:
        """
        Get chat-specific preferences (convenience method).

        Args:
            user_id: The user's UUID

        Returns:
            Chat preferences dictionary
        """
        preferences = self.get_user_preferences(user_id)
        return preferences.get('chat', {})

    def update_chat_preferences(self, user_id: UUID, chat_preferences: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update chat-specific preferences (convenience method).

        Args:
            user_id: The user's UUID
            chat_preferences: Chat preferences to set

        Returns:
            Updated full preferences, or None if user not found
        """
        # Add timestamp to chat preferences
        chat_preferences['lastUpdated'] = datetime.utcnow().isoformat()
        return self.update_preference_key(user_id, 'chat', chat_preferences)

