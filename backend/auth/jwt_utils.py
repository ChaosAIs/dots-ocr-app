"""
JWT token generation and validation utilities.
"""
import os
import jwt
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from uuid import UUID

logger = logging.getLogger(__name__)


class JWTUtils:
    """Utility class for JWT token operations."""

    JWT_ALGORITHM = "HS256"

    @classmethod
    def _get_secret_key(cls) -> str:
        """Get JWT secret key from environment at runtime."""
        return os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")

    @classmethod
    def _get_access_token_expire_minutes(cls) -> int:
        """Get access token expiration minutes from environment at runtime."""
        return int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

    @classmethod
    def _get_refresh_token_expire_days(cls) -> int:
        """Get refresh token expiration days from environment at runtime."""
        return int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    
    @classmethod
    def create_access_token(
        cls,
        user_id: UUID,
        username: str,
        role: str,
        additional_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a JWT access token.
        
        Args:
            user_id: User ID
            username: Username
            role: User role
            additional_claims: Additional claims to include
            
        Returns:
            JWT token as string
        """
        now = datetime.utcnow()
        expires_at = now + timedelta(minutes=cls._get_access_token_expire_minutes())

        payload = {
            "sub": str(user_id),
            "username": username,
            "role": role,
            "exp": expires_at,
            "iat": now,
            "type": "access"
        }

        if additional_claims:
            payload.update(additional_claims)

        token = jwt.encode(payload, cls._get_secret_key(), algorithm=cls.JWT_ALGORITHM)
        return token
    
    @classmethod
    def create_refresh_token(
        cls,
        user_id: UUID,
        username: str
    ) -> Tuple[str, datetime]:
        """
        Create a JWT refresh token.
        
        Args:
            user_id: User ID
            username: Username
            
        Returns:
            Tuple of (token, expiration_datetime)
        """
        now = datetime.utcnow()
        expires_at = now + timedelta(days=cls._get_refresh_token_expire_days())

        payload = {
            "sub": str(user_id),
            "username": username,
            "exp": expires_at,
            "iat": now,
            "type": "refresh"
        }

        token = jwt.encode(payload, cls._get_secret_key(), algorithm=cls.JWT_ALGORITHM)
        return token, expires_at
    
    @classmethod
    def decode_token(cls, token: str) -> Optional[Dict[str, Any]]:
        """
        Decode and validate a JWT token.

        Args:
            token: JWT token

        Returns:
            Decoded payload or None if invalid
        """
        try:
            payload = jwt.decode(
                token,
                cls._get_secret_key(),
                algorithms=[cls.JWT_ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning(f"Token expired: {token[:20]}...")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {token[:20]}... - Error: {str(e)}")
            return None
    
    @classmethod
    def verify_access_token(cls, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify an access token.
        
        Args:
            token: JWT access token
            
        Returns:
            Decoded payload or None if invalid
        """
        payload = cls.decode_token(token)
        if payload and payload.get("type") == "access":
            return payload
        return None
    
    @classmethod
    def verify_refresh_token(cls, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify a refresh token.
        
        Args:
            token: JWT refresh token
            
        Returns:
            Decoded payload or None if invalid
        """
        payload = cls.decode_token(token)
        if payload and payload.get("type") == "refresh":
            return payload
        return None

