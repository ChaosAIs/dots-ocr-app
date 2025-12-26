"""
Redis Session Manager for Analytics

Provides fast session state caching with automatic expiration
and pub/sub for real-time updates.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID

logger = logging.getLogger(__name__)


class RedisSessionManager:
    """
    Redis-backed session cache for analytics conversations.

    Provides:
    - Fast session state access (< 1ms)
    - Automatic expiration (TTL)
    - Pub/sub for real-time updates
    - Fallback to PostgreSQL on cache miss
    """

    def __init__(self):
        """Initialize Redis connection."""
        self.redis = None
        self.ttl = int(os.getenv("REDIS_SESSION_TTL", "86400"))
        self.prefix = os.getenv("REDIS_SESSION_PREFIX", "analytics:session:")
        self.pubsub_channel = os.getenv("REDIS_PUBSUB_CHANNEL", "analytics:updates")

        self._connect()

    def _connect(self):
        """Establish Redis connection."""
        try:
            import redis

            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", "6379"))
            db = int(os.getenv("REDIS_DB", "1"))
            password = os.getenv("REDIS_PASSWORD") or None

            self.redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )

            # Test connection
            self.redis.ping()
            logger.info(f"Redis session manager connected to {host}:{port} db={db}")

        except ImportError:
            logger.warning("Redis package not installed. Session caching disabled.")
            self.redis = None
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Session caching disabled.")
            self.redis = None

    @property
    def is_connected(self) -> bool:
        """Check if Redis is available."""
        if not self.redis:
            return False
        try:
            self.redis.ping()
            return True
        except Exception:
            return False

    def _key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"{self.prefix}{session_id}"

    def _serialize(self, data: Dict[str, Any]) -> str:
        """Serialize session data to JSON."""
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, UUID):
                return str(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        return json.dumps(data, default=json_serializer)

    def _deserialize(self, data: str) -> Dict[str, Any]:
        """Deserialize JSON to session data."""
        return json.loads(data)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session from Redis cache.

        Args:
            session_id: Session UUID as string

        Returns:
            Session data dict or None if not found
        """
        if not self.redis:
            return None

        try:
            data = self.redis.get(self._key(session_id))
            if data:
                return self._deserialize(data)
        except Exception as e:
            logger.error(f"Redis get error for session {session_id}: {e}")

        return None

    def set_session(
        self,
        session_id: str,
        session_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store session in Redis cache.

        Args:
            session_id: Session UUID as string
            session_data: Session data to cache
            ttl: Optional custom TTL in seconds

        Returns:
            True if successful, False otherwise
        """
        if not self.redis:
            return False

        try:
            self.redis.setex(
                self._key(session_id),
                ttl or self.ttl,
                self._serialize(session_data)
            )
            return True
        except Exception as e:
            logger.error(f"Redis set error for session {session_id}: {e}")
            return False

    def update_session_field(
        self,
        session_id: str,
        field: str,
        value: Any
    ) -> bool:
        """
        Update a single field in cached session.

        Args:
            session_id: Session UUID as string
            field: Field name to update
            value: New value for the field

        Returns:
            True if successful, False otherwise
        """
        session = self.get_session(session_id)
        if session:
            session[field] = value
            session["updated_at"] = datetime.utcnow().isoformat()
            return self.set_session(session_id, session)
        return False

    def update_session_fields(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update multiple fields in cached session.

        Args:
            session_id: Session UUID as string
            updates: Dictionary of field-value pairs to update

        Returns:
            True if successful, False otherwise
        """
        session = self.get_session(session_id)
        if session:
            session.update(updates)
            session["updated_at"] = datetime.utcnow().isoformat()
            return self.set_session(session_id, session)
        return False

    def delete_session(self, session_id: str) -> bool:
        """
        Remove session from cache.

        Args:
            session_id: Session UUID as string

        Returns:
            True if deleted, False otherwise
        """
        if not self.redis:
            return False

        try:
            self.redis.delete(self._key(session_id))
            return True
        except Exception as e:
            logger.error(f"Redis delete error for session {session_id}: {e}")
            return False

    def extend_ttl(self, session_id: str, additional_seconds: int = 3600) -> bool:
        """
        Extend session TTL.

        Args:
            session_id: Session UUID as string
            additional_seconds: Seconds to add to current TTL

        Returns:
            True if successful, False otherwise
        """
        if not self.redis:
            return False

        try:
            key = self._key(session_id)
            current_ttl = self.redis.ttl(key)
            if current_ttl > 0:
                self.redis.expire(key, current_ttl + additional_seconds)
                return True
        except Exception as e:
            logger.error(f"Redis TTL extend error for session {session_id}: {e}")

        return False

    def get_ttl(self, session_id: str) -> int:
        """
        Get remaining TTL for session.

        Args:
            session_id: Session UUID as string

        Returns:
            Remaining TTL in seconds, -1 if no expiry, -2 if not found
        """
        if not self.redis:
            return -2

        try:
            return self.redis.ttl(self._key(session_id))
        except Exception as e:
            logger.error(f"Redis TTL get error for session {session_id}: {e}")
            return -2

    def publish_update(
        self,
        session_id: str,
        event_type: str,
        data: Dict[str, Any]
    ) -> bool:
        """
        Publish real-time update for WebSocket clients.

        Args:
            session_id: Session UUID as string
            event_type: Type of event (e.g., 'state_change', 'progress', 'complete')
            data: Event data payload

        Returns:
            True if published, False otherwise
        """
        if not self.redis:
            return False

        try:
            message = self._serialize({
                "session_id": session_id,
                "event_type": event_type,
                "data": data,
                "timestamp": datetime.utcnow().isoformat()
            })
            self.redis.publish(self.pubsub_channel, message)
            return True
        except Exception as e:
            logger.error(f"Redis publish error: {e}")
            return False

    def get_active_sessions(self, pattern: str = "*") -> List[str]:
        """
        Get list of active session IDs matching pattern.

        Args:
            pattern: Redis key pattern (default: all sessions)

        Returns:
            List of session IDs
        """
        if not self.redis:
            return []

        try:
            keys = self.redis.keys(f"{self.prefix}{pattern}")
            return [key.replace(self.prefix, "") for key in keys]
        except Exception as e:
            logger.error(f"Redis keys error: {e}")
            return []

    def flush_expired(self) -> int:
        """
        Redis automatically handles expiration via TTL.
        This method is a no-op but provided for interface consistency.

        Returns:
            0 (Redis handles expiration automatically)
        """
        return 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get Redis connection and usage statistics.

        Returns:
            Dictionary with stats
        """
        if not self.redis:
            return {"connected": False, "error": "Redis not available"}

        try:
            info = self.redis.info("clients")
            keys_count = len(self.redis.keys(f"{self.prefix}*"))

            return {
                "connected": True,
                "host": os.getenv("REDIS_HOST", "localhost"),
                "port": int(os.getenv("REDIS_PORT", "6379")),
                "db": int(os.getenv("REDIS_DB", "1")),
                "active_sessions": keys_count,
                "connected_clients": info.get("connected_clients", 0),
                "default_ttl": self.ttl,
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}
