"""
Database connection and session management for PostgreSQL.
"""
import os
import logging
from contextlib import contextmanager
from typing import Generator
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Database configuration from environment
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "6400")
POSTGRES_DB = os.getenv("POSTGRES_DB", "dots_ocr")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

# URL-encode the password to handle special characters
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{quote_plus(POSTGRES_PASSWORD)}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Create engine with connection pooling
_engine = None
_SessionLocal = None


def init_db():
    """Initialize database connection."""
    global _engine, _SessionLocal
    
    if _engine is not None:
        return
    
    logger.info(f"Connecting to PostgreSQL at {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}")
    
    _engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=10,  # Increased from 5 to handle concurrent chat sessions
        max_overflow=20,  # Increased from 10 to handle burst traffic
        pool_pre_ping=True,
        pool_recycle=3600,  # Recycle connections after 1 hour to prevent stale connections
        echo=False,
    )
    
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    
    # Test connection
    try:
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection successful")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise


def close_db():
    """Close database connection."""
    global _engine, _SessionLocal
    
    if _engine is not None:
        _engine.dispose()
        _engine = None
        _SessionLocal = None
        logger.info("Database connection closed")


def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    if _SessionLocal is None:
        init_db()
    
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for database session."""
    if _SessionLocal is None:
        init_db()

    db = _SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_db_session() -> Session:
    """
    Create a new database session (caller is responsible for closing it).
    Use get_db_session() context manager instead when possible.
    """
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()

