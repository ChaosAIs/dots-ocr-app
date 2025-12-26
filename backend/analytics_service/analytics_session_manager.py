"""
Analytics Session Manager

Manages analytics conversation sessions with Redis caching
and PostgreSQL persistence.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime, timedelta

from sqlalchemy.orm import Session as DBSession

from db.models import AnalyticsSession, AnalyticsState
from .redis_session_manager import RedisSessionManager
from .intent_classifier import IntentClassifier, IntentClassification, QueryIntent

logger = logging.getLogger(__name__)


class AnalyticsSessionManager:
    """
    Manages analytics conversation sessions.

    Features:
    - Redis caching for fast access
    - PostgreSQL persistence for durability
    - State machine management
    - Plan versioning and history
    """

    def __init__(self, db: DBSession, redis_manager: Optional[RedisSessionManager] = None):
        """
        Initialize the session manager.

        Args:
            db: SQLAlchemy database session
            redis_manager: Optional Redis session manager (creates one if not provided)
        """
        self.db = db
        self.redis = redis_manager or RedisSessionManager()
        self.intent_classifier = IntentClassifier()
        self.session_expiry_hours = int(os.getenv("ANALYTICS_SESSION_EXPIRY_HOURS", "24"))
        self.max_plan_iterations = int(os.getenv("ANALYTICS_MAX_PLAN_ITERATIONS", "5"))

    def get_or_create_session(
        self,
        chat_session_id: UUID,
        user_id: UUID,
        workspace_id: UUID,
        query: str
    ) -> Dict[str, Any]:
        """
        Get existing analytics session or create new one.

        Args:
            chat_session_id: Associated chat session ID
            user_id: User ID
            workspace_id: Workspace ID
            query: Initial query

        Returns:
            Session data dictionary
        """
        session_key = str(chat_session_id)

        # Check Redis cache first
        cached = self.redis.get_session(session_key)
        if cached and cached.get("state") not in ["COMPLETE", "EXPIRED", "ERROR"]:
            logger.debug(f"Found cached session: {session_key}")
            return cached

        # Check PostgreSQL
        db_session = self.db.query(AnalyticsSession).filter(
            AnalyticsSession.chat_session_id == chat_session_id,
            AnalyticsSession.state.notin_(["COMPLETE", "EXPIRED", "ERROR"])
        ).first()

        if db_session:
            session_data = db_session.to_dict()
            self.redis.set_session(session_key, session_data)
            logger.debug(f"Loaded session from DB: {session_key}")
            return session_data

        # Create new session
        new_session = AnalyticsSession(
            chat_session_id=chat_session_id,
            user_id=user_id,
            workspace_id=workspace_id,
            original_query=query,
            state=AnalyticsState.INITIAL.value,
            expires_at=datetime.utcnow() + timedelta(hours=self.session_expiry_hours)
        )
        self.db.add(new_session)
        self.db.commit()
        self.db.refresh(new_session)

        session_data = new_session.to_dict()
        self.redis.set_session(session_key, session_data)

        logger.info(f"Created new analytics session: {new_session.id}")
        return session_data

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session by ID.

        Args:
            session_id: Session ID (can be session ID or chat_session_id)

        Returns:
            Session data or None
        """
        # Try Redis first
        cached = self.redis.get_session(session_id)
        if cached:
            return cached

        # Try PostgreSQL
        try:
            uuid_id = UUID(session_id)
        except ValueError:
            return None

        # Try by session ID first, then by chat_session_id
        db_session = self.db.query(AnalyticsSession).filter(
            AnalyticsSession.id == uuid_id
        ).first()

        if not db_session:
            db_session = self.db.query(AnalyticsSession).filter(
                AnalyticsSession.chat_session_id == uuid_id
            ).order_by(AnalyticsSession.created_at.desc()).first()

        if db_session:
            session_data = db_session.to_dict()
            self.redis.set_session(session_id, session_data)
            return session_data

        return None

    def transition_state(
        self,
        session_id: str,
        new_state: AnalyticsState,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Transition session to new state.

        Args:
            session_id: Session ID
            new_state: Target state
            additional_data: Optional data to update

        Returns:
            Updated session data
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        old_state = session.get("state")
        logger.info(f"Session {session_id} transitioning: {old_state} -> {new_state.value}")

        # Update session
        session["state"] = new_state.value
        session["state_entered_at"] = datetime.utcnow().isoformat()
        session["updated_at"] = datetime.utcnow().isoformat()

        if additional_data:
            session.update(additional_data)

        # Update Redis
        self.redis.set_session(session_id, session)

        # Update PostgreSQL
        self._persist_to_db(session_id, {
            "state": new_state.value,
            "state_entered_at": datetime.utcnow(),
            **(additional_data or {})
        })

        # Publish real-time update
        self.redis.publish_update(session_id, "state_change", {
            "old_state": old_state,
            "new_state": new_state.value,
            "additional_data": additional_data
        })

        return session

    def classify_intent(
        self,
        session_id: str,
        query: str,
        available_schemas: Optional[List[str]] = None
    ) -> IntentClassification:
        """
        Classify the intent of a query.

        Args:
            session_id: Session ID
            query: User query
            available_schemas: Available schema types

        Returns:
            Intent classification result
        """
        classification = self.intent_classifier.classify(query, available_schemas)

        # Update session with classification
        self.update_session(session_id, {
            "intent_classification": classification.dict()
        })

        return classification

    def update_gathered_info(
        self,
        session_id: str,
        info_type: str,
        info_data: Any
    ) -> Dict[str, Any]:
        """
        Update gathered information in session.

        Args:
            session_id: Session ID
            info_type: Type of info (e.g., 'entities', 'time_range', 'metrics')
            info_data: The gathered information

        Returns:
            Updated session data
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        gathered_info = session.get("gathered_info", {})
        gathered_info[info_type] = info_data
        gathered_info["last_updated"] = datetime.utcnow().isoformat()

        return self.update_session(session_id, {"gathered_info": gathered_info})

    def add_clarification(
        self,
        session_id: str,
        question: str,
        answer: str
    ) -> Dict[str, Any]:
        """
        Add a clarification Q&A to gathered info.

        Args:
            session_id: Session ID
            question: The clarification question
            answer: User's answer

        Returns:
            Updated session data
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        gathered_info = session.get("gathered_info", {})
        clarifications = gathered_info.get("clarifications", [])
        clarifications.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.utcnow().isoformat()
        })
        gathered_info["clarifications"] = clarifications

        return self.update_session(session_id, {"gathered_info": gathered_info})

    def set_plan(
        self,
        session_id: str,
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Set or update the current plan.

        Args:
            session_id: Session ID
            plan: Plan data

        Returns:
            Updated session data
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        # Archive current plan if exists
        current_plan = session.get("current_plan")
        if current_plan:
            plan_history = session.get("plan_history", [])
            plan_history.append({
                "version": session.get("plan_version", 0),
                "plan": current_plan,
                "archived_at": datetime.utcnow().isoformat()
            })
            session["plan_history"] = plan_history

        # Set new plan
        new_version = session.get("plan_version", 0) + 1
        plan["version"] = new_version
        plan["created_at"] = datetime.utcnow().isoformat()

        return self.update_session(session_id, {
            "current_plan": plan,
            "plan_version": new_version,
            "plan_history": session.get("plan_history", [])
        })

    def update_execution_progress(
        self,
        session_id: str,
        current_step: int,
        total_steps: int,
        step_result: Optional[Any] = None,
        status: str = "in_progress"
    ) -> Dict[str, Any]:
        """
        Update execution progress.

        Args:
            session_id: Session ID
            current_step: Current step number
            total_steps: Total number of steps
            step_result: Result of current step
            status: Progress status

        Returns:
            Updated session data
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        progress = session.get("execution_progress", {
            "started_at": datetime.utcnow().isoformat(),
            "completed_results": []
        })

        progress["current_step"] = current_step
        progress["total_steps"] = total_steps
        progress["status"] = status
        progress["updated_at"] = datetime.utcnow().isoformat()

        if step_result is not None:
            progress["completed_results"].append({
                "step": current_step,
                "result": step_result,
                "completed_at": datetime.utcnow().isoformat()
            })

        result = self.update_session(session_id, {"execution_progress": progress})

        # Publish progress update
        self.redis.publish_update(session_id, "progress", {
            "current_step": current_step,
            "total_steps": total_steps,
            "status": status
        })

        return result

    def set_results(
        self,
        session_id: str,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Set cached results and mark session complete.

        Args:
            session_id: Session ID
            results: Query results

        Returns:
            Updated session data
        """
        return self.update_session(session_id, {
            "cached_results": results,
            "result_generated_at": datetime.utcnow().isoformat()
        })

    def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update session with arbitrary data.

        Args:
            session_id: Session ID
            updates: Data to update

        Returns:
            Updated session data
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        session.update(updates)
        session["updated_at"] = datetime.utcnow().isoformat()

        # Update Redis
        self.redis.set_session(session_id, session)

        # Update PostgreSQL
        self._persist_to_db(session_id, updates)

        return session

    def _persist_to_db(self, session_id: str, updates: Dict[str, Any]):
        """
        Persist updates to PostgreSQL.

        Args:
            session_id: Session ID
            updates: Data to persist
        """
        try:
            uuid_id = UUID(session_id)
        except ValueError:
            # session_id might be chat_session_id
            db_session = self.db.query(AnalyticsSession).filter(
                AnalyticsSession.chat_session_id == UUID(session_id)
            ).first()
            if db_session:
                uuid_id = db_session.id
            else:
                logger.warning(f"Session not found for persistence: {session_id}")
                return

        db_session = self.db.query(AnalyticsSession).filter(
            AnalyticsSession.id == uuid_id
        ).first()

        if not db_session:
            db_session = self.db.query(AnalyticsSession).filter(
                AnalyticsSession.chat_session_id == uuid_id
            ).first()

        if db_session:
            for key, value in updates.items():
                if hasattr(db_session, key):
                    setattr(db_session, key, value)
            db_session.updated_at = datetime.utcnow()
            self.db.commit()
        else:
            logger.warning(f"Session not found in DB: {session_id}")

    def recover_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Recover a session after disconnection.

        Args:
            session_id: Session ID

        Returns:
            Session data with recovery info, or None
        """
        session = self.get_session(session_id)
        if not session:
            return None

        state = session.get("state")

        # Add recovery context based on state
        recovery_info = {
            "recovered": True,
            "recovered_at": datetime.utcnow().isoformat(),
            "recovery_state": state
        }

        if state == AnalyticsState.QUESTIONING.value:
            pending_questions = session.get("gathered_info", {}).get("pending_questions", [])
            recovery_info["pending_question"] = pending_questions[0] if pending_questions else None

        elif state == AnalyticsState.REVIEWING.value:
            recovery_info["current_plan"] = session.get("current_plan")
            recovery_info["plan_version"] = session.get("plan_version")

        elif state == AnalyticsState.EXECUTING.value:
            progress = session.get("execution_progress", {})
            recovery_info["execution_progress"] = progress

        elif state == AnalyticsState.COMPLETE.value:
            recovery_info["cached_results"] = session.get("cached_results")

        session["recovery_info"] = recovery_info
        return session

    def cleanup_expired(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        now = datetime.utcnow()

        # Find expired sessions
        expired = self.db.query(AnalyticsSession).filter(
            AnalyticsSession.expires_at < now,
            AnalyticsSession.state.notin_(["EXPIRED"])
        ).all()

        count = 0
        for session in expired:
            session.state = AnalyticsState.EXPIRED.value
            self.redis.delete_session(str(session.chat_session_id))
            count += 1

        if count > 0:
            self.db.commit()
            logger.info(f"Cleaned up {count} expired analytics sessions")

        return count

    def get_user_sessions(
        self,
        user_id: UUID,
        include_completed: bool = False,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get all sessions for a user.

        Args:
            user_id: User ID
            include_completed: Whether to include completed sessions
            limit: Maximum number of sessions to return

        Returns:
            List of session data dictionaries
        """
        query = self.db.query(AnalyticsSession).filter(
            AnalyticsSession.user_id == user_id
        )

        if not include_completed:
            query = query.filter(
                AnalyticsSession.state.notin_(["COMPLETE", "EXPIRED", "ERROR"])
            )

        sessions = query.order_by(
            AnalyticsSession.updated_at.desc()
        ).limit(limit).all()

        return [s.to_dict() for s in sessions]
