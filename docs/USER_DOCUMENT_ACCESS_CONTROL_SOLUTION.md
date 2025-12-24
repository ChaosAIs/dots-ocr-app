# User Document Access Control Solution

## Executive Summary

This document outlines a comprehensive solution for implementing role-based access control (RBAC) and document-level permissions for the Dots OCR application. The solution introduces a `user_documents` bridge table to manage granular permissions, enabling document sharing while maintaining security.

---

## Table of Contents

1. [Database Design](#1-database-design)
2. [Permission Model](#2-permission-model)
3. [Backend Implementation](#3-backend-implementation)
4. [Vector & Graph Search Filtering](#4-vector--graph-search-filtering)
5. [API Endpoints](#5-api-endpoints)
6. [Frontend UI Design](#6-frontend-ui-design)
7. [Migration Strategy](#7-migration-strategy)

---

## 1. Database Design

### 1.1 New Tables

#### `user_documents` (Bridge Table)

```sql
-- Migration: 012_create_user_documents_table.sql

-- Permission type enum
CREATE TYPE document_permission AS ENUM ('read', 'update', 'delete', 'share', 'full');

-- Access origin enum (how permission was granted)
CREATE TYPE permission_origin AS ENUM ('owner', 'shared', 'admin_granted', 'public');

CREATE TABLE user_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Foreign keys
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Permission details
    permissions document_permission[] NOT NULL DEFAULT '{read}',
    origin permission_origin NOT NULL DEFAULT 'shared',
    is_owner BOOLEAN NOT NULL DEFAULT FALSE,

    -- Sharing metadata
    shared_by UUID REFERENCES users(id) ON DELETE SET NULL,
    shared_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,  -- Optional: time-limited access

    -- Access tracking
    last_accessed_at TIMESTAMP WITH TIME ZONE,
    access_count INTEGER DEFAULT 0,

    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT unique_user_document UNIQUE (user_id, document_id),
    CONSTRAINT valid_owner_permissions CHECK (
        (is_owner = TRUE AND permissions @> '{full}') OR is_owner = FALSE
    )
);

-- Indexes for performance
CREATE INDEX idx_user_documents_user_id ON user_documents(user_id);
CREATE INDEX idx_user_documents_document_id ON user_documents(document_id);
CREATE INDEX idx_user_documents_user_doc ON user_documents(user_id, document_id);
CREATE INDEX idx_user_documents_owner ON user_documents(document_id) WHERE is_owner = TRUE;
CREATE INDEX idx_user_documents_expires ON user_documents(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX idx_user_documents_permissions ON user_documents USING GIN(permissions);

-- Trigger for updated_at
CREATE TRIGGER update_user_documents_updated_at
    BEFORE UPDATE ON user_documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

#### `document_share_invitations` (Optional: For Pending Invites)

```sql
CREATE TABLE document_share_invitations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    invited_by UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,

    -- Invitation target (either user_id OR email for external invite)
    invited_user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    invited_email VARCHAR(255),

    -- Proposed permissions
    permissions document_permission[] NOT NULL DEFAULT '{read}',

    -- Invitation status
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'accepted', 'declined', 'expired')),
    message TEXT,  -- Optional message from sender

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (CURRENT_TIMESTAMP + INTERVAL '7 days'),
    responded_at TIMESTAMP WITH TIME ZONE,

    CONSTRAINT valid_invitation_target CHECK (
        (invited_user_id IS NOT NULL) OR (invited_email IS NOT NULL)
    )
);

CREATE INDEX idx_share_invitations_user ON document_share_invitations(invited_user_id);
CREATE INDEX idx_share_invitations_email ON document_share_invitations(invited_email);
CREATE INDEX idx_share_invitations_document ON document_share_invitations(document_id);
CREATE INDEX idx_share_invitations_status ON document_share_invitations(status) WHERE status = 'pending';
```

### 1.2 Updates to Existing Tables

#### `documents` Table Updates

```sql
-- Add visibility column to documents
ALTER TABLE documents ADD COLUMN visibility VARCHAR(20) DEFAULT 'private'
    CHECK (visibility IN ('private', 'shared', 'public'));

-- Add owner reference (if not exists)
ALTER TABLE documents ADD COLUMN owner_id UUID REFERENCES users(id) ON DELETE SET NULL;

-- Index for owner queries
CREATE INDEX idx_documents_owner ON documents(owner_id);
CREATE INDEX idx_documents_visibility ON documents(visibility);
```

### 1.3 Entity Relationship Diagram

```
┌─────────────────┐         ┌─────────────────────┐         ┌─────────────────┐
│     users       │         │   user_documents    │         │   documents     │
├─────────────────┤         ├─────────────────────┤         ├─────────────────┤
│ id (PK)         │◄───────┤ user_id (FK)        │         │ id (PK)         │
│ username        │         │ document_id (FK)    ├────────►│ filename        │
│ email           │         │ permissions[]       │         │ owner_id (FK)   │
│ role            │         │ origin              │         │ visibility      │
│ ...             │         │ is_owner            │         │ ...             │
└─────────────────┘         │ shared_by (FK)      │         └─────────────────┘
        │                   │ shared_at           │                 │
        │                   │ expires_at          │                 │
        │                   │ last_accessed_at    │                 │
        │                   │ access_count        │                 │
        │                   └─────────────────────┘                 │
        │                                                           │
        │         ┌──────────────────────────────┐                  │
        │         │  document_share_invitations  │                  │
        │         ├──────────────────────────────┤                  │
        └────────►│ invited_by (FK)              │                  │
                  │ invited_user_id (FK)         │◄─────────────────┘
                  │ document_id (FK)             │
                  │ permissions[]                │
                  │ status                       │
                  └──────────────────────────────┘
```

---

## 2. Permission Model

### 2.1 Permission Types

| Permission | Description | Allows |
|------------|-------------|--------|
| `read` | View document | View content, chat with document, see in search results |
| `update` | Modify document | Edit markdown, re-index, update metadata |
| `delete` | Remove document | Delete document and all related data |
| `share` | Share with others | Grant read/update permissions to other users |
| `full` | Full access | All permissions (owner default) |

### 2.2 Permission Hierarchy

```
full (owner)
  ├── read
  ├── update
  ├── delete
  └── share
        └── Can grant: read, update (but NOT delete, share, or full)
```

### 2.3 Role-Based Access Matrix

| Action | Admin | Owner | Shared (read) | Shared (update) | Shared (share) | No Access |
|--------|-------|-------|---------------|-----------------|----------------|-----------|
| View document | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Chat with document | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| See in vector search | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| See in graph search | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| Edit markdown | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| Re-index document | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| Delete document | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Share document | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| Revoke access | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| View all documents | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Manage all shares | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |

### 2.4 Access Resolution Logic

```python
def check_document_access(user_id: UUID, document_id: UUID, required_permission: str) -> bool:
    """
    Access resolution order:
    1. Admin bypass - admins have full access to all documents
    2. Owner check - document owner has full access
    3. Explicit permission - check user_documents table
    4. Default deny - no record = no access
    """

    # 1. Admin bypass
    user = get_user(user_id)
    if user.role == 'admin':
        return True

    # 2. Check user_documents table
    user_doc = get_user_document(user_id, document_id)

    if user_doc is None:
        return False  # No access record = denied

    # 3. Check expiration
    if user_doc.expires_at and user_doc.expires_at < now():
        return False

    # 4. Check permission
    if 'full' in user_doc.permissions:
        return True

    return required_permission in user_doc.permissions
```

---

## 3. Backend Implementation

### 3.1 New Repository: `UserDocumentRepository`

```python
# backend/db/user_document_repository.py

from typing import List, Optional
from uuid import UUID
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

class UserDocumentRepository:
    """Repository for managing user-document access permissions."""

    def __init__(self, db: Session):
        self.db = db

    # ==================== CRUD Operations ====================

    def grant_access(
        self,
        user_id: UUID,
        document_id: UUID,
        permissions: List[str],
        origin: str = 'shared',
        shared_by: Optional[UUID] = None,
        expires_at: Optional[datetime] = None,
        is_owner: bool = False
    ) -> UserDocument:
        """Grant document access to a user."""
        pass

    def revoke_access(self, user_id: UUID, document_id: UUID) -> bool:
        """Remove user's access to a document."""
        pass

    def update_permissions(
        self,
        user_id: UUID,
        document_id: UUID,
        permissions: List[str]
    ) -> UserDocument:
        """Update existing permissions."""
        pass

    def set_owner(self, user_id: UUID, document_id: UUID) -> UserDocument:
        """Set user as document owner with full permissions."""
        pass

    # ==================== Query Operations ====================

    def get_user_document(self, user_id: UUID, document_id: UUID) -> Optional[UserDocument]:
        """Get specific user-document permission record."""
        pass

    def get_user_accessible_documents(
        self,
        user_id: UUID,
        include_expired: bool = False
    ) -> List[UUID]:
        """Get all document IDs user can access."""
        pass

    def get_document_users(self, document_id: UUID) -> List[UserDocument]:
        """Get all users with access to a document."""
        pass

    def get_document_owner(self, document_id: UUID) -> Optional[UUID]:
        """Get the owner user_id of a document."""
        pass

    # ==================== Permission Checks ====================

    def has_permission(
        self,
        user_id: UUID,
        document_id: UUID,
        permission: str
    ) -> bool:
        """Check if user has specific permission on document."""
        pass

    def has_any_access(self, user_id: UUID, document_id: UUID) -> bool:
        """Check if user has any access to document."""
        pass

    def get_accessible_document_ids_for_search(
        self,
        user_id: UUID,
        user_role: str
    ) -> Optional[List[UUID]]:
        """
        Get document IDs for search filtering.
        Returns None for admin (no filter needed).
        Returns list of accessible document IDs for regular users.
        """
        pass

    # ==================== Sharing Operations ====================

    def share_document(
        self,
        document_id: UUID,
        owner_id: UUID,
        target_user_id: UUID,
        permissions: List[str],
        expires_at: Optional[datetime] = None
    ) -> UserDocument:
        """Share document with another user."""
        pass

    def get_shared_with_me(self, user_id: UUID) -> List[DocumentWithPermission]:
        """Get documents shared with user (not owned)."""
        pass

    def get_shared_by_me(self, user_id: UUID) -> List[DocumentShareInfo]:
        """Get documents user has shared with others."""
        pass

    # ==================== Admin Operations ====================

    def admin_grant_access(
        self,
        admin_id: UUID,
        user_id: UUID,
        document_id: UUID,
        permissions: List[str]
    ) -> UserDocument:
        """Admin grants access to any document."""
        pass

    def get_all_document_permissions(
        self,
        document_id: UUID
    ) -> List[UserDocument]:
        """Admin view: all permissions for a document."""
        pass

    # ==================== Cleanup Operations ====================

    def cleanup_expired_permissions(self) -> int:
        """Remove expired permission records."""
        pass

    def transfer_ownership(
        self,
        document_id: UUID,
        current_owner_id: UUID,
        new_owner_id: UUID
    ) -> bool:
        """Transfer document ownership to another user."""
        pass
```

### 3.2 Permission Service Layer

```python
# backend/services/permission_service.py

class PermissionService:
    """Service for document permission operations."""

    def __init__(self, db: Session):
        self.db = db
        self.user_doc_repo = UserDocumentRepository(db)
        self.user_repo = UserRepository(db)
        self.doc_repo = DocumentRepository(db)

    async def check_access(
        self,
        user_id: UUID,
        document_id: UUID,
        required_permission: str = 'read'
    ) -> bool:
        """
        Main access check method.
        Handles admin bypass, owner check, and permission lookup.
        """
        # Get user for role check
        user = await self.user_repo.get_by_id(user_id)
        if not user:
            return False

        # Admin bypass
        if user.role == 'admin':
            return True

        # Check user_documents
        return self.user_doc_repo.has_permission(
            user_id, document_id, required_permission
        )

    async def get_accessible_documents(
        self,
        user_id: UUID,
        include_metadata: bool = True
    ) -> List[DocumentWithAccess]:
        """Get all documents user can access with their permissions."""
        user = await self.user_repo.get_by_id(user_id)

        if user.role == 'admin':
            # Admin sees all documents
            documents = await self.doc_repo.get_all()
            return [
                DocumentWithAccess(
                    document=doc,
                    permissions=['full'],
                    is_owner=False,
                    is_admin_access=True
                )
                for doc in documents
            ]

        # Regular user - get from user_documents
        user_docs = self.user_doc_repo.get_user_accessible_documents(user_id)
        # ... join with documents table

    async def share_document(
        self,
        sharer_id: UUID,
        document_id: UUID,
        target_user_id: UUID,
        permissions: List[str]
    ) -> ShareResult:
        """Share document with another user."""
        # Verify sharer has share permission
        if not await self.check_access(sharer_id, document_id, 'share'):
            raise PermissionDenied("You don't have permission to share this document")

        # Validate permissions (can't grant more than you have, except admin)
        user = await self.user_repo.get_by_id(sharer_id)
        if user.role != 'admin':
            # Sharers can only grant read/update, not delete/share/full
            allowed_to_grant = {'read', 'update'}
            invalid = set(permissions) - allowed_to_grant
            if invalid:
                raise InvalidPermission(f"Cannot grant permissions: {invalid}")

        # Create permission record
        return self.user_doc_repo.share_document(
            document_id=document_id,
            owner_id=sharer_id,
            target_user_id=target_user_id,
            permissions=permissions
        )

    async def on_document_created(
        self,
        document_id: UUID,
        creator_id: UUID
    ):
        """Called when a new document is created."""
        # Automatically grant full access to creator as owner
        self.user_doc_repo.set_owner(creator_id, document_id)

    async def on_document_deleted(self, document_id: UUID):
        """Called when a document is deleted."""
        # Cascade delete will handle user_documents cleanup
        pass
```

### 3.3 Dependency Injection for Route Protection

```python
# backend/auth/dependencies.py

from fastapi import Depends, HTTPException, status
from typing import Optional, List

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Extract and validate current user from JWT."""
    pass

def require_document_permission(permission: str = 'read'):
    """
    Dependency factory for document permission checks.

    Usage:
        @router.get("/documents/{document_id}")
        async def get_document(
            document_id: UUID,
            _: bool = Depends(require_document_permission('read'))
        ):
    """
    async def check_permission(
        document_id: UUID,
        current_user: User = Depends(get_current_user),
        permission_service: PermissionService = Depends(get_permission_service)
    ) -> bool:
        has_access = await permission_service.check_access(
            current_user.id,
            document_id,
            permission
        )
        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"You don't have {permission} permission for this document"
            )
        return True

    return check_permission

def require_admin():
    """Dependency to require admin role."""
    async def check_admin(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role != 'admin':
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        return current_user
    return check_admin
```

---

## 4. Vector & Graph Search Filtering

### 4.1 Search Filter Service

```python
# backend/rag_service/search_filter.py

from typing import List, Optional, Set
from uuid import UUID
from qdrant_client.models import Filter, FieldCondition, MatchAny

class SearchFilterService:
    """Service for building access-filtered search queries."""

    def __init__(self, permission_service: PermissionService):
        self.permission_service = permission_service

    async def get_accessible_sources(
        self,
        user_id: UUID,
        user_role: str
    ) -> Optional[Set[str]]:
        """
        Get set of source filenames user can access.
        Returns None for admin (no filtering needed).
        Returns empty set if user has no document access.
        """
        if user_role == 'admin':
            return None  # No filter for admin

        accessible_doc_ids = await self.permission_service.get_accessible_document_ids(
            user_id
        )

        if not accessible_doc_ids:
            return set()  # Empty set = no results

        # Get filenames from document IDs
        filenames = await self.doc_repo.get_filenames_by_ids(accessible_doc_ids)
        return set(filenames)

    def build_qdrant_filter(
        self,
        accessible_sources: Optional[Set[str]],
        additional_filters: Optional[dict] = None
    ) -> Optional[Filter]:
        """
        Build Qdrant filter for vector search.

        Args:
            accessible_sources: Set of accessible source filenames, or None for no filter
            additional_filters: Any other filters to apply

        Returns:
            Qdrant Filter object or None
        """
        if accessible_sources is None:
            # Admin - no source filter needed
            return self._build_additional_filters(additional_filters)

        if not accessible_sources:
            # Empty set - user has no access, return impossible filter
            return Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchAny(any=["__NO_ACCESS__"])
                    )
                ]
            )

        # Build source filter
        source_filter = FieldCondition(
            key="source",
            match=MatchAny(any=list(accessible_sources))
        )

        conditions = [source_filter]

        if additional_filters:
            conditions.extend(self._parse_additional_filters(additional_filters))

        return Filter(must=conditions)

    def build_neo4j_filter(
        self,
        accessible_sources: Optional[Set[str]]
    ) -> str:
        """
        Build Neo4j WHERE clause for graph search.

        Returns Cypher WHERE clause string.
        """
        if accessible_sources is None:
            return ""  # No filter for admin

        if not accessible_sources:
            return "WHERE false"  # No access

        sources_list = ", ".join([f"'{s}'" for s in accessible_sources])
        return f"WHERE n.source IN [{sources_list}]"
```

### 4.2 Modified Vector Search

```python
# backend/rag_service/vectorstore.py (modifications)

class VectorStoreManager:

    async def search_with_access_filter(
        self,
        query: str,
        user_id: UUID,
        user_role: str,
        top_k: int = 10,
        score_threshold: float = 0.5
    ) -> List[SearchResult]:
        """
        Perform vector search with access filtering.
        """
        # Get accessible sources
        filter_service = SearchFilterService(self.permission_service)
        accessible_sources = await filter_service.get_accessible_sources(
            user_id, user_role
        )

        # Handle no access case
        if accessible_sources is not None and len(accessible_sources) == 0:
            return []

        # Build filter
        qdrant_filter = filter_service.build_qdrant_filter(accessible_sources)

        # Perform search with filter
        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=self.embed_query(query),
            query_filter=qdrant_filter,
            limit=top_k,
            score_threshold=score_threshold
        )

        return self._process_results(results)

    async def get_retriever_with_access_filter(
        self,
        user_id: UUID,
        user_role: str,
        **kwargs
    ) -> VectorStoreRetriever:
        """Get retriever that respects user's document access."""
        accessible_sources = await self.filter_service.get_accessible_sources(
            user_id, user_role
        )

        search_kwargs = kwargs.copy()
        if accessible_sources is not None:
            search_kwargs['filter'] = self.filter_service.build_qdrant_filter(
                accessible_sources
            )

        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
```

### 4.3 Modified Graph Search

```python
# backend/rag_service/graph_rag/graph_rag.py (modifications)

class GraphRAG:

    async def query_with_access_filter(
        self,
        query: str,
        user_id: UUID,
        user_role: str,
        mode: QueryMode = QueryMode.AUTO
    ) -> GraphRAGResponse:
        """
        Query graph with access filtering.
        """
        # Get accessible sources
        filter_service = SearchFilterService(self.permission_service)
        accessible_sources = await filter_service.get_accessible_sources(
            user_id, user_role
        )

        # Handle no access case
        if accessible_sources is not None and len(accessible_sources) == 0:
            return GraphRAGResponse(
                entities=[],
                relationships=[],
                chunks=[],
                message="No accessible documents found"
            )

        # Build Neo4j filter clause
        neo4j_filter = filter_service.build_neo4j_filter(accessible_sources)

        # Modified entity retrieval with filter
        entities = await self._get_entities_with_filter(query, neo4j_filter)
        relationships = await self._get_relationships_with_filter(
            entities, neo4j_filter
        )

        # ... rest of query processing

    async def _get_entities_with_filter(
        self,
        query: str,
        filter_clause: str
    ) -> List[Entity]:
        """Retrieve entities with access filter."""
        cypher = f"""
        MATCH (e:Entity)
        {filter_clause}
        WITH e, vector.similarity.cosine(e.embedding, $query_embedding) AS score
        WHERE score > $threshold
        RETURN e, score
        ORDER BY score DESC
        LIMIT $limit
        """
        # ... execute query
```

### 4.4 Chat Integration

```python
# backend/rag_service/chat_service.py (modifications)

class ChatService:

    async def process_message(
        self,
        session_id: UUID,
        user_id: UUID,
        user_role: str,
        message: str
    ) -> ChatResponse:
        """Process chat message with access-filtered retrieval."""

        # Get relevant chunks with access filtering
        chunks = await self.vectorstore.search_with_access_filter(
            query=message,
            user_id=user_id,
            user_role=user_role,
            top_k=self.retrieval_top_k
        )

        # Get graph context with access filtering (if enabled)
        if self.graph_rag_enabled:
            graph_context = await self.graph_rag.query_with_access_filter(
                query=message,
                user_id=user_id,
                user_role=user_role
            )

        # Build context from filtered results
        context = self._build_context(chunks, graph_context)

        # Generate response
        response = await self._generate_response(message, context)

        # Include source information (only from accessible documents)
        sources = self._extract_sources(chunks)

        return ChatResponse(
            content=response,
            sources=sources,
            session_id=session_id
        )
```

---

## 5. API Endpoints

### 5.1 Document Sharing Endpoints

```python
# backend/routes/document_sharing.py

from fastapi import APIRouter, Depends, HTTPException
from uuid import UUID
from typing import List

router = APIRouter(prefix="/api/documents", tags=["document-sharing"])

# ==================== Share Operations ====================

@router.post("/{document_id}/share")
async def share_document(
    document_id: UUID,
    request: ShareDocumentRequest,
    current_user: User = Depends(get_current_user),
    _: bool = Depends(require_document_permission('share'))
):
    """
    Share a document with another user.

    Request body:
    {
        "user_id": "uuid",           # Target user
        "permissions": ["read"],      # Permissions to grant
        "expires_at": "2024-12-31"   # Optional expiration
    }
    """
    pass

@router.delete("/{document_id}/share/{user_id}")
async def revoke_share(
    document_id: UUID,
    user_id: UUID,
    current_user: User = Depends(get_current_user),
    _: bool = Depends(require_document_permission('share'))
):
    """Revoke a user's access to a document."""
    pass

@router.get("/{document_id}/shares")
async def list_document_shares(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    _: bool = Depends(require_document_permission('share'))
) -> List[ShareInfo]:
    """List all users who have access to a document."""
    pass

@router.put("/{document_id}/share/{user_id}")
async def update_share_permissions(
    document_id: UUID,
    user_id: UUID,
    request: UpdateShareRequest,
    current_user: User = Depends(get_current_user),
    _: bool = Depends(require_document_permission('share'))
):
    """Update permissions for a shared user."""
    pass

# ==================== User's Shared Documents ====================

@router.get("/shared-with-me")
async def get_shared_with_me(
    current_user: User = Depends(get_current_user)
) -> List[SharedDocumentInfo]:
    """Get documents shared with current user."""
    pass

@router.get("/shared-by-me")
async def get_shared_by_me(
    current_user: User = Depends(get_current_user)
) -> List[MySharedDocumentInfo]:
    """Get documents current user has shared with others."""
    pass

@router.get("/my-documents")
async def get_my_documents(
    current_user: User = Depends(get_current_user)
) -> List[OwnedDocumentInfo]:
    """Get documents owned by current user."""
    pass

# ==================== Admin Operations ====================

@router.get("/admin/all-permissions", dependencies=[Depends(require_admin())])
async def admin_list_all_permissions(
    document_id: Optional[UUID] = None,
    user_id: Optional[UUID] = None
) -> List[PermissionRecord]:
    """Admin: List all permission records with optional filters."""
    pass

@router.post("/admin/grant-access", dependencies=[Depends(require_admin())])
async def admin_grant_access(
    request: AdminGrantAccessRequest,
    current_user: User = Depends(get_current_user)
):
    """Admin: Grant any user access to any document."""
    pass

@router.post("/admin/transfer-ownership", dependencies=[Depends(require_admin())])
async def admin_transfer_ownership(
    request: TransferOwnershipRequest,
    current_user: User = Depends(get_current_user)
):
    """Admin: Transfer document ownership between users."""
    pass
```

### 5.2 Request/Response Models

```python
# backend/models/sharing_models.py

from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID
from datetime import datetime

# ==================== Requests ====================

class ShareDocumentRequest(BaseModel):
    user_id: UUID
    permissions: List[str] = ["read"]  # read, update
    expires_at: Optional[datetime] = None
    message: Optional[str] = None  # Optional message to recipient

class UpdateShareRequest(BaseModel):
    permissions: List[str]
    expires_at: Optional[datetime] = None

class AdminGrantAccessRequest(BaseModel):
    user_id: UUID
    document_id: UUID
    permissions: List[str]
    expires_at: Optional[datetime] = None

class TransferOwnershipRequest(BaseModel):
    document_id: UUID
    current_owner_id: UUID
    new_owner_id: UUID

# ==================== Responses ====================

class ShareInfo(BaseModel):
    user_id: UUID
    username: str
    email: str
    permissions: List[str]
    shared_at: datetime
    shared_by: UUID
    expires_at: Optional[datetime]
    is_owner: bool

class SharedDocumentInfo(BaseModel):
    document_id: UUID
    filename: str
    original_filename: str
    permissions: List[str]
    shared_by_username: str
    shared_at: datetime
    expires_at: Optional[datetime]

class MySharedDocumentInfo(BaseModel):
    document_id: UUID
    filename: str
    shared_with: List[ShareInfo]
    share_count: int

class OwnedDocumentInfo(BaseModel):
    document_id: UUID
    filename: str
    original_filename: str
    created_at: datetime
    shared_with_count: int
    visibility: str

class PermissionRecord(BaseModel):
    id: UUID
    user_id: UUID
    username: str
    document_id: UUID
    filename: str
    permissions: List[str]
    origin: str
    is_owner: bool
    shared_by: Optional[UUID]
    created_at: datetime
    expires_at: Optional[datetime]
```

### 5.3 Updated Document Endpoints

```python
# Modify existing document routes to include permission checks

@router.get("/documents")
async def list_documents(
    current_user: User = Depends(get_current_user),
    include_shared: bool = True
) -> List[DocumentWithAccess]:
    """
    List documents accessible to current user.
    - Admin sees all documents
    - Regular users see owned + shared documents
    """
    pass

@router.get("/documents/{document_id}")
async def get_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    _: bool = Depends(require_document_permission('read'))
) -> DocumentDetail:
    """Get document details (requires read permission)."""
    pass

@router.put("/documents/{document_id}")
async def update_document(
    document_id: UUID,
    request: UpdateDocumentRequest,
    current_user: User = Depends(get_current_user),
    _: bool = Depends(require_document_permission('update'))
):
    """Update document (requires update permission)."""
    pass

@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    _: bool = Depends(require_document_permission('delete'))
):
    """Delete document (requires delete permission)."""
    pass

@router.post("/upload")
async def upload_document(
    file: UploadFile,
    current_user: User = Depends(get_current_user)
):
    """
    Upload new document.
    Automatically grants owner (full) permissions to uploader.
    """
    # Create document
    document = await doc_repo.create(...)

    # Set uploader as owner
    await permission_service.on_document_created(document.id, current_user.id)

    return document
```

---

## 6. Frontend UI Design

### 6.1 Component Architecture

```
src/components/
├── documents/
│   ├── documentList.jsx          # Updated with access indicators
│   ├── DocumentCard.jsx          # NEW: Individual document with permissions
│   ├── ShareDialog.jsx           # NEW: Share document modal
│   ├── PermissionBadge.jsx       # NEW: Permission indicator badge
│   ├── SharedWithMe.jsx          # NEW: Shared documents section
│   └── DocumentAccessManager.jsx # NEW: Manage document permissions
├── sharing/
│   ├── ShareButton.jsx           # NEW: Trigger share dialog
│   ├── UserSearchSelect.jsx      # NEW: Search/select users to share with
│   ├── PermissionSelector.jsx    # NEW: Select permissions to grant
│   ├── SharedUsersList.jsx       # NEW: List users with access
│   └── ShareNotification.jsx     # NEW: Notification for new shares
├── admin/
│   ├── DocumentPermissionsAdmin.jsx # NEW: Admin permission management
│   ├── UserDocumentsAdmin.jsx       # NEW: View user's documents
│   └── OwnershipTransfer.jsx        # NEW: Transfer ownership UI
└── common/
    ├── PermissionGuard.jsx        # NEW: Conditional render by permission
    └── AccessDenied.jsx           # NEW: Access denied state
```

### 6.2 Document List UI Updates

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  📁 My Documents                                               [+ Upload]   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─ Filter: [All ▾] [My Documents] [Shared with Me] ──────────────────┐    │
│  │                                                                     │    │
│  │  Search: [                                         ] 🔍             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 📄 Financial_Report_Q4.pdf                                           │  │
│  │ ─────────────────────────────────────────────────────────────────── │  │
│  │ 👤 Owner: You                    📅 Dec 15, 2024                    │  │
│  │ 📊 Status: ✅ Indexed            📏 Size: 2.4 MB                    │  │
│  │                                                                      │  │
│  │ 🏷️ [Full Access]   Shared with: 3 users                            │  │
│  │                                                                      │  │
│  │ [View] [Edit] [Share] [Delete]                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 📄 Marketing_Strategy_2025.docx                    🔗 Shared        │  │
│  │ ─────────────────────────────────────────────────────────────────── │  │
│  │ 👤 Owner: john.doe@company.com   📅 Dec 10, 2024                    │  │
│  │ 📊 Status: ✅ Indexed            📏 Size: 1.8 MB                    │  │
│  │                                                                      │  │
│  │ 🏷️ [Read Only]   Shared by: John Doe                               │  │
│  │                                                                      │  │
│  │ [View] [Chat]                                                        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 📄 Technical_Specs_v2.pdf                          🔗 Shared        │  │
│  │ ─────────────────────────────────────────────────────────────────── │  │
│  │ 👤 Owner: jane.smith@company.com 📅 Dec 8, 2024                     │  │
│  │ 📊 Status: ✅ Indexed            📏 Size: 5.2 MB                    │  │
│  │                                                                      │  │
│  │ 🏷️ [Read + Edit]   Shared by: Jane Smith                           │  │
│  │                                                                      │  │
│  │ [View] [Edit] [Chat]                                                │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Share Dialog UI

```
┌─────────────────────────────────────────────────────────────────┐
│  Share Document                                            [✕]  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📄 Financial_Report_Q4.pdf                                     │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  Share with user:                                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 🔍 Search by name or email...                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 👤 john.doe@company.com (John Doe)              [Add]   │   │
│  │ 👤 jane.smith@company.com (Jane Smith)          [Add]   │   │
│  │ 👤 bob.wilson@company.com (Bob Wilson)          [Add]   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Selected: jane.smith@company.com                               │
│                                                                 │
│  Permissions:                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ [✓] Read    - View document and chat with it            │   │
│  │ [ ] Edit    - Modify document content                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Access expires: [Never ▾]                                      │
│                 [1 Week] [1 Month] [3 Months] [Custom Date]     │
│                                                                 │
│  Add message (optional):                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Here's the Q4 report for your review.                   │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  👥 Currently shared with:                                      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 👤 bob.wilson    [Read]     Shared Dec 10   [Revoke]    │   │
│  │ 👤 alice.jones   [Read+Edit] Shared Dec 8   [Revoke]    │   │
│  │ 👤 mike.brown    [Read]     Expires Jan 15  [Revoke]    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│                              [Cancel]  [Share Document]         │
└─────────────────────────────────────────────────────────────────┘
```

### 6.4 Permission Badge Component

```jsx
// PermissionBadge.jsx

const PermissionBadge = ({ permissions, isOwner, isShared, className }) => {
  const getBadgeConfig = () => {
    if (isOwner) {
      return {
        label: 'Full Access',
        color: 'bg-green-100 text-green-800',
        icon: '👑'
      };
    }

    if (permissions.includes('full')) {
      return {
        label: 'Full Access',
        color: 'bg-green-100 text-green-800',
        icon: '✨'
      };
    }

    const hasEdit = permissions.includes('update');
    const hasRead = permissions.includes('read');

    if (hasEdit && hasRead) {
      return {
        label: 'Read + Edit',
        color: 'bg-blue-100 text-blue-800',
        icon: '✏️'
      };
    }

    if (hasRead) {
      return {
        label: 'Read Only',
        color: 'bg-gray-100 text-gray-800',
        icon: '👁️'
      };
    }

    return {
      label: 'Limited',
      color: 'bg-yellow-100 text-yellow-800',
      icon: '⚠️'
    };
  };

  const config = getBadgeConfig();

  return (
    <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${config.color} ${className}`}>
      <span className="mr-1">{config.icon}</span>
      {config.label}
      {isShared && <span className="ml-1">🔗</span>}
    </span>
  );
};
```

### 6.5 Admin Document Permissions Panel

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🔐 Document Permissions Administration                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─ Filters ────────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  Document: [All Documents ▾]     User: [All Users ▾]                 │  │
│  │                                                                       │  │
│  │  Permission: [Any ▾]   Origin: [Any ▾]   Status: [Active ▾]         │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Document              │ User           │ Permissions │ Origin │ Actions│ │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │  Financial_Q4.pdf      │ john.doe      │ Full        │ Owner  │ [···] │  │
│  │  Financial_Q4.pdf      │ jane.smith    │ Read        │ Shared │ [···] │  │
│  │  Financial_Q4.pdf      │ bob.wilson    │ Read+Edit   │ Shared │ [···] │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │  Marketing_2025.docx   │ jane.smith    │ Full        │ Owner  │ [···] │  │
│  │  Marketing_2025.docx   │ john.doe      │ Read        │ Shared │ [···] │  │
│  ├───────────────────────────────────────────────────────────────────────┤  │
│  │  Tech_Specs_v2.pdf     │ bob.wilson    │ Full        │ Owner  │ [···] │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─ Quick Actions ──────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  [+ Grant Access]  [🔄 Transfer Ownership]  [🗑️ Bulk Revoke]         │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.6 Chat Interface Updates

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  💬 Chat with Documents                                                     │
├───────────────────────┬─────────────────────────────────────────────────────┤
│                       │                                                     │
│  📚 Available Docs    │  Conversation                                       │
│  ─────────────────    │  ────────────────────────────────────────────────── │
│                       │                                                     │
│  ┌─────────────────┐  │  🧑 You: What are the Q4 revenue projections?      │
│  │ 📄 Financial_Q4 │  │                                                     │
│  │    [Full] 👑    │  │  🤖 Assistant: Based on the Financial Report Q4,   │
│  │                 │  │  the revenue projections show...                    │
│  │ 📄 Marketing    │  │                                                     │
│  │    [Read] 🔗    │  │  📎 Sources:                                        │
│  │                 │  │  ├─ Financial_Report_Q4.pdf (page 12)              │
│  │ 📄 Tech_Specs   │  │  └─ Financial_Report_Q4.pdf (page 15)              │
│  │    [Read] 🔗    │  │                                                     │
│  └─────────────────┘  │  ────────────────────────────────────────────────── │
│                       │                                                     │
│  ⚠️ Note: Chat will  │  🧑 You: Compare this with the marketing budget     │
│  only search docs     │                                                     │
│  you have access to.  │  🤖 Assistant: Comparing the Q4 projections with   │
│                       │  the Marketing Strategy 2025 document...            │
│  ─────────────────    │                                                     │
│                       │  📎 Sources:                                        │
│  [Select documents    │  ├─ Financial_Report_Q4.pdf (page 12)              │
│   to include in       │  └─ Marketing_Strategy_2025.docx (page 5)          │
│   chat context]       │                                                     │
│                       │  ────────────────────────────────────────────────── │
│  ☐ Financial_Q4       │                                                     │
│  ☐ Marketing          │  ┌─────────────────────────────────────────────────┐│
│  ☐ Tech_Specs         │  │ Type your message...                      [Send]││
│                       │  └─────────────────────────────────────────────────┘│
└───────────────────────┴─────────────────────────────────────────────────────┘
```

### 6.7 Shared With Me Section

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  🔗 Shared With Me                                           [Mark All Read]│
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  📌 New Shares (2)                                                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 📄 Project_Roadmap_2025.pdf                                 🆕 NEW   │  │
│  │ ─────────────────────────────────────────────────────────────────── │  │
│  │ 👤 Shared by: john.doe@company.com                                   │  │
│  │ 📅 Shared: 2 hours ago                                              │  │
│  │ 💬 "Please review the roadmap and let me know your thoughts"        │  │
│  │                                                                      │  │
│  │ 🏷️ [Read + Edit]        Expires: Never                              │  │
│  │                                                                      │  │
│  │ [View Document] [Start Chat]                                        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Previously Shared                                                          │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 📄 Marketing_Strategy_2025.docx                                      │  │
│  │ 👤 Shared by: jane.smith  │ 📅 Dec 10, 2024  │ 🏷️ [Read Only]       │  │
│  │ [View] [Chat]                                                        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ 📄 Technical_Specs_v2.pdf                      ⏰ Expires: Jan 15    │  │
│  │ 👤 Shared by: bob.wilson  │ 📅 Dec 8, 2024   │ 🏷️ [Read + Edit]    │  │
│  │ [View] [Edit] [Chat]                                                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.8 State Management (Redux)

```javascript
// frontend/src/store/slices/documentAccessSlice.js

import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';

// Async thunks
export const fetchAccessibleDocuments = createAsyncThunk(
  'documentAccess/fetchAccessible',
  async (_, { getState }) => {
    const response = await documentService.getAccessibleDocuments();
    return response.data;
  }
);

export const shareDocument = createAsyncThunk(
  'documentAccess/share',
  async ({ documentId, userId, permissions, expiresAt }, { rejectWithValue }) => {
    try {
      const response = await documentService.shareDocument(
        documentId, userId, permissions, expiresAt
      );
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response.data);
    }
  }
);

export const revokeAccess = createAsyncThunk(
  'documentAccess/revoke',
  async ({ documentId, userId }) => {
    await documentService.revokeAccess(documentId, userId);
    return { documentId, userId };
  }
);

const documentAccessSlice = createSlice({
  name: 'documentAccess',
  initialState: {
    // Documents organized by access type
    ownedDocuments: [],
    sharedWithMe: [],
    allAccessible: [],  // Combined for search/chat

    // Permission lookup cache
    permissionCache: {},  // { [documentId]: { permissions: [], isOwner: bool } }

    // UI state
    loading: false,
    error: null,
    shareDialogOpen: false,
    selectedDocumentForShare: null,
  },
  reducers: {
    openShareDialog: (state, action) => {
      state.shareDialogOpen = true;
      state.selectedDocumentForShare = action.payload;
    },
    closeShareDialog: (state) => {
      state.shareDialogOpen = false;
      state.selectedDocumentForShare = null;
    },
    updatePermissionCache: (state, action) => {
      const { documentId, permissions, isOwner } = action.payload;
      state.permissionCache[documentId] = { permissions, isOwner };
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchAccessibleDocuments.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchAccessibleDocuments.fulfilled, (state, action) => {
        state.loading = false;
        state.ownedDocuments = action.payload.filter(d => d.isOwner);
        state.sharedWithMe = action.payload.filter(d => !d.isOwner);
        state.allAccessible = action.payload;

        // Update permission cache
        action.payload.forEach(doc => {
          state.permissionCache[doc.id] = {
            permissions: doc.permissions,
            isOwner: doc.isOwner
          };
        });
      })
      .addCase(shareDocument.fulfilled, (state, action) => {
        // Update local state with new share
        const docIndex = state.ownedDocuments.findIndex(
          d => d.id === action.payload.documentId
        );
        if (docIndex !== -1) {
          state.ownedDocuments[docIndex].sharedWith.push(action.payload);
        }
      });
  }
});

// Selectors
export const selectHasPermission = (documentId, permission) => (state) => {
  const cached = state.documentAccess.permissionCache[documentId];
  if (!cached) return false;

  // Owner or full access has all permissions
  if (cached.isOwner || cached.permissions.includes('full')) return true;

  return cached.permissions.includes(permission);
};

export const selectAccessibleDocumentIds = (state) =>
  state.documentAccess.allAccessible.map(d => d.id);

export const { openShareDialog, closeShareDialog, updatePermissionCache } =
  documentAccessSlice.actions;

export default documentAccessSlice.reducer;
```

### 6.9 Permission Guard Component

```jsx
// frontend/src/components/common/PermissionGuard.jsx

import { useSelector } from 'react-redux';
import { selectHasPermission } from '../../store/slices/documentAccessSlice';

/**
 * Conditionally renders children based on document permission.
 *
 * Usage:
 * <PermissionGuard documentId={docId} permission="update">
 *   <EditButton />
 * </PermissionGuard>
 */
const PermissionGuard = ({
  documentId,
  permission,
  children,
  fallback = null,
  adminBypass = true
}) => {
  const hasPermission = useSelector(selectHasPermission(documentId, permission));
  const userRole = useSelector(state => state.auth.user?.role);

  // Admin bypass
  if (adminBypass && userRole === 'admin') {
    return children;
  }

  if (hasPermission) {
    return children;
  }

  return fallback;
};

/**
 * Hook version for programmatic checks.
 */
export const useDocumentPermission = (documentId, permission) => {
  const hasPermission = useSelector(selectHasPermission(documentId, permission));
  const userRole = useSelector(state => state.auth.user?.role);

  return userRole === 'admin' || hasPermission;
};

export default PermissionGuard;
```

---

## 7. Migration Strategy

### 7.1 Migration Steps

```sql
-- Step 1: Create new tables and types
-- (Run 012_create_user_documents_table.sql)

-- Step 2: Add owner_id to documents if not exists
ALTER TABLE documents ADD COLUMN IF NOT EXISTS owner_id UUID REFERENCES users(id);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS visibility VARCHAR(20) DEFAULT 'private';

-- Step 3: Migrate existing documents
-- Option A: Assign all existing documents to a default admin user
UPDATE documents
SET owner_id = (SELECT id FROM users WHERE role = 'admin' LIMIT 1)
WHERE owner_id IS NULL;

-- Option B: Assign to creator based on created_by field
UPDATE documents
SET owner_id = created_by
WHERE owner_id IS NULL AND created_by IS NOT NULL;

-- Step 4: Create user_documents records for existing owners
INSERT INTO user_documents (user_id, document_id, permissions, origin, is_owner, created_at)
SELECT
    owner_id,
    id,
    ARRAY['full']::document_permission[],
    'owner',
    TRUE,
    COALESCE(created_at, CURRENT_TIMESTAMP)
FROM documents
WHERE owner_id IS NOT NULL
ON CONFLICT (user_id, document_id) DO NOTHING;

-- Step 5: Create indexes
CREATE INDEX IF NOT EXISTS idx_documents_owner ON documents(owner_id);
CREATE INDEX IF NOT EXISTS idx_documents_visibility ON documents(visibility);
```

### 7.2 Rollback Strategy

```sql
-- Rollback if needed
DROP TABLE IF EXISTS document_share_invitations;
DROP TABLE IF EXISTS user_documents;
DROP TYPE IF EXISTS document_permission;
DROP TYPE IF EXISTS permission_origin;

ALTER TABLE documents DROP COLUMN IF EXISTS owner_id;
ALTER TABLE documents DROP COLUMN IF EXISTS visibility;
```

### 7.3 Implementation Phases

#### Phase 1: Database & Backend Core (Week 1-2)
- [ ] Create migration files
- [ ] Implement `UserDocumentRepository`
- [ ] Implement `PermissionService`
- [ ] Add dependency injection for permission checks
- [ ] Update document upload to create owner record
- [ ] Update document delete to cascade permissions

#### Phase 2: Search Filtering (Week 2-3)
- [ ] Implement `SearchFilterService`
- [ ] Update vector search with access filtering
- [ ] Update graph search with access filtering
- [ ] Update chat service to pass user context
- [ ] Add tests for filtered search

#### Phase 3: API Endpoints (Week 3-4)
- [ ] Implement sharing endpoints
- [ ] Update existing document endpoints with permission checks
- [ ] Add admin endpoints
- [ ] API documentation

#### Phase 4: Frontend - Document List (Week 4-5)
- [ ] Add Redux slice for document access
- [ ] Update document list with permission badges
- [ ] Add filter for owned/shared documents
- [ ] Implement `PermissionGuard` component

#### Phase 5: Frontend - Sharing UI (Week 5-6)
- [ ] Implement share dialog
- [ ] User search/select component
- [ ] Permission selector
- [ ] Shared users list with revoke

#### Phase 6: Frontend - Chat & Admin (Week 6-7)
- [ ] Update chat to show accessible documents
- [ ] Add document context selector
- [ ] Admin permission management panel
- [ ] Ownership transfer UI

#### Phase 7: Testing & Polish (Week 7-8)
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Error handling improvements
- [ ] Documentation

---

## 8. Security Considerations

### 8.1 Permission Validation

1. **Always validate on backend** - Never trust frontend permission checks
2. **Use database constraints** - Enforce owner must have full permissions
3. **Audit logging** - Log all permission changes
4. **Rate limiting** - Prevent permission enumeration attacks

### 8.2 Data Leakage Prevention

1. **Filter at query level** - Never fetch unauthorized data
2. **Validate document sources in responses** - Ensure returned sources are accessible
3. **Sanitize error messages** - Don't reveal document existence to unauthorized users

### 8.3 Admin Safety

1. **Require re-authentication for sensitive operations**
2. **Log all admin actions**
3. **Implement approval workflow for bulk operations**

---

## 9. Future Enhancements

### 9.1 Potential Features

- **Group/Team sharing** - Share with groups instead of individuals
- **Public links** - Generate public read-only links with optional password
- **Permission templates** - Predefined permission sets (Viewer, Editor, Co-Owner)
- **Activity feed** - Track document access and modifications
- **Email notifications** - Notify users when documents are shared
- **Version-aware permissions** - Different permissions per document version

### 9.2 Scalability Considerations

- **Caching layer** - Redis cache for permission lookups
- **Batch operations** - Optimize bulk permission grants
- **Async permission propagation** - Queue permission updates for large shares

---

## Summary

This solution provides a comprehensive document access control system with:

1. **Bridge table design** (`user_documents`) for granular permission management
2. **Role-based access** with admin bypass and user-level permissions
3. **Permission types**: read, update, delete, share, full
4. **Filtered search** in both vector (Qdrant) and graph (Neo4j) databases
5. **Complete API** for sharing, revoking, and managing permissions
6. **Frontend components** for document management, sharing dialogs, and admin panels
7. **Migration strategy** for existing documents

The design ensures that:
- Users can only access documents explicitly shared with them
- Owners have full control over their documents
- Admins can manage all documents and permissions
- Search results respect user permissions
- Chat only retrieves from accessible documents
