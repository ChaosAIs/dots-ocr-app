"""
Integration bridges to existing services.

This module provides bridge classes that connect the new agentic
system to existing services in the application.
"""

from agents.integration.document_router import DocumentRouterBridge
from agents.integration.schema_service_bridge import SchemaServiceBridge
from agents.integration.sql_generator_bridge import SQLGeneratorBridge
from agents.integration.vector_bridge import VectorServiceBridge
from agents.integration.graph_bridge import GraphServiceBridge
from agents.integration.formatter_bridge import FormatterBridge

__all__ = [
    "DocumentRouterBridge",
    "SchemaServiceBridge",
    "SQLGeneratorBridge",
    "VectorServiceBridge",
    "GraphServiceBridge",
    "FormatterBridge",
]
