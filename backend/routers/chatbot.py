"""
Chatbot router for the RAG Chatbot API.

This router handles chatbot-related endpoints including
query processing, conversation history, and content retrieval.
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import uuid
from datetime import datetime

# Import our modules
from backend.rag import get_rag_pipeline, RAGResult
from backend.database import get_database

# Create router instance
router = APIRouter()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatQuery(BaseModel):
    """Request model for chat queries"""
    query: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    """Response model for chat queries"""
    response: str
    sources: List[str]
    confidence: float
    session_id: str
    timestamp: datetime


class SelectedTextQuery(BaseModel):
    """Request model for selected text queries"""
    selected_text: str
    question: str
    session_id: Optional[str] = None


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(query: ChatQuery):
    """
    Main chat endpoint that processes user queries using RAG.

    Args:
        query: The user's query with optional context and history

    Returns:
        ChatResponse: The response with sources and confidence score
    """
    logger.info(f"Received query: {query.query}")

    # Generate session ID if not provided
    session_id = query.session_id or str(uuid.uuid4())

    try:
        # Get RAG pipeline
        rag_pipeline = await get_rag_pipeline()

        # Process query using RAG
        rag_result: RAGResult = await rag_pipeline.query(query.query)

        # Get database instance
        db = await get_database()

        # Save user message
        await db.save_message(
            session_id=session_id,
            role="user",
            content=query.query,
            sources=[],
            confidence=1.0  # User input always has full confidence
        )

        # Save assistant response
        await db.save_message(
            session_id=session_id,
            role="assistant",
            content=rag_result.response,
            sources=rag_result.sources,
            confidence=rag_result.confidence
        )

        # Save/update conversation
        await db.save_conversation(session_id)

        # Create response
        response = ChatResponse(
            response=rag_result.response,
            sources=rag_result.sources,
            confidence=rag_result.confidence,
            session_id=session_id,
            timestamp=datetime.utcnow()
        )

        logger.info(f"Processed query successfully for session: {session_id}")
        return response

    except Exception as e:
        logger.error(f"Error processing chat query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/query-selected-text", response_model=ChatResponse)
async def query_selected_text_endpoint(selected_query: SelectedTextQuery):
    """
    Endpoint for querying selected text from the book.

    Args:
        selected_query: The selected text and question about it

    Returns:
        ChatResponse: The response with sources and confidence score
    """
    logger.info(f"Received selected text query: {selected_query.question}")

    # Generate session ID if not provided
    session_id = selected_query.session_id or str(uuid.uuid4())

    try:
        # Combine selected text with question to form the full query
        full_query = f"Based on this text: '{selected_query.selected_text}', {selected_query.question}"

        # Get RAG pipeline
        rag_pipeline = await get_rag_pipeline()

        # Process query using RAG
        rag_result: RAGResult = await rag_pipeline.query(full_query)

        # Get database instance
        db = await get_database()

        # Save user message (selected text query)
        await db.save_message(
            session_id=session_id,
            role="user",
            content=f"Selected text: {selected_query.selected_text}\nQuestion: {selected_query.question}",
            sources=[],
            confidence=1.0
        )

        # Save assistant response
        await db.save_message(
            session_id=session_id,
            role="assistant",
            content=rag_result.response,
            sources=rag_result.sources,
            confidence=rag_result.confidence
        )

        # Save/update conversation
        await db.save_conversation(session_id)

        # Create response
        response = ChatResponse(
            response=rag_result.response,
            sources=rag_result.sources,
            confidence=rag_result.confidence,
            session_id=session_id,
            timestamp=datetime.utcnow()
        )

        logger.info(f"Processed selected text query successfully for session: {session_id}")
        return response

    except Exception as e:
        logger.error(f"Error processing selected text query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing selected text query: {str(e)}")


@router.get("/conversations/{session_id}")
async def get_conversation_history(session_id: str):
    """
    Get conversation history for a specific session.

    Args:
        session_id: The session ID to retrieve history for

    Returns:
        List of messages in the conversation
    """
    try:
        # Get database instance
        db = await get_database()

        # Get conversation history
        history = await db.get_conversation_history(session_id)

        return {
            "session_id": session_id,
            "messages": history
        }

    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting conversation history: {str(e)}")


@router.get("/conversations")
async def get_recent_conversations(limit: int = 10):
    """
    Get recent conversations.

    Args:
        limit: Number of recent conversations to return

    Returns:
        List of recent conversations
    """
    try:
        # Get database instance
        db = await get_database()

        # Get recent conversations
        conversations = await db.get_recent_conversations(limit=limit)

        return {
            "conversations": conversations,
            "limit": limit
        }

    except Exception as e:
        logger.error(f"Error getting recent conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting recent conversations: {str(e)}")


@router.get("/health")
async def router_health():
    """Health check for the chatbot router"""
    # Check if RAG pipeline is available
    try:
        rag_pipeline = await get_rag_pipeline()
        rag_healthy = True
    except:
        rag_healthy = False

    # Check if database is available
    try:
        db = await get_database()
        db_healthy = await db.health_check()
    except:
        db_healthy = False

    return {
        "status": "healthy",
        "service": "chatbot router",
        "dependencies": {
            "rag_pipeline": "healthy" if rag_healthy else "unhealthy",
            "database": "healthy" if db_healthy else "unhealthy"
        }
    }