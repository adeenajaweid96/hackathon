"""
Database Module for Neon Postgres Connection

This module provides database connection and operations for the RAG chatbot system
using Neon Postgres as the database provider.
"""

import asyncio
import asyncpg
import logging
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class Database:
    """Database class for Neon Postgres operations"""

    def __init__(
        self,
        host: str = os.getenv("NEON_HOST", "ep-icy-salad-123456.us-east-1.aws.neon.tech"),
        database: str = os.getenv("NEON_DATABASE", "neondb"),
        user: str = os.getenv("NEON_USER", "neondb_owner"),
        password: str = os.getenv("NEON_PASSWORD", ""),
        port: int = int(os.getenv("NEON_PORT", 5432)),
        ssl_required: bool = True
    ):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.ssl_required = ssl_required
        self.pool = None

    async def connect(self):
        """Establish connection to the database"""
        try:
            # Create connection pool
            dsn = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

            # Add SSL options if required
            if self.ssl_required:
                self.pool = await asyncpg.create_pool(
                    dsn,
                    command_timeout=60,
                    ssl="require" if self.ssl_required else None,
                    min_size=1,
                    max_size=10
                )
            else:
                self.pool = await asyncpg.create_pool(
                    dsn,
                    command_timeout=60,
                    min_size=1,
                    max_size=10
                )

            logger.info(f"Connected to Neon Postgres database: {self.database}")

            # Initialize tables if they don't exist
            await self._initialize_tables()

        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    async def _initialize_tables(self):
        """Initialize required tables if they don't exist"""
        async with self.pool.acquire() as conn:
            # Create conversations table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB DEFAULT '{}'
                )
            """)

            # Create messages table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    role VARCHAR(50) NOT NULL, -- 'user' or 'assistant'
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    sources JSONB DEFAULT '[]',
                    confidence FLOAT DEFAULT 0.0,
                    FOREIGN KEY (session_id) REFERENCES conversations(session_id) ON DELETE CASCADE
                )
            """)

            # Create book_content table (for storing processed book content metadata)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS book_content (
                    id SERIAL PRIMARY KEY,
                    source_path VARCHAR(500) NOT NULL,
                    section_title VARCHAR(500),
                    content_hash VARCHAR(255) UNIQUE,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding_id VARCHAR(255), -- Reference to Qdrant ID
                    metadata JSONB DEFAULT '{}'
                )
            """)

            # Create user preferences table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) UNIQUE NOT NULL,
                    preferences JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            logger.info("Database tables initialized successfully")

    async def close(self):
        """Close the database connection"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection closed")

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool"""
        conn = await self.pool.acquire()
        try:
            yield conn
        finally:
            await self.pool.release(conn)

    async def save_conversation(self, session_id: str, metadata: Optional[Dict] = None) -> bool:
        """Save or update a conversation session"""
        try:
            async with self.get_connection() as conn:
                # Check if session exists
                existing = await conn.fetchrow(
                    "SELECT id FROM conversations WHERE session_id = $1",
                    session_id
                )

                if existing:
                    # Update existing session
                    await conn.execute(
                        """
                        UPDATE conversations
                        SET updated_at = CURRENT_TIMESTAMP, metadata = $2
                        WHERE session_id = $1
                        """,
                        session_id, metadata or {}
                    )
                else:
                    # Create new session
                    await conn.execute(
                        """
                        INSERT INTO conversations (session_id, metadata)
                        VALUES ($1, $2)
                        """,
                        session_id, metadata or {}
                    )

                logger.info(f"Conversation session saved: {session_id}")
                return True

        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return False

    async def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[str]] = None,
        confidence: float = 0.0
    ) -> bool:
        """Save a message to the conversation"""
        try:
            # Ensure conversation exists
            await self.save_conversation(session_id)

            async with self.get_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO messages (session_id, role, content, sources, confidence)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    session_id, role, content, sources or [], confidence
                )

            logger.info(f"Message saved for session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving message: {e}")
            return False

    async def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get conversation history for a session"""
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch(
                    """
                    SELECT role, content, timestamp, sources, confidence
                    FROM messages
                    WHERE session_id = $1
                    ORDER BY timestamp ASC
                    LIMIT $2
                    """,
                    session_id, limit
                )

                messages = []
                for row in rows:
                    messages.append({
                        "role": row["role"],
                        "content": row["content"],
                        "timestamp": row["timestamp"],
                        "sources": row["sources"],
                        "confidence": row["confidence"]
                    })

                return messages

        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []

    async def get_recent_conversations(self, limit: int = 10) -> List[Dict]:
        """Get recent conversations"""
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch(
                    """
                    SELECT session_id, created_at, updated_at, metadata
                    FROM conversations
                    ORDER BY updated_at DESC
                    LIMIT $1
                    """,
                    limit
                )

                conversations = []
                for row in rows:
                    conversations.append({
                        "session_id": row["session_id"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "metadata": row["metadata"]
                    })

                return conversations

        except Exception as e:
            logger.error(f"Error getting recent conversations: {e}")
            return []

    async def save_book_content(
        self,
        source_path: str,
        section_title: str,
        content_hash: str,
        embedding_id: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Save book content metadata to track processed content"""
        try:
            async with self.get_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO book_content (source_path, section_title, content_hash, embedding_id, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (content_hash) DO UPDATE SET
                        updated_at = CURRENT_TIMESTAMP,
                        embedding_id = $4,
                        metadata = $5
                    """,
                    source_path, section_title, content_hash, embedding_id, metadata or {}
                )

                logger.info(f"Book content saved: {source_path}")
                return True

        except Exception as e:
            logger.error(f"Error saving book content: {e}")
            return False

    async def get_processed_content(self, content_hash: str) -> Optional[Dict]:
        """Check if content has been processed before"""
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT source_path, section_title, processed_at, embedding_id, metadata
                    FROM book_content
                    WHERE content_hash = $1
                    """,
                    content_hash
                )

                if row:
                    return {
                        "source_path": row["source_path"],
                        "section_title": row["section_title"],
                        "processed_at": row["processed_at"],
                        "embedding_id": row["embedding_id"],
                        "metadata": row["metadata"]
                    }

                return None

        except Exception as e:
            logger.error(f"Error getting processed content: {e}")
            return None

    async def save_user_preferences(self, user_id: str, preferences: Dict) -> bool:
        """Save user preferences"""
        try:
            async with self.get_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO user_preferences (user_id, preferences)
                    VALUES ($1, $2)
                    ON CONFLICT (user_id) DO UPDATE SET
                        preferences = $2,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    user_id, preferences
                )

                logger.info(f"User preferences saved for: {user_id}")
                return True

        except Exception as e:
            logger.error(f"Error saving user preferences: {e}")
            return False

    async def get_user_preferences(self, user_id: str) -> Dict:
        """Get user preferences"""
        try:
            async with self.get_connection() as conn:
                row = await conn.fetchrow(
                    "SELECT preferences FROM user_preferences WHERE user_id = $1",
                    user_id
                )

                if row:
                    return row["preferences"]
                else:
                    return {}

        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return {}

    async def health_check(self) -> bool:
        """Perform a database health check"""
        try:
            async with self.get_connection() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database instance
db_instance = None


async def get_database() -> Database:
    """Get or create the global database instance"""
    global db_instance
    if db_instance is None:
        db_instance = Database()
        await db_instance.connect()
    return db_instance


# Example usage and testing function
async def test_database():
    """Test function to verify database functionality"""
    db = Database()

    try:
        # Connect to database
        await db.connect()

        # Test health check
        health = await db.health_check()
        print(f"Database health: {health}")

        # Test saving a conversation
        session_id = "test_session_123"
        success = await db.save_conversation(session_id, {"test": True})
        print(f"Save conversation success: {success}")

        # Test saving a message
        msg_success = await db.save_message(
            session_id, "user", "Hello, world!", ["source1"], 0.95
        )
        print(f"Save message success: {msg_success}")

        # Test getting conversation history
        history = await db.get_conversation_history(session_id)
        print(f"Conversation history: {history}")

        # Close connection
        await db.close()

    except Exception as e:
        print(f"Error in test: {e}")


if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_database())