"""
Content Ingestion Pipeline for Book Content

This module implements the content ingestion pipeline that processes
book content and stores it in the vector database for RAG retrieval.
"""

import asyncio
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import aiofiles
from bs4 import BeautifulSoup
import markdown

# Import our modules
from backend.rag import get_rag_pipeline, DocumentChunk
from backend.database import get_database
from backend.qdrant_client import QdrantClientWrapper

logger = logging.getLogger(__name__)


class ContentIngestor:
    """Main class for ingesting book content into the RAG system"""

    def __init__(self):
        self.rag_pipeline = None
        self.db = None
        self.qdrant_client = None

    async def initialize(self):
        """Initialize all required components"""
        self.rag_pipeline = await get_rag_pipeline()
        self.db = await get_database()
        self.qdrant_client = self.rag_pipeline.qdrant_client

    async def ingest_book_directory(self, directory_path: str, recursive: bool = True) -> bool:
        """
        Ingest all content from a book directory.

        Args:
            directory_path: Path to the book content directory
            recursive: Whether to process subdirectories

        Returns:
            bool: True if ingestion was successful
        """
        try:
            directory = Path(directory_path)
            if not directory.exists():
                logger.error(f"Directory does not exist: {directory_path}")
                return False

            # Get all markdown files in the directory
            if recursive:
                md_files = list(directory.rglob("*.md"))
            else:
                md_files = list(directory.glob("*.md"))

            logger.info(f"Found {len(md_files)} markdown files to process")

            # Process each file
            success_count = 0
            for md_file in md_files:
                try:
                    success = await self.ingest_single_file(str(md_file))
                    if success:
                        success_count += 1
                except Exception as e:
                    logger.error(f"Error ingesting file {md_file}: {e}")

            logger.info(f"Successfully ingested {success_count}/{len(md_files)} files")
            return success_count == len(md_files)

        except Exception as e:
            logger.error(f"Error in directory ingestion: {e}")
            return False

    async def ingest_single_file(self, file_path: str) -> bool:
        """
        Ingest a single content file.

        Args:
            file_path: Path to the content file

        Returns:
            bool: True if ingestion was successful
        """
        try:
            logger.info(f"Ingesting file: {file_path}")

            # Read the file content
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()

            # Extract title from the file (first heading)
            title = self._extract_title(content)

            # Generate content hash to check if already processed
            content_hash = hashlib.md5(content.encode()).hexdigest()

            # Check if content has already been processed
            existing_content = await self.db.get_processed_content(content_hash)
            if existing_content:
                logger.info(f"Content already processed: {file_path}")
                return True

            # Process the content into chunks
            chunks = await self._process_content(content, file_path, title)

            # Ingest chunks using RAG pipeline
            success = await self.rag_pipeline.ingest_document(
                content=content,
                source=file_path
            )

            if success:
                # Save to database to track processed content
                await self.db.save_book_content(
                    source_path=file_path,
                    section_title=title,
                    content_hash=content_hash,
                    embedding_id=f"{file_path}_ingested",
                    metadata={"title": title, "file_path": file_path}
                )

            logger.info(f"Completed ingestion for: {file_path}")
            return success

        except Exception as e:
            logger.error(f"Error ingesting single file {file_path}: {e}")
            return False

    def _extract_title(self, content: str) -> str:
        """
        Extract the title from content (first heading).

        Args:
            content: The content to extract title from

        Returns:
            str: The extracted title
        """
        lines = content.split('\n')
        for line in lines:
            # Look for markdown heading
            if line.strip().startswith('# '):
                return line.strip()[2:]  # Remove '# ' prefix
            elif line.strip().startswith('## '):
                return line.strip()[3:]  # Remove '## ' prefix
            elif line.strip().startswith('### '):
                return line.strip()[4:]  # Remove '### ' prefix

        # If no heading found, use first line as title
        first_line = lines[0].strip() if lines else "Untitled"
        return first_line[:100]  # Limit to 100 characters

    async def _process_content(
        self,
        content: str,
        source_path: str,
        title: str,
        chunk_size: int = 512,
        overlap: int = 128
    ) -> List[DocumentChunk]:
        """
        Process content into chunks for ingestion.

        Args:
            content: The content to process
            source_path: Path of the source file
            title: Title of the content
            chunk_size: Size of each chunk
            overlap: Overlap between chunks

        Returns:
            List[DocumentChunk]: List of processed document chunks
        """
        # Convert markdown to plain text if needed
        plain_content = self._markdown_to_text(content)

        # Split content into sentences first
        sentences = self._split_into_sentences(plain_content)

        chunks = []
        current_chunk = ""
        chunk_num = 0

        for sentence in sentences:
            # Add sentence to current chunk
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + " "
            else:
                # Create a new chunk
                if current_chunk.strip():
                    chunk = DocumentChunk(
                        id=f"{source_path}_chunk_{chunk_num}",
                        content=current_chunk.strip(),
                        source=source_path,
                        section=title,
                    )
                    chunks.append(chunk)
                    chunk_num += 1

                # Start new chunk with overlap
                # Take the last few sentences from the previous chunk as overlap
                words = current_chunk.split()
                overlap_text = " ".join(words[-overlap:]) if len(words) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence + " "

        # Add the last chunk if it has content
        if current_chunk.strip():
            chunk = DocumentChunk(
                id=f"{source_path}_chunk_{chunk_num}",
                content=current_chunk.strip(),
                source=source_path,
                section=title,
            )
            chunks.append(chunk)

        logger.info(f"Processed content into {len(chunks)} chunks from {source_path}")
        return chunks

    def _markdown_to_text(self, markdown_content: str) -> str:
        """
        Convert markdown content to plain text.

        Args:
            markdown_content: Markdown content to convert

        Returns:
            str: Plain text content
        """
        try:
            # Convert markdown to HTML first
            html = markdown.markdown(markdown_content)
            # Then extract text from HTML
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text()
        except Exception:
            # If conversion fails, return original content
            logger.warning("Markdown conversion failed, returning original content")
            return markdown_content

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List[str]: List of sentences
        """
        import re

        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        # Remove empty strings and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    async def ingest_from_text(
        self,
        content: str,
        source: str = "manual_input",
        title: str = "Manual Content"
    ) -> bool:
        """
        Ingest content from a text string.

        Args:
            content: The content to ingest
            source: Source identifier
            title: Title of the content

        Returns:
            bool: True if ingestion was successful
        """
        try:
            logger.info(f"Ingesting content from: {source}")

            # Generate content hash to check if already processed
            content_hash = hashlib.md5(content.encode()).hexdigest()

            # Check if content has already been processed
            existing_content = await self.db.get_processed_content(content_hash)
            if existing_content:
                logger.info(f"Content already processed: {source}")
                return True

            # Process the content into chunks
            chunks = await self._process_content(content, source, title)

            # Ingest chunks using RAG pipeline
            success = await self.rag_pipeline.ingest_document(
                content=content,
                source=source
            )

            if success:
                # Save to database to track processed content
                await self.db.save_book_content(
                    source_path=source,
                    section_title=title,
                    content_hash=content_hash,
                    embedding_id=f"{source}_ingested",
                    metadata={"title": title, "source": source}
                )

            logger.info(f"Completed ingestion for: {source}")
            return success

        except Exception as e:
            logger.error(f"Error ingesting text content: {e}")
            return False

    async def rebuild_index(self, directory_path: str) -> bool:
        """
        Rebuild the entire content index from scratch.

        Args:
            directory_path: Path to the book content directory

        Returns:
            bool: True if rebuilding was successful
        """
        try:
            logger.info("Starting index rebuild...")

            # Clear existing content from database (optional)
            # This would involve deleting processed content records

            # Clear Qdrant collection (optional)
            # await self.qdrant_client.delete_collection()

            # Reinitialize Qdrant client to recreate collection
            await self.qdrant_client.init_client()

            # Process all content
            success = await self.ingest_book_directory(directory_path)

            logger.info("Index rebuild completed")
            return success

        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            return False


async def get_ingestor() -> ContentIngestor:
    """Get or create the global content ingestor instance"""
    ingestor = ContentIngestor()
    await ingestor.initialize()
    return ingestor


# Example usage and testing function
async def test_ingestion():
    """Test function to verify ingestion functionality"""
    try:
        # Create ingestor
        ingestor = await get_ingestor()

        # Example: Ingest a sample content
        sample_content = """
        # Introduction to Physical AI

        Physical AI is an emerging field that combines artificial intelligence with physical systems.

        ## Key Concepts

        The main concepts in Physical AI include:
        - Embodied Intelligence
        - Sensorimotor Learning
        - Real-world Interaction

        These concepts enable AI systems to interact with the physical world effectively.
        """

        success = await ingestor.ingest_from_text(
            content=sample_content,
            source="test_content",
            title="Introduction to Physical AI"
        )

        print(f"Ingestion test result: {success}")

    except Exception as e:
        print(f"Error in ingestion test: {e}")


if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_ingestion())