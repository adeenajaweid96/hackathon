"""
RAG (Retrieval-Augmented Generation) Pipeline Implementation

This module implements the core RAG pipeline that retrieves relevant
book content and generates responses using LLMs.
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import required libraries
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "Please install required packages: "
        "pip install transformers torch sentence-transformers"
    )

from backend.qdrant_client import QdrantClientWrapper
from backend.database import Database

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata"""
    id: str
    content: str
    source: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    embedding: Optional[List[float]] = None


@dataclass
class RAGResult:
    """Result of a RAG query"""
    response: str
    sources: List[str]
    confidence: float
    retrieved_chunks: List[DocumentChunk]


class RAGPipeline:
    """Main RAG pipeline class that orchestrates retrieval and generation"""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "gpt-3.5-turbo",  # This would be configurable
        collection_name: str = "book_content"
    ):
        self.collection_name = collection_name
        self.qdrant_client = QdrantClientWrapper()
        self.db = Database()

        # Initialize sentence transformer for embeddings
        self.encoder = SentenceTransformer(model_name)

        # Initialize tokenizer and model for generation (simplified)
        # In a real implementation, you'd use an LLM API or local model
        self.llm_model = llm_model

        logger.info("RAG Pipeline initialized successfully")

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a text"""
        embedding = self.encoder.encode([text])
        return embedding[0].tolist()

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = self.encoder.encode(texts)
        return [emb.tolist() for emb in embeddings]

    async def retrieve_relevant_chunks(
        self,
        query: str,
        top_k: int = 5
    ) -> List[DocumentChunk]:
        """Retrieve relevant document chunks based on the query"""
        try:
            # Generate embedding for the query
            query_embedding = await self.embed_text(query)

            # Search in Qdrant
            results = await self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k
            )

            # Convert results to DocumentChunk objects
            chunks = []
            for result in results:
                chunk = DocumentChunk(
                    id=result.id,
                    content=result.payload.get("content", ""),
                    source=result.payload.get("source", ""),
                    page_number=result.payload.get("page_number"),
                    section=result.payload.get("section", ""),
                    embedding=result.vector
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Error retrieving relevant chunks: {e}")
            return []

    async def generate_response(
        self,
        query: str,
        context_chunks: List[DocumentChunk]
    ) -> str:
        """Generate a response based on the query and context"""
        try:
            # Combine context chunks into a single context string
            context = "\n\n".join([chunk.content for chunk in context_chunks])

            # Generate response using enhanced method
            response = await self._generate_response_with_llm(query, context)

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Sorry, I encountered an error generating a response: {str(e)}"

    def _create_prompt(self, query: str, context: str) -> str:
        """Create a prompt for the LLM"""
        prompt = f"""
        You are an expert assistant for the Physical AI & Humanoid Robotics book.
        Use the following context to answer the question.
        If the context doesn't contain enough information, say so.

        Context:
        {context}

        Question: {query}

        Answer:
        """
        return prompt.strip()

    async def _generate_response_with_llm(
        self,
        query: str,
        context: str,
        max_tokens: int = 500
    ) -> str:
        """Generate response using LLM API (placeholder implementation)"""
        try:
            # In a real implementation, you would call an LLM API here
            # For example: OpenAI, Anthropic, or local model

            # Create a prompt for the LLM
            prompt = self._create_prompt(query, context)

            # Placeholder response - in real implementation, call actual LLM
            # response = await self._call_llm_api(prompt, max_tokens)

            # For now, create a more sophisticated response based on context
            response = self._create_contextual_response(query, context)
            return response

        except Exception as e:
            logger.error(f"Error in LLM response generation: {e}")
            return f"Sorry, I encountered an error generating a response: {str(e)}"

    def _create_contextual_response(self, query: str, context: str) -> str:
        """Create a contextual response based on query and context"""
        import re

        # Clean up the context
        clean_context = re.sub(r'\s+', ' ', context.strip())

        # If context is too long, summarize key points
        if len(clean_context) > 1000:
            # Extract key sentences that are most relevant to the query
            sentences = clean_context.split('. ')
            query_words = set(query.lower().split())

            relevant_sentences = []
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                # Calculate relevance based on word overlap with query
                relevance = len(query_words.intersection(sentence_words))
                if relevance > 0:
                    relevant_sentences.append((sentence, relevance))

            # Sort by relevance and take top sentences
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s[0] for s in relevant_sentences[:3]]  # Top 3 sentences
            clean_context = '. '.join(top_sentences)

        # Create response based on context
        response = f"Based on the Physical AI & Humanoid Robotics book content:\n\n{clean_context}\n\nIn summary, this addresses your question about '{query}'."
        return response

    def _simple_response_generation(self, query: str, context: str) -> str:
        """Simple response generation (fallback method)"""
        # This is a fallback implementation
        if context:
            return f"Based on the book content: {context[:500]}... [truncated for brevity]"
        else:
            return "I couldn't find relevant information in the book to answer your question."

    async def query(self, query: str, top_k: int = 5) -> RAGResult:
        """Main query method that orchestrates the RAG process"""
        logger.info(f"Processing RAG query: {query}")

        try:
            # Retrieve relevant chunks
            relevant_chunks = await self.retrieve_relevant_chunks(query, top_k)
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")

            if not relevant_chunks:
                return RAGResult(
                    response="I couldn't find relevant information in the book to answer your question.",
                    sources=[],
                    confidence=0.0,
                    retrieved_chunks=[]
                )

            # Generate response
            response = await self.generate_response(query, relevant_chunks)

            # Extract sources
            sources = list(set([chunk.source for chunk in relevant_chunks if chunk.source]))

            # Calculate confidence based on number of relevant chunks and their scores
            confidence = min(len(relevant_chunks) * 0.2, 1.0)  # Simple confidence calculation

            return RAGResult(
                response=response,
                sources=sources,
                confidence=confidence,
                retrieved_chunks=relevant_chunks
            )

        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            return RAGResult(
                response=f"Sorry, I encountered an error processing your query: {str(e)}",
                sources=[],
                confidence=0.0,
                retrieved_chunks=[]
            )

    async def ingest_document(
        self,
        content: str,
        source: str,
        chunk_size: int = 512,
        overlap: int = 128
    ) -> bool:
        """Ingest a document into the RAG system"""
        try:
            logger.info(f"Ingesting document from source: {source}")

            # Split document into chunks
            chunks = self._split_document(content, chunk_size, overlap)

            # Prepare points for Qdrant
            points = []
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = await self.embed_text(chunk.content)

                # Create point for Qdrant
                point = {
                    "id": f"{source}_chunk_{i}",
                    "vector": embedding,
                    "payload": {
                        "content": chunk.content,
                        "source": chunk.source,
                        "section": chunk.section or f"chunk_{i}",
                        "page_number": chunk.page_number
                    }
                }
                points.append(point)

            # Upload to Qdrant
            success = await self.qdrant_client.upload_points(
                collection_name=self.collection_name,
                points=points
            )

            if success:
                logger.info(f"Successfully ingested {len(chunks)} chunks from {source}")
                return True
            else:
                logger.error(f"Failed to upload chunks to Qdrant for {source}")
                return False

        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            return False

    def _split_document(
        self,
        content: str,
        chunk_size: int,
        overlap: int
    ) -> List[DocumentChunk]:
        """Split document content into chunks"""
        # Split content into sentences first
        sentences = content.split('. ')
        chunks = []
        current_chunk = ""
        chunk_num = 0

        for sentence in sentences:
            # Add sentence to current chunk
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                # Create a new chunk
                if current_chunk.strip():
                    chunk = DocumentChunk(
                        id=f"chunk_{chunk_num}",
                        content=current_chunk.strip(),
                        source="unknown",  # Will be set during ingestion
                    )
                    chunks.append(chunk)
                    chunk_num += 1

                # Start new chunk with overlap
                # Take the last few sentences from the previous chunk as overlap
                words = current_chunk.split()
                overlap_text = " ".join(words[-overlap:]) if len(words) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence + ". "

        # Add the last chunk if it has content
        if current_chunk.strip():
            chunk = DocumentChunk(
                id=f"chunk_{chunk_num}",
                content=current_chunk.strip(),
                source="unknown",
            )
            chunks.append(chunk)

        return chunks


# Global RAG pipeline instance
rag_pipeline = None


async def get_rag_pipeline() -> RAGPipeline:
    """Get or create the global RAG pipeline instance"""
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline()
        await rag_pipeline.qdrant_client.init_client()
    return rag_pipeline