"""
Main FastAPI application for the RAG Chatbot system.

This file sets up the FastAPI application with CORS middleware
and includes the chatbot router for handling chatbot requests.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import chatbot

# Create FastAPI app instance
app = FastAPI(
    title="Physical AI & Humanoid Robotics RAG Chatbot API",
    description="API for the RAG chatbot system that answers questions from the Physical AI & Humanoid Robotics book content",
    version="1.0.0"
)

# Add CORS middleware to allow requests from the Docusaurus frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include chatbot router
app.include_router(chatbot.router, prefix="/api/v1", tags=["chatbot"])

@app.get("/")
async def root():
    """Root endpoint for health check"""
    return {"message": "Physical AI & Humanoid Robotics RAG Chatbot API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "RAG Chatbot API"}