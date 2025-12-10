---
id: 1
title: Physical AI Humanoid Robotics Book Plan
stage: plan
date: 2025-12-10
surface: agent
model: claude-sonnet-4-5-20250929
feature: physical-ai-humanoid-robotics-book
branch: 001-spec-refinement
user: user
command: /sp.plan Implementation Plan — Physical AI & Humanoid Robotics Book
labels: ["planning","book-writing","docusaurus"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - null
tests:
  - null
---

## Prompt

```text
Implementation Plan — Physical AI & Humanoid Robotics Book
1. Project Overview

This project produces an AI-native textbook for the course Physical AI & Humanoid Robotics, built using:

Docusaurus

Spec-Kit Plus

Claude Code

GitHub Pages Deployment

OpenAI Agents/ChatKit SDK

FastAPI backend

Neon Serverless Postgres

Qdrant Cloud Free Tier

(Optional) better-auth for Signup/Signin + personalization.

The book will also include an embedded RAG Chatbot capable of answering questions from the book text and optionally from user-selected passages.

2. Implementation Phases
Phase 1 — Docusaurus Book Structure

Goal: Prepare the full outline and folder structure.

Tasks:

Set up Docusaurus with sidebar navigation.

Create chapter folders for:

Introduction to Physical AI

ROS 2 Fundamentals

Gazebo Simulation

Unity Integration

NVIDIA Isaac

Humanoid Development

Vision-Language-Action

Capstone Humanoid Project

Hardware Requirements

Tools & Lab Setup

Implement markdown template for each chapter:

Learning objectives

Concepts

Code samples

Diagrams

Exercises

Deliverables:

/docs folder fully structured

Sidebars.js configured

All initial chapter files created (empty or drafted)

Phase 2 — Writing Chapters Using Spec-Kit + Claude Code

Goal: AI-driven writing workflow.

Tasks:

Use Spec-Kit commands:
/sp.clarity — refine and expand content
/sp.feature — generate chapter-level chunks
/sp.page — create chapter pages
/subagent — reusable writing intelligence
/sp.section — break chapters into sections

Write all chapters using the course details provided.

Deliverables:

All chapters written in markdown

Consistent style, diagrams in Mermaid where possible

Glossary + index

Phase 3 — Frontend Integration (Docusaurus Enhancements)

Tasks:

Add custom React components:

Chapter-level buttons:

Personalize chapter

Translate to Urdu

RAG chatbot widget (floating or sidebar)

Create /components folder with:

ChatbotWidget.jsx

PersonalizeButton.jsx

TranslateButton.jsx

Add theme styling (Tailwind or CSS modules)

Deliverables:

Working UI components

Buttons appear at start of each chapter

Phase 4 — Backend for RAG Chatbot

Tasks:

Create FastAPI backend:

/ingest — extract book markdown → vectors → Qdrant

/query — retrieve relevant chunks

/selected-text-query — for user-selected text

Integrate OpenAI Agents (ChatKit SDK)

Neon serverless Postgres for user metadata

Qdrant Cloud for vector store

Create environment configs (.env)

Deliverables:

backend/ folder with:

main.py

rag.py

database.py

qdrant_client.py

routers/auth.py

routers/chatbot.py

Phase 5 — Embedding Chatbot in Docusaurus

Tasks:

Add API calls from React widget to FastAPI backend

Support 2 modes:

General RAG QA

Selected text QA (book text highlighted by user)

Add streaming responses (SSE or WebSockets)

Deliverables:

Chatbot appears inside book

Fully functional QA

Phase 6 — Optional Bonus Features (+50 each)
6A. Personalization (better-auth)

Tasks:

Implement signup + signin

Ask user background:

Programming experience

Hardware (Jetson? GPU?)

Robotics background

Store profile in Neon Postgres

Modify Docusaurus chapter rendering to adapt text based on user role/background

Deliverables:

/auth API

Personalized chapter experience

6B. Translate to Urdu

Tasks:

Add translation API via OpenAI GPT-4/5 model

Store translated version in frontend state

Toggle between English/Urdu

Deliverables:

Instant Urdu translation for any chapter

6C. Claude Code Subagents (Reusable Intelligence)

Examples:

chapterWriter subagent

exampleGenerator

tutorialExpander

quizMaker

Deliverables:

/claude/subagents/*.json

Demonstrated usage in writing workflow

Phase 7 — Deployment

Tasks:

Deploy Docusaurus to GitHub Pages

Deploy FastAPI to:

Render

Railway

Fly.io

Vercel (using serverless FastAPI framework)

Set up CORS for frontend

Final testing

Deliverables:

Live book URL

Live chatbot backend

Integration working end-to-end

Phase 8 — Final Hackathon Packaging

Tasks:

Record 90-second demo video

Include:

Book walkthrough

Chatbot demo

Personalization

Translation

Submit:

GitHub repo

Book link

Backend link

Video link

WhatsApp number

Deliverables:

Submission ready for Nov 30 deadline

3. Architecture Diagram (High Level)
Docusaurus Frontend
 ├── Chatbot Widget → FastAPI → OpenAI Agent
 ├── Personalize Button → Auth API → Neon DB
 └── Translate Button → Translation API

FastAPI Backend
 ├── Auth (better-auth)
 ├── RAG Pipeline
 │      ├── Markdown Loader
 │      ├── Embeddings
 │      └── Qdrant Cloud Vector Store
 └── User Profiles (Neon Postgres)

LLM Stack
 ├── OpenAI ChatKit Agents
 └── Claude Code (for authoring)

4. Risks & Mitigation
Risk    Mitigation
Backend deployment issues    Use Render free tier or Railway
CORS errors    Configure CORS in FastAPI
Slow RAG responses    Reduce chunk size, use smaller embeddings
Chapter overload    Use subagents to auto-generate content
5. Success Criteria

Complete book with all modules

Wor this will be the plan for writing book on docusaurus
```

## Response snapshot

PHR created for the implementation plan.

## Outcome

- ✅ Impact: N/A
- 🧪 Tests: N/A
- 📁 Files: N/A
- 🔁 Next prompts: N/A
- 🧠 Reflection: N/A

## Evaluation notes (flywheel)

- Failure modes observed: N/A
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A
