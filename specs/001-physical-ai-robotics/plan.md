# Implementation Plan: Physical AI & Humanoid Robotics Book

**Branch**: `001-physical-ai-robotics` | **Date**: 2025-12-10 | **Spec**: [specs/001-physical-ai-robotics/spec.md](../001-physical-ai-robotics/spec.md)
**Input**: Feature specification from `/specs/[001-physical-ai-robotics]/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Create an AI-native textbook for the course Physical AI & Humanoid Robotics using Docusaurus, Spec-Kit Plus, Claude Code, GitHub Pages, OpenAI Agents/ChatKit SDK, FastAPI backend, Neon Serverless Postgres, Qdrant Cloud Free Tier, and optionally better-auth for Signup/Signin + personalization. The book will include an embedded RAG Chatbot capable of answering questions from the book text and optionally from user-selected passages.

## Technical Context

**Language/Version**: Python 3.11, JavaScript/TypeScript, Markdown
**Primary Dependencies**: Docusaurus, FastAPI, OpenAI SDK, Qdrant, Neon Postgres, Claude Code, Spec-Kit Plus
**Storage**: Neon Serverless Postgres for user metadata, Qdrant Cloud for vector store, GitHub Pages for static content
**Testing**: pytest for backend, Docusaurus built-in testing for frontend
**Target Platform**: Web-based (Docusaurus frontend + FastAPI backend)
**Project Type**: Web application with frontend and backend components
**Performance Goals**: Sub-200ms response time for RAG queries, 95% uptime for book access
**Constraints**: Must support Docusaurus deployment to GitHub Pages, FastAPI backend with vector search capabilities, RAG functionality for book content
**Scale/Scope**: Educational textbook for robotics course with interactive chatbot, expected to serve 100-1000 concurrent users during course delivery

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Based on the constitution file, this project must:
- Follow Spec-Driven Development principles
- Remain highly structured and technically accurate
- Align with industry standards
- Be suitable for a hackathon-grade academic project
- Include all modules as specified in the constitution
- Contain 28+ complete chapters with diagrams
- Include capstone project specification and weekly labs
- Provide hardware guides that match specifications

## Project Structure

### Documentation (this feature)

```text
specs/001-physical-ai-robotics/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
book/
├── docs/                    # Book content chapters
│   ├── intro-to-physical-ai/
│   ├── ros-2-fundamentals/
│   ├── gazebo-simulation/
│   ├── unity-integration/
│   ├── nvidia-isaac/
│   ├── humanoid-development/
│   ├── vision-language-action/
│   ├── capstone-humanoid-project/
│   ├── hardware-requirements/
│   └── tools-lab-setup/
├── src/
│   ├── components/          # Custom React components
│   │   ├── ChatbotWidget.jsx
│   │   ├── PersonalizeButton.jsx
│   │   └── TranslateButton.jsx
│   └── pages/
├── static/                  # Static assets
├── docusaurus.config.js     # Docusaurus configuration
└── sidebars.js              # Navigation structure

backend/
├── main.py                  # FastAPI main application
├── rag.py                   # RAG pipeline implementation
├── database.py              # Database connection and models
├── qdrant_client.py         # Vector store integration
├── routers/
│   ├── auth.py              # Authentication endpoints (optional)
│   └── chatbot.py           # Chatbot endpoints
└── requirements.txt         # Backend dependencies

.history/
├── prompts/                 # Prompt History Records
└── adr/                     # Architecture Decision Records

.specify/
├── memory/                  # Project constitution
├── scripts/                 # Automation scripts
└── templates/               # Template files
```

**Structure Decision**: This is a web application with both frontend (Docusaurus) and backend (FastAPI) components. The book content will be served statically via Docusaurus on GitHub Pages, while the RAG chatbot functionality will be powered by a FastAPI backend with vector storage in Qdrant and user data in Neon Postgres.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Multi-component architecture (Docusaurus + FastAPI + Qdrant + Neon) | Educational book requires static content delivery with dynamic RAG capabilities | Single static site insufficient for interactive chatbot functionality |
| Complex deployment (GitHub Pages + separate backend) | Need to serve static book content cost-effectively while maintaining dynamic backend | Single deployment option would require expensive hosting for entire application |