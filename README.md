# Physical AI & Humanoid Robotics Book

Welcome to the Physical AI & Humanoid Robotics educational book project! This AI-native textbook is designed to guide you through the essential concepts, technologies, and practical implementations needed to understand and develop advanced robotic systems.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
- [Development](#development)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project is an interactive, AI-native textbook for Physical AI and Humanoid Robotics. It combines traditional educational content with modern web technologies, including:

- Interactive modules with card-based navigation
- RAG (Retrieval Augmented Generation) chatbot for answering questions
- Dark aesthetic theme optimized for readability
- Comprehensive curriculum covering robotics fundamentals to advanced topics
- Integration with simulation environments and real-world robotics platforms

## Features

### Educational Content
- Complete curriculum from Physical AI fundamentals to advanced humanoid development
- Interactive modules with hands-on exercises
- Capstone project integrating all concepts
- Assessment tools and quizzes

### Technical Features
- Dark theme with optimized color contrast for readability
- Responsive design for all screen sizes
- Advanced animations and interactive elements
- RAG chatbot for content-based Q&A
- Personalization and translation features
- Search functionality with Algolia integration

### Technology Stack
- **Frontend**: Docusaurus v3, React, TypeScript
- **Styling**: Infima, CSS Modules, Framer Motion for animations
- **Backend**: FastAPI, Python
- **Database**: Neon Postgres
- **Vector Store**: Qdrant Cloud
- **Deployment**: GitHub Pages (frontend), separate backend hosting

## Getting Started

### Prerequisites

- Node.js (v18 or higher)
- npm or yarn
- Python (for backend services)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/hackathon.git
   cd hackathon
   ```

2. Navigate to the book directory:
   ```bash
   cd book
   ```

3. Install dependencies:
   ```bash
   npm install
   ```

4. Start the development server:
   ```bash
   npm start
   ```

The book will be available at `http://localhost:3000/hackathon/`

### Backend Setup (Optional)

To use the RAG chatbot functionality:

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Start the backend:
   ```bash
   uvicorn main:app --reload
   ```

## Development

### Adding Content

Content is organized in the `book/docs/` directory by topic. Each topic has its own subdirectory with markdown files.

### Component Development

Custom React components are located in `book/src/components/`:
- `ModuleCard/` - Interactive module cards for the homepage
- `ChatbotWidget/` - RAG chatbot interface
- `PersonalizeButton/` - User preference controls
- `TranslateButton/` - Translation functionality

### Styling

Custom styles are in `book/src/css/custom.css` with:
- Dark theme variables optimized for readability
- Responsive design for all screen sizes
- Animation and interaction styles
- Accessibility enhancements

### Running Tests

Frontend:
```bash
npm test
```

Backend (from backend directory):
```bash
pytest
```

## Deployment

### GitHub Pages

The frontend is deployed to GitHub Pages using GitHub Actions:

1. Update the `docusaurus.config.ts` with your repository details
2. Push changes to the main branch
3. GitHub Actions will automatically build and deploy

The workflow is defined in `.github/workflows/deploy.yml`.

### Backend Deployment

The backend needs to be deployed separately to a cloud provider that supports Python applications (e.g., Heroku, Render, Railway).

## Contributing

We welcome contributions to improve the Physical AI & Humanoid Robotics book!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Guidelines

- Follow the existing code style
- Write clear, descriptive commit messages
- Update documentation as needed
- Add tests for new functionality
- Ensure all tests pass before submitting

## Tech Stack Details

### Frontend Architecture
- **Docusaurus**: Static site generator with excellent documentation features
- **React**: Component-based UI development
- **TypeScript**: Type safety for better development experience
- **Framer Motion**: Advanced animations and interactions
- **Infima**: CSS framework for consistent styling

### Backend Architecture
- **FastAPI**: High-performance Python web framework
- **RAG Pipeline**: Retrieval Augmented Generation for Q&A
- **Qdrant**: Vector database for semantic search
- **Neon**: Serverless Postgres for metadata

### AI Integration
- **OpenAI Whisper**: Voice recognition capabilities
- **LLM Integration**: Cognitive planning and reasoning
- **Vector Embeddings**: Semantic search and retrieval

## Project Structure

```
hackathon/
├── book/                    # Docusaurus frontend
│   ├── docs/               # Documentation content
│   ├── src/                # Custom React components
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Custom pages
│   │   └── css/            # Custom styles
│   ├── static/             # Static assets
│   ├── docusaurus.config.ts # Docusaurus configuration
│   └── sidebars.ts         # Navigation structure
├── backend/                # FastAPI backend
│   ├── main.py            # Application entry point
│   ├── rag.py             # RAG pipeline
│   ├── database.py        # Database connection
│   └── routers/           # API routes
└── specs/                  # Project specifications
    └── 001-physical-ai-robotics/
```

## Support

If you encounter any issues or have questions:

1. Check the existing documentation
2. Search existing issues
3. Create a new issue with detailed information
4. Join our community discussions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Built with ❤️ for the robotics community.

[![Docusaurus](https://img.shields.io/badge/Built%20with-Docusaurus-1a1a1a?style=for-the-badge&logo=docusaurus&logoColor=white)](https://docusaurus.io/)
[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)