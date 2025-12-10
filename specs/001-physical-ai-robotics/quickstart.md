# Quickstart Guide: Physical AI & Humanoid Robotics Book

## Overview
This guide will help you set up the development environment for the Physical AI & Humanoid Robotics Book project, including both the Docusaurus frontend and FastAPI backend components.

## Prerequisites

### System Requirements
- Node.js 18+ (for Docusaurus)
- Python 3.11+ (for FastAPI backend)
- Git
- Access to OpenAI API key
- Access to Qdrant Cloud account
- Access to Neon Postgres account (optional, for personalization features)

### Recommended Development Environment
- Ubuntu 22.04 LTS (for ROS 2 compatibility, as per constitution)
- VS Code with appropriate extensions
- Git configured with your GitHub account

## Frontend Setup (Docusaurus Book)

### 1. Clone the Repository
```bash
git clone [repository-url]
cd [repository-name]
```

### 2. Install Docusaurus Dependencies
```bash
cd book
npm install
```

### 3. Start Development Server
```bash
npm start
```
This will start the Docusaurus development server at `http://localhost:3000`.

### 4. Build for Production
```bash
npm run build
```

## Backend Setup (FastAPI)

### 1. Navigate to Backend Directory
```bash
cd backend
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the backend directory:
```env
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
NEON_DATABASE_URL=your_neon_database_url
SECRET_KEY=your_secret_key_for_auth
```

### 5. Start Backend Server
```bash
uvicorn main:app --reload --port 8000
```
The backend will be available at `http://localhost:8000`.

### 6. Access API Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Database Setup

### Neon Postgres Setup
1. Create a Neon account at https://neon.tech
2. Create a new project
3. Get the connection string from the project dashboard
4. Add it to your `.env` file as `NEON_DATABASE_URL`

### Qdrant Setup
1. Create a Qdrant Cloud account at https://cloud.qdrant.io
2. Create a new cluster
3. Get the cluster URL and API key
4. Add them to your `.env` file as `QDRANT_URL` and `QDRANT_API_KEY`

## Running the Complete Application

### 1. Terminal 1 - Start Backend
```bash
cd backend
source venv/bin/activate  # If using virtual environment
uvicorn main:app --reload --port 8000
```

### 2. Terminal 2 - Start Frontend
```bash
cd book
npm start
```

### 3. Access the Application
- Book: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- Backend Docs: `http://localhost:8000/docs`

## Initial Data Setup

### 1. Ingest Book Content to Vector Database
```bash
# From backend directory
python -m scripts.ingest_book_content
```

This script will:
- Read all book chapters from the `book/docs` directory
- Process and chunk the content
- Generate embeddings using OpenAI
- Store in Qdrant vector database

### 2. Initialize Database Tables
The backend will automatically create required tables on first run if they don't exist.

## Testing the RAG Functionality

### 1. Test Backend API
```bash
curl -X POST "http://localhost:8000/chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key components of ROS 2 architecture?",
    "session_id": "test-session-123"
  }'
```

### 2. Verify Frontend Integration
The chatbot widget should be accessible from any book page and connect to the backend API.

## Environment Configuration

### Frontend Environment Variables
The Docusaurus frontend may need these environment variables in a `.env` file:
```env
REACT_APP_BACKEND_URL=http://localhost:8000
REACT_APP_QDRANT_ENABLED=true
```

### Backend Environment Variables
Required environment variables for the backend (in `.env` file):
```env
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
NEON_DATABASE_URL=your_neon_database_url
SECRET_KEY=your_secret_key
DEBUG=true  # Set to false in production
ALLOWED_ORIGINS=http://localhost:3000  # Frontend URL for CORS
```

## Development Workflow

### Adding New Book Content
1. Create new markdown files in `book/docs/[module]/`
2. Update `book/sidebars.js` to include the new content in navigation
3. Run the ingestion script to update the vector database

### Backend Development
1. Make changes to backend code
2. The server will automatically reload due to `--reload` flag
3. Test API endpoints using the built-in documentation

### Frontend Development
1. Make changes to React components in `book/src/components/`
2. The development server will automatically reload
3. Test the integration with backend APIs

## Running Tests

### Backend Tests
```bash
cd backend
python -m pytest
```

### Frontend Tests
```bash
cd book
npm test
```

## Deployment

### Frontend Deployment (GitHub Pages)
```bash
cd book
GIT_USER=your-username CURRENT_BRANCH=main USE_SSH=true npm run deploy
```

### Backend Deployment Options
1. **Render**: Connect your GitHub repository to Render
2. **Railway**: Deploy using Railway's GitHub integration
3. **Fly.io**: Deploy using Fly's CLI tool

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   - Change ports in startup commands: `npm start -- --port 3001` or `uvicorn main:app --port 8001`

2. **API Key Issues**
   - Verify all API keys are correctly set in environment variables
   - Check for typos in key values

3. **Database Connection Issues**
   - Verify database URLs are correct
   - Check firewall settings if connecting from restricted networks

4. **Vector Search Not Working**
   - Ensure the ingestion script has been run
   - Verify Qdrant connection details

### Useful Commands

```bash
# Check backend API status
curl http://localhost:8000/health

# Check frontend build status
cd book && npm run build && npm run serve

# Run backend in production mode
uvicorn main:app --host 0.0.0.0 --port 8000

# View backend logs
tail -f logs/app.log
```

## Next Steps

1. Review the [API contracts](./contracts/) for detailed endpoint specifications
2. Check the [tasks.md](./tasks.md) file for specific implementation tasks
3. Follow the [specification](./spec.md) for detailed feature requirements
4. Refer to the [data model](./data-model.md) for database schema details