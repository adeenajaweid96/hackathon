# Data Model: Physical AI & Humanoid Robotics Book

## Overview
This document defines the key data entities and their relationships for the Physical AI & Humanoid Robotics Book project.

## Core Entities

### User
Represents a student or instructor using the book platform.

**Attributes:**
- `id`: Unique identifier (UUID)
- `email`: User's email address
- `name`: Full name
- `role`: User's role (student, instructor)
- `background`: Programming experience, hardware access, robotics background
- `preferences`: Language preference, personalization settings
- `createdAt`: Account creation timestamp
- `updatedAt`: Last update timestamp

**Relationships:**
- One-to-many with `UserProgress`
- One-to-many with `ChatHistory`

### Chapter
Represents a chapter in the Physical AI & Humanoid Robotics book.

**Attributes:**
- `id`: Unique identifier (UUID)
- `title`: Chapter title
- `slug`: URL-friendly identifier
- `content`: Chapter content in markdown format
- `learningObjectives`: List of learning objectives
- `exercises`: List of exercises for the chapter
- `order`: Chapter position in the book sequence
- `module`: Module this chapter belongs to (e.g., "ROS 2 Fundamentals")
- `createdAt`: Creation timestamp
- `updatedAt`: Last update timestamp

**Relationships:**
- One-to-many with `UserProgress`
- One-to-many with `BookContent` (for RAG indexing)

### BookContent
Represents processed book content for the RAG system.

**Attributes:**
- `id`: Unique identifier (UUID)
- `chapterId`: Reference to the parent chapter
- `content`: Content chunk for vector search
- `chunkIndex`: Position of this chunk within the chapter
- `embedding`: Vector embedding of the content
- `metadata`: Additional metadata (section, page, etc.)
- `createdAt`: Creation timestamp
- `updatedAt`: Last update timestamp

**Relationships:**
- Many-to-one with `Chapter`

### ChatSession
Represents a conversation session with the RAG chatbot.

**Attributes:**
- `id`: Unique identifier (UUID)
- `userId`: Reference to the user
- `title`: Session title (auto-generated from first query)
- `createdAt`: Creation timestamp
- `updatedAt`: Last update timestamp

**Relationships:**
- One-to-many with `ChatMessage`
- Many-to-one with `User`

### ChatMessage
Represents a single message in a chat conversation.

**Attributes:**
- `id`: Unique identifier (UUID)
- `sessionId`: Reference to the chat session
- `role`: Message role (user, assistant)
- `content`: Message content
- `timestamp`: When the message was created
- `contextChunks`: IDs of content chunks used for response
- `source`: Source of the information (book content, general knowledge)

**Relationships:**
- Many-to-one with `ChatSession`

### UserProgress
Tracks user progress through the book.

**Attributes:**
- `id`: Unique identifier (UUID)
- `userId`: Reference to the user
- `chapterId`: Reference to the chapter
- `completed`: Whether the chapter is completed
- `progressPercentage`: Progress in the chapter (0-100)
- `timeSpent`: Time spent on the chapter (in seconds)
- `lastAccessed`: When the chapter was last accessed
- `notes`: User's notes on the chapter
- `quizScores`: Scores for chapter quizzes
- `createdAt`: Creation timestamp
- `updatedAt`: Last update timestamp

**Relationships:**
- Many-to-one with `User`
- Many-to-one with `Chapter`

### PersonalizationProfile
Stores user preferences for content personalization.

**Attributes:**
- `id`: Unique identifier (UUID)
- `userId`: Reference to the user
- `programmingExperience`: Level of programming experience (beginner, intermediate, advanced)
- `hardwareAccess`: Hardware available to user (Jetson, GPU, etc.)
- `roboticsBackground`: Experience with robotics
- `learningGoals`: User's learning objectives
- `preferredTopics`: Topics of particular interest
- `createdAt`: Creation timestamp
- `updatedAt`: Last update timestamp

**Relationships:**
- One-to-one with `User`

### APIKey
Stores API keys for external services (secure implementation).

**Attributes:**
- `id`: Unique identifier (UUID)
- `service`: Service name (openai, qdrant, etc.)
- `encryptedKey`: Encrypted API key value
- `createdAt`: Creation timestamp
- `updatedAt`: Last update timestamp

## Relationships Summary

```
User ||--o{ UserProgress
User ||--o{ ChatSession
User ||--|| PersonalizationProfile

Chapter ||--o{ UserProgress
Chapter ||--o{ BookContent

ChatSession ||--o{ ChatMessage

BookContent }o--|| Chapter
```

## Validation Rules

### User
- Email must be valid
- Name cannot be empty
- Role must be one of: student, instructor

### Chapter
- Title cannot be empty
- Slug must be unique
- Order must be positive integer
- Content must be valid markdown

### BookContent
- Content cannot be empty
- Embedding must be a valid vector
- ChapterId must reference an existing chapter

### ChatSession
- UserId must reference an existing user
- Title can be auto-generated if empty

### ChatMessage
- SessionId must reference an existing session
- Role must be either 'user' or 'assistant'
- Content cannot be empty

### UserProgress
- UserId must reference an existing user
- ChapterId must reference an existing chapter
- ProgressPercentage must be between 0 and 100

### PersonalizationProfile
- UserId must reference an existing user
- ProgrammingExperience must be one of: beginner, intermediate, advanced
- HardwareAccess can include multiple values

## State Transitions

### UserProgress States
- `not_started` → `in_progress` → `completed`
- Can transition back to `in_progress` from `completed` if user revisits

### ChatSession States
- `active` (default) - session is ongoing
- `archived` - session is no longer active but preserved
- `deleted` - session marked for deletion (retention policy applies)

## Indexes for Performance

### BookContent
- Index on `chapterId` for efficient chapter-based queries
- Index on `embedding` for vector search operations

### ChatMessage
- Index on `sessionId` for efficient session retrieval
- Index on `timestamp` for chronological ordering

### UserProgress
- Composite index on `userId` and `chapterId` for efficient lookups
- Index on `completed` for progress tracking queries