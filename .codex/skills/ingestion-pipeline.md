# Ingestion Pipeline Skill

Description: Strictly follow Test Driven Development for the Document Ingestion Pipeline

When to use: For all work related to document ingestion (Scanner, Parser, Chunker, Embedder, Storer)

## Rules
- Always follow strict Red → Green → Refactor cycle
- Write exactly one small, focused test at a time
- Never write implementation code before a failing test
- Test each component independently when possible
- Use mocks for Embedder and Storer (Vector DB)
- Prioritize structure preservation and metadata quality

## Core Components to Test

### 1. Scanner / Document Loader
- Loads different file types
- Returns raw text + basic metadata

### 2. Parser
- Cleans text
- Extracts and preserves document structure (headings, sections, tiers)

### 3. Chunker
- Creates semantically meaningful chunks
- Preserves section boundaries
- Respects chunk size and overlap

### 4. Embedder
- Generates embeddings for chunks
- Handles batching and errors gracefully

### 5. Storer
- Saves chunks + embeddings + metadata to vector database
- Ensures idempotency (no duplicate entries)

## Development Phases (Recommended Order)

### Phase 1: Scanner + Parser
- Empty document handling
- Basic text extraction
- Structure detection (headings, sections)

### Phase 2: Chunker
- Basic chunking
- Section boundary preservation
- Chunk size + overlap

### Phase 3: Metadata Handling
- Rich metadata attachment to chunks

### Phase 4: Embedder
- Embedding generation and validation

### Phase 5: Storer
- Storage with idempotency

## Workflow
1. Read the ingestion spec first
2. Add one failing test focused on one component/behavior
3. Run the test (should fail)
4. Write minimal code to make it pass
5. Refactor only after tests are green
6. Move to the next test