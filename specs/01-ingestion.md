# 01 - Document Ingestion Pipeline

## Overview
Process documents (starting with Tata Nexon Brochure) into vector embeddings for semantic search. The pipeline must be scalable for lengthy and complex documents.

## Core Components

### 1. Scanner / Document Loader
- Scans and loads the input document (PDF, TXT, MD, etc.)
- Extracts raw text while preserving basic structure
- Handles different file formats
- Returns raw text + basic metadata (filename, page count, etc.)

### 2. Parser
- Cleans the extracted text
- Identifies and preserves document hierarchy (headings, sub-headings, sections, product tiers)
- Maintains section numbers and logical boundaries

### 3. Chunker
- Splits the document into semantically coherent chunks
- Preserves section boundaries — does not break inside a section or important paragraph
- Supports configurable chunk size and overlap
- Designed to scale for long, complex documents

### 4. Embedder
- Converts each chunk into a high-quality vector embedding
- Uses embedding model (e.g., OpenAI, Sentence-Transformers, etc.)
- Ensures consistent embedding dimensions

### 5. Storer (Vector Store)
- Stores chunks + embeddings + metadata into the vector database
- Ensures idempotency (same document processed twice does not create duplicates)
- Stores rich metadata for better retrieval (section, heading, tier, source, etc.)

## Functional Requirements

- Accept Tata Nexon Brochure (PDF) as input
- Preserve section boundaries, headings, and product tiers during chunking
- Every chunk must carry rich metadata
- Indexing must be optimized for high-quality semantic search
- Pipeline must scale efficiently for lengthy documents

## Acceptance Criteria
- Tata Nexon brochure is successfully parsed without losing structural information
- Chunks respect section boundaries and contain proper metadata
- Empty document returns zero chunks
- Same document can be processed multiple times without creating duplicates
- Pipeline works end-to-end: Scanner → Parser → Chunker → Embedder → Storer