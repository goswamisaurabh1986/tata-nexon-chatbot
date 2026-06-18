# 02 - Retrieval Module (Search Tool for LangGraph Agent)

## Overview
A dedicated retrieval/search tool that the LangGraph agent uses to query the ChromaDB vector database and retrieve the most relevant document chunks with **full citation support**.

## Core Components

### 1. Retriever (Main Class)
- Main entry point used by the LangGraph agent
- Accepts user query and returns ranked relevant chunks with citations

### 2. Embedder
- Uses the same Embedder as the ingestion pipeline

### 3. VectorStore Client
- Interfaces with ChromaDB for similarity search

### 4. Ranker / Post-Processor
- Ranks results by relevance
- Adds citation metadata

## Functional Requirements

- Accept a user query (string)
- Convert query into embedding
- Perform similarity search on ChromaDB
- Return top_k most relevant chunks, sorted by relevance (highest score first)
- **Each chunk must include proper citation support**:
  - `citation_id`: Unique identifier in format `"source:chunk_index"` (e.g., `"Tata_Nexon_Brochure.pdf:5"`)
  - `text`: The chunk content
  - `metadata`: Full original metadata (source, section_title, chunk_index, page_number, etc.)
  - `score`: Similarity score (0.0 to 1.0)

## Citation Requirements
- Every returned chunk **must** have a `citation_id`
- `citation_id` should be stable and unique for each chunk
- Metadata must preserve ingestion-time information (section_title, page_number if available)
- Designed to be easily used by LangGraph agent for grounded answers with citations

## Acceptance Criteria
- Empty or meaningless query returns empty list
- Results are sorted by relevance score (descending)
- Every chunk contains `citation_id`, `text`, `metadata`, and `score`
- Returned chunks can be directly used to generate citations like [1], [2] in final answers
- Supports configurable `top_k` and `similarity_threshold`

## Non-Functional Requirements
- Low latency
- Consistent embedding model with ingestion
- Stable citation IDs across multiple retrievals