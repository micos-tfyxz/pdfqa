# Optimized PDF Q&A Systems

This repository provides two Python implementations for querying PDF documents using OpenAI embeddings and FAISS for similarity search. Both tools segment PDF content into manageable chunks, compute embeddings, and retrieve contextually relevant text for user queries. The new updates introduce performance improvements, enhanced memory management, and context expansion to better support single and multi-session querying scenarios.

## Overview

### pdfqas.py – Fast One-Time Querying with Enhanced Performance
- **Purpose**: Optimized for users who need a fast, one-off query against a PDF.
- **Key Improvements**:
  - **Concurrent PDF Parsing**: Uses multi-threading (via `ThreadPoolExecutor`) to extract text from PDF pages in parallel, significantly reducing loading time.
  - **Caching Mechanisms**: Caches the full PDF text and question embeddings to avoid redundant computations during a single session.
  - **Fixed-size Chunking**: Splits the PDF text into fixed-size chunks for quick processing.
- **Usage Scenario**: Best suited for quick, single-session queries where persistent storage is not required. Each new session reloads and reprocesses the PDF.

### pdfqal.py – Multi-Session Querying with Persistent Index and Context Expansion
- **Purpose**: Designed for scenarios where the same PDF document is queried repeatedly.
- **Key Improvements**:
  - **Advanced Text Segmentation**: Utilizes NLTK to split text into sentences and intelligently merge them into chunks based on a maximum size, ensuring better context retention.
  - **Persistent Indexing**: Builds a FAISS index and saves it to disk (along with text chunks via `pickle`) to eliminate the need for reprocessing in subsequent sessions.
  - **Dynamic Index Selection**: Automatically selects an appropriate FAISS index type based on the amount of data.
  - **Context Expansion**: Implements context-aware querying by evaluating adjacent chunks using configurable similarity thresholds, enhancing the relevance of returned results.
  - **Memory Management and Logging**: Offers options to free up embeddings from memory post-index construction and provides logging for error handling and warnings.
- **Usage Scenario**: Ideal when you expect to run multiple queries on the same PDF, benefiting from reduced response time and improved result quality after the initial setup.

## Common Features
- **PDF Loading & Text Extraction**: Uses `PyPDF2` for reading PDFs and NLTK for tokenizing text.
- **Embeddings Generation**: Leverages OpenAI's API (`text-embedding-3-small`) for generating text embeddings.
- **Similarity Search**: Employs FAISS for fast and accurate nearest neighbor search.
- **Configurable Query Parameters**: Both implementations allow adjusting parameters like chunk size, top-K results, and similarity thresholds to tailor search results.

## Prerequisites

- **Python 3.7+**
- **OpenAI API Key**: Must be set as an environment variable `OPENAI_API_KEY`.
- **Required Python Libraries**:
  - `numpy`
  - `faiss` (or `faiss-cpu`)
  - `openai`
  - `nltk`
  - `PyPDF2`
  - `pickle` (included in Python standard library)
  - `logging` (included in Python standard library)
  
Install the dependencies via pip:
```bash
pip install numpy faiss-cpu openai nltk PyPDF2
```

## Usage

### Running pdfqas.py (Fast One-Time Query)
```bash
python pdfqas.py
```
- **Workflow**:
  - Enter the PDF file path when prompted.
  - The script extracts text concurrently from PDF pages, splits the text into fixed-size chunks, and builds a FAISS Flat index.
  - For each query, it generates and caches the question's embedding, performs a similarity search, and returns the most relevant chunks.
- **Note**: Every session processes the PDF from scratch; hence, it is best for one-off queries.

### Running pdfqal.py (Multi-Session Querying with Persistent Index)
```bash
python pdfqal.py
```
- **Workflow**:
  - On startup, the script attempts to load a pre-built FAISS index and associated text chunks from disk.
  - If the index is not found, it will prompt for a PDF file path, load and split the text (with advanced sentence merging), build the index dynamically, and save it for future sessions.
  - The `query_with_context` function then leverages context expansion by considering adjacent text chunks, returning combined context based on configurable similarity thresholds.
- **Note**: After the initial index construction, subsequent queries are much faster since the PDF does not need to be reprocessed.

## Comparison Table

| Feature                     | **pdfqas.py**                                | **pdfqal.py**                                   |
|-----------------------------|----------------------------------------------|-------------------------------------------------|
| **Ideal For**               | One-time, quick queries                      | Repeated queries with persistent storage        |
| **PDF Parsing**             | Parallel extraction using ThreadPoolExecutor | Sentence splitting with intelligent merging       |
| **Index Persistence**       | No (rebuilds index each session)             | Yes (saves FAISS index and text chunks to disk)   |
| **Embedding Caching**       | Caches question embeddings within session    | Not explicitly cached per query (persistent index)|
| **Context Handling**        | Returns fixed-size text chunks               | Dynamically expands context by including adjacent chunks |
| **Index Type**              | Always uses Flat FAISS index                 | Chooses between Flat or IVF index based on data size  |
| **Memory Management**       | Maintains embeddings during session          | Optionally frees embeddings from memory post-indexing |

## Conclusion
- **Use `pdfqas.py`** when you need a lightweight, fast solution for single-session PDF queries.
- **Use `pdfqal.py`** for more frequent queries on the same document, where persistent indexing and context expansion yield faster and more relevant responses.

---
