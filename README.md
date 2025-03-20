# PDF Q&A Systems

This repository contains two optimized Python implementations for querying PDF documents using OpenAI embeddings and FAISS for similarity search. Both tools segment PDF content into manageable text chunks, compute embeddings, and retrieve contextually relevant text for user queries. The two versions cater to different usage scenarios:

- **pdfqas.py**: Best for quick, one-time queries with a focus on speed. Every time a question is asked, the script reloads the PDF and reprocesses the data.
- **pdfqal.py**: Designed for repeated queries with persistent indexing and efficient memory management. Once the index is built and saved, it can be reused across multiple queries without reloading the PDF.

## Overview

### pdfqas.py – Fast One-Time Querying
- **Purpose**: Optimized for users who need a fast, one-off query against a PDF.
- **Advantages**:
  - **Lightweight**: Loads a PDF, builds an index, and queries in a single session.
  - **Fixed-size chunking**: Uses simple segmentation without sentence merging.
  - **High-speed search**: Always employs a Flat FAISS index for maximum performance.
- **Limitations**: Each time a query is made, the PDF must be reprocessed, making it inefficient for repeated use.
- **When to use**: Choose this version for quick, single-session queries where you do not need to save the index.

### pdfqal.py – Multi-Session Querying with Persistent Index
- **Purpose**: Built for scenarios where a PDF is queried multiple times.
- **Advantages**:
  - **Persistent Indexing**: Saves the FAISS index and text chunks to disk, reducing redundant computations.
  - **Optimized Chunking**: Dynamically merges sentences for better context handling.
  - **Memory-Efficient**: Frees embeddings from memory after index construction.
- **Efficiency**: After the first use, subsequent queries run significantly faster since the document does not need to be reprocessed.
- **When to use**: Use this version when you expect to query the same PDF repeatedly, as it avoids reprocessing the document in each session.

## Common Features
- **PDF Loading & Text Segmentation**: Uses `PyPDF2` for text extraction and `nltk` for text tokenization.
- **Embeddings**: Leverages OpenAI's embedding API (`text-embedding-3-small`).
- **Similarity Search**: Uses FAISS for fast and accurate retrieval.
- **Context Expansion**: Retrieves adjacent text chunks for better contextual understanding.

## Prerequisites

- **Python 3.7+**
- **OpenAI API Key**: Set as an environment variable `OPENAI_API_KEY`.
- **Required Python Libraries**:
  - `numpy`
  - `faiss`
  - `openai`
  - `nltk`
  - `PyPDF2`
  - `pickle` (for pdfqal.py)
  ```bash
  pip install numpy faiss-cpu openai nltk PyPDF2 pickle
  ```

## Usage

### Using pdfqas.py (Fast One-Time Query)
```bash
python pdfqas.py
```
- Enter the PDF file path when prompted.
- The script will extract text from the PDF, process it, and build an index.
- Each time a new question is asked, the entire document is reprocessed, which can be time-consuming.
- Input your question, and the script will return relevant text from the document.

### Using pdfqal.py (Multi-Use with Persistent Indexing)
```bash
python pdfqal.py
```
- The script attempts to load a pre-built FAISS index.
- If no index is found, it processes the PDF and saves the index to disk for future use.
- Queries run significantly faster in subsequent uses since the index is precomputed and does not require reprocessing the PDF.
- Once the index is built, you can ask multiple questions without reloading the document.

## Comparison Table

| Feature           | **pdfqas.py** | **pdfqal.py** |
|------------------|--------------|--------------|
| **Best for**      | One-time quick queries | Repeated queries with persistent storage |
| **Index Persistence** | No | Yes (saves to disk) |
| **Processing Speed** | Rebuilds index every time | Faster after initial setup |
| **Chunking Strategy** | Fixed-size chunks | Sentence-aware merging |
| **Data Reuse** | No, reprocesses document for every query | Yes, reuses stored index |
| **Use Case** | Quickly extracting information from a PDF | Efficient querying over multiple sessions |

## Conclusion
- **Use `pdfqas.py`** for simple, one-time queries where processing speed is not critical.
- **Use `pdfqal.py`** if you need to query the same document multiple times efficiently, as it avoids unnecessary recomputation and significantly reduces response time.




