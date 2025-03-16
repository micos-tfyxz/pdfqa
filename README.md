# PDF Q&A Systems

This repository contains two Python implementations for querying PDF documents using OpenAI embeddings and FAISS for similarity search. Both tools segment PDF content into manageable text chunks, compute embeddings, and then retrieve contextually relevant text for user questions. The two versions are designed for different usage scenarios:

- **pdfqas.py**: Best for quick, one-time usage.
- **pdfqal.py**: Ideal for repeated or multiple queries, with the added advantage of persistent indexing.

## Overview

### pdfqas.py – One-Time Quick Use
- **Purpose**: This version is streamlined for users who want a fast, one-off query against a PDF document.
- **Advantages**:
  - **Ease of use**: Simply load the PDF, build the index, and ask your question.
  - **Dynamic Similarity Threshold**: Uses a dynamic threshold to merge adjacent sentences for context expansion.
- **When to use**: Choose this version if you only need to perform a single query or a few queries in one session and do not require persistent storage of the index.

### pdfqal.py – Multi-Use with Persistent Indexing
- **Purpose**: This version is designed for scenarios where you expect to query the same PDF repeatedly.
- **Advantages**:
  - **Persistent Indexing**: After the initial index is built and saved to disk, subsequent sessions can load the saved FAISS index and text chunks, greatly speeding up query response time.
  - **Resource Efficiency**: Avoids redundant computation of embeddings by reusing the stored index.
- **When to use**: Opt for this version when you plan on making multiple queries over time, as it improves efficiency by not requiring the index to be rebuilt for every session.

## Common Features

Both implementations share the following features:
- **PDF Loading & Text Segmentation**: Uses `PyPDF2` to extract text and `nltk` to split text into sentences and merge them into chunks.
- **Embeddings**: Leverages the OpenAI API to compute text embeddings.
- **Similarity Search**: Uses FAISS for fast and accurate similarity search.
- **Context Expansion**: Retrieves adjacent text chunks to provide more context around the most similar text.

## Prerequisites

- **Python 3.7+**
- **OpenAI API Key**: Set the environment variable `OPENAI_API_KEY` with your API key.
- **Required Python Libraries**:
  - `numpy`
  - `faiss`
  - `openai`
  - `nltk`
  - `PyPDF2`
  - `pickle` (for pdfqal.py)

You can install the required libraries with:
```bash
pip install numpy faiss-cpu openai nltk PyPDF2
```

## Setup

1. **Set up your OpenAI API key**:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   ```

2. **Download NLTK Resources**:
   Both scripts automatically check and download the necessary NLTK resource (`punkt`). Make sure you have internet connectivity the first time you run the scripts.

## Usage

### Using pdfqas.py (One-Time Quick Use)
1. **Load and Query PDF**:
   - Edit the file path in the script to point to your PDF document.
   - Run the script:
     ```bash
     python pdfqas.py
     ```
2. **Example Query**:
   - When prompted, enter the content you want to learn from the PDF.
   - The script will display the similarity score and the expanded context from the PDF.

### Using pdfqal.py (Multi-Use with Persistent Index)
1. **Build and Save the Index**:
   - Edit the file path in the script to your PDF.
   - The script will first try to load an existing FAISS index and text chunks. If not found, it will load the PDF and build the index, saving both to disk.
   - Run the script:
     ```bash
     python pdfqal.py
     ```
2. **Subsequent Uses**:
   - For future queries, the script will load the saved index from disk, speeding up the query process.
3. **Example Query**:
   - After the index is loaded, input your question at the prompt.
   - The script returns results along with similarity scores.


---

test file: "Quantifying streambed advection and conduction heat fluxes"

### **Using pdfqas.py (One-Time Quick Use)**
1. **Run the script and provide the PDF path**:
   ```bash
   python pdfqas.py
   ```
2. **Example Query**:
   ```
   Enter the path to your PDF file: C:/Users/汪有为/Downloads/test.pdf
   Enter your question (or type 'exit' to quit): heat conduction affect
   ```
3. **Example Output**:
   ```
   Similarity Score: 0.472
   Expanded Content:
   Conductive heat ﬂuxes are generally estimated as nominally instantaneous functions of current bed temperature gradients[e.g., Hondzo and Stefan , 1994, Brown , 1969], and advective ﬂuxes are given nominally immediate action based on current temperatures in the bed and an estimate of the groundwater ﬂux [e.g., Kurylyk et al ., 2016].
   ```

---


### **Using pdfqal.py (Multi-Use with Persistent Index)**

#### **Step 1: First-Time Setup (Building the Index)**
- Run the script and specify the path to your PDF file:
  ```bash
  python pdfqal.py
  ```
- If no existing index is found, you will be prompted to enter the PDF file path:
  ```
  ⚠ Index or text chunk files not found
  Enter the path to your PDF file: C:/Users/汪有为/Downloads/test.pdf
  ```
- The script will process the PDF and save the index:
  ```
  Index saved to pdf_index.faiss
  Text chunks saved to pdf_index.faiss.pkl
  ```

#### **Step 2: Querying the PDF**
- After the index is built, you can query the PDF:
  ```
  Please enter your question (type 'exit' to quit): heat conduction affect
  ```
- Since the index is now stored, subsequent runs will load it directly:
  ```
  Index loaded from pdf_index.faiss
  Text chunks loaded from pdf_index.faiss.pkl
  ```

#### **Example Query Output**
```text
Similarity: 0.472
Thorough reviews by Anderson [2005], Constantz [2008], Rau et al . [2014], Halloran et al . [2016a], and Irvine et al . [2016] provide a comprehensive overview of how heat tracer techniques can be applied to study surface-groundwater interactions. Although many studies have used streambed temperature as a tracer to study surface water-groundwater interactions, rigorous coupling of heat and ﬂuid ﬂow in the bed to drive stream energy balances has not been done. The bed is generally treated as an external boundary condition to the stream. Because bed heat contributions can seem a small contribution to the overall energy ﬂuxes, these approximations have been made as adjunct parameterizations on mass and energy conservation equations for the stream itself without reference to the energy and mass balance equations of the bed...
```

#### **Step 3: Reusing the Index in Future Sessions**
- The next time you run `pdfqal.py`, it will load the saved index instead of rebuilding it:
  ```
  Index loaded from pdf_index.faiss
  Text chunks loaded from pdf_index.faiss.pkl
  ```
- This significantly reduces processing time and makes multiple queries more efficient.

---

## Summary

| Feature           | **pdfqas.py** | **pdfqal.py** |
|------------------|--------------|--------------|
| **Best for**      | One-time quick queries | Repeated queries with persistent storage |
| **Index Persistence** | No | Yes (saves index to disk) |
| **Processing Speed** | Needs to reprocess PDF each time | Faster after the first run |
| **Context Expansion** | Yes | Yes (with enhanced retrieval logic) |
| **Use Case** | Quickly extracting information from a PDF without saving data | Efficient querying over multiple sessions |


- **pdfqas.py**: Quick and straightforward for one-off queries, with dynamic context expansion.
- **pdfqal.py**: Enhanced for multiple queries with persistent indexing, reducing overhead by reusing the pre-built index.

Choose the tool that best fits your needs:
- For a single session or occasional query, use **pdfqas.py**.
- For repeated use or when working with large documents over multiple sessions, use **pdfqal.py**.

