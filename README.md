PDF Chatbot with OpenAI and FAISS

This project implements a PDF-based Question & Answer (Q&A) system using OpenAI's embeddings and FAISS for efficient similarity search. The chatbot allows users to load a PDF document, build a searchable index, and query the document with natural language questions. It dynamically expands the context of the query results by considering adjacent sentences based on a similarity threshold.

Features
PDF Text Extraction: Extracts text from PDF files and intelligently segments it into manageable chunks.

Semantic Search: Uses OpenAI's embeddings to generate vector representations of text chunks and queries.

Efficient Similarity Search: Utilizes FAISS for fast and accurate similarity search.

Dynamic Context Expansion: Expands query results by including adjacent sentences that meet a dynamic similarity threshold.

Customizable Parameters: Allows customization of chunk size, similarity thresholds, and the number of top results to return.

Requirements
To run this project, you need the following dependencies:

Python 3.7 or higher

Libraries:

PyPDF2 (for PDF text extraction)

nltk (for sentence tokenization)

numpy (for numerical operations)

faiss (for similarity search)

openai (for generating text embeddings)

You can install the required libraries using pip:


pip install PyPDF2 nltk numpy faiss-cpu openai

Setup

Set OpenAI API Key:

Obtain an API key from OpenAI.

Set the API key as an environment variable:


export OPENAI_API_KEY="your_openai_api_key_here"
Download NLTK Resources:

The project uses NLTK's punkt tokenizer. If not already downloaded, it will be automatically installed.

Usage
Step 1: Initialize the Chatbot
from pdf_chatbot import PDFChatBot
# Initialize the chatbot with your OpenAI API key
bot = PDFChatBot(openai_api_key="your_openai_api_key_here")

Step 2: Load a PDF Document
# Load a PDF file and specify the maximum chunk size (default: 500 characters)
bot.load_pdf("path/to/your/document.pdf", max_chunk_size=600)

Step 3: Build the FAISS Index
# Build the FAISS index for similarity search
bot.build_index()

Step 4: Query the Document
# Query the document with a question
results = bot.query_with_context(
    question="smartphone addicted harmful",
    top_k=2,  # Number of top results to return
    similarity_ratio=0.85,  # Scaling factor for dynamic threshold
    min_return_threshold=0.60  # Minimum similarity score for results
)

# Display the results
for text, score in results:
    print(f"Similarity Score: {score:.3f}\nExpanded Content:\n{text}\n{'-'*50}")

# Example Output
Similarity Score: 0.644
Expanded Content:
It is also reported that smartphone addiction causes accidents at home, workplace, traffic, and so forth, due to its dis- tracting features (Ghazizadeh & Boyle, 2009 , Nasar et al., 2008 , Vladisavljevic et al., 2009 ). It is considered that this situation may also endanger patient safety and the safety of healthcare employees themselves.
--------------------------------------------------
Similarity Score: 0.605

Expanded Content:
One of the leading problems is smartphone addiction (nomophobia) (Hosgor et al., 2017 ), which is associated with a desire toconstantly check notifications (Oulasvirta et al., 2012 ), not turning off the phone all day, spending a long time on the phone before bed (Bragazzi & Del Puente, 2014 ), obsessive use, and increased anxiety level (Matusik & Mickel, 2011 ).
Customization
Chunk Size: Adjust the max_chunk_size parameter in load_pdf to control the size of text chunks.

Similarity Thresholds: Modify similarity_ratio and min_return_threshold in query_with_context to fine-tune the results.

Embedding Model: Change the embedding_model attribute in the PDFChatBot class to use a different OpenAI embedding model.

Limitations
PDF Quality: The accuracy of text extraction depends on the quality of the PDF file. Scanned or image-based PDFs may require OCR preprocessing.

API Costs: Using OpenAI's API for embeddings incurs costs. Monitor your usage to avoid unexpected charges.

Performance: Large PDF files may take longer to process and require more memory.
