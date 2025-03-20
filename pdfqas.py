import os
import numpy as np
import faiss
import openai
import nltk
from PyPDF2 import PdfReader
from typing import List, Tuple


class PDFChatBot:
    """Optimized PDF-based Q&A system for maximum single-query speed"""
    _nltk_checked = False

    def __init__(self, openai_api_key: str):
        self._check_nltk_resources()
        openai.api_key = openai_api_key
        self.text_chunks = []
        self.index = None
        self.embeddings = None
        self.embedding_model = "text-embedding-3-small"

    def _check_nltk_resources(self):
        if not self.__class__._nltk_checked:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            self.__class__._nltk_checked = True

    def load_pdf(self, file_path: str, max_chunk_size: int = 500) -> None:
        """Fixed window chunking without sentence merging"""
        reader = PdfReader(file_path)
        full_text = ' '.join([page.extract_text().replace('\n', ' ') 
                             for page in reader.pages if page.extract_text()])
        
        # Direct fixed-size chunking
        self.text_chunks = [full_text[i:i+max_chunk_size] 
                           for i in range(0, len(full_text), max_chunk_size)]

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Batch processing with maximum API batch size"""
        embeddings = []
        batch_size = 2048  # OpenAI max batch size
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = openai.embeddings.create(
                input=batch,
                model=self.embedding_model
            )
            embeddings.extend([data.embedding for data in response.data])
        return np.array(embeddings, dtype=np.float32)

    def build_index(self) -> None:
        """Force Flat index for fastest search"""
        if not self.text_chunks:
            raise ValueError("Please load a PDF file first.")

        self.embeddings = self._get_embeddings(self.text_chunks)
        dimension = self.embeddings.shape[1]
        
        # Always use Flat index
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.embeddings)  # Keep if embeddings not pre-normalized
        self.index.add(self.embeddings)

    def query(self, question: str, top_k: int = 2) -> List[Tuple[str, float]]:
        """Simplified query without context expansion"""
        if self.index is None or self.embeddings is None:
            raise ValueError("Index not built. Please call build_index() first.")

        # Single API call for query embedding
        query_embedding = self._get_embeddings([question])[0]
        query_embedding = np.expand_dims(query_embedding, axis=0)
        faiss.normalize_L2(query_embedding)

        # Direct search
        scores, indices = self.index.search(query_embedding, top_k)
        return [(self.text_chunks[idx], float(score)) 
               for idx, score in zip(indices[0], scores[0])]


# Optimized Example Usage --------------------------------------------------
if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    bot = PDFChatBot(openai_api_key=api_key)
    
    pdf_path = input("Enter PDF path: ")
    bot.load_pdf(pdf_path, max_chunk_size=600)
    bot.build_index()
    
    while True:
        question = input("Question (type 'exit'): ")
        if question.lower() == 'exit':
            break
        
        results = bot.query(question, top_k=2)
        for text, score in results:
            print(f"Score: {score:.3f}\nResult:\n{text}\n{'-'*50}")
