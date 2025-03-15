import os
import numpy as np
import faiss
import openai
import nltk
import io
import pickle 
from PyPDF2 import PdfReader
from typing import List, Tuple

class PDFChatBot:
    def __init__(self, openai_api_key: str, index_path: str = "pdf_index.faiss"):
        self._check_nltk_resources()
        openai.api_key = openai_api_key
        self.text_chunks = []
        self.text_embeddings = None  # Only used during index building
        self.embedding_model = "text-embedding-3-small"
        self.index_path = index_path
        self.index = None  # Only loaded during query

    def _check_nltk_resources(self):
        """Ensure nltk resources are available"""
        required_resources = ['punkt']
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)

    def load_pdf(self, file_path: str, max_chunk_size: int = 500) -> None:
        """Load PDF and split text into small chunks"""
        reader = PdfReader(file_path)
        full_text = ' '.join([
            page.extract_text().replace('\n', ' ').strip()
            for page in reader.pages if page.extract_text()
        ])
        
        sentences = self._split_into_sentences(full_text)
        self.text_chunks = self._merge_sentences(sentences, max_chunk_size)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if len(s.strip()) > 0]

    def _merge_sentences(self, sentences: List[str], max_size: int) -> List[str]:
        """Merge sentences to ensure each text chunk does not exceed max_size"""
        chunks = []
        buffer = io.StringIO()
        current_length = 0

        for sentence in sentences:
            sent_length = len(sentence)

            if sent_length > max_size:
                if buffer.tell() > 0:
                    chunks.append(buffer.getvalue().strip())
                    buffer = io.StringIO()
                chunks.append(sentence[:max_size])
                continue

            if current_length + sent_length <= max_size:
                buffer.write(sentence + " ")
                current_length += sent_length
            else:
                chunks.append(buffer.getvalue().strip())
                buffer = io.StringIO()
                buffer.write(sentence + " ")
                current_length = sent_length

        if buffer.tell() > 0:
            chunks.append(buffer.getvalue().strip())

        return chunks

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute OpenAI embeddings for text chunks"""
        batch_size = 16
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = [text for text in texts[i:i + batch_size] if isinstance(text, str) and text.strip()]
            if not batch_texts:
                continue

            try:
                response = openai.embeddings.create(input=batch_texts, model=self.embedding_model)
                embeddings.extend([data.embedding for data in response.data])
            except Exception as e:
                print(f"⚠ Embedding computation failed: {e}")
                return np.empty((0, 1536), dtype=np.float32)

        return np.array(embeddings, dtype=np.float32) if embeddings else np.empty((0, 1536), dtype=np.float32)

    def build_index(self) -> None:
        """Build FAISS index and save to disk"""
        if not self.text_chunks:
            raise ValueError("Please load a PDF file first!")

        self.text_embeddings = self._get_embeddings(self.text_chunks)
        faiss.normalize_L2(self.text_embeddings)
        dimension = self.text_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(self.text_embeddings)

        # Save index & text chunks
        faiss.write_index(index, self.index_path)
        print(f"Index saved to {self.index_path}")

        with open(self.index_path + ".pkl", "wb") as f:
            pickle.dump(self.text_chunks, f)
        print(f"Text chunks saved to {self.index_path}.pkl")

        # Free memory
        self.text_embeddings = None
        self.index = None  

    def load_index(self):
        """Load FAISS index and text chunks from disk"""
        if os.path.exists(self.index_path) and os.path.exists(self.index_path + ".pkl"):
            self.index = faiss.read_index(self.index_path)
            print(f"Index loaded from {self.index_path}")

            with open(self.index_path + ".pkl", "rb") as f:
                self.text_chunks = pickle.load(f)
            print(f"Text chunks loaded from {self.index_path}.pkl")
        else:
            print("⚠ Index or text chunk files not found")

    def query_with_context(self, question: str, top_k: int = 1, 
                           similarity_ratio: float = 0.75,
                           min_return_threshold: float = 0.75) -> List[Tuple[str, float]]:
        """Query PDF and expand context"""
        if self.index is None:
            self.load_index()

        if self.index is None or not self.text_chunks:
            raise ValueError("Index or text chunks not loaded correctly, please call build_index() first")

        query_embedding = self._get_embeddings([question])[0]
        query_embedding = np.expand_dims(query_embedding, axis=0)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= len(self.text_chunks):  # Avoid index out of range
                continue  

            selected_text = self.text_chunks[idx]
            dynamic_threshold = score * similarity_ratio  
            expanded_text = selected_text
            context_sentences = []

            prev_text = self.text_chunks[idx - 1] if idx > 0 else None
            next_text = self.text_chunks[idx + 1] if idx < len(self.text_chunks) - 1 else None

            if prev_text:
                context_sentences.append(prev_text)
            if next_text:
                context_sentences.append(next_text)

            if context_sentences:
                expanded_text = " ".join(context_sentences + [selected_text])
            
            results.append((expanded_text, score))
        
        return [res for res in results if res[1] >= min_return_threshold] or [max(results, key=lambda x: x[1])]

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    bot = PDFChatBot(openai_api_key=api_key)

    bot.load_index()

    if bot.index is None or not bot.text_chunks:
        pdf_path = input("Enter the path to your PDF file: ")
        bot.load_pdf(pdf_path, max_chunk_size=600)
        bot.build_index()

    while True:
        question = input("Please enter your question (type 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        results = bot.query_with_context(question, top_k=3)
        for text, score in results:
            print(f"Similarity: {score:.3f}\n{text}\n{'-'*50}")

    bot.index = None  
