import os
import numpy as np
import faiss
import openai
import nltk
import pickle
from PyPDF2 import PdfReader
from typing import List, Tuple


class PDFChatBot:
    """
    Optimized PDF-based Q&A system with memory management enhancements
    """
    _nltk_checked = False  # Class-level flag for NLTK resource check

    def __init__(self, openai_api_key: str, index_path: str = "pdf_index.faiss"):
        self._check_nltk_resources()
        openai.api_key = openai_api_key
        self.text_chunks = []
        self.embeddings = None  # Precomputed embeddings
        self.embedding_model = "text-embedding-3-small"
        self.index_path = index_path  # Path to the index file
        self.index = None  # Lazy-loaded index

    def _check_nltk_resources(self):
        """Check NLTK resources once"""
        if not self.__class__._nltk_checked:
            required_resources = ['punkt']
            for resource in required_resources:
                try:
                    nltk.data.find(f'tokenizers/{resource}')
                except LookupError:
                    print(f"Downloading NLTK resource: {resource}")
                    nltk.download(resource, quiet=True)
            self.__class__._nltk_checked = True

    def load_pdf(self, file_path: str, max_chunk_size: int = 500) -> None:
        """Load and chunk PDF text"""
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

    def _merge_sentences(self, sentences: List[str], max_size: int, recursion_depth: int = 0) -> List[str]:
        """Intelligent chunking algorithm"""
        MAX_RECURSION_DEPTH = 10
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sent_length = len(sentence)

            if sent_length > max_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                if recursion_depth >= MAX_RECURSION_DEPTH:
                    chunks.append(sentence[:max_size].rsplit(' ', 1)[0])
                    continue
                    
                sub_sentences = self._split_into_sentences(sentence)
                if len(sub_sentences) == 1:
                    truncated = sentence[:max_size].rsplit(' ', 1)[0]
                    chunks.append(truncated)
                    remaining = sentence[len(truncated):].lstrip()
                    if remaining:
                        chunks.extend(self._merge_sentences([remaining], max_size, recursion_depth + 1))
                else:
                    chunks.extend(self._merge_sentences(sub_sentences, max_size, recursion_depth + 1))
                continue
                    
            if current_length + sent_length <= max_size:
                current_chunk.append(sentence)
                current_length += sent_length
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sent_length
                    
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Batch get embeddings"""
        batch_size = 16
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = [t for t in texts[i:i+batch_size] if t.strip()]
            if not batch:
                continue
            response = openai.embeddings.create(input=batch, model=self.embedding_model)
            embeddings.extend([data.embedding for data in response.data])
        return np.array(embeddings, dtype=np.float32)

    def build_index(self) -> None:
        """Build and persist index"""
        if not self.text_chunks:
            raise ValueError("Please load a PDF file first")

        self.embeddings = self._get_embeddings(self.text_chunks)
        dimension = self.embeddings.shape[1]
        num_data = len(self.text_chunks)

        # Dynamic indexing strategy
        if num_data < 1000:
            self.index = faiss.IndexFlatIP(dimension)
        else:
            nlist = min(int(np.sqrt(num_data)), 1000, num_data // 39)
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            if num_data < nlist * 39:
                print(f"Warning: Training data may be insufficient (current {num_data}, recommended {nlist*39})")
            
            self.index.train(self.embeddings)

        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = min(4, nlist//10)

        # Persist storage
        faiss.write_index(self.index, self.index_path)
        with open(self.index_path + ".pkl", "wb") as f:
            pickle.dump((self.text_chunks, self.embeddings), f)
        print(f"Index saved to {self.index_path}")

        # Memory cleanup
        self.embeddings = None
        self.index = None

    def load_index(self):
        """Load pre-built index"""
        if os.path.exists(self.index_path) and os.path.exists(self.index_path + ".pkl"):
            self.index = faiss.read_index(self.index_path)
            with open(self.index_path + ".pkl", "rb") as f:
                self.text_chunks, self.embeddings = pickle.load(f)
            print(f"Loaded pre-built index: {self.index_path}")
        else:
            print("⚠ Index file not found, please build the index first")

    def query_with_context(self, question: str, top_k: int = 1, 
                          similarity_ratio: float = 0.75,
                          min_return_threshold: float = 0.75) -> List[Tuple[str, float]]:
        """Context-aware query"""
        if self.index is None:
            self.load_index()

        if self.index is None or self.embeddings is None:
            raise ValueError("Index not loaded correctly, please build the index first")

        query_embed = self._get_embeddings([question])[0]
        query_embed = np.expand_dims(query_embed, axis=0)
        faiss.normalize_L2(query_embed)

        scores, indices = self.index.search(query_embed, top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= len(self.text_chunks):
                continue

            main_text = self.text_chunks[idx]
            dynamic_thresh = score * similarity_ratio
            context = []

            # Filter context based on precomputed embeddings
            if idx > 0:
                prev_sim = np.dot(self.embeddings[idx-1], query_embed[0])
                if prev_sim >= dynamic_thresh:
                    context.append(self.text_chunks[idx-1])
            
            if idx < len(self.text_chunks)-1:
                next_sim = np.dot(self.embeddings[idx+1], query_embed[0])
                if next_sim >= dynamic_thresh:
                    context.append(self.text_chunks[idx+1])

            combined = " ".join(context + [main_text])
            results.append((combined, float(score)))
        
        filtered = [r for r in results if r[1] >= min_return_threshold]
        return filtered if filtered else [max(results, key=lambda x: x[1])]


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    bot = PDFChatBot(openai_api_key=api_key)
    
    # Try to load existing index first
    bot.load_index()
    
    if not bot.index:
        pdf_path = input("Enter PDF file path: ")
        bot.load_pdf(pdf_path, max_chunk_size=600)
        bot.build_index()
    
    while True:
        query = input("Enter your question (type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        results = bot.query_with_context(
            query, 
            top_k=2,
            similarity_ratio=0.85,
            min_return_threshold=0.60
        )
        
        for text, score in results:
            print(f"▏Similarity: {score:.3f}")
            print(f"▏Context: \n{text}\n{'-'*50}")
