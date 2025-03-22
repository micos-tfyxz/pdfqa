import os
import numpy as np
import faiss
import openai
import nltk
import pickle
import logging
from PyPDF2 import PdfReader
from typing import List, Tuple

# Configure logging to show warnings and above only
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class PDFChatBot:
    """
    Optimized PDF-based Q&A system with memory management and efficiency improvements.
    """
    _nltk_checked = False  # Class-level flag for NLTK resource check

    def __init__(self, openai_api_key: str, index_path: str = "pdf_index.faiss",
                 embedding_batch_size: int = 16, keep_in_memory: bool = True):
        self._check_nltk_resources()
        openai.api_key = openai_api_key
        self.text_chunks = []
        self.embeddings = None  # Precomputed embeddings
        self.embedding_model = "text-embedding-3-small"
        self.index_path = index_path  # Path to the index file
        self.index = None  # Lazy-loaded index
        self.embedding_batch_size = embedding_batch_size
        self.keep_in_memory = keep_in_memory

    def _check_nltk_resources(self):
        """Check NLTK resources only once."""
        if not self.__class__._nltk_checked:
            required_resources = ['punkt']
            for resource in required_resources:
                try:
                    nltk.data.find(f'tokenizers/{resource}')
                except LookupError:
                    nltk.download(resource, quiet=True)
            self.__class__._nltk_checked = True

    def load_pdf(self, file_path: str, max_chunk_size: int = 500) -> None:
        """Load PDF and split text into chunks."""
        reader = PdfReader(file_path)
        full_text = ' '.join([
            page.extract_text().replace('\n', ' ').strip()
            for page in reader.pages if page.extract_text()
        ])

        sentences = self._split_into_sentences(full_text)
        self.text_chunks = self._merge_sentences(sentences, max_chunk_size)
        # Print an indicator that PDF loading is complete
        print(f"PDF loaded successfully.")

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]

    def _merge_sentences(self, sentences: List[str], max_size: int) -> List[str]:
        """
        Iteratively merge sentences to avoid recursion.
        For sentences that exceed max_size, split by spaces into smaller chunks.
        """
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
                # Split long sentence by words
                words = sentence.split()
                sub_chunk = []
                sub_length = 0
                for word in words:
                    if sub_length + len(word) + 1 <= max_size:
                        sub_chunk.append(word)
                        sub_length += len(word) + 1
                    else:
                        chunks.append(" ".join(sub_chunk))
                        sub_chunk = [word]
                        sub_length = len(word) + 1
                if sub_chunk:
                    chunks.append(" ".join(sub_chunk))
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
        """Batch get embeddings using the specified batch size."""
        embeddings = []
        for i in range(0, len(texts), self.embedding_batch_size):
            batch = [t for t in texts[i:i+self.embedding_batch_size] if t.strip()]
            if not batch:
                continue
            try:
                response = openai.embeddings.create(input=batch, model=self.embedding_model)
            except Exception as e:
                logger.error(f"Error fetching embeddings: {e}")
                continue
            embeddings.extend([data.embedding for data in response.data])
        return np.array(embeddings, dtype=np.float32)

    def build_index(self) -> None:
        """Build and persist the index."""
        if not self.text_chunks:
            raise ValueError("Please load a PDF file first.")
        self.embeddings = self._get_embeddings(self.text_chunks)
        dimension = self.embeddings.shape[1]
        num_data = len(self.text_chunks)

        # Dynamic index selection
        if num_data < 1000:
            self.index = faiss.IndexFlatIP(dimension)
        else:
            nlist = min(int(np.sqrt(num_data)), 1000, num_data // 39)
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            if num_data < nlist * 39:
                logger.warning(f"Insufficient training data (current {num_data}, recommended {nlist * 39})")
            self.index.train(self.embeddings)

        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = min(4, nlist // 10 if num_data >= 1000 else 4)

        # Persist the index and embeddings
        faiss.write_index(self.index, self.index_path)
        with open(self.index_path + ".pkl", "wb") as f:
            pickle.dump((self.text_chunks, self.embeddings), f)

        # Optionally keep index and embeddings in memory
        if not self.keep_in_memory:
            self.embeddings = None
            self.index = None

    def load_index(self):
        """Load a pre-built index."""
        if os.path.exists(self.index_path) and os.path.exists(self.index_path + ".pkl"):
            self.index = faiss.read_index(self.index_path)
            with open(self.index_path + ".pkl", "rb") as f:
                self.text_chunks, self.embeddings = pickle.load(f)
        else:
            logger.warning("Index file not found, please build the index first.")

    def query_with_context(self, question: str, top_k: int = 1,
                           similarity_ratio: float = 0.75,
                           min_return_threshold: float = 0.75) -> List[Tuple[str, float]]:
        """Perform a context-aware query."""
        if self.index is None:
            self.load_index()

        if self.index is None or self.embeddings is None:
            raise ValueError("Index not loaded correctly, please build the index first.")

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

            if idx > 0:
                prev_sim = np.dot(self.embeddings[idx-1], query_embed[0])
                if prev_sim >= dynamic_thresh:
                    context.append(self.text_chunks[idx-1])
            if idx < len(self.text_chunks) - 1:
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
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")

    bot = PDFChatBot(openai_api_key=api_key, embedding_batch_size=16, keep_in_memory=True)

    # Try loading existing index
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
            print(f"Similarity: {score:.3f}")
            print(f"Context:\n{text}\n{'-'*50}")
