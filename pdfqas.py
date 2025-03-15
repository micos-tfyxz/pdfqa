import os
import numpy as np
import faiss
import openai
import nltk
from PyPDF2 import PdfReader
from typing import List, Tuple


class PDFChatBot:
    """
    Core class for the PDF-based Q&A system (Dynamic similarity threshold version)
    """

    def __init__(self, openai_api_key: str):
        self._check_nltk_resources()
        openai.api_key = openai_api_key
        self.text_chunks = []
        self.index = None
        self.embedding_model = "text-embedding-3-small"

    def _check_nltk_resources(self):
        """Verify and automatically download necessary NLTK resources"""
        required_resources = ['punkt']
        for resource in required_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                print(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True)

    def load_pdf(self, file_path: str, max_chunk_size: int = 500) -> None:
        """
        Improved PDF loading method: text segmentation based on semantics.
        Parameters:
            file_path: Path to the PDF file.
            max_chunk_size: Maximum number of characters per text chunk (default: 500).
        """
        reader = PdfReader(file_path)
        full_text = ' '.join([page.extract_text().replace('\n', ' ') 
                              for page in reader.pages if page.extract_text()])
        
        sentences = self._split_into_sentences(full_text)
        self.text_chunks = self._merge_sentences(sentences, max_chunk_size)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Use NLTK to split text into sentences and clean up."""
        sentences = nltk.sent_tokenize(text)
        return [s.strip() for s in sentences if len(s.strip()) > 0]

    def _merge_sentences(self, sentences: List[str], max_size: int) -> List[str]:
        """Intelligently merge sentences into text chunks."""
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sent_length = len(sentence)

            # Handle the case where a single sentence is too long
            if sent_length > max_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                chunks.append(sentence[:max_size])
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
        """Generate text embeddings using OpenAI API."""
        response = openai.embeddings.create(
            input=texts,
            model=self.embedding_model
        )
        return np.array([data.embedding for data in response.data], dtype=np.float32)

    def build_index(self) -> None:
        """Build FAISS index for efficient similarity search."""
        if not self.text_chunks:
            raise ValueError("Please load a PDF file first.")

        embeddings = self._get_embeddings(self.text_chunks)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def query_with_context(self, question: str, top_k: int = 1, 
                           similarity_ratio: float = 0.75,
                           min_return_threshold: float = 0.75) -> List[Tuple[str, float]]:
        """
        Query the text and attempt to expand its context dynamically.
        - First, find the most similar text chunk.
        - Then, dynamically adjust the similarity threshold for adjacent sentences.
        - Finally, if one or more candidate results have a similarity score above
          'min_return_threshold', return all of them; otherwise, return only the
          candidate with the highest similarity score.
        
        Parameters:
            question: The query text.
            top_k: Number of top similar sentences to consider.
            similarity_ratio: A scaling factor for calculating the dynamic threshold for
                              adjacent sentences (default: 0.75).
            min_return_threshold: The minimum similarity score for a candidate to be returned.
        
        Returns:
            List[Tuple(expanded_text, similarity_score)]
        """
        if self.index is None:
            raise ValueError("Index not built. Please call build_index() first.")

        # Compute query embedding
        query_embedding = self._get_embeddings([question])[0]
        query_embedding = np.expand_dims(query_embedding, axis=0)
        faiss.normalize_L2(query_embedding)  # Normalize embeddings

        # Perform similarity search
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            selected_text = self.text_chunks[idx]
            
            # Compute dynamic similarity threshold for adjacent sentences
            dynamic_threshold = score * similarity_ratio  

            expanded_text = selected_text  # Initialize expanded text
            context_sentences = []

            # Retrieve adjacent sentences if available
            prev_text = self.text_chunks[idx - 1] if idx > 0 else None
            next_text = self.text_chunks[idx + 1] if idx < len(self.text_chunks) - 1 else None

            # Compute similarity for adjacent sentences and decide whether to merge
            if prev_text:
                prev_embedding = self._get_embeddings([prev_text])[0]
                prev_sim = np.dot(prev_embedding, query_embedding[0])  # Cosine similarity
                if prev_sim >= dynamic_threshold:
                    context_sentences.append(prev_text)

            if next_text:
                next_embedding = self._get_embeddings([next_text])[0]
                next_sim = np.dot(next_embedding, query_embedding[0])  # Cosine similarity
                if next_sim >= dynamic_threshold:
                    context_sentences.append(next_text)

            # Merge final text
            if context_sentences:
                expanded_text = " ".join(context_sentences + [selected_text])
            
            results.append((expanded_text, score))
        
        # New feature: Filtering results based on the min_return_threshold
        filtered_results = [res for res in results if res[1] >= min_return_threshold]
        if filtered_results:
            return filtered_results
        else:
            best_result = max(results, key=lambda x: x[1])
            return [best_result]


# Example Usage --------------------------------------------------
if __name__ == "__main__":
    # Retrieve API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the environment variable OPENAI_API_KEY")
    
    # Initialize the chatbot instance
    bot = PDFChatBot(openai_api_key=api_key)
    
    # Load the PDF document
    pdf_path = input("Enter the path to your PDF file: ")
    bot.load_pdf(pdf_path, max_chunk_size=600)
    
    # Build vector index
    bot.build_index()
    
    # Perform query
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        
        results = bot.query_with_context(question, top_k=2, 
                                         similarity_ratio=0.85, min_return_threshold=0.60)
        
        for text, score in results:
            print(f"Similarity Score: {score:.3f}\nExpanded Content:\n{text}\n{'-'*50}")
