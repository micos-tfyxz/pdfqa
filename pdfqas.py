import os
import numpy as np
import faiss
import nltk
import openai
from PyPDF2 import PdfReader
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

class PDFChatBot:
    """优化后的基于 PDF 的问答系统，重点提升单次查询速度"""
    def __init__(self, openai_api_key: str):
        openai.api_key = openai_api_key
        self.text_chunks = []
        self.index = None
        self.embeddings = None
        self.embedding_model = "text-embedding-3-small"
        self.embedding_cache = {}  # 缓存问题的嵌入向量
        self.pdf_text = None       # 缓存 PDF 全文，避免重复解析

    def _check_nltk_resources(self):
        if not self.__class__._nltk_checked:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            self.__class__._nltk_checked = True

    def load_pdf(self, file_path: str, max_chunk_size: int = 500) -> None:
        """
        并行提取 PDF 各页文本，并进行固定窗口切分。
        如果 PDF 内容较大，并行提取能提高加载速度。
        """
        reader = PdfReader(file_path)

        def extract_page_text(page):
            text = page.extract_text()
            return text.replace('\n', ' ') if text else ''
        
        with ThreadPoolExecutor() as executor:
            pages_text = list(executor.map(extract_page_text, reader.pages))
        
        self.pdf_text = ' '.join([text for text in pages_text if text])
        self.text_chunks = [self.pdf_text[i:i+max_chunk_size] 
                            for i in range(0, len(self.pdf_text), max_chunk_size)]

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        批量调用 OpenAI 接口生成文本嵌入（每批次最多 2048 条）。
        """
        embeddings = []
        batch_size = 2048  # OpenAI 的最大批处理大小
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            response = openai.embeddings.create(
                input=batch,
                model=self.embedding_model
            )
            embeddings.extend([data.embedding for data in response.data])
        return np.array(embeddings, dtype=np.float32)

    def build_index(self) -> None:
        """
        生成 PDF 文本块的嵌入并构建 FAISS Flat 索引。
        由于 PDF 内容不变，嵌入结果被缓存以提高后续查询速度。
        """
        if not self.text_chunks:
            raise ValueError("请先加载 PDF 文件。")
        if self.embeddings is None:
            self.embeddings = self._get_embeddings(self.text_chunks)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

    def query(self, question: str, top_k: int = 2) -> List[Tuple[str, float]]:
        """
        查询时对问题生成嵌入，并利用缓存减少重复调用。
        采用 FAISS 索引直接返回最匹配的文本块。
        """
        if self.index is None or self.embeddings is None:
            raise ValueError("索引尚未构建，请先调用 build_index()。")
        
        # 使用缓存避免重复生成嵌入
        if question in self.embedding_cache:
            query_embedding = self.embedding_cache[question]
        else:
            query_embedding = self._get_embeddings([question])[0]
            self.embedding_cache[question] = query_embedding

        query_embedding = np.expand_dims(query_embedding, axis=0)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)
        return [(self.text_chunks[idx], float(score)) for idx, score in zip(indices[0], scores[0])]

# 优化后的示例使用方式 --------------------------------------------------
if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请设置 OPENAI_API_KEY 环境变量")
    
    bot = PDFChatBot(openai_api_key=api_key)
    
    pdf_path = input("请输入 PDF 文件路径：")
    bot.load_pdf(pdf_path, max_chunk_size=600)
    bot.build_index()
    
    while True:
        question = input("问题（输入 'exit' 退出）：")
        if question.lower() == 'exit':
            break
        
        results = bot.query(question, top_k=2)
        for text, score in results:
            print(f"Score: {score:.3f}\nResult:\n{text}\n{'-'*50}")
