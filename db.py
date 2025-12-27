# db.py
import os
from typing import List, Dict
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

DEFAULT_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_news_db")

class NewsVectorDB:
    def __init__(self, persist_dir: str = None, embedding_model: str = "models/gemini-embedding-001"):
        self.persist_dir = persist_dir or DEFAULT_PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        self.embedding_model = embedding_model
        self.embeddings = GoogleGenerativeAIEmbeddings(model=self.embedding_model)

    def _doc_from_article(self, article: Dict) -> Document:
        metadata = {
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "published_at": article.get("published_at", ""),
            "source": article.get("source", ""),
        }
        content = f"{article.get('title','')}\n\n{article.get('content','')}"
        return Document(page_content=content, metadata=metadata)

    def add_articles(self, articles: List[Dict], replace: bool = False):
        docs = [self._doc_from_article(a) for a in articles if a.get("content") or a.get("title")]
        if not docs:
            return 0
        if replace:
            # create new DB
            vectordb = Chroma.from_documents(documents=docs, embedding=self.embeddings, persist_directory=self.persist_dir)
        else:
            # load if exists else create
            try:
                vectordb = Chroma(embedding_function=self.embeddings, persist_directory=self.persist_dir)
                vectordb.add_documents(docs)
            except Exception:
                vectordb = Chroma.from_documents(documents=docs, embedding=self.embeddings, persist_directory=self.persist_dir)
        vectordb.persist()
        return len(docs)

    def as_retriever(self, k: int = 4):
        vectordb = Chroma(embedding_function=self.embeddings, persist_directory=self.persist_dir)
        return vectordb.as_retriever(search_kwargs={"k": k})
