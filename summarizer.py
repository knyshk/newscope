# summarizer.py
import os
from typing import Tuple, List
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

def get_gemini_llm(model_name: str = "gemini-2.0-flash", temperature: float = 0.0):
    # ChatGoogleGenerativeAI uses GOOGLE_API_KEY from env
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

def make_qa_chain(retriever, model_name: str = "gemini-2.0-flash"):
    llm = get_gemini_llm(model_name=model_name, temperature=0.0)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="map_reduce", retriever=retriever, return_source_documents=True)
    return qa

def answer_query(qa_chain, query: str) -> Tuple[str, List]:
    # returns (answer_text, source_documents_list)
    answer = qa_chain.run(query)
    # RetrievalQA.from_chain_type(..., return_source_documents=True) usually returns dict or tuple depending on version
    if isinstance(answer, dict):
        text = answer.get("result") or answer.get("answer") or str(answer)
        sources = answer.get("source_documents") or []
    else:
        # The run() can return plain text while qa_chain(...) returns a dict
        text = answer
        # Try to call qa_chain with return_source_documents using __call__
        try:
            out = qa_chain({"query": query}, return_only_outputs=False)
            sources = out.get("source_documents", [])
        except Exception:
            sources = []
    return text, sources
