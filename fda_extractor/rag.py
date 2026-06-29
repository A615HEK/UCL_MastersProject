# RAG Portions
import faiss
import numpy as np
from time import perf_counter
from functools import lru_cache
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from .schema import DETAILS_REQUIRED

RETRIEVAL_QUERY_PROMPT = "You are part of an automated machine interface that specialises in scanning through FDA regulatory documents to find the requested fields that are present in the document provided to you and packaging these outputs into only valid JSON outputs. Given above are the contents from various regulatory documents on a particular product. I require the following details from the document: " + ", ".join(DETAILS_REQUIRED) + ". Return None for the CE_Classification if it is not specified. Give me these information in a JSON format only."

@lru_cache(maxsize=4)
def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Cached -> repeated calls (e.g. from a web backend)
    don't reload model weights from disk every time
    """
    return SentenceTransformer(model_name)

## Chunker
def chunker(doc_contents: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """
    Split the document string into chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(doc_contents)

## Embedding and Vector DB Creation
def embedder(chunks: list[str], model_name: str = "all-MiniLM-L6-v2"):
    """
    Embeds the chunks and builds a FAISS L2 vector search index
    """
    embedding_model = load_embedding_model(model_name)
    start_time = perf_counter()
    chunk_embeddings = embedding_model.encode(chunks)
    print(f"Embedding time: {perf_counter() - start_time}")
    index = faiss.IndexFlatL2(len(chunk_embeddings[0]))
    index.add(chunk_embeddings.astype(np.float32))
    return index, embedding_model

## Similarity Search and Retrieval
def retriever(index, model: SentenceTransformer, query_prompt: str, num_rel_chunks_to_return: int = 7):
    """
    Retrieves the relevant documents based off of the passed query prompt
    """
    query_embeddings = model.encode(query_prompt).reshape((1,-1))
    distances, indices = index.search(query_embeddings.astype(np.float32), num_rel_chunks_to_return)
    return distances, indices

def rag(doc_contents: str, query: str = RETRIEVAL_QUERY_PROMPT, num_rel_chunks_to_return: int = 7, model_name: str = "all-MiniLM-L6-v2"):
    """
    Consolidated function to perform RAG using all-MiniLM-L6-v2
    """
    chunks_after_split = chunker(doc_contents)
    index, embedding_model = embedder(chunks_after_split, model_name)
    _, indices = retriever(index, embedding_model, query, num_rel_chunks_to_return)
    relevant_chunks = "".join([chunks_after_split[i] for i in indices[0]])
    return relevant_chunks