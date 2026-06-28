from time import perf_counter
# RAG Imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# RAG Portions
## Chunker
def Chunker(doc_contents:str):
    """
    Split the document string into chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_after_split = text_splitter.split_text(doc_contents)
    return docs_after_split

## Embedding and Vector DB Creation
def Embedder(docs_after_split:list[str], model_name:str = "all-MiniLM-L6-v2"):
    embedding_model = SentenceTransformer(model_name)
    details_required = ["AI Product name", "AI Company name", "Sub-specialty", "Imaging Modality", "Intended Purpose", "Summary paragraph/sentence of Clinical use", "Summary paragraph/sentence of Product output", "CE Classification", "FDA Classification", "Manufacturer"]
    trial_ollama_prompt = "You are part of an automated machine interface that specialises in scanning through FDA regulatory documents to find the requested fields that are present in the document provided to you and packaging these outputs into only valid JSON outputs. Given above are the contents from various regulatory documents on a particular product. I require the following details from the document: " + ", ".join(details_required) + ". Return None for the CE_Classification if it is not specified. Give me these information in a JSON format only."
    start_time = perf_counter()
    doc_embeddings = embedding_model.encode(docs_after_split)
    print(f"Embedding time: {perf_counter() - start_time}")
    index = faiss.IndexFlatL2(len(doc_embeddings[0]))
    index.add(doc_embeddings.astype(np.float32))
    query_embeddings = embedding_model.encode(trial_ollama_prompt).reshape((1,-1))
    return index, query_embeddings

## Similarity Search and Retrieval
def Retriever(index, query_embeddings, num_rel_docs_to_return:int = 7):
    distances, indices = index.search(query_embeddings.astype(np.float32), num_rel_docs_to_return)
    return distances, indices

def RAG(doc_contents:str, num_rel_docs_to_return:int = 7, model_name:str = "all-MiniLM-L6-v2"):
    """
    Consolidated function to perform RAG using all-MiniLM-L6-v2
    """
    docs_after_split = Chunker(doc_contents)
    index, query_embeddings = Embedder(docs_after_split, model_name)
    _, indices = Retriever(index, query_embeddings, num_rel_docs_to_return)
    relevant_doc_contents = "".join([docs_after_split[i] for i in indices[0]])
    return relevant_doc_contents