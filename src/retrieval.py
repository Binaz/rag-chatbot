
# Using 'all-MiniLM-L6-v2' model from huggingFace.
# This is a sentence-transformers model and it maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.
# Using FAISS developed by meta. Faiss is a library for efficient similarity search and clustering of dense vectors.


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def build_index(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Create embeddings and build FAISS index"""
    model = SentenceTransformer(model_name)
    texts = [c["chunk"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    return model, index

def retrieve(query, model, index, chunks, top_k=5):
    """Retrieve top-k relevant chunks for a query"""
    query_emb = model.encode([query])
    D, I = index.search(query_emb, top_k)
    results = [chunks[i] for i in I[0]]
    return results
