
def generate_answer(query, retrieved_chunks, tokenizer, rag_pipeline, max_chunks=3, max_chunk_tokens=500):
    """
    Generate answer using retrieved chunks and LLM
    """
    retrieved_chunks = retrieved_chunks[:max_chunks]
    truncated_chunks = []
    
    for c in retrieved_chunks:
        tokens = tokenizer(c["chunk"], truncation=True, max_length=max_chunk_tokens)["input_ids"]
        truncated_chunks.append(tokenizer.decode(tokens, skip_special_tokens=True))
    
    context = "\n\n".join(truncated_chunks)
    
    prompt = f"""You are a helpful research assistant that focuses on the area of Energy & Sustainability.
Use the context below to answer the question in 5-10 complete sentences.

Context:
{context}

Question: {query}

Answer:"""
    
    response = rag_pipeline(prompt)[0]["generated_text"]
    
    if "Answer:" in response:
        answer = response.split("Answer:")[-1].strip()
    else:
        answer = response.strip()
    
    return answer
