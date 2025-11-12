# -*- coding: utf-8 -*-

# main.py
# -------------------------------
# Energy & Sustainability RAG Chatbot
# -------------------------------

# Install required packages (for Colab)
import sys
!{sys.executable} -m pip install pymupdf sentence-transformers faiss-cpu transformers accelerate bitsandbytes gradio torch --quiet

# -------------------------------
# Imports
# -------------------------------
from src.extract_text import load_pdfs
from src.retrieval import build_index, retrieve
from src.generate_answer import generate_answer
from src.chat_interface import create_gradio

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -------------------------------
# Step 1: Load PDFs
# -------------------------------
pdf_folder = "data/pdfs"  # make sure PDFs are here
pdf_texts = load_pdfs(pdf_folder)

chunks = []
for pdf in pdf_texts:
    # naive chunking: split text into ~500 word chunks
    words = pdf["text"].split()
    for i in range(0, len(words), 500):
        chunks.append({"filename": pdf["filename"], "chunk": " ".join(words[i:i+500])})

print(f"Loaded {len(pdf_texts)} PDFs, created {len(chunks)} chunks.")

# -------------------------------
# Step 2: Build FAISS index
# -------------------------------
model_emb, index = build_index(chunks)
print(f"FAISS index built with {index.ntotal} vectors.")

# -------------------------------
# Step 3: Load LLM (Phi-2)
# -------------------------------
model_name = "microsoft/phi-2"

print("Loading LLM... this may take a few minutes.")
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

rag_pipeline = pipeline(
    "text-generation",
    model=llm,
    tokenizer=tokenizer,
    max_new_tokens=350,
    temperature=0.4,      # higher = more diverse
    top_p=0.9,            # nucleus sampling
    do_sample=False,       # ensures sampling instead of greedy
    return_full_text=False
)

# -------------------------------
# Step 4: Define chat function
# -------------------------------
def chat_fn(user_input):
    retrieved = retrieve(user_input, model_emb, index, chunks)
    return generate_answer(user_input, retrieved, tokenizer, rag_pipeline)

# -------------------------------
# Step 5: Launch Gradio interface
# -------------------------------
create_gradio(chat_fn, title="Energy & Sustainability RAG Chatbot")

