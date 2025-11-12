
# Extract Text from PDFs

# Using PyMuPDF (fitz) to extract Text from PDFs.

import fitz
import os

def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def load_pdfs(pdf_folder):
    """Load all PDFs from a folder and extract text"""
    pdf_texts = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            path = os.path.join(pdf_folder, file)
            text = extract_text_from_pdf(path)
            pdf_texts.append({"filename": file, "text": text})
    return pdf_texts
