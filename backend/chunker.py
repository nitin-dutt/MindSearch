import nltk
from nltk.tokenize import sent_tokenize
import os
import chardet
import pdfplumber
from docx import Document

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def extract_text_from_file(file_path):
    """Extract text from various file formats"""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    try:
        if ext == '.pdf':
            return extract_text_from_pdf(file_path)
        elif ext == '.docx':
            return extract_text_from_docx(file_path)
        elif ext == '.txt':
            return extract_text_from_txt(file_path)
        else:
            # Try as text with encoding detection
            return extract_text_from_txt(file_path)
    except Exception as e:
        print(f"Error extracting text from {file_path}: {str(e)}")
        return ""

def extract_text_from_pdf(file_path):
    """Extract text from PDF files"""
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(file_path):
    """Extract text from DOCX files"""
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error reading DOCX: {str(e)}")
        return ""

def extract_text_from_txt(file_path):
    """Extract text from TXT files with encoding detection"""
    try:
        # First try UTF-8
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            # Detect encoding
            with open(file_path, 'rb') as f:
                raw = f.read()
                result = chardet.detect(raw)
                encoding = result.get('encoding', 'utf-8')
            
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            print(f"Error detecting encoding: {str(e)}")
            # Last resort: ignore errors
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

def run_chunking(file_path, chunk_size=400):
    """Chunk text from various file formats"""
    text = extract_text_from_file(file_path)
    
    if not text:
        return []
    
    sentences = sent_tokenize(text)
    chunks = []
    current = []

    for sent in sentences:
        current.append(sent)
        if sum(len(s.split()) for s in current) >= chunk_size:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks
