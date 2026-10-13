import nltk
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

def run_chunking(file_path, chunk_size=150, overlap=20):
    """Semantic chunking that preserves newlines and formatting"""
    text = extract_text_from_file(file_path)
    
    if not text:
        return []
    
    # Split by newlines to preserve semantic boundaries (paragraphs, bullet points)
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_len = len(para.split())
        
        if current_length + para_len > chunk_size and current_chunk:
            # Save the accumulated chunk, preserving newlines
            chunks.append("\n".join(current_chunk))
            
            # Create the overlap by keeping the last few paragraphs
            overlap_chunk = []
            overlap_length = 0
            
            for prev_para in reversed(current_chunk):
                if overlap_length + len(prev_para.split()) <= overlap:
                    overlap_chunk.insert(0, prev_para)
                    overlap_length += len(prev_para.split())
                else:
                    break
                    
            current_chunk = overlap_chunk
            current_length = overlap_length
            
        current_chunk.append(para)
        current_length += para_len
        
    if current_chunk:
        chunks.append("\n".join(current_chunk))
        
    return chunks
