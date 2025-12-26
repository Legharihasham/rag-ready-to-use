import os
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdfs_from_directory(directory_path):
    """
    Load all PDFs from a specified directory
    
    Args:
        directory_path: Path to the directory containing PDF files
        
    Returns:
        Dictionary with filename as key and text content as value
    """
    pdf_contents = {}
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page_num in range(len(pdf_reader.pages)):
                            text += pdf_reader.pages[page_num].extract_text()
                        
                        if text.strip():  # Only add if text was extracted
                            pdf_contents[file] = text
                        else:
                            print(f"Warning: No text extracted from {file}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
    return pdf_contents

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """
    Split text into smaller chunks for processing
    
    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    return chunks

def process_pdf_directory(directory_path, chunk_size=800, chunk_overlap=250):
    """
    Process all PDFs in a directory and return chunks with metadata
    
    Args:
        directory_path: Path to directory with PDFs
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of dictionaries with text chunks and metadata
    """
    pdf_contents = load_pdfs_from_directory(directory_path)
    all_chunks = []
    
    for filename, content in pdf_contents.items():
        chunks = split_text_into_chunks(content, chunk_size, chunk_overlap)
        
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "source": filename,
                    "chunk_id": i,
                    "type": "pdf"
                }
            })
    
    return all_chunks 