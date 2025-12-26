import os
from pdf_loader import process_pdf_directory
from web_scraper import main as process_web_links
from embeddings_manager import EmbeddingsManager

def main():
    """
    Process all PDFs and web links, then generate combined embeddings
    """
    # Process the main PDFs directory with optimized chunk sizes
    pdf_dir = os.path.join("Data", "PDF's")
    print(f"Processing PDFs in {pdf_dir}...")
    # Smaller chunk size and higher overlap for better context preservation
    main_chunks = process_pdf_directory(pdf_dir, chunk_size=500, chunk_overlap=200)
    print(f"Processed {len(main_chunks)} chunks from main PDFs")
    
    # Process the fee structure PDFs - using smaller chunks for fee tables
    fee_dir = os.path.join("Data", "Fee_structure")
    print(f"Processing PDFs in {fee_dir}...")
    fee_chunks = process_pdf_directory(fee_dir, chunk_size=400, chunk_overlap=200)
    print(f"Processed {len(fee_chunks)} chunks from fee structure PDFs")
    
    # Process web links - web content often needs larger chunks
    links_file = os.path.join("Data", "Links.txt")
    print(f"Processing web links from {links_file}...")
    web_chunks = process_web_links(links_file, chunk_size=600, chunk_overlap=200)
    print(f"Processed {len(web_chunks)} chunks from web links")
    
    # Combine all chunks
    all_chunks = main_chunks + fee_chunks + web_chunks
    print(f"Total chunks: {len(all_chunks)}")
    
    # Create embeddings with new model
    print("Creating embeddings with BGE model...")
    embeddings_manager = EmbeddingsManager(model_name="BAAI/bge-base-en-v1.5")
    embeddings = embeddings_manager.create_embeddings(all_chunks)
    print(f"Created embeddings with shape: {embeddings.shape}")
    
    # Save embeddings
    print("Saving embeddings...")
    index_path, chunks_path = embeddings_manager.save_embeddings(filename_prefix="university_combined")
    print(f"Saved index to {index_path}")
    print(f"Saved chunks to {chunks_path}")
    
    print("Processing complete!")

if __name__ == "__main__":
    main() 