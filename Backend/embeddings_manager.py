import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util

class EmbeddingsManager:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        """
        Initialize the embeddings manager with the specified model
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = None
        self.embeddings_folder = "embeddings"
        self.relevance_threshold = 0.65  # Minimum similarity score for relevance
        
        # Create embeddings folder if it doesn't exist
        if not os.path.exists(self.embeddings_folder):
            os.makedirs(self.embeddings_folder)
    
    def create_embeddings(self, chunks):
        """
        Create embeddings for text chunks and build FAISS index
        
        Args:
            chunks: List of dictionaries with text and metadata
        """
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        # Create FAISS index - using IndexFlatIP for cosine similarity
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
        return embeddings
    
    def save_embeddings(self, filename_prefix="university_combined"):
        """
        Save the embeddings and chunks to disk
        
        Args:
            filename_prefix: Prefix for the saved files
        """
        if self.index is None or self.chunks is None:
            raise ValueError("No embeddings or chunks to save")
        
        # Save the FAISS index
        index_path = os.path.join(self.embeddings_folder, f"{filename_prefix}_index.faiss")
        faiss.write_index(self.index, index_path)
        
        # Save the chunks
        chunks_path = os.path.join(self.embeddings_folder, f"{filename_prefix}_chunks.pkl")
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        
        return index_path, chunks_path
    
    def load_embeddings(self, filename_prefix="university_combined"):
        """
        Load embeddings and chunks from disk
        
        Args:
            filename_prefix: Prefix for the saved files
            
        Returns:
            True if successful, False otherwise
        """
        index_path = os.path.join(self.embeddings_folder, f"{filename_prefix}_index.faiss")
        chunks_path = os.path.join(self.embeddings_folder, f"{filename_prefix}_chunks.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            return False
        
        # Load the FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load the chunks
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        
        return True
    
    def combine_embeddings(self, sources):
        """
        Combine embeddings from multiple sources
        
        Args:
            sources: List of filename prefixes to combine
            
        Returns:
            True if successful, False otherwise
        """
        all_chunks = []
        
        # Load chunks from each source
        for source in sources:
            chunks_path = os.path.join(self.embeddings_folder, f"{source}_chunks.pkl")
            if not os.path.exists(chunks_path):
                return False
            
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
                all_chunks.extend(chunks)
        
        # Create new embeddings from combined chunks
        self.create_embeddings(all_chunks)
        
        # Save combined embeddings
        self.save_embeddings(filename_prefix="university_combined")
        
        return True
    
    def filter_relevant_chunks(self, query, chunks, scores):
        """
        Filter chunks to remove those that don't meet relevance threshold
        
        Args:
            query: Query text
            chunks: List of chunks to filter
            scores: Similarity scores for each chunk
            
        Returns:
            List of relevant chunks that pass the threshold
        """
        if not chunks:
            return []
            
        relevant_chunks = []
        
        for i, (chunk, score) in enumerate(zip(chunks, scores)):
            # Check if score is above threshold
            if score >= self.relevance_threshold:
                # Add the score to the chunk metadata for debugging
                chunk["metadata"]["relevance_score"] = float(score)
                relevant_chunks.append(chunk)
                
        # If all chunks were filtered out but we had results, 
        # return the top result regardless of score
        if not relevant_chunks and chunks:
            top_idx = np.argmax(scores)
            chunks[top_idx]["metadata"]["relevance_score"] = float(scores[top_idx])
            relevant_chunks = [chunks[top_idx]]
                
        return relevant_chunks
    
    def search_similar_chunks(self, query, k=15):
        """
        Search for chunks most similar to the query
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of relevant chunks with their metadata
        """
        if self.index is None or self.chunks is None:
            raise ValueError("Index or chunks not loaded")
        
        # Encode the query - normalize for cosine similarity
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search the index
        scores, indices = self.index.search(query_embedding, k)
        
        # Get the corresponding chunks
        results = []
        result_scores = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
                result_scores.append(scores[0][i])
        
        # Filter chunks for relevance
        relevant_chunks = self.filter_relevant_chunks(query, results, result_scores)
        
        return relevant_chunks
    
    def get_chunks_by_source_type(self, source_type):
        """
        Get chunks filtered by source type (pdf or web)
        
        Args:
            source_type: Type of source ("pdf" or "web")
            
        Returns:
            List of chunks from the specified source type
        """
        if not self.chunks:
            return []
        
        return [chunk for chunk in self.chunks if chunk["metadata"].get("type") == source_type] 