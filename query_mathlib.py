import pickle
import numpy as np
from langchain_openai import OpenAIEmbeddings
from typing import List, Tuple
import json

class MathlibRAG:
    def __init__(self, embeddings_path: str = "mathlib_embeddings.pkl", tree_path: str = "mathlib_tree.json"):
        """Initialize the RAG system with saved embeddings"""
        print("Loading embeddings...")
        with open(embeddings_path, 'rb') as f:
            self.data = pickle.load(f)
        
        self.documents = self.data['documents']
        self.embeddings = np.array(self.data['embeddings'])
        self.metadata = self.data['metadata']
        
        # Load file tree
        with open(tree_path, 'r') as f:
            self.file_tree = json.load(f)
        
        # Initialize OpenAI embeddings for queries
        self.embedding_model = OpenAIEmbeddings(
            model=self.metadata['embedding_model']
        )
        
        print(f"Loaded {len(self.documents)} document chunks")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float, dict]]:
        """
        Search for similar documents using cosine similarity
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of tuples (content, similarity_score, metadata)
        """
        # Get query embedding
        query_embedding = np.array(self.embedding_model.embed_query(query))
        
        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top k results
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            similarity = similarities[idx]
            results.append((doc.page_content, similarity, doc.metadata))
        
        return results
    
    def search_by_file(self, filename: str) -> List[str]:
        """
        Get all chunks from a specific file
        
        Args:
            filename: Name of the file to search for
            
        Returns:
            List of document contents from that file
        """
        results = []
        for doc in self.documents:
            if doc.metadata.get('file_name') == filename:
                results.append(doc.page_content)
        return results
    
    def get_file_info(self):
        """Get information about the loaded dataset"""
        files = set()
        for doc in self.documents:
            files.add(doc.metadata.get('file_name', 'Unknown'))
        
        return {
            'total_chunks': len(self.documents),
            'unique_files': len(files),
            'embedding_model': self.metadata['embedding_model'],
            'chunk_size': self.metadata['chunk_size'],
            'chunk_overlap': self.metadata['chunk_overlap']
        }

def main():
    # Example usage
    rag = MathlibRAG()
    
    # Get dataset info
    info = rag.get_file_info()
    print(f"Dataset info: {info}")
    
    # Example query
    query = "topological space definition"
    results = rag.similarity_search(query, k=3)
    
    print(f"\nTop 3 results for '{query}':")
    for i, (content, score, metadata) in enumerate(results):
        print(f"\n--- Result {i+1} (Score: {score:.4f}) ---")
        print(f"File: {metadata['file_name']}")
        print(f"Content preview: {content[:200]}...")

if __name__ == "__main__":
    main() 