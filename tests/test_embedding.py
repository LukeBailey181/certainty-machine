"""
Test script to demonstrate text-to-vector transformation
Shows what text looks like before embedding and what the vector looks like after
"""

import os
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

def test_embedding_process():
    """Test the embedding process with sample Lean code"""
    
    # Sample Lean code (similar to what would be in mathlib)
    with open("mathlib4/Mathlib/CategoryTheory/Bicategory/Functor/Oplax.lean", "r") as f:
        sample_lean_code = f.read()

    print("=" * 60)
    print("EMBEDDING TEST DEMONSTRATION")
    print("=" * 60)
    
    # Step 1: Show original text
    print("\n1. ORIGINAL TEXT:")
    print("-" * 40)
    print(sample_lean_code)
    print(f"\nText length: {len(sample_lean_code)} characters")
    
    # Step 2: Show chunking process (same as in process_mathlib.py)
    print("\n2. CHUNKING PROCESS:")
    print("-" * 40)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True
    )
    
    chunks = splitter.split_text(sample_lean_code)
    print(f"Number of chunks created: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"Length: {len(chunk)} characters")
        print(f"Content preview: {repr(chunk[:100])}...")
    
    # Step 3: Initialize OpenAI embeddings
    print("\n3. INITIALIZING EMBEDDING MODEL:")
    print("-" * 40)

    import dotenv
    dotenv.load_dotenv()
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large", 
        api_key=os.getenv("OPENAI_API_KEY")
    )
    print("Model: text-embedding-3-large")
    
    # Step 4: Create embeddings
    print("\n4. CREATING EMBEDDINGS:")
    print("-" * 40)
    
    print("Calling OpenAI API to create embeddings...")
    chunk_embeddings = embeddings.embed_documents(chunks)
    
    print(f"Number of embedding vectors created: {len(chunk_embeddings)}")
    
    # Step 5: Show embedding details
    print("\n5. EMBEDDING DETAILS:")
    print("-" * 40)
    
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        print(f"\nChunk {i+1} Embedding:")
        print(f"  Text length: {len(chunk)} characters")
        print(f"  Vector dimension: {len(embedding)}")
        print(f"  Vector type: {type(embedding)}")
        print(f"  Vector dtype: {type(embedding[0])}")
        print(f"  Vector range: [{min(embedding):.6f}, {max(embedding):.6f}]")
        print(f"  Vector norm: {np.linalg.norm(embedding):.6f}")
        print(f"  First 5 values: {embedding[:5]}")
        print(f"  Last 5 values: {embedding[-5:]}")
    
    # Step 6: Test similarity
    print("\n6. SIMILARITY TEST:")
    print("-" * 40)
    
    # Create a query embedding
    query = "topological space definition"
    query_embedding = embeddings.embed_query(query)
    
    print(f"Query: '{query}'")
    print(f"Query embedding dimension: {len(query_embedding)}")
    
    # Calculate similarity with each chunk
    for i, chunk_embedding in enumerate(chunk_embeddings):
        similarity = np.dot(chunk_embedding, query_embedding) / (
            np.linalg.norm(chunk_embedding) * np.linalg.norm(query_embedding)
        )
        print(f"Similarity with chunk {i+1}: {similarity:.6f}")
    
    # Step 7: Show storage format
    print("\n7. STORAGE FORMAT:")
    print("-" * 40)
    
    # This is what gets stored in the pickle file
    storage_format = {
        'text_chunks': chunks,
        'embeddings': chunk_embeddings,
        'metadata': {
            'model': 'text-embedding-3-large',
            'chunk_size': 1000,
            'chunk_overlap': 200
        }
    }
    
    print("Data structure stored in pickle file:")
    print(f"  - text_chunks: {len(storage_format['text_chunks'])} items")
    print(f"  - embeddings: {len(storage_format['embeddings'])} vectors")
    print(f"  - metadata: {storage_format['metadata']}")
    
    print("\n" + "=" * 60)
    print("EMBEDDING TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    # # Check if OpenAI API key is set
    # if not os.environ.get("OPENAI_API_KEY") and not dotenv.get_key(".env", "OPENAI_API_KEY"):
    #     print("ERROR: Please set your OPENAI_API_KEY environment variable")
    #     print("Example: export OPENAI_API_KEY='your-api-key-here'")
    #     exit(1)
    
    test_embedding_process() 