"""
Script to inspect the saved embeddings pickle file
Shows the actual structure and content of the saved data
"""

import pickle
import numpy as np
import os

def inspect_embeddings(embeddings_path: str = "mathlib_embeddings.pkl"):
    """Inspect the contents of the saved embeddings file"""
    
    if not os.path.exists(embeddings_path):
        print(f"ERROR: File {embeddings_path} not found!")
        print("Run process_mathlib.py first to create the embeddings file.")
        return
    
    print("=" * 60)
    print("INSPECTING SAVED EMBEDDINGS FILE")
    print("=" * 60)
    
    # Load the pickle file
    print(f"\nLoading {embeddings_path}...")
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    
    print("✓ File loaded successfully!")
    
    # Show top-level structure
    print(f"\n1. TOP-LEVEL DATA STRUCTURE:")
    print("-" * 40)
    print(f"Keys in data: {list(data.keys())}")
    print(f"Data type: {type(data)}")
    
    # Show metadata
    if 'metadata' in data:
        print(f"\n2. METADATA:")
        print("-" * 40)
        for key, value in data['metadata'].items():
            print(f"  {key}: {value}")
    
    # Show documents info
    if 'documents' in data:
        print(f"\n3. DOCUMENTS:")
        print("-" * 40)
        documents = data['documents']
        print(f"  Number of documents: {len(documents)}")
        print(f"  Document type: {type(documents[0]) if documents else 'N/A'}")
        
        if documents:
            # Show first document details
            first_doc = documents[0]
            print(f"  First document:")
            print(f"    Content length: {len(first_doc.page_content)} characters")
            print(f"    Metadata keys: {list(first_doc.metadata.keys()) if hasattr(first_doc, 'metadata') else 'N/A'}")
            print(f"    Content preview: {repr(first_doc.page_content[:100])}...")
    
    # Show embeddings info
    if 'embeddings' in data:
        print(f"\n4. EMBEDDINGS:")
        print("-" * 40)
        embeddings = data['embeddings']
        print(f"  Number of embeddings: {len(embeddings)}")
        print(f"  Embedding type: {type(embeddings[0]) if embeddings else 'N/A'}")
        
        if embeddings:
            first_embedding = np.array(embeddings[0])
            print(f"  First embedding:")
            print(f"    Dimension: {len(first_embedding)}")
            print(f"    Data type: {first_embedding.dtype}")
            print(f"    Shape: {first_embedding.shape}")
            print(f"    Range: [{first_embedding.min():.6f}, {first_embedding.max():.6f}]")
            print(f"    Norm: {np.linalg.norm(first_embedding):.6f}")
            print(f"    First 5 values: {first_embedding[:5]}")
            print(f"    Last 5 values: {first_embedding[-5:]}")
    
    # Show some example document-embedding pairs
    if 'documents' in data and 'embeddings' in data:
        print(f"\n5. DOCUMENT-EMBEDDING PAIRS EXAMPLES:")
        print("-" * 40)
        
        documents = data['documents']
        embeddings = data['embeddings']
        
        # Show first 3 pairs
        for i in range(min(3, len(documents))):
            doc = documents[i]
            emb = np.array(embeddings[i])
            
            print(f"\nPair {i+1}:")
            print(f"  File: {doc.metadata.get('file_name', 'Unknown')}")
            print(f"  Text (first 80 chars): {repr(doc.page_content[:80])}...")
            print(f"  Embedding summary: {len(emb)}-dim vector, norm={np.linalg.norm(emb):.4f}")
    
    # File size analysis
    file_size = os.path.getsize(embeddings_path)
    print(f"\n6. FILE STATISTICS:")
    print("-" * 40)
    print(f"  File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    if 'documents' in data and 'embeddings' in data:
        num_docs = len(data['documents'])
        num_embs = len(data['embeddings'])
        avg_size_per_doc = file_size / num_docs if num_docs > 0 else 0
        
        print(f"  Documents: {num_docs:,}")
        print(f"  Embeddings: {num_embs:,}")
        print(f"  Average size per document: {avg_size_per_doc:.0f} bytes")
    
    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    inspect_embeddings() 