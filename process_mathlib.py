import os
from pathlib import Path
from typing import List, Dict
import json
import pickle
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
import numpy as np

def get_file_tree(start_path: str) -> Dict:
    """
    Create a dictionary representing the file tree structure
    """
    tree = {}
    for root, dirs, files in tqdm(os.walk(start_path), desc="Building file tree"):
        current = tree
        path = os.path.relpath(root, start_path)
        if path != '.':
            for part in path.split(os.sep):
                current = current.setdefault(part, {})
        for file in files:
            if file.endswith('.lean'):
                current[file] = None
    return tree

def save_file_tree(tree: Dict, output_file: str):
    """
    Save the file tree structure to a JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(tree, f, indent=2)

def process_lean_file(file_path: str) -> List[Document]:
    """
    Process a single Lean file and return a list of document chunks
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Create a text splitter that respects Lean code structure
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
        keep_separator=True
    )

    # Create metadata for the document
    metadata = {
        "source": file_path,
        "file_name": os.path.basename(file_path)
    }

    # Split the document
    texts = splitter.create_documents(
        texts=[content],
        metadatas=[metadata]
    )
    
    return texts

def main():
    # Configure paths
    mathlib_path = "mathlib4/Mathlib"
    embeddings_path = "mathlib_embeddings.pkl"
    tree_path = "mathlib_tree.json"

    import dotenv
    dotenv.load_dotenv()

    # Initialize embedding model
    print("Initializing OpenAI embedding model...")
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-3-large"
    )

    # Get and save file tree
    print("Building and saving file tree...")
    file_tree = get_file_tree(mathlib_path)
    save_file_tree(file_tree, tree_path)

    # Process all Lean files
    all_files = []
    print("Finding Lean files...")
    for root, _, files in tqdm(os.walk(mathlib_path), desc="Finding .lean files"):
        for file in files:
            if file.endswith('.lean'):
                all_files.append(os.path.join(root, file))

    # Process files with progress bar
    all_documents = []
    for file_path in tqdm(all_files, desc="Processing files"):
        try:
            documents = process_lean_file(file_path)
            all_documents.extend(documents)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    # Create embeddings for all documents
    print("Creating embeddings...")
    document_texts = [doc.page_content for doc in all_documents]
    document_embeddings = []
    
    # Process in batches to avoid API rate limits
    batch_size = 100
    for i in tqdm(range(0, len(document_texts), batch_size), desc="Creating embeddings"):
        batch_texts = document_texts[i:i + batch_size]
        batch_embeddings = embeddings.embed_documents(batch_texts)
        document_embeddings.extend(batch_embeddings)
    
    # Prepare data structure for saving
    embedding_data = {
        'documents': all_documents,
        'embeddings': document_embeddings,
        'metadata': {
            'total_documents': len(all_documents),
            'embedding_model': 'text-embedding-3-large',
            'chunk_size': 1000,
            'chunk_overlap': 200
        }
    }
    
    # Save embeddings to pickle file
    print("Saving embeddings to pickle file...")
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embedding_data, f)

    print(f"Successfully processed {len(all_documents)} document chunks")
    print(f"File tree saved to: {tree_path}")
    print(f"Embeddings saved to: {embeddings_path}")

if __name__ == "__main__":
    main()
