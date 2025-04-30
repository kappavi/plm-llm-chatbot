import faiss
import numpy as np
from typing import Dict, List, Optional
import json
import os

class VectorRetriever:
    def __init__(self, index_path: Optional[str] = None):
        self.dimension = 1280  # Dimension of ESM embeddings
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        
        if index_path and os.path.exists(index_path):
            self.load_index(index_path)
            
    def add_documents(self, documents: List[Dict]):
        """
        Add documents to the index
        """
        embeddings = np.array([doc['embeddings'] for doc in documents])
        self.index.add(embeddings)
        self.documents.extend(documents)
        
    def retrieve(self, query: str, context: Dict, k: int = 5) -> Dict:
        """
        Retrieve relevant documents based on the query and context
        """
        # Use the protein embeddings from context
        query_embedding = context['embeddings']
        
        # Search for similar documents
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'),
            k
        )
        
        # Get relevant documents
        relevant_docs = [self.documents[i] for i in indices[0]]
        
        # Format the response
        return {
            'relevant_documents': relevant_docs,
            'distances': distances[0].tolist()
        }
        
    def save_index(self, path: str):
        """
        Save the index and documents to disk
        """
        # Save FAISS index
        faiss.write_index(self.index, f"{path}/index.faiss")
        
        # Save documents
        with open(f"{path}/documents.json", 'w') as f:
            json.dump(self.documents, f)
            
    def load_index(self, path: str):
        """
        Load the index and documents from disk
        """
        # Load FAISS index
        self.index = faiss.read_index(f"{path}/index.faiss")
        
        # Load documents
        with open(f"{path}/documents.json", 'r') as f:
            self.documents = json.load(f) 